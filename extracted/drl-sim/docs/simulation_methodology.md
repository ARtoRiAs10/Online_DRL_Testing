# Simulation methodology: what is run, under which conditions, and how each number is computed

This document explains

1. **what** the simulator models (the environment shared by every algorithm),
2. **how each algorithm** of the paper and of the multi-resource proposal is implemented, line by line,
3. **under which experimental conditions** every experiment is run (workload, sizes, seeds, budgets),
4. **how every reported quantity** (latency, cost, hits, β_up/β_down, Λ, utilization, regret, ratio γ) is computed
   and why the computation is the correct one,
5. **how the code follows the two documents to the letter** — a claim → code mapping table,
6. **where the documents are silent** and which assumption was made instead (A1–A4, and the known
   off-by-one in the paper's OPT formula).

Notation follows the paper (*Edge Service Caching With Delayed Hits: Minimizing Latency and
Reducing Cost*, Deka & Sukapuram, IEEE TCC 2025) and the proposal (*Randomized DRL with
Multi-Resource (CPU, RAM, Disk) Capacity: From Slots to Resource Vectors*).

---

## 1. The environment (shared by all algorithms)

`sim/engine.py` is an event-driven edge node. Nothing in it knows about any algorithm: it only
implements the environment described in Sec. 1.1 of the paper, so that every algorithm sees
exactly the same requests, the same clock and the same accounting.

### 1.1 Catalogue (`sim/catalog.py`)

For every service `n` the catalogue stores

| symbol | meaning | how it is computed (`from_physical`) |
|---|---|---|
| `l_n` | round-trip forwarding latency to the cloud | `rtt + req/U` (RTT + push the request over the uplink) |
| `M_n` | download time (request + transfer) | `rtt/2 + req/U + S_n·8000/D` — i.e. eq. (26) of the proposal: `M_n = M^req_n + S_n/D` with `M^req_n = rtt/2 + req/U` |
| `a_n` | normalized demand vector (CPU, RAM, disk) | `raw_n / caps`, `caps = max(cap_mult·mean(raw), max(raw))` per resource, so every `a_{n,r} ∈ (0,1]` |
| `raw_n`, `caps` | physical demands and capacities | kept for the "capacity × mean demand" sweeps of exp04 |

The three resources are used exactly as in the proposal: **CPU** (cores), **RAM** (GB),
**disk** (MB, converted with `1 MB = 8000 kbit` so that `S/D` is in ms).

### 1.2 Traces (`sim/trace.py`)

* `synthetic_world(N, T, zipf, rate, seed, imbalance, ...)`
  Zipf popularity over `N` services (exponent 1.2 by default), Poisson arrivals
  (rate 0.5 req/ms ⇒ mean inter-arrival 2 ms), log-normal raw demands.
  `imbalance > 1` multiplies one randomly chosen resource of every service by that factor, which
  is what makes a *slot* (dominant-resource) reduction wasteful — the phenomenon the proposal
  studies in Sec. 2.
* `load_trace_csv(...)` — generic loader for the real Google / SHARCNet CSVs (`data/README.md`
  describes the expected column layout). Zeros in CPU/RAM/disk are replaced by the column median
  (as in the paper), the resources of a service are taken from its first occurrence and the
  timestamps are made strictly increasing.
* `large_sequence(k, n_reps, ...)` — the **adversarial sequence of the proof of Theorem 1**:
  `BS1 | BS2 | BS3 | MS` repeated `n` times, where `BS_i` holds `k/3` services dominated by disk /
  RAM / CPU respectively and `MS` needs `(k/3)×` the large demand of *every* resource, so caching
  `MS` evicts the whole cache. It also returns the paper's optimum `OPT = (k + 2n)·l` (eq. 17).

### 1.3 The request loop (Sec. 1.1 of the paper)

For request `t` of service `n = s_t` the engine performs, in this order:

1. **complete** every download whose completion time `≤ t` (heap of `(completion_time, service)`);
   the cache policy is notified (`on_complete`) and the *miss epoch* of that service is reset;
2. **classify** the request:

   | state | event | latency | counters |
   |---|---|---|---|
   | `n` cached | **Hit** | `0` | `hits` |
   | download of `n` in flight, remaining time `rem ≤ l_n` | **Delayed hit (D)** | `min(l_n, rem) = rem` | `delayed_hits`, `buf_srv[n]++` |
   | download of `n` in flight, `rem > l_n` | **Request forwarding (R)** | `l_n` | `forwards`, `fwd_srv[n]++`, `fwd_inflight++` |
   | not cached, no download in flight | **Miss** | `l_n` (unless a download is triggered now) | `forwards` |

   This is the paper's rule "if the remaining download time is below `l_n` the request is buffered,
   otherwise it is forwarded, so the latency is `min(l_n, rem)`".
3. **download decision** on a miss (only on a miss, i.e. at the *first* request of a miss epoch):

   * the epoch counters are `T_i = t − miss_start` (time since the epoch began) and
     `L_i = l_n · (#requests of n in this epoch)`;
   * the **threshold** `θ_n` is drawn: deterministic `θ_n = θ·M_n` for Online-DRL,
     random `θ_n ~ f(θ) = e^{θ/M_n} / (M_n (e−1))` on `[0, M_n]` for R3;
   * a download starts iff `T_i ≥ θ_n` **or** `L_i ≥ θ_n` (Algorithm 1 of the paper), the policy
     allows it (`allow_download`), the budget is not exceeded (`cost + M_n ≤ B_max`) and the
     (budget-aware) trigger probability accepts it;
   * the download completes at `t + M_n`; its cost `M_n` is added to the total **cost**;
4. **accounting**: `latency` (sum of per-request latencies), `cost` (sum of `M_n` of the started
   downloads), `hits/delayed_hits/forwards/downloads`, per-request latency vector `lat_per_req`,
   per-step resource utilization `util[t] = Σ_{n cached at t} a_n` (3 components).

**Correctness of the delayed-hit rule.** The engine never uses `l_n` alone when a download is in
flight: the latency is literally `min(l_n, rem)` and the *event* (buffered vs. forwarded) is
decided by the same comparison, so the two counters (`delayed_hits` vs. `forwards`) partition the
in-flight requests exactly. A service can have at most one download in flight
(`self.pending` is a dict keyed by the service), which is also a constraint of the offline MILP.

**Cost vs. latency are kept separate** (as the paper insists): latency counts only what the user
waits, cost counts only download time. A hit after a download contributes `0` latency but the
download that made it possible was already charged to the cost.

---

## 2. The algorithms

### 2.1 Online-DRL and LL-RC (paper, Algorithms 1 + 2) — `sim/landlord.py`, `sim/algos.py`

`LandlordPolicy` is Algorithm 2 of the paper, with the paper's *multi-resource* admission test:

* every cached service `f` has a **credit** `credit[f]` (initialised to `M_f`, i.e. the download
  cost, and reset to `M_f` on every hit — line 25 of Algorithm 2);
* admission of a downloaded service `g` is allowed iff, for **all three resources**,
  `Σ_f a_{f,r} + a_{g,r} ≤ 1` **and** the number of cached services stays `≤ C`;
* otherwise credits are decreased proportionally to the size of each cached service
  (`d = min_f credit[f]/size[f]`, `credit[f] −= d·size[f]`), the services whose credit reaches `0`
  become *eligible*, and eligible services are evicted in increasing order of
  `Σ_r max(a_{f,r} − need_r, 0)` (the paper's "prefer the item whose resources beyond the unmet
  need are smallest" rule) until the new service fits.

`online_drl(cat, tr, C, theta=1.0)` uses the deterministic threshold; `theta = 0` degenerates to
"download on the first request of every miss epoch", which is exactly the **LL-RC** baseline
(`ll_rc`). exp02 asserts numerically that `online_drl(..., 0)` and `ll_rc` produce the same cost.

### 2.2 R3 — randomized download threshold (proposal)

`RandThreshold` draws the threshold from `f(θ) = e^{θ/M_n} / (M_n (e−1))` on `[0, M_n]`.
Inverse CDF: `F(θ) = (e^{θ/M_n} − 1)/(e−1)` ⇒ `θ = M_n·ln(1 + (e−1)U)`; `draw` returns the
fraction of `M_n` and the engine multiplies by `M_n`. The density integrates to one on `[0, M_n]`,
which is checked numerically in exp06.

### 2.3 R1 + R2 — fractional OGD state and randomized rounding (proposal) — `sim/rand_drl.py`

`FracPolicy` keeps a fractional cache state `p ∈ D_vec` with

```
D_vec = { p ∈ [0,1]^N : Σ_n p_n ≤ C ,  Σ_n a_{n,r} p_n ≤ 1  ∀r }      (proposal eq. 8)
```

* **R1 — projected OGD step (eqs. 19–22).** The per-request surrogate gradient is
  `g_t = (1 − q_t)·l_{n_t} + q_t·min(l_{n_t}, dm_t)`, where `q_t ∈ {0,1}` says "a download of the
  requested service is in flight" and `dm_t` is the remaining download time. The step is
  `p ← Π_{D_vec}( p + η·g_t e_{n_t} )`. The Euclidean projection is computed by **accelerated
  dual ascent** on the multipliers `ρ` with `p(ρ) = clip(z − A^Tρ, 0, 1)` (eq. 22) and a final
  scaling `p ← p / max(1, max_r A p)` (eq. 25) that makes the result exactly feasible.
  `sim/projection.py::project_exact` solves the same QP with **cvxpy**; exp07 checks that the
  two agree to `~1e-9` and that the KKT residual `|ρ^T(Ap − 1)|` is at machine precision.
* **R2 — randomized rounding with repair (Sec. 3.4).** `K` persistent uniform vectors
  `U_k ∈ [0,1]^N` are drawn once; the sample paths are `s_k = 1[U_k < p]`. Because the `U_k` are
  persistent, `E[s_k] = p` and `E|s_{k,t} − s_{k,t−1}| = |p_t − p_{t−1}|_1` — the coupling the
  proposal asks for. A path that violates `A s_k ≤ 1` is repaired by evicting the services with
  the smallest `p_{n,t}/max_r a_{n,r}` (or, with `repair_order="alignment"`, the ones that fit the
  residual capacity best) until the path is feasible. The *serving* path `k*` is drawn once
  (or every `resample_every` steps); the **real** cache is
  `X_t = (downloaded services) ∩ s_{k*,t}`: a sampled "cached" service that was never downloaded
  is **not** a hit, and a service is dropped as soon as the serving path switches it off.
* **Step size.** `η = √N / (l_max √T)` (Proposition 1 of the proposal), computed from the
  horizon `T` known to the policy.

### 2.4 Budget-aware trigger (goal G2)

`FracPolicy.download_prob` lowers the probability of *starting* a download to
`p = clip( budget_left / (M_n · requests_left), 0, 1 )`. With at most one download per request,
`E[#downloads] ≤ requests_left`, hence the expected download cost stays inside the remaining
budget. The engine additionally refuses a download that would break the hard budget
(`cost + M_n > B_max`).

### 2.5 Offline OPT (reference) — `sim/opt.py`

A MILP over `x_{n,t} ∈ {0,1}` ("service `n` is cached at request `t`") and `u_s ∈ {0,1}`
("a download is started at request `s`", charged `M_{s}` once), solved with **HiGHS**
(`scipy.optimize.milp`, no Gurobi):

* cache dynamics `x_{n,t} ≤ x_{n,t−1} + Σ_{s: comp(s)=t, n_s=n} u_s` (evictions are free);
* `x` and "download in flight" are mutually exclusive; at most one download in flight per service;
* per-step capacity: three resource constraints `Σ_n a_{n,r} x_{n,t} ≤ 1` and the cardinality
  constraint `Σ_n x_{n,t} ≤ C`;
* optional budget `Σ_s M_{s} u_s ≤ B_max`;
* objective: `Σ_t l_{n_t}` minus the savings obtained by caching or by delaying (`min(l, rem)`).

The paper's constraints (4)–(6) write `(1 − x)`; this repo uses `x` (the sensible reading, as in
the proposal) — see the README.

---

## 3. Experimental conditions (per experiment)

All experiments share `experiments/common.py`:
`--quick` gives small sizes for a smoke run, `--seeds` sets the number of repetitions,
`--out` chooses the output directory. Default (full) sizes: `N = 40`, `T = 8000`, `seeds = 10`.

| script | purpose | workload | sizes | seeds | figures / data |
|---|---|---|---|---|---|
| `exp01_baseline_vs_llrc` | paper's headline: Online-DRL vs LL-RC (latency ↑, cost ↓) | Zipf/Poisson synthetic, `C = 15` | 10 seeds | 10 | `exp01_baseline_vs_llrc.{png,json}` |
| `exp02_theta` | `θ` trades cost against latency; `θ = 0` ≡ LL-RC | one Zipf trace, `C = 15` | `θ ∈ {0,.25,.5,1,2,4,8,16}` | 1 | `exp02_theta.*` |
| `exp03_bandwidth` | delayed hits vs uplink `U` and downlink `D` | Zipf, `C = 15` | `U ∈ {20..400}` Mbps, `D ∈ {250..4000}` Mbps | 1 | `exp03_bandwidth.*` |
| `exp04_capacity` | latency vs slot count `C` and vs the CPU/RAM/disk limits | Zipf | `C ∈ {1..32}`, `cap_mult ∈ {2..16}` | 1 | `exp04_capacity.*` |
| `exp05_large_sequence` | Theorem 1: `γ` on the adversarial Large Sequence | `large_sequence(k, 2k)` | `k ∈ {3,6,9,12,15,18,21,24}` | 10 | `exp05_large_sequence.*` |
| `exp06_ski_rental` | R3 rent-or-buy: `2` → `e/(e−1)` | analytical + Monte-Carlo | 200k threshold draws | — | `exp06_ski_rental.*` |
| `exp07_projection` | dual-ascent projection ≡ exact QP | random `A`, `z` | 60 random instances | — | `exp07_projection.*` |
| `exp08_feasibility_budget` | G1 feasibility (with/without repair), G2 budget | Zipf, `imbalance = 3`, `C = 12` | 10 seeds | 10 | `exp08_feasibility_budget.*` |
| `exp09_regret` | G3 regret: exogenous / endogenous / literal | Zipf with a popularity shift | `T ∈ {500..8000}` | 1 | `exp09_regret.*` |
| `exp10_expected_ratio` | G4 expected ratio vs the MILP optimum | small traces (`T = 120`), `C = 3` | 5 traces | 10 | `exp10_expected_ratio.*` |
| `exp11_slot_reduction` | G5 slot reductions: `Λ`, stranded capacity, violations | Zipf, `imbalance = 4`, `C = 40` | 10 seeds | 10 | `exp11_slot_reduction.*` |
| `exp12_multiresource` | multi-resource comparison (proposal Sec. 4) | balanced + imbalanced Zipf, `C = 12` | 2 workloads | 10 | `exp12_multiresource_{balanced,imbalanced}.*` |
| `exp13_gamma_complexity` | is `γ = O(k(1+3β_up M_x))` reached / improved by randomization? | (i) Large Sequence, (ii) Zipf vs MILP OPT | `k ∈ {3..24}` / `k ∈ {2..12}` | 10 | `exp13_gamma_complexity*.{png,json}` |
| `exp14_time_complexity` | computational cost of each algorithm | Zipf | `N ∈ {20..320}`, `C ∈ {4..64}` | 3 | `exp14_time_complexity.*` |
| `exp15_budget_regret` | G2 budget + G3 regret with the multi-resource projection | Zipf, `imbalance = 3`, `C = 12` | 3 budget levels | 10 | `exp15_budget_regret.*` |

---

## 4. How each reported quantity is computed (and why it is right)

| quantity | definition in the code | why it is the correct one |
|---|---|---|
| **latency** `L` | `Σ_t lat_t` with `lat_t = 0` (hit), `min(l_n, rem)` (delayed hit) or `l_n` (forward) | exactly the objective of the paper (3): the user-visible delay |
| **cost** | `Σ M_n` over the *started* downloads | the paper's cost = download time; request forwarding is charged through the latency, not the cost |
| **hit / delayed hit / forward** | three disjoint counters | the paper's taxonomy (1)(2)(3) of Sec. 1.1 |
| **downloads** | number of started downloads | needed for the budget and for the "wasted downloads" diagnostic |
| **`β_up`, `β_down`** | per service `β_i = fwd_i / M_i` where `fwd_i` = requests of service `i` that were forwarded *while a download of `i` was in flight*; `β_up = max_i β_i` (used with `M_x` in the O-bound) and `β_down = mean_i β_i` (used with `M_min` in the Ω-bound) | the paper's β is "the fraction of `M_x` (`M_min`) requests that are forwarded to the cloud instead of buffered"; the counters are exact because the engine tags every in-flight request as buffered or forwarded |
| **Theorem-1 bounds** | `O_bound = k(1 + 3β_up M_x)`, `Ω_bound = k(1 + 3β_down M_min)` | the paper's Theorem 1 evaluated with the *measured* β, `M_x = max_n M_n`, `M_min = min_n M_n` |
| **competitive ratio `γ`** | `ALG(I) / OPT(I)`; `OPT = (k+2n)l` on the Large Sequence, `OPT` = MILP optimum elsewhere | the paper's definition; the MILP is exact (HiGHS), `opt_status = 1` means "time limit" ⇒ the ratio is a *lower* bound |
| **growth of `γ(k)`** | least-squares fits: linear `γ = a·k + b`, logarithmic `γ = a·log k + b`, power law `γ = a·k^b` (log-log), with `R²` | decides empirically whether the linear-in-`k` behaviour of Theorem 1 is removed by randomization (it is not a proof) |
| **utilization `U_{r,t}`** | `Σ_{n cached at t} a_{n,r}` (3 resources) | the proposal's per-resource occupancy |
| **`U^max_t`, imbalance `I_t`, stranded `W_t`** | `U^max = max_r U_r`, `I_t = U^max_t − min_r U_r`, `W_t = Σ_r (U^max_t − U_{r,t})` | proposal eqs. (30)–(33): imbalance = how unequal the resource usage is, stranded capacity = how much of the bottleneck resource is unusable by the other resources |
| **violation rate** | fraction of steps with `max_r U_{r,t} > 1 + 1e-9` | goal G1: the *true* multi-resource constraints must hold, not just the reduced ones |
| **latency per unit resource** | `L / Σ_t U^max_t` | proposal eq. (34): latency normalized by the bottleneck resource actually used |
| **`Λ_φ` (theoretical)** | `max_n φ(a_n) / max_r a_{n,r}` | proposal eq. (18): the factor by which the slot reduction `φ` over-charges a service relative to its true bottleneck demand |
| **`Λ̂` (empirical)** | `E[U^max] / E[mean_r U_r]` | the slot view charges the *largest* demand, so the slot occupancy is `U^max` while the balanced usage is the mean; the ratio is the wasted fraction of the bottleneck |
| **raw violation rate of the sample** | `#{k : A s_k > 1} / (K·T)` | how often the *un-repaired* rounding violates the constraints (G1) |
| **surrogate regret (G3)** | `Σ_t F_t(P_t) − min_{P∈D_vec} Σ_t F_t(P)` with `F_t(P) = (1 − p_{n_t})·g_t`; the comparator is the best *fixed* fractional cache, computed exactly by an LP (`scipy.optimize.linprog`) | the proposal's Proposition 1 is stated against the best fixed `P`; the comparator is solved to optimality, so the regret is an upper bound on nothing and a lower bound on nothing — it is the exact regret |
| **Proposition-1 bound** | `√N · l_max · √T` | the diameter of `D_vec` is `≤ √N`, so projected OGD with `η = √N/(l_max√T)` has regret `≤ √N·l_max·√T` |
| **running time** | `1e6 · wall_time / T` µs per request, wall time measured with `time.perf_counter()` around the request loop only | the *algorithmic* cost of each policy, excluding catalogue/trace construction |
| **MILP optimum** | `scipy.optimize.milp` with HiGHS, `mip_rel_gap = 1e-4`, `time_limit = 60 s` | an exact reference; `status = 1` (time limit) ⇒ OPT is an upper bound ⇒ the reported ratios are lower bounds |

---

## 5. Fidelity to the two documents (claim → code)

| # | statement | where it comes from | where it is implemented | where it is checked |
|---|---|---|---|---|
| P1 | A request is a Hit, a Delayed Hit (buffered) or a Request Forwarded, with latency `min(l, rem)` in the in-flight case | paper Sec. 1.1 | `sim/engine.py` request loop | `tests/test_core.py` |
| P2 | A download starts when `T_i ≥ θ M_i` or `L_i ≥ θ M_i` | paper Alg. 1 | `sim/engine.py` | `tests/test_core.py`, exp02 |
| P3 | Landlord: credits `M_i`, reset on hit, proportional decrease, eviction by "resources beyond the unmet need" | paper Alg. 2 | `sim/landlord.py` | `tests/test_core.py` |
| P4 | Landlord admission must respect the three resources and the cardinality `C` | paper Sec. 1.2 / proposal Sec. 1 | `sim/landlord.py::_violated` | exp12, exp04 |
| P5 | `θ = 0` degenerates to LL-RC | paper Sec. III | `sim/algos.py::ll_rc` | exp02 (assertion) |
| P6 | `γ ∈ O(k(1+3β_up M_x))` and `γ ∈ Ω(k(1+3β_down M_min))` | paper Thm. 1 | `sim/analysis.py::ratio_bounds` | exp13 |
| P7 | The adversarial Large Sequence has `OPT = (k+2n)l` | paper eq. 17 | `sim/trace.py::large_sequence` | `tests/test_core.py` (MILP vs. `(k+2n)l − l`) |
| R1 | Fractional state `p ∈ D_vec` updated by projected OGD | proposal eqs. (8), (19)–(22) | `sim/rand_drl.py::_ogd`, `sim/projection.py` | exp07, exp09, exp15 |
| R2 | Randomized rounding with repair; `E[s] = p`, `E|Δs| = |Δp|` | proposal Sec. 3.4 | `sim/rand_drl.py::_sample`, `_repair_row` | exp08 (raw vs repaired violations) |
| R3 | Randomized download threshold `f(θ) = e^{θ/M_n}/(M_n(e−1))`, ratio `e/(e−1)` | proposal Sec. 3.3 | `sim/engine.py::RandThreshold` | exp06 |
| R4 | Slot reductions `φ`: dominant, linear `α`, `ℓ_p`, shadow price; `Λ_φ` | proposal Sec. 2, eq. (18) | `sim/slots.py` | exp11, exp12 |
| R5 | Vector formulation vs slots: utilization, imbalance, stranded capacity, violations | proposal Sec. 4.4, eqs. (30)–(34) | `sim/metrics.py`, `sim/slots.py::lambda_estimate` | exp11, exp12 |
| R6 | Expected download budget `E[Σ M_n] ≤ B_max` | proposal goal G2 | `sim/rand_drl.py::download_prob` + engine budget check | exp08, exp15 |
| R7 | Regret `o(T)` w.r.t. the best fixed fractional cache, bound `√N l_max √T` | proposal goal G3, Prop. 1 | `sim/regret.py` | exp09, exp15 |
| R8 | Offline OPT as a MILP with delayed hits, forwarding, multi-resource capacity and a budget | proposal Sec. 1.2 | `sim/opt.py` | exp10, exp13 (family 2) |

---

## 6. Where the documents are silent — the assumptions we made

The proposal PDF that should have contained the full statement
(`randomized_drl_problem_statement.pdf`) is **corrupted** (its binary streams were destroyed by a
lossy text round-trip; no PDF reader can open it). Everything below is therefore an explicit,
documented modelling decision; all of them are switchable in `sim/rand_drl.py`.

* **A1 — `q_{n,t}`.** The documents give no update rule for the "download in flight" indicator.
  We use `q_{n,t} = 1` iff the node itself has a download of the requested service in flight
  (exogenous to `p`). This is what makes the OGD gradient well defined.
* **A2 — rounding.** `K` persistent uniform vectors, `s = 1[U < p]`. Marginals are exact and
  `E|s_t − s_{t−1}| = |p_t − p_{t−1}|_1` (the coupling the proposal asks for).
* **A3 — serving path.** The serving path `k*` is fixed per run (`resample_every` redraws it).
  The real cache is (downloaded) ∧ (sampled on); a sampled "cached" service that was never
  downloaded is **not** a hit.
* **A4 — downloads.** Downloads are triggered by the randomized threshold and, with
  `gate_downloads=True`, only if the serving sample currently wants the service (otherwise the
  download would be immediately discarded). Eviction comes from the sample/projection, not from
  the Landlord.
* **Peek vs. no-peek.** `peek=True` follows the proposal (update with request `t`'s gradient,
  then serve); `peek=False` is textbook OGD.
* **`(1 − x)` in the paper's constraints (4)–(6).** Read as `x` here (sensible reading, as in the
  proposal).
* **OPT off-by-one.** The paper's `OPT_t = (k+2n)l` is off by one `l` at the end of the trace,
  because in the last repetition OPT can download `MS`, flush the cache and hit `MS`'s second
  request. `tests/test_core.py` therefore checks the MILP against `(k+2n)l − l`.

---

## 7. How to reproduce

```bash
pip install -r requirements.txt && pip install -e .
make test                        # unit tests (~3 s)
python experiments/run_all.py --quick    # smoke run of every experiment (~1 min)
python experiments/run_all.py            # full run; writes results/*.json and results/*.png
python experiments/exp12_multiresource.py # one experiment
```

Every experiment writes a JSON file with the raw numbers and a PNG with the figures; the numbers
quoted in `docs/results.md` are read from those JSON files (`results/`), and the figures are
copied to `docs/figures/`.
