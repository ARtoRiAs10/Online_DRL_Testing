# final_exp — updated Landlord-Repair randomized DRL vs the original paper

This folder is a self-contained experiment suite for the algorithm in

* **`Randomized_DRL_Pseudocode__updated_landlord_algo.pdf`** — *Randomized DRL with
  Multi-Resource (CPU, RAM, Disk) Capacity: From Slots to Resource Vectors* (updated
  Landlord-Repair, October 2026),

evaluated on the dataset family of

* **S. Deka and R. Sukapuram, "Edge Service Caching With Delayed Hits: Minimizing Latency
  and Reducing Cost," IEEE Trans. Cloud Computing 13(4), 2025**
  (`Edge_Service_Caching_With_Delayed_Hits_Minimizing_Latency_and_Reducing_Cost.pdf`).

The original paper evaluates on the **Google cluster traces 2011** (`dataset0..dataset10`,
`dataset2`) and on **SHARCNet**.  This workspace contains the Google traces prepared with
the paper's recipe (`data/prepare_google.py`); SHARCNet is not reachable from this
codespace and is documented as an open item.

---

## 1. What is new relative to the previous simulator

The previous randomized-DRL implementation repaired a rounded sample with a **stateless**
rule: evict in increasing order of `p_n / max_r a_{n,r}`.  The updated pseudocode replaces
this with **Algorithm 3 — Landlord-Based Feasibility Repair**:

* every sample path `k` keeps a persistent **credit** `cr[k,n]`;
* a service rounded into the path with no credit receives its **download cost** `M_n` as
  credit; a service absent from the path has its credit cleared (Step 0);
* while the path is infeasible, every cached service pays **rent proportional to its
  bottleneck size** `w_n = max_r a_{n,r}`; the rent step is the largest one that keeps all
  credits non-negative, so at least one service goes bankrupt;
* only **bankrupt (zero-credit)** services may be evicted; among them the one whose
  eviction leaves the smallest **residual overflow** `D_f` is evicted (ties: smaller
  `p_f/w_f`), and `D_f` is recomputed after every eviction.

The implementation is in `sim/landlord_repair.py`; `sim/rand_drl.py` calls it for every
sample path on every request (credit synchronization is done even when the sample is
feasible, as in Step 0 of Algorithm 3).

Two repair modes are available for comparison:

```python
rand_drl(..., repair_mode="landlord")   # updated Algorithm 3 (default)
rand_drl(..., repair_mode="stateless")  # previous v1 rule
rand_drl(..., repair_mode="none")       # raw randomized rounding (no repair)
```

The download gate of the earlier proposal (assumption A4: start a download only when the
serving sample currently wants the service) is **on by default**, so that the comparison
with the previous randomized DRL isolates the repair.  `gate_downloads=False` gives the
literal reading of Algorithm 1 in the updated PDF; exp08 reports both.

The serving path `k*` is kept **fixed for the whole run** (assumption A3 of the earlier
proposal).  Algorithm 1 line 15 says "select `k* ~ Uniform{1..K}`" inside the request loop;
we also tested the literal per-request redraw (`kstar_policy="step"`) and it is much worse
(fewer hits, higher latency) because the real cache is dropped every time the serving path
changes.  The sticky path is therefore the useful reading of R2 and is used in all
experiments.

The download trigger keeps the original paper's rule `T_i >= theta_n` **or**
`L_i >= theta_n` (Algorithm 1 of the paper).  The updated PDF writes only the accumulated
forwarding latency `acc_n` (i.e. `L_i`); the time condition is kept because it is part of
the paper's delayed-hit mechanism and is what makes `theta` meaningful for short traces.

---

## 2. Layout

```
final_exp/
├── sim/                     # simulator (engine, catalog, traces, projection, OPT, algorithms)
│   ├── landlord_repair.py   # Algorithm 3 of the updated PDF
│   └── rand_drl.py          # R1 + R2 + R3, with repair_mode
├── experiments/
│   ├── exp01_baseline_vs_llrc.py
│   ├── exp02_theta.py
│   ├── exp03_bandwidth.py
│   ├── exp04_capacity.py
│   ├── exp05_large_sequence.py
│   ├── exp06_ski_rental.py
│   ├── exp07_projection.py
│   ├── exp08_feasibility_budget.py
│   ├── exp09_regret.py
│   ├── exp10_expected_ratio.py
│   ├── exp11_slot_reduction.py
│   ├── exp12_multiresource.py
│   ├── exp13_gamma_complexity.py
│   ├── exp14_time_complexity.py
│   ├── exp15_budget_regret.py
│   ├── exp16_real_traces.py
│   ├── exp17_paper_synthetic_params.py
│   ├── paper_claims.py      # reference values from the original paper
│   ├── summarize.py         # builds docs/REPORT.md from results/*.json
│   └── run_all.py
├── data/                    # Google cluster traces prepared with the paper's recipe
│   ├── google.csv                       # 50 000 requests, 4 619 services
│   ├── google_dataset2_30.csv           # 1 000 requests, 30 services (paper-like dataset2)
│   ├── google_dataset2_paper.csv        # 1 000-row slice of the paper recipe
│   ├── google_dataset2.csv              # small 1 000-row slice
│   └── prepare_google.py
├── tests/                   # unit tests (core + Landlord-Repair)
├── results/                 # generated JSON + PNG
└── docs/
    ├── REPORT.md            # generated comparison with the original paper (all 17 experiments)
    └── PAPER_COMPARISON.md  # short head-to-head summary
```

---

## 2.1 Phase-1 diagnostics (code-vs-algorithm checks)

`FracPolicy` now supports three diagnostic switches:

- `log_steps=True` — per-step record of `sum_n p_n`, `sum_n a_{n,r} p_n` (normalized and
  raw), hit rank, fraction of rounded-in services evicted by repair, `c_t`, the OGD
  gradient `c~_t`, and `p_{n*,t-1} - xbar_{n*,t-1}`;
- `topup=True` — after repair, add the largest `p_n/w_n` services that still fit;
- `lazy=True` — keep the previous sample path and evict only when the union is infeasible.

Two entry points exercise them:

```bash
python experiments/diagnose_phase1.py --quick --seeds 3   # eta / repair / top-up / lazy
python experiments/phase2_bound_check.py --quick --seeds 3 # G3b bound terms
```

The phase-2 utilities live in `sim/regret_bound.py` (comparator LP + RHS extraction).

## 3. Run

```bash
cd final_exp
pip install -r requirements.txt && pip install -e .
make test

# smoke run (small sizes, ~5 min)
python experiments/run_all.py --quick

# full run (N=40, T=8000, 10 seeds; ~30-60 min)
python experiments/run_all.py

# build the comparison report from results/*.json
python experiments/summarize.py
```

Individual experiments:

```bash
python experiments/exp08_feasibility_budget.py
python experiments/exp16_real_traces.py --csv data/google.csv --kind google --rows 50000
```

---

## 4. The 17 experiments

| # | script | question | paper reference |
|---|---|---|---|
| 01 | `exp01_baseline_vs_llrc` | Online-DRL vs LL-RC latency/cost, with the updated randomized DRL | headline trade-off (latency `+0.016..+7.41 %`, cost `-77.57..-92.54 %`) |
| 02 | `exp02_theta` | deterministic threshold `theta` trades cost vs latency; `theta=0` = LL-RC | Sec. III |
| 03 | `exp03_bandwidth` | delayed hits vs uplink `U` / downlink `D` | Sec. IV |
| 04 | `exp04_capacity` | latency vs slots `C` and vs CPU/RAM/disk limits | Sec. IV |
| 05 | `exp05_large_sequence` | competitive ratio on the adversarial Large Sequence | Theorem 1, eq. (17) |
| 06 | `exp06_ski_rental` | randomized threshold: `2 -> e/(e-1)` | Sec. III |
| 07 | `exp07_projection` | dual-ascent projection equals the exact QP | eqs. (21)-(25) |
| 08 | `exp08_feasibility_budget` | G1 feasibility (with/without repair) and G2 budget | goals G1/G2 |
| 09 | `exp09_regret` | G3 surrogate regret vs `sqrt(N) l_max sqrt(T)` | Proposition 2 |
| 10 | `exp10_expected_ratio` | G4 expected ratio vs the MILP optimum | G4 |
| 11 | `exp11_slot_reduction` | G5 slot reductions, `Lambda_phi`, violations | Sec. 2, G5 |
| 12 | `exp12_multiresource` | CPU/RAM/disk vs slots, utilization, imbalance | Sec. 4 |
| 13 | `exp13_gamma_complexity` | is the linear-in-`k` ratio removed? | Theorem 1 |
| 14 | `exp14_time_complexity` | computational cost per request | Sec. 5 |
| 15 | `exp15_budget_regret` | G2 budget + G3 regret | G2/G3 |
| 16 | `exp16_real_traces` | the paper's Google traces | Sec. IV |
| 17 | `exp17_paper_synthetic_params` | sensitivity to Zipf `rho` and request rate | paper's synthetic dataset |

### Phase-3 experiments (18–22)

| # | script | question |
|---|---|---|
| 18 | `exp18_ablation` | 20-seed paired ablation: R1+R2, +R3, R3 only, Landlord/stateless/top-up |
| 19 | `exp19_adversarial` | cyclic scan, Large Sequence with varied `M/l`, deterministic-eviction trace, `k` up to 64 |
| 20 | `exp20_milp` | 30 random traces (`N=10..16`) against the offline MILP |
| 21 | `exp21_g3b_comparator` | G3b comparator LP vs realized latency + downloads |
| 22 | `exp22_real_traces_phase3` | cross `M_n` and `l_n` on two Google traces, with CIs |

### Phase-5 experiment (35)

| # | script | question |
|---|---|---|
| 35 | `exp35_phase5_theory_check` | Proposition 2(i)/(ii) regret check, `ζ`/`δ` measurements, and repair-induced fetches vs rounding churn (gate-off probe) |

### Phase-6 experiments (36–37)

| # | script | question |
|---|---|---|
| 36 | `exp36_real_data_phase6` | multi-seed real-trace comparison on two Google traces, with `M/l` crossing and paired CIs |
| 37 | `exp37_real_k_sensitivity` | is the real-trace result a `K` artifact? (stateless, K=2/4/8) |

---

## 5. Headline comparison (full run, N=40, T=8000, 10 seeds)

See `docs/REPORT.md` for the complete tables.  The short version:

* **Feasibility (G1)**: the updated Landlord-Repair removes every true-resource violation.
  In exp08 the raw randomized rounding violates the constraints in **40.4 %** of the
  `(path, step)` pairs; after the repair the true-resource violation rate is **0.000** and
  the maximum utilization is **0.941**.
* **Budget (G2)**: with the proposal's download gate the updated randomized DRL respects
  the budget in **10/10 seeds** (mean cost 55 397 vs `B_max` 191 128).  Without the gate
  (literal Algorithm 1) it starts downloads the serving sample does not want: mean cost
  245 307, `B_max` exceeded in 10/10 seeds.
* **Latency**: the credit-based Landlord-Repair is *more conservative* than the previous
  stateless `p_n / max_r a_{n,r}` repair.  In exp08 the gated updated repair has latency
  **155 747** vs **91 697** for the stateless repair (and 145 467 for deterministic
  Online-DRL).  In exp01 the updated randomized DRL is **+55.1 %** vs LL-RC on latency and
  **-72.5 %** on cost.  The cost win is real; the latency penalty is the main empirical
  difference introduced by the updated repair.
* **Large Sequence (exp05/exp13)**: the updated randomized DRL stays at `gamma ≈ 3-5` for
  `k = 3..24`, while deterministic Online-DRL is exactly linear (`gamma = 2.00 k + 0.33`,
  `R² = 1.000`).  The linear fit of the randomized `gamma(k)` has slope **0.070** and
  `R² = 0.22`, i.e. the linear-in-`k` blow-up is removed on this adversarial family.
* **Paper's headline (exp01/exp16)**: the paper's exact latency/cost percentages are obtained
  on the Google traces with the paper's own bandwidth parameters.  On synthetic Zipf the
  deterministic Online-DRL is **-2.41 %** on latency and **-50.5 %** on cost vs LL-RC
  (paper: `+0.016..+7.41 %`, `-77.57..-92.54 %`).  On the bundled Google traces `M_n < l_n`
  for every service, so the `theta=1` and `theta=0` triggers coincide: Online-DRL and LL-RC
  are identical and the paper's latency split cannot be reproduced from the bundled trace
  alone (see exp16).  The paper's PDF text layer is damaged, so its exact bandwidth table
  could not be recovered.

The report distinguishes clearly between (i) confirmed, (ii) refuted and (iii) not testable
with the bundled data.