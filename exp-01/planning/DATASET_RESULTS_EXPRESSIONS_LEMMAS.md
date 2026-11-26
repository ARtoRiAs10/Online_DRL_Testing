# Dataset, results, expressions and lemmas

**Scope:** every dataset used, every result table from Phases 1–6, and the expressions/lemmas
behind the claims.  This file is a working reference.  **It is not committed** (it lives in
`planning/`, which is git-ignored).

---

## 1. Datasets

### 1.1 Synthetic world

`sim.trace.synthetic_world` — Zipf popularity, Poisson arrivals, lognormal demands.

| parameter | default |
|---|---|
| `N` | 40 (headline), 20 (theory), 10–16 (MILP) |
| `T` | 8,000 (headline), 2,000 (ablation), 120–1,200 (MILP) |
| Zipf `rho` | 1.2 |
| request rate | 0.5 requests/ms |
| `U` (uplink) | 100 Mbps |
| `D` (downlink) | 1,000 Mbps (default), 0.1 Mbps (natural crossing) |
| RTT | 20 ms |
| demands | lognormal, `cap_mult=4`, `imbalance` 1.0–4.0 |
| cache capacity | `C = 12` (headline), `C = 50` (Google), `C = k` (Large Sequence) |

Derived:
- `l_n = RTT + req/U = 20 + 100/100 = 21 ms`.
- `M_n = RTT/2 + req/U + S_n/D = 10 + 1 + S_n·8000/D`.
- Raw Google disk: `M_n < l_n` for every service; `D=0.1` with disk column as MB gives a natural crossing.

### 1.2 Large Sequence (adversarial)

`sim.trace.large_sequence(k, n_reps, M, Ml, l)` — the construction from the Online-DRL lower bound.

| parameter | values |
|---|---|
| `k` | 3, 6, 9, 12, 24, 48 |
| `n_reps` | `2k` (headline) |
| `M/l` | 2, 4, 8 |
| `l` | 1 |
| closed-form OPT | `(k + 2 n_reps) l` |
| true MILP OPT | 1 unit lower on small `k` (14 vs 15 at `k=3`; 29 vs 30 at `k=6`) |

### 1.3 Adversarial families

| family | generator | description |
|---|---|---|
| shuffled cyclic scan | `shuffled_cyclic_scan(k, n_reps)` | `k+1` services, fresh random permutation each cycle, `k ≤ 128` |
| Markov-modulated Zipf | `markov_zipf_world` | popularity alternates between Zipf 0.8 and 1.6 |
| bursty | `bursty_world` | bursts of 20 requests centred on one service with random outliers |
| hotspot | `hotspot_world` | Zipf + a hotspot that moves with probability `switch` |
| random-walk Zipf | `random_walk_zipf_world` | log-popularity follows a Gaussian random walk |
| periodic burst | `periodic_burst_world` | Zipf popularity with a sinusoidal arrival rate |

### 1.4 Real traces (Google cluster traces 2011)

| file | requests | services | use |
|---|---:|---:|---|
| `data/google.csv` | 50,000 | 4,619 | main real trace |
| `data/google.csv` first 10,000 | 10,000 | 4,619 | multi-seed slice |
| `data/google_dataset2_30.csv` | 1,000 | 30 | second real trace |
| `data/google_dataset2_paper.csv` | 1,000 | 6 | paper recipe |

Preprocessing: keep `SUBMIT` events, one row per `(JobID, distinct Timestamp)`, first-occurrence resources, zeros/missing → column median, timestamps µs → ms.  SHARCNet is not reachable.

### 1.5 Real-trace regimes

| regime | `D` | `disk_unit_mb` | `disk_scale` | `M<ℓ` (Google 10k) | `M<ℓ` (Google 50k) | `M<ℓ` (dataset2_30) |
|---|---:|---:|---:|---:|---:|---:|
| raw | 1000 | 1/1024 | 1 | 1.000 | 1.000 | 1.000 |
| natural crossing | 0.1 | 1.0 | 1 | 0.653 | 0.664 | 0.067 |
| rescaled | 1000 | 1/1024 | 1e7 | 0.654 | 0.664 | 0.067 |

---

## 2. Expressions and lemmas

### 2.1 Notation

- `N` services; `A ∈ R^{m×N}` resource matrix; `b ∈ R^m_{>0}` capacities.
- `D_vec = {p ∈ [0,1]^N : A p ≤ b}`.
- `p_t` fractional state after round `t`; `p_{t-1}` is played at round `t`.
- `n*_t` request at round `t`; `l_n` forwarding latency; `M_n` download time.
- `q_t ∈ {0,1}` in-flight indicator; `dm_t` remaining download time.
- `c_t = (1 - q_t) l_{n*_t} + q_t min(l_{n*_t}, dm_t)` realized cost coefficient.
- `c~_t` surrogate cost coefficient used by the algorithm.
- `ℓ_max = max_n l_n`; `M_max = max_n M_n`.
- `F_t(p) = (1 - p_{n*_t}) c~_t`; `L_t(p) = (1 - p_{n*_t}) c_t`.
- `η = sqrt(N) / (ℓ_max sqrt(T))`.
- `ζ` predictor bias; `δ_t` rounding distortion.

### 2.2 Algorithm

**R1 (fractional OGD):**
```
p_t = Π_{D_vec}( p_{t-1} + η c~_t e_{n*_t} ).
```

**R2 (rounding + repair):**
```
x_{k,n,t} = 1[ U_{k,n} < p_{n,t} ],   U_{k,n} ~ Uniform[0,1] persistent.
```
If `{n : x_{k,n,t}=1}` violates `A x ≤ b`, evict services:
- stateless: increasing `p_n / max_r a_{n,r}`;
- Landlord: persistent credits, rent proportional to `max_r a_{n,r}`, evict bankrupt services.

**R3 (randomized threshold):**
```
f(θ) = e^{θ/M_n} / (M_n (e - 1)),   θ ∈ [0, M_n],
F(θ) = (e^{θ/M_n} - 1)/(e - 1),
θ = M_n ln(1 + (e - 1) U).
```

**Download gate:** start a download only if the serving sample contains the service:
```
allow_download(n,t) = True  iff  x_{k*,n,t} = 1.
```

### 2.3 Lemma 1 (feasibility, G1)

> Any repair that evicts at least one cached service in every pass and stops as soon as the
> state is feasible returns a state with `A x ≤ b + ε` after at most `N` passes.  This holds
> for the stateless and Landlord rules.

**Proof idea:** each pass removes at least one service; at most `N` services are cached, so
the loop ends in at most `N` passes.  Each eviction costs `O(N)` to recompute the
overfulls.  ∎

### 2.4 Lemma 2 (budget, G2)

> Let `B_consumed` be the running sum of `M_n` for downloads triggered by R3.  With the
> budget check of Algorithm 1 and the download gate, `B_consumed ≤ B_max` at all times on
> every sample path.

**Proof:** a download starts only if `B_consumed + M_n ≤ B_max`, and only then is `M_n`
added; induct from `B_consumed = 0`.  Without the gate the budget can be exceeded.  ∎

### 2.5 Proposition 1 (projection)

> Projected dual ascent with step `1/L` (`L = ||A||_2^2`) satisfies the same first-order
> and second-order conditions as the exact Euclidean projection; the error is `O(1/t)` and
> the feasibility residual is zero.

The implementation matches the exact QP to ~`1e-9` (exp07).

### 2.6 Proposition 2 (surrogate regret)

> Let `η = sqrt(N)/(ℓ_max sqrt(T))` and `L_t(P) = (1 - p_{n*_t}) c_t`.
>
> **(i)** For every realization, the pre-update projected-OGD sequence satisfies
> ```
> Σ_t ⟨ ∇F_t(p_{t-1}), p_{t-1} − P* ⟩ ≤ sqrt(N) ℓ_max sqrt(T).
> ```
> **(ii)** Under the consistency assumption `|E[c_t | F_{t-1}] − c~_t| ≤ ζ`,
> ```
> E[ Σ_t L_t(p_{t-1}) ] − E[ min_P Σ_t L_t(P) ]
>     ≤ 2 sqrt(N) ℓ_max sqrt(T) + ζ T.
> ```
> The regret is `o(T)` if `ζ → 0`; a persistent bias yields linear regret.

**Proof (i):** standard projected OGD.  With `g_t = ∇F_t(p_{t-1}) = -c~_t e_{n*_t}`,
```
||p_t − P*||² ≤ ||p_{t-1} − P*||² − 2η⟨g_t, p_{t-1} − P*⟩ + η²||g_t||².
```
Rearrange, sum, telescope:
```
Σ_t ⟨g_t, p_{t-1} − P*⟩ ≤ ||p_0 − P*||²/(2η) + (η/2) Σ_t ||g_t||²
                       ≤ N/(2η) + (η/2) ℓ_max² T
                       = sqrt(N) ℓ_max sqrt(T).
```

**Proof (ii):** write `c_t = c~_t + ζ_t + ε_t`, with
`ζ_t = E[c_t|F_{t-1}] − c~_t` (`|ζ_t| ≤ ζ`) and `ε_t = c_t − E[c_t|F_{t-1}]`
(`E[ε_t|F_{t-1}]=0`, `|ε_t| ≤ ℓ_max`).  Then
```
Σ_t L_t(p_{t-1}) − L_t(P*)
 = Σ_t c~_t (P*_{n*_t} − p_{n*_t,t-1})       [surrogate]
 + Σ_t ζ_t (P*_{n*_t} − p_{n*_t,t-1})         [bias]
 + Σ_t ε_t (P*_{n*_t} − p_{n*_t,t-1}).        [noise]
```
The surrogate term is ≤ `sqrt(N) ℓ_max sqrt(T)` by (i); the bias term is ≤ `ζT`.
The noise term is
`-Σ_t ε_t p_{n*_t,t-1} + Σ_n P*_n S_n`, `S_n = Σ_{t:n*_t=n} ε_t`.
The first sum has mean zero; the second is bounded by
`E[Σ_n max(S_n,0)] ≤ ℓ_max sqrt(NT)` (Cauchy–Schwarz + orthogonality).  ∎

### 2.7 Lemma 3 (rounding churn) — exact and approximation

**Exact (independent uniforms).**  If `U_{k,n}` is independent of the past,
```
Pr[x_{k,n,t-1}=0, x_{k,n,t}=1] = (p_{n,t} − p_{n,t-1})_+.
```
Since `p_t, p_{t-1} ∈ D_vec` and the projection is nonexpansive,
`||p_t − p_{t-1}||_2 ≤ η c~_t ≤ η ℓ_max`, so
`||p_t − p_{t-1}||_1 ≤ sqrt(N) η ℓ_max`.  With `η = sqrt(N)/(ℓ_max sqrt(T))`,
```
Σ_t Σ_n M_n Pr[x_{k,n,t-1}=0, x_{k,n,t}=1]
   ≤ M_max Σ_t ||p_t − p_{t-1}||_1
   ≤ M_max N sqrt(T).
```

**Approximation (Algorithm 2).**  In Algorithm 2, `p_t` depends on `q_t`, which depends on
the sample states, so the independence assumption fails.  A rigorous version is
```
E[Σ_t Σ_n M_n 1{x_{k,n,t-1}=0, x_{k,n,t}=1}]
  ≤ M_max Σ_t ||p_t − p_{t-1}||_1
    + M_max Σ_t P( q_t changes the update at t ).
```
The second term is the dependence correction; it is not bounded by the L1 movement alone.
The lemma is therefore an **approximation** for the algorithm.

### 2.8 Download-cost lemma (R3, e/(e−1))

> For any fixed cache-state trajectory, the randomized threshold R3 satisfies
> ```
> E[ download cost of R3 ] ≤ (e/(e−1)) · OPT_download.
> ```

The deterministic rule has worst-case ratio 2 (`2 M_n`); the randomized threshold has
density `f(θ) = e^{θ/M_n}/(M_n(e−1))` and expected cost
`E[cost(t)] = ∫_0^{min(t,M_n)} (θ+M_n) f(θ)dθ + t(1−F(min(t,M_n)))`,
which gives `E[cost(t)] / min(t,M_n) ≤ e/(e−1)` with equality at `t = M_n`.
This is a component-level result; it does not prove a full competitive ratio.

### 2.9 G3b (download-charged regret) — removed

The old draft's bound
```
E[latency + downloads]
 ≤ min_P [Σ_t F~_t(P) + Σ_n M_n P_n]
   + 2 sqrt(N) ℓ_max sqrt(T) + ζT + ℓ_max Σ_t δ_t + M_max N sqrt(T)
```
is **not** claimed.  The `M_max N sqrt(T)` term does not cover repair-induced fetches:
Lemma 3 bounds only rounding churn, and a repair eviction can remove a service that the
next draw would have kept, causing a new download.  The paper states the download-charged
regret is open.

### 2.10 `ζ` and `δ` definitions

**Predictor bias `ζ`:**
- `ζ_signed = mean_t (c_t − c~_t)`;
- `ζ_mae = mean_t |c_t − c~_t|` (used in the `ζT` term);
- `ζ_pred_mae = mean_t |c_t − c_pred,t|`, with
  `c_pred,t = (1 − q_ema,t−1) l_n + q_ema,t−1 min(l_n, dl_ema,t−1)`.

**Rounding distortion `δ`:**
- `δ_l1,t = ||p_t − p_{t-1}||_1`;
- `δ_round,t = |p_{n*_t,used} − x_{k*,n*_t,t}|`;
- `δ_churn,t = 1[x_{k*,n*_t}(t-1)=0, x_{k*,n*_t}(t)=1]`;
- `δ_churn_all,t = Σ_n 1[x_{k*,n}(t-1)=0, x_{k*,n}(t)=1]`.

---

## 3. Results

### 3.1 Phase 1 — deterministic frontier (`exp38`)

**Synthetic (N=40, T=8000, 10 seeds).**

| θ | Online-DRL latency | Online-DRL cost |
|---:|---:|---:|
| 0 | 144,537 | 366,644 |
| 0.25 | 140,543 | 254,503 |
| 0.5 | 141,548 | 223,834 |
| 1 | 141,598 | 184,001 |
| 2 | 141,210 | 135,412 |
| 4 | 142,294 | 89,424 |
| 8 | 144,962 | 51,024 |
| 16 | 143,712 | 24,691 |

Randomized stateless:

| variant | latency | cost |
|---|---:|---:|
| default (η×1, α=0.1, K=4) | **93,237 ± 20,785** | **12,011 ± 6,363** |
| η×4 | 119,024 ± 27,767 | 66,347 ± 16,621 |
| K=8 | 99,988 ± 25,936 | 15,528 ± 10,051 |

**Google 10k (3 seeds, K=4).**

| θ | latency | cost |
|---:|---:|---:|
| 0 | 161,818 | 82,510 |
| 1 | 161,833 | 44,956 |
| 4 | 161,982 | 3,044 |
| 16 | 163,253 | 2,830 |

Randomized stateless: **161,669 ± 22,536**, **34,002 ± 5,737**.

**Google 50k (3 seeds, K=2).**

| θ | latency | cost |
|---:|---:|---:|
| 0 | 689,898 | 107,107 |
| 1 | 690,095 | 64,688 |
| 4 | 690,530 | 13,259 |
| 16 | 697,240 | 12,687 |

Randomized stateless: **680,384 ± 181,273**, **24,831 ± 5,985**.

**Budget-matched (synthetic, 10 seeds):**

| `B_max` | Online-DRL lat | Randomized lat | paired Δlat | paired Δcost |
|---:|---:|---:|---:|---:|
| 20% | 154,637 | 97,257 | **−57,380 ± 16,679** | −28,572 ± 3,597 |
| 50% | 148,803 | 94,712 | **−54,091 ± 11,515** | −82,591 ± 5,154 |
| 100% | 141,598 | 93,941 | **−47,658 ± 11,721** | −174,005 ± 7,873 |

### 3.2 Phase 2 — horizon and consistency (`exp39`, `exp31`)

**Trace prefixes (50k Google, 3 seeds, K=2):**

| prefix | raw Δlat | raw Δcost | crossing Δlat | crossing Δcost |
|---:|---:|---:|---:|---:|
| 10k | +221 ± 169 | −18,498 ± 79 | +353 ± 143 | −1,676 ± 1,741 |
| 20k | +50 ± 129 | −27,551 ± 158 | −4 ± 140 | −14,371 ± 887 |
| 30k | +159 ± 109 | −32,732 ± 187 | −54 ± 164 | −20,902 ± 813 |
| 50k | **−2,448 ± 676** | **−42,658 ± 206** | **−2,215 ± 782** | **−33,867 ± 1,021** |

**MILP ALG/OPT vs T:**

| T | Online-DRL | R3 only | Landlord | Stateless |
|---:|---:|---:|---:|---:|
| 120 | 1.168 | 1.152 | 1.427 | 1.301 |
| 300 | 1.456 | 1.485 | 1.981 | 1.631 |
| 600 | 1.717 | 1.761 | 2.251 | 1.675 |
| 1200 | 1.753 | 1.819 | 2.188 | 1.622 |

Stateless gap to Online-DRL: **+11% → +12% → −2% → −7%**.  Warm-up supported.

### 3.3 Phase 3 — adversarial families and ablation (`exp40`)

**Shuffled cyclic scan ablation:**

| variant | latency | ratio vs Online-DRL |
|---|---:|---:|
| Online-DRL | 90 | 1.000 |
| R3 only | 90 | 1.000 |
| R1+R2 stateless (det threshold) | 128 | 1.490 |
| R1+R2 stateless + gate | 128 | 1.490 |
| R1+R2+R3 stateless + gate | 128 | 1.490 |
| R1+R2+R3 stateless, no gate | 128 | 1.490 |
| R1+R2+R3 Landlord + gate | 128 | 1.483 |

**Loss is caused by R1+R2 (rounding/repair), not R3 or the gate.**

**Large Sequence (median latency ratio to closed-form OPT):**

| k | M/l | R3 only | R1+R2 stateless | R1+R2+R3 stateless | R1+R2+R3 Landlord |
|---:|---:|---:|---:|---:|---:|
| 9 | 2 | 7.867 | 2.444 | 2.200 | 2.533 |
| 9 | 4 | 12.844 | 3.000 | 2.778 | 3.222 |
| 9 | 8 | 22.444 | 3.133 | 2.844 | 2.844 |
| 12 | 2 | 10.267 | 1.900 | 1.817 | 2.117 |
| 12 | 4 | 16.633 | 2.283 | 2.117 | 2.300 |
| 12 | 8 | 29.067 | 2.883 | 2.817 | 2.817 |
| 24 | 2 | 19.675 | 2.067 | 1.967 | 2.108 |
| 24 | 4 | 32.825 | 2.483 | 2.400 | 2.517 |
| 24 | 8 | 58.433 | 3.117 | 3.025 | 3.258 |
| 48 | 2 | 38.317 | 2.592 | 2.475 | 2.533 |
| 48 | 4 | 64.692 | 3.121 | 2.962 | 2.962 |
| 48 | 8 | 116.033 | 4.033 | 3.900 | 3.900 |

**Additional families (median latency, ratio vs Online-DRL):**

| family | Online-DRL | R3 only | R1+R2 stateless | R1+R2+R3 stateless | Landlord |
|---|---:|---:|---:|---:|---:|
| hotspot | 136,468 | 0.990 | **0.862** | 0.890 | 1.150 |
| markov_zipf | 141,701 | 1.007 | **0.709** | 0.716 | 1.147 |
| periodic_burst | 138,933 | 0.982 | **0.633** | 0.642 | 1.151 |
| random_walk_zipf | 87,933 | 1.001 | **0.739** | 0.748 | 1.356 |
| bursty | 153,050 | 0.992 | 1.034 | 1.043 | 1.086 |

**OPT check:**

| k | M/l | closed `(k+2n)l` | MILP | MILP/closed |
|---:|---:|---:|---:|---:|
| 3 | 2 | 15 | 14 | 0.933 |
| 3 | 4 | 15 | 14 | 0.933 |
| 6 | 2 | 30 | 29 | 0.967 |
| 6 | 4 | 30 | 29 | 0.967 |

**The closed form is an upper bound, not exact; the MILP is 1 unit lower.**

### 3.4 Phase 4 — real data (`exp41`)

**Google dataset2_30 (10 seeds, K=4):**

| regime | M<ℓ | Online-DRL lat | stateless Δlat | stateless Δcost | Landlord Δlat | Landlord Δcost |
|---|---:|---:|---:|---:|---:|---:|
| raw | 1.000 | 9,834 | +1,327 ± 108 | −163 ± 28 | +1,767 ± 264 | +153 ± 89 |
| natural crossing | 0.067 | 18,740 | +256 ± 84 | −30 ± 78 | +1,051 ± 259 | +566 ± 138 |

**Google dataset2_paper (10 seeds, K=4):**

| regime | M<ℓ | Online-DRL lat | stateless Δlat | stateless Δcost | Landlord Δlat | Landlord Δcost |
|---|---:|---:|---:|---:|---:|---:|
| raw | 1.000 | 1,016 | +228 ± 205 | −303 ± 61 | +841 ± 775 | −117 ± 207 |
| natural crossing | 0.500 | 1,089 | +193 ± 233 | −301 ± 182 | +895 ± 813 | +99 ± 471 |

**Google 10k (5 seeds, K=4, stateless only):**

| regime | M<ℓ | Online-DRL lat | stateless Δlat | stateless Δcost |
|---|---:|---:|---:|---:|
| raw | 1.000 | 138,921 | +309 ± 152 | −20,044 ± 305 |
| natural crossing | 0.653 | 183,907 | +437 ± 179 | −3,568 ± 908 |

**Google 50k (3 seeds, K=2, stateless only):**

| regime | M<ℓ | Online-DRL lat | stateless Δlat | stateless Δcost |
|---|---:|---:|---:|---:|
| raw | 1.000 | 499,143 | **−2,448 ± 676** | **−42,658 ± 206** |
| natural crossing | 0.664 | 861,620 | **−2,505 ± 764** | **−34,229 ± 1,251** |

### 3.5 Phase 5 — theory measurements (`exp35`, `exp42`)

**Proposition 2 check (N=40, T=8000, 10 seeds):**

| config | surrogate regret | Prop-2(i) bound | realized regret | Prop-2(ii) bound | ζ_mae | δ_round | δ_churn | hold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| realized/stateless | 3,883 | 11,879 | 3,883 | 23,759 | 0.000 | 0.106 | 0.0155 | 1.00 |
| realized/Landlord | 3,859 | 11,879 | 3,859 | 23,759 | 0.000 | 0.199 | 0.1067 | 1.00 |
| expectation/stateless | 3,891 | 11,879 | 3,869 | 24,151 | 0.049 | 0.105 | 0.0151 | 1.00 |
| expectation/Landlord | 3,891 | 11,879 | 3,826 | 26,070 | 0.289 | 0.196 | 0.1071 | 1.00 |

Surrogate regret ≈ **0.33×** the bound; realized regret ≈ **0.15–0.16×**.

**EMA ζ over longer horizons (N=20, stateless, K=4):**

| T | ζ_mae (realized) | ζ_mae (expected) | ζ_signed (expected) | rounding churn | repair fetches (gate off) | refetch/churn |
|---:|---:|---:|---:|---:|---:|---:|
| 8,000 | 0.0000 | 0.0335 | −0.0335 | 485 | 1 | 0.002 |
| 16,000 | 0.0000 | 0.0400 | −0.0400 | 817 | 5 | 0.006 |
| 32,000 | 0.0000 | 0.0523 | −0.0523 | 1,563 | 11 | 0.007 |

**ζ does not go to 0; it grows slowly with T.**

**Repair-induced fetches (N=40, T=8000, gate off):**

| repair rule | repair evictions | repair-evicted requests | repair-induced refetches | refetch cost | rounding churn |
|---|---:|---:|---:|---:|---:|
| stateless | 18,740 | 283.7 | 45.3 | 6,352 | 154.3 |
| Landlord | 21,811 | 556.7 | 42.0 | 5,992 | 651.7 |

### 3.6 Phase 6 — claims

| # | claim | status | evidence |
|---|---|---|---|
| G1 | Repaired state satisfies CPU/RAM/disk and cardinality | **proved** | Lemma 1; exp08 true violations 0 |
| G2 | Expected download cost respects `B_max` | **proved with the gate** | Lemma 2; exp08/15 |
| G3 | Surrogate regret `O(sqrt(N) ℓ_max sqrt(T) + ζT)` | **proved** | Proposition 2; exp35 ratio ≈ 0.33; literal realized-latency regret **not bounded**; G3b removed |
| G4 | Randomization improves the cost–latency tradeoff | **empirical** | exp38/39/40/41: on/below frontier in synthetic and moderate-cost real traces; flat ratio on Large Sequence; loses on shuffled cyclic; trace-dependent latency |

**Bottom line:** G1/G2/G3 proved (with the gate and surrogate-loss caveat); G4 empirical only.
The flat ratio in `k` is not a general competitive ratio.

---

## 4. File map

| result file | content |
|---|---|
| `results/exp38_theta_frontier.json` | Phase 1 frontier + budget-matched |
| `results/exp39_horizon_prefixes.json` | Phase 2 prefix sweep |
| `results/exp31_alg_opt_vs_T.json` | Phase 2 MILP ALG/OPT |
| `results/exp40_adversarial_ablation.json` | Phase 3 adversarial/ablation/OPT |
| `results/exp41_real_data_phase4.json` | Phase 4 real data |
| `results/exp42_theory_phase5_v2.json` | Phase 5 ζ/churn |
| `results/exp35_phase5_theory_check.json` | Phase 5 Prop-2 check |
| `results/reference_check.json` | 39 verified references |

All randomized numbers carry per-seed rows, sample sd and paired 95% CIs in the JSON.