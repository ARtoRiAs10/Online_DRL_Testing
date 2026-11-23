# Development change log

**Not committed.** All paths are relative to the repository root.

## Repository cleanup

- Removed research-paper PDFs/PPTX, manuscript drafts, text extractions, `docs_bundle/`,
  the superseded `extracted/drl-sim/` tree, unrelated `scripts/`, and old working notes.
- Added `final_exp/` as the canonical simulator and experiment suite.
- Added `.gitignore` for papers, manuscripts, and Python artifacts.
- Moved all working documentation into `docs/`:
  - `docs/NEXT_STEPS.md`
  - `docs/PLAN.md`
  - `docs/ROADMAP.md`
  - `docs/SCHEDULE.md`
  - `docs/PHASE1_STATUS.md`
  - `docs/REFERENCE_CHECK.md`
  - `docs/DEV_CHANGES.md`
  - `docs/PHASE2_PHASE3_REPORT.md`
  - `docs/randomized_online_drl_paper.pdf`

## Phase 0 — integrity fixes

### Code

- `final_exp/sim/rand_drl.py`
  - Gradient now uses `Q_{t-1}`; `_update_q_ema` runs after the gradient.
  - Added `log_steps` with `p_sum`, resource loads, hit rank, repair-eviction fraction,
    `c_t`, `c~_t`, `p_{n*,t-1} - xbar_{n*,t-1}`, `q_ema`, `dl_ema`.
  - Added `topup`, `lazy`, `charge_churn`, `download_time_mode`.
  - Added `evict_mode` (deficit/popularity/hybrid) and `repair_pop_weight`.
  - Added eviction-popularity logging: `evicted_p_mean`, `evicted_p_max`,
    `evicted_top1_fraction`.
  - `diagnostics()` now reports `repair_mode`, `evict_mode`, `churn_credit`,
    `churn_events`, `mean_dl_ema`.
- `final_exp/sim/engine.py`
  - Budget checks include `churn_credit`.
  - Records `last_outcome` and empirical popularity rank.
- `final_exp/sim/landlord_repair.py`
  - Caller passes raw `A`, `b=[C,1,1,1]`.
  - Added `evict_mode` and `pop_weight`.

### New experiments / utilities

- `exp28_landlord_evictions.py` — logs which services the repair evicts and their `p_n`.
- `verify_references.py` — OpenAlex reference triage; writes `results/reference_check.json`.
- `exp29_stateless_headline.py` — phase-2 reruns with stateless repair.
- `exp30_hyperparameters.py` — eta/alpha/K/projection sweeps.
- `exp31_alg_opt_vs_T.py` — ALG/OPT vs trace length.

### Tests

- `tests/test_landlord_repair.py` — added a test proving the three eviction modes differ.
- `tests/test_phase1_diagnostics.py`, `tests/test_phase3_utils.py`,
  `tests/test_phase4.py` — existing logging/top-up/lazy/bound tests.

### Results

- `results/exp28_landlord_evictions.json`
- `results/reference_check.json`
- `results/exp29_stateless_headline.json`
- `results/exp30_hyperparameters.json`
- `results/exp31_alg_opt_vs_T.json`
- `final_exp/docs/REPORT.md` regenerated with sections 18–31.

## Phase 2 — stateless reruns

- `exp29_stateless_headline.py` reruns:
  - exp01 vs LL-RC;
  - exp04 capacity sweep;
  - exp12 balanced/imbalanced multi-resource;
  - exp17 Zipf/rate sensitivity.
- All randomized runs use `repair_mode="stateless"`.

Key results (full run, `N=40`, `T=8000`, 10 seeds unless noted):

- exp01: randomized stateless latency **-27.82%** vs LL-RC, cost **-94.28%**.
- exp04: lower latency than Online-DRL at every capacity.
- exp12 balanced: stateless latency **89,750** vs Online-DRL **120,110**, cost **17,420** vs
  **167,401**; Landlord latency **154,018**, cost **58,806**.
- exp12 imbalanced: stateless **91,058** / **15,527** vs Online-DRL **132,142** / **190,686**.
- exp17: randomized stateless latency **-16% to -37%** and cost **-89% to -95%** across
  `rho` and request rate.

The low-utilization problem is specific to the Landlord repair:
- Landlord max utilization ~0.10–0.18;
- stateless max utilization ~0.62–0.72.

## Phase 2 — hyperparameters

- `exp30_hyperparameters.py` sweeps:
  - `eta_mult ∈ {1,4,16,64}`;
  - `alpha ∈ {0.05,0.1,0.2,0.4}`;
  - `K ∈ {2,4,8}`;
  - `proj_tol ∈ {1e-6,1e-7,1e-8}`;
  - `proj_skip_tol ∈ {0,1e-3}`.
- Best in the quick/full run:
  - `eta_mult=1`;
  - `alpha=0.05`;
  - `K=8`;
  - `proj_tol=1e-6`;
  - `proj_skip_tol=0`.
- `alpha` has only a small effect because the synthetic world has few in-flight downloads.

## Phase 3 — reconcile the evidence

- `exp31_alg_opt_vs_T.py` measures `ALG/OPT` at `T ∈ {120,300,600,1200}`.
- 30 traces at T=120, 15 at T=300, 10 at T=600, 5 at T=1200; 2 seeds per randomized variant.
- Mean ratios:

| T | Online-DRL | R3 only | Landlord | Stateless |
|---:|---:|---:|---:|---:|
| 120 | 1.168 | 1.152 | 1.427 | 1.301 |
| 300 | 1.456 | 1.485 | 1.981 | 1.631 |
| 600 | 1.717 | 1.761 | 2.251 | 1.675 |
| 1200 | 1.753 | 1.819 | 2.188 | 1.622 |

- The absolute ratio grows with `T` for all algorithms, but the randomized stateless gap to
  Online-DRL closes:
  - T=120: **+11%** worse;
  - T=300: **+12%** worse;
  - T=600: **-2%** better;
  - T=1200: **-7%** better.
- This supports the warm-up explanation for the short-trace MILP result.
- Variance is moderate (`sd ≈ 0.09–0.31`), so no additional MILP instances were added.

## Not committed

All files under `docs/` are excluded via `.git/info/exclude`. The paper PDF is ignored by
`.gitignore` (`*.pdf`).

## Phase 5 — theory clean-up

### Theory

- `docs/PHASE5_THEORY.md` — full re-derivation of Proposition 2:
  - fixes the draft's off-by-one: the bound is for the **pre-update** OGD sequence;
  - states the exact-projection assumption and the adaptive-adversary condition;
  - explains the factor 2 as OGD + martingale, and notes that `grad_mode="realized"`
    gives the one-term bound;
  - decides G3b is **not** a theorem: the `M_max N sqrt(T)` term does not cover
    repair-induced fetches; the paper should state the download-charged regret is open;
  - re-derives Lemma 3 (rounding churn) and separates it from repair-induced fetches;
  - defines `ζ` (signed bias, MAE, one-step EMA error) and `δ` (L1 movement, per-request
    rounding error, 0→1 churn).
- `docs/PHASE5_REPORT.md` — results and decisions.

### Code

- `sim/regret.py`
  - `ogd_run` now returns the **pre-update** state in both `peek` modes; the old version
    returned the post-update state when `peek=True`, which was inconsistent with the OGD
    bound.
- `sim/rand_drl.py`
  - per-step log now includes `c_realized`, `c_pred`, `zeta`, `zeta_pred`, `delta_l1`,
    `delta_round`, `delta_churn`, `delta_churn_all`, `p_minus_xbar_post`, `kstar_step`,
    `repair_evicted_request`, `repair_refetch`;
  - `P_pre_hist` / `P_post_hist` recorded separately;
  - repair-evicted-request and repair-induced-refetch counters
    (`repair_evicted_requests`, `repair_refetches`, `repair_refetch_cost`);
  - `diagnostics()` reports `mean_zeta`, `mean_abs_zeta`, `mean_delta_l1`,
    `mean_delta_round`, `delta_churn_sum`, `delta_churn_all_sum`.
- `sim/regret_bound.py`
  - `regret_bound_terms()` now returns the Proposition-2 quantities:
    `surrogate_regret_pre`, `realized_regret_pre`, `surrogate_comparator`,
    `realized_comparator`, `prop2_bound_1`, `prop2_bound_2`, `zeta_signed`, `zeta_mae`,
    `zeta_pred_mae`, `delta_l1_mean`, `delta_round_mean`, `delta_churn_rate`,
    `delta_churn_sum`, `delta_churn_all_sum`;
  - the G3b fields are kept as a diagnostic only.
- `experiments/exp35_phase5_theory_check.py` — new: deterministic Prop-2(i) check,
  realized/expectation × stateless/Landlord regret check, ζ/δ summaries, and a
  gate-off repair-induced-fetch probe.
- `tests/test_phase5.py` — new tests for pre-update OGD, phase-5 log fields, ζ modes,
  Prop-2 terms, repair-refetch counters, and churn consistency.

### Key results (N=40, T=8000, 10 seeds)

- Surrogate regret ≈ 3,883 vs Prop-2(i) bound 11,879 (ratio ≈ 0.33).
- Realized regret ≈ 3,883 (realized) / 3,826–3,869 (expectation) vs Prop-2(ii) bound
  23,759–26,070 (ratio ≈ 0.15–0.16).
- `ζ_mae` = 0 (realized) / 0.049 (expectation, stateless) / 0.289 (expectation, Landlord).
- `δ_l1` mean ≈ 0.077; `δ_round` mean ≈ 0.106 (stateless) / 0.199 (Landlord);
  `δ_churn` rate ≈ 0.0155 (stateless) / 0.107 (Landlord).
- Repair-induced refetches (gate off): 45.3 (stateless) / 42.0 (Landlord), versus
  rounding churn 154.3 / 651.7.  With the gate on they are zero by construction.
## Phase 4 — adversarial families and ratio

### New experiments

- `exp32_large_sequence_ablation.py`
  - R3 only, R1+R2 stateless, R1+R2+R3 stateless, R1+R2+R3 Landlord.
  - `k ∈ {9,12,24,48}`, `M/l ∈ {2,4,8}`, 5 seeds (3 for k=48).
  - Writes `results/exp32_large_sequence_ablation.json`.
- `exp33_adversarial_families.py`
  - Shuffled cyclic scan (`k` up to 128), Markov-modulated Zipf, bursty random.
  - 10 seeds, 5 algorithms.
  - Writes `results/exp33_adversarial_families.json`.
- `exp34_large_sequence_dynamics.py`
  - Records `P_t` dynamics on the Large Sequence.
  - Writes `results/exp34_large_sequence_dynamics.json`.

### New trace generators

- `sim/trace.py`
  - `shuffled_cyclic_scan(k, n_reps, seed)`
  - `markov_zipf_world(N, T, seed, rho1, rho2, switch)`
  - `bursty_world(N, T, seed, burst_len)`

### Key results

- R3 alone remains linear in `k`; R1+R2 (randomized rounding) is what flattens the ratio.
- R1+R2+R3 stateless has latency ratio ~2–4 for k=9..48 and cost ratio ~0.6–3.5.
- Randomized stateless dominates on Markov-modulated Zipf (latency -26%, cost -90%).
- Randomized stateless trades latency for cost on bursty traffic (+5.6% latency, -55% cost).
- Randomized stateless is worse on the shuffled cyclic scan (+46% latency, -13% cost) —
  an honest negative result.
- `P_t` stays near the capacity simplex on the Large Sequence; the ratio is controlled by
  the cyclic request pattern.

### Theory

- `docs/PHASE4_THEORY.md` states the `e/(e−1)` download-cost lemma and explains where the
  factor 2 enters the Online-DRL proof.
- The lemma is component-level; it does not prove a full competitive ratio.

### Reports

- `docs/PHASE4_REPORT.md` — full phase-4 results and analysis.
- `final_exp/docs/REPORT.md` — regenerated with sections 32–34.
