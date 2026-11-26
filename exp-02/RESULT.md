# Result — R3 + Marking-Gated Landlord on Online-DRL

Generated from the exp-02 run after the three requested fixes.  Thresholds are
frozen in `ACCEPTANCE/`.  Full details: [`docs/REPORT.md`](docs/REPORT.md);
machine-readable: [`results/acceptance.json`](results/acceptance.json).

## Short answer

| Request | Effect |
|---|---|
| **#1 `M_i > l_i` calibration** | **WORKS** — row #0 flips **FAIL → PASS**; pipeline unblocked. |
| **#3 Soft-MG, τ∈[0.1,0.25]** | **WORKS** — #6 **PASS** (0.00% Google cost regression) while #5 is retained (59–79% cyclic latency reduction, p<0.05). |
| **#2 Bayes-R3 at matched cost** | **DOES NOT WORK** — #4 remains **FAIL** (2/4 synthetic, 0/4 dataset2, 0/2 50k levels). |
| **#4 Pure Randomized Marking / O(log k)** | **NOT PROVED** — #8 remains **UNSUPPORTED** (ratio still grows roughly linearly). |

## Expected outcomes

| # | Expected outcome | Evidence | Status |
|---|---|---|---|
| 0 | **Baseline reproduced** (`M_i > l_i` calibration) | `D=1e-4`, 50k Google, `C=5`: latency **+2.046%**, cost **−88.895%**, `M>l` for **100%** of services. | **PASS** |
| 1 | Reduction holds (`lam=0` = LL-RC) | Exact match; seed-independent. | **PASS** |
| 2 | Invariants hold | 0 resource/cardinality/budget/marking violations. | **PASS** |
| 3 | R3 trigger ratio 1.58 ± 0.02 | 1.5928. | **ACCEPT** (isolated) |
| 4 | R3 / Bayes-R3 beats deterministic at matched cost | R3: 0/4, 0/4, 2/4. Bayes-R3: 2/4 synthetic, 0/4 dataset2, 0/2 50k. No dataset passes. | **FAIL** |
| 5 | MG breaks eviction lockstep (Soft-MG) | Cyclic scan 59.3/71.8/78.9%; burst 27.2–48.3%; all p<0.05 over 30 seeds. | **ACCEPT** |
| 6 | MG does no harm on real traces (Soft-MG) | Google natural/calibrated: latency ≤0.23%, cost **0.00%**; hard MG was +2.8–5.8%. | **ACCEPT** |
| 7 | R3+MG lowers E[ALG]/OPT_t | 9.889→9.226, 19.889→18.718, 39.889→37.319, 79.889→74.996; slope 1.667→1.565. | **ACCEPT** |
| 8 | Improves toward O(log k) | Log-fit R²=0.920, slope 31.15 vs 33.18; ratio still ~linear. | **UNSUPPORTED** |
| 9 | Overhead ≤2× deterministic | Max **1.00×**. | **ACCEPT** |
| 10 | Statistical standard | 30 seeds, bootstrap 95% CI, paired Wilcoxon, Holm. | **PASS** |

## What changed

### #1 — `M_i > l_i` calibration

The bundled Google disk column is normalised, so the literal KB recipe gives
`M_i < l_i` for every service and Online-DRL = LL-RC.  The calibrated regime
uses `D=1e-4` (disk column as MB).  On the 50k trace with `C=5`:

* `M_i > l_i` for **1.000** of services;
* latency **+2.046%** vs LL-RC (published 0.016–7.41%);
* cost **−88.895%** vs LL-RC (published −77.57 to −92.54%).

Row #0 passes and the acceptance pipeline is unblocked.  SHARCNet remains
**BLOCKED** (trace unavailable).

### #3 — Soft-MG

`soft_tau` makes marked services pay `(1−τ)` rent and allows them to be
evicted.  The sweep τ∈{0.1,0.15,0.2,0.25} gives identical real-trace
behaviour; **τ=0.15** is frozen.

* Cyclic lockstep: **59–79%** lower latency, p<0.05 (retained).
* Google natural/calibrated: **0.00%** latency and cost change (hard MG was
  +2.8–5.8% cost).
* Zipf: 0.00% change (hard MG had up to +12% latency).

### #2 — Bayes-R3

Posterior-rate threshold `θ_n/M_n = λ/(1+c·E[λ_n]·M_n)`, `c=0.1`, swept over
`λ`.  It is deterministic given the trace, so all seeds coincide.  It wins
only 2/4 levels on synthetic and 0/4 on dataset2; on the 50k calibrated trace
only 2 levels are covered and it loses both.  **#4 fails.**

## Decision

**GO for the calibrated baseline + Soft-MG package.  NO-GO for the
matched-cost R3/Bayes-R3 claim.**

The paper is positioned as an empirical MG/calibration result with a negative
matched-cost result.  The `O(log k)` claim (#8) remains a conjecture; proving
Pure Randomized Marking in the slot-constrained regime is the remaining
theoretical option.