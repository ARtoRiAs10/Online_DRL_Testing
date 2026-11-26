# exp-02 — R3 + Marking-Gated Landlord on Online-DRL

This folder is the working area for the roadmap in
[`EXPERIMENTATION_PLAN.md`](EXPERIMENTATION_PLAN.md).  It contains a
self-contained simulator, the frozen acceptance criteria, the experiment
pipeline and the generated report.

The previous workspace (the randomized-DRL / Landlord-Repair code, the older
experiment suite and the planning notes) is in [`../exp-01/`](../exp-01/).

## Layout

| path | content |
|---|---|
| `online_drl_r3.py` | simulator: Online-DRL, LL-RC, R3, MG, R3+MG, traces, statistics |
| `EXPERIMENTATION_PLAN.md` | the roadmap (copied from exp-01, leading dot removed) |
| `ACCEPTANCE/` | frozen preconditions, claim table, statistical standard, prior evidence status |
| `experiments/` | phase 0–7 scripts and `run_all.py` |
| `tests/` | unit tests (`pytest`) |
| `data/` | bundled Google traces (copied from exp-01) |
| `results/` | generated JSON + PNG |
| `docs/` | generated markdown reports |

## Quick start

```bash
cd exp-02
python -m pytest -q
python experiments/run_all.py          # full pipeline (about 15 min)
python experiments/run_all.py --many   # phase 6 with 100 seeds
```

`run_all.py --only phase1_correctness.py phase7_report.py` runs a subset.

## What the current run shows

The **M_i > l_i calibration** (`D=1e-4`, 50k Google, C=5) reproduces the
paper's baseline: latency **+2.046%**, cost **−88.895%**, `M>l` for 100% of
services.  Row #0 **PASSES** and the pipeline is unblocked.  SHARCNet remains
**BLOCKED** (trace unavailable).

| check | result |
|---|---|
| H0 reduction, `lam=0` = LL-RC, invariants, budget | **PASS** |
| threshold mean `1/(e−1)` | **PASS** (0.581871 vs 0.581977) |
| pure ski-rental worst ratio | **PASS** (1.5928 vs 1.58 ± 0.02) |
| R3 at matched cost (3–4 levels, ≥2/3 datasets) | **FAIL** (0/4 synthetic, 0/4 50k, 2/4 dataset2) |
| Bayes-R3 at matched cost (#4) | **FAIL** (2/4 synthetic, 0/4 dataset2, 0/2 50k) |
| MG breaks lockstep, Soft-MG τ=0.15 (≥20%, p<0.05, ≥30 seeds) | **PASS** (59–79% lower on cyclic scan/burst) |
| MG no harm on real traces, Soft-MG (≤2%) | **PASS** (0.00% Google latency/cost change) |
| R3+MG Large Sequence (lower every k, flatter slope) | **PASS** (9.889→9.226, 19.889→18.718, 39.889→37.319, 79.889→74.996; slope 1.667→1.565) |
| O(log k) | **UNSUPPORTED** (ratio still grows roughly linearly) |
| overhead within 2× deterministic | **PASS** (max 1.00×) |

The full evaluation is in [`docs/REPORT.md`](docs/REPORT.md); the machine
readable version is `results/acceptance.json`.  The decision is **GO for the
calibrated baseline + Soft-MG package; NO-GO for the matched-cost R3/Bayes-R3
claim**.

## Implementation notes

* `R3` uses `theta_n = lam * M_n * ln(1 + u(e−1))`, redrawn after a download
  or an eviction.  `lam=0` is exactly LL-RC.
* `MG` adds one mark bit per cached service.  Rent is charged only to
  unmarked services; hits and admissions set the mark; when all cached
  services are marked and the cache is still infeasible the marks are cleared
  (a new phase); zero-credit victims are drawn uniformly.
* `Soft-MG` (`soft_tau`, frozen at 0.15) charges marked services
  `(1−τ)` rent and allows them to be evicted.  It retains the cyclic lockstep
  gain while removing the real-trace cost regression.
* `Bayes-R3` uses a Gamma posterior on each service's request rate and the
  threshold `θ_n/M_n = λ/(1+c·E[λ_n]·M_n)` (`c=0.1`).  It is the candidate
  for the matched-cost claim #4; it does not pass.
* The `calibrated` regime (`D=1e-4`) is the `M_i > l_i` calibration used for
  row #0 and the real-trace comparisons.
* The engine checks cardinality, CPU/RAM/disk and budget invariants after
  every eviction/admission.  A run with any violation is reported as a failed
  precondition.
* The deterministic tie-break is the paper's residual-deficit sort; LRU is
  available for the ablation.

## Reproducibility

* Python 3.14, numpy 2.5, scipy 1.18, pandas 3.0, matplotlib 3.11.
* Seeds: 30 for randomized results (phase 6 can use 100 with `--many`).
* Bootstrap: 2000 resamples; paired Wilcoxon; Holm correction.
* The Google CSVs are the ones prepared in exp-01 (`data/prepare_google.py`).
  SHARCNet is not bundled and could not be fetched in this environment.