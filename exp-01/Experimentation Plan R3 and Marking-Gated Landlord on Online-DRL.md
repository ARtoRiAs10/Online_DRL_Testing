.# Experimentation Plan: R3 and Marking-Gated Landlord on Online-DRL

*Baseline paper: Deka and Sukapuram, "Edge Service Caching With Delayed Hits: Minimizing Latency and Reducing Cost," IEEE TCC, vol. 13, no. 4, 2025.*

*Code: `online_drl_r3.py` (simulator, unit tests, demo workloads).*

## 1. Objective

Test whether two randomized changes to the paper's Online-DRL algorithm improve it without altering its system model, offline optimum (OPT) or objective:

- **R3 (randomized ski-rental download trigger).** In Algorithm 1 line 14, the threshold `M_i` (scaled by the paper's `θ`) is replaced by a random threshold `θ_i = lam · M_i · ln(1 + u(e−1))`, with `u ~ U[0,1)`. It is redrawn after each download or eviction.
- **MG (marking-gated Landlord).** Algorithm 2 gets a 1-bit `marked` flag per cached service. Rent (`DecreaseCredit`) runs only over unmarked items. Hits and admissions set the flag. When every item is marked and the cache is still infeasible, all flags clear (a new phase). Among zero-credit items the victim is drawn uniformly.

With R3, MG and random eviction switched off, the code must reduce to the original Online-DRL. With `lam = 0` it is LL-RC.

## 2. Hypotheses and pass criteria

| ID | Hypothesis | Pass criterion |
| --- | --- | --- |
| H0 | Randomization off reproduces the paper's Online-DRL. | Identical metrics regardless of seed. |
| H1 | R3 improves the rent-or-buy ratio from 2 to e/(e−1). | Pure ski-rental worst ratio ≈ 1.58. |
| H2 | At equal download cost, R3 gives lower latency than deterministic `θ`. | R3 cost-latency frontier lies below the deterministic one. |
| H3 | MG breaks the deterministic eviction lockstep. | Lower latency on cyclic and adversarial patterns, with no meaningful regression on benign traces. |
| H4 | R3+MG lowers E\[ALG\]/OPT\_t on the Large Sequence. | Lower ratio, and a flatter growth with k. |
| H5 | Per-request overhead stays small. | Measured µs/request within about 2× of deterministic. |

**Important caveat for H2.** `E[θ] = M/(e−1) ≈ 0.58·M`, so at the same `lam` R3 downloads earlier and costs more. Comparing at equal `lam` is unfair to the deterministic baseline. H2 must be judged at matched cost.

**Scope caveat.** The e/(e−1) result applies to the download decision in isolation. The paper's trigger also uses wall-clock `T_i`, and the budget `Mmax` gates downloads, so the end-to-end competitive ratio is not simply e/(e−1). An O(log k) bound for the multi-resource, delayed-hit setting is not proven; treat it as an empirical question.

## 3. Variants under test

| Variant | Trigger | Eviction |
| --- | --- | --- |
| LL-RC | download on every miss (`lam = 0`) | Landlord, deterministic tie-break |
| Online-DRL | deterministic `θ·M_i` | Landlord, deterministic tie-break |
| DRL+R3 | randomized threshold | Landlord, deterministic tie-break |
| DRL+MG | deterministic `θ·M_i` | marking-gated, uniform victim |
| DRL+R3+MG | randomized threshold | marking-gated, uniform victim |

Deterministic variants run once. Randomized variants run 30 to 100 seeds.

## 4. Metrics

- Total latency (µs), total download cost (µs), hits, delayed hits, misses, number of downloads.
- Ratio to the theoretical OPT\_t = (k + 2n)·l on the Large Sequence.
- Ratio to Gurobi OPT on SHARCNet (first 1000 rows only, as in the paper).
- Per-request wall-clock time (µs).
- For randomized variants: mean, standard deviation, bootstrap 95% CI, worst seed and 95th percentile.

## 5. Phased plan

### Phase 0: reproduce the paper's baseline

Do not proceed until the baseline matches the published numbers.

- Rebuild Online-DRL and LL-RC from Algorithms 1 and 2. Replace the simulator's simplified LRU tie-break with the paper's residual-deficit sort (Algorithm 2, lines 8 to 16).
- Take defaults from Table II of the paper.
- Reproduce the Google trace results: Online-DRL latency 0.016% (best) to 7.41% (worst) above LL-RC, and cost 77.57% to 92.54% lower.
- Reproduce the SHARCNet results against OPT: latency 9.34% to 26.62% higher, cost 87.30% and 41.31% higher.
- Preprocessing must match the paper: Google disk in KB, zero CPU/RAM/disk replaced by the column median; SHARCNet disk copied from UsedMemory, timestamps offset to start at zero with duplicates incremented by 1, first occurrence's resources used.

**Exit criterion:** baseline metrics within a stated tolerance (for example 2%) of the published values.

### Phase 1: correctness tests

- H0 regression: deterministic mode is seed-independent.
- Threshold mean: `E[θ] = M/(e−1)` (measured 6.977 vs 6.984 theory in the demo).
- Ski-rental ratio: randomized worst case ≈ 1.582, deterministic 2.0 (passes in the demo).
- Invariants checked after every eviction: cardinality, CPU, RAM and disk within limits; `Bmax` never exceeded.
- Marking invariant: a marked item is never evicted while an unmarked item exists.

### Phase 2: adversarial and synthetic workloads

| Workload | Sweep | Purpose |
| --- | --- | --- |
| Large Sequence (Theorem 1) | k ∈ {6, 12, 24, 48}; n ∈ {k, 4k}; β | H4: ratio to OPT\_t vs k |
| Cyclic C+1 adversary | C; request spacing | H3: lockstep breaking |
| Zipf/Poisson synthetic (Exp. 15, 16) | ρ ∈ {1.2, 1.4, 1.6}; ξ; C = 1 to 9 | No-regression check on skewed traffic |

Before running the Large Sequence, resolve the resource inconsistency in the paper's footnote 1: its disk limit of 10 contradicts its own formula, which gives 14. The current simulator uses a scaled adaptation, so absolute ratios should be treated loosely.

### Phase 3: real traces (paper Experiments 1 to 14)

| Dataset | Experiments | Sweeps |
| --- | --- | --- |
| Google dataset2 (DS1 to DS4) | 1 to 8 | C, time horizon T, uplink U, downlink D, request size, resource limits, `θ` |
| SHARCNet | 9 to 14 | C, U, D, request size, resource limits, `θ`; OPT on 1000 rows |

Notes:

- In the paper's Google data, 1000 rows contain only 30 unique services, grouped together, which makes the cache-size constraint C less meaningful. Use the longer horizons for cache-size claims.
- For SHARCNet, the paper sets OPT's budget `Tmax` to Online-DRL's cost. Set it to the deterministic cost for all variants so OPT is a common reference.
- OPT has no feasible solution at some sweep points (Exp. 10 to 13). Mark these as gaps; do not impute.

### Phase 4: matched-cost comparison (decides H2)

1. Sweep `lam` over the paper's range (0 to 25) for the deterministic and R3 variants.
2. For each, record (cost, latency).
3. Interpolate latency at 3 to 4 fixed cost levels.
4. Report the latency gap between R3 and deterministic at each level, with CIs over seeds.

### Phase 5: ablation and sensitivity

- Compare the five variants listed in Section 3.
- Vary the size weight `w` (max of normalized resources vs sum).
- Vary the victim choice (uniform vs weighted by `w/cr`).
- Vary the `θ` redraw point (per miss epoch vs per request).
- Vary the deterministic tie-break (LRU vs residual-deficit sort).

### Phase 6: statistics and overhead

- 30 to 100 seeds per randomized configuration.
- Paired Wilcoxon signed-rank test against the deterministic baseline.
- Report tail behavior (worst seed, 95th percentile), because the competitive ratio is defined in expectation but operators experience the tail.
- Time µs/request with a profiler on the same hardware for all variants. The "9 to 28 µs" figure in earlier notes is not from the paper; measure it.

### Phase 7: reporting and go/no-go

- Reproduce the paper's figures (cost, latency, hits and delayed hits for each sweep) with the new variants overlaid.
- Keep MG if it wins on adversarial and cyclic workloads and is within about 1 to 2% of deterministic latency on real traces.
- Keep R3 only if it wins at matched cost (Phase 4).
- If either fails, report it as a negative result.

## 6. Results so far (simulator demo)

These are from `--demo` on scaled synthetic workloads. They are early signals, not paper-level evidence.

| Workload | Online-DRL | DRL+R3 | DRL+MG | DRL+R3+MG |
| --- | --- | --- | --- | --- |
| Cyclic C+1 adversary, latency | 7200 | 7200 | about 2453 | about 2469 |
| Large Sequence (k=6, n=40), ratio to OPT\_t | 8.37 | 7.72 | 8.37 | 7.68 |
| Zipf/Poisson, latency | 13343 | 13295 | 13475 | 13312 |
| Zipf/Poisson, cost | 88119 | 92677 | 88825 | 94171 |

Reading:

- MG removes the eviction lockstep on the cyclic adversary (about 66% lower latency).
- R3 gives only about 8% improvement on the Large Sequence, where the large factor comes from eviction and resource coupling rather than the download decision.
- R3 raises cost by roughly 5 to 7% at equal `lam`, consistent with `E[θ] = 0.58·M`. Phase 4 is required to judge it fairly.
- MG alone showed no change on the Large Sequence because its zero-credit ties are rare there.

## 7. Deliverables

- Phase 0 reproduction table: published vs reproduced numbers.
- One CSV per experiment, with all variants and seeds.
- Figures for every sweep, with variants overlaid and CIs shown.
- Cost-latency Pareto plot with matched-cost gaps (Phase 4).
- Ratio-vs-k plot for the Large Sequence (H4).
- Overhead table (H5).
- Final go/no-go summary for R3 and MG.

## 8. Known risks

- The e/(e−1) improvement should be claimed for the download trigger only.
- The Large Sequence in the paper has an internally inconsistent footnote example, so absolute ratios are approximate.
- The simulator's deterministic tie-break is a simplification until Phase 0 swaps in the paper's residual-deficit sort.
- Google and SHARCNet preprocessing choices (median fill, disk copied from memory) directly change results; document them.
- With randomization, results depend on seed count. Report CIs, not single runs.
