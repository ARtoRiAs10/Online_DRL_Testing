# Updated Landlord-Repair vs the original paper — short comparison

This is the short version of [`REPORT.md`](REPORT.md), which is generated from
`results/*.json` by `experiments/summarize.py`.

## What was implemented

`sim/landlord_repair.py` implements **Algorithm 3 (Landlord-Based Feasibility Repair)** of
`Randomized_DRL_Pseudocode__updated_landlord_algo.pdf`:

1. persistent per-path credit `cr[k,n]`;
2. a service entering a path receives its download cost `M_n` as credit, a service leaving
   the path loses it;
3. while the path is infeasible, every cached service pays rent proportional to its
   bottleneck size `w_n = max_r a_{n,r}`;
4. only zero-credit (bankrupt) services are evicted, choosing the one that leaves the
   smallest residual overflow `D_f` (ties: smaller `p_f/w_f`).

The previous stateless repair (`evict in increasing p_n / max_r a_{n,r}`) is kept as
`repair_mode="stateless"` for comparison; `"none"` gives raw randomized rounding.

## Data

The paper's Google cluster traces, prepared with the paper's recipe
(`data/prepare_google.py`):

| file | requests | services |
|---|---|---|
| `data/google_dataset2_30.csv` | 1 000 | 30 |
| `data/google_dataset2_paper.csv` | 1 000 | 6 |
| `data/google.csv` | 50 000 | 4 619 |

SHARCNet is not reachable from this codespace.  The paper's PDF text layer is damaged, so
its exact bandwidth table could not be recovered.

## Head-to-head (full run: N=40, T=8000, 10 seeds)

| quantity | Online-DRL | LL-RC | Randomized DRL (updated landlord) |
|---|---|---|---|
| exp01 latency vs LL-RC | **-2.41 %** | 0 | **+55.14 %** |
| exp01 cost vs LL-RC | **-50.52 %** | 0 | **-72.55 %** |
| exp08 latency (imbalanced, C=12) | 145 467 | — | **155 747** |
| exp08 cost | 191 128 | — | **55 397** |
| exp08 true-resource violations | 0 | 0 | **0** |
| exp08 raw rounding violations | — | — | **0.405** |

For comparison, the **stateless** repair on the same exp08 workload gives latency
**91 697** and cost **11 209** — both better than the updated Landlord-Repair, but the
stateless rule is not the algorithm of the updated PDF.

## What the 37 experiments show

* **G1 (feasibility)**: confirmed.  The raw sample violates the true constraints in ~40 %
  of the `(path, step)` pairs; after Landlord-Repair the violation rate is 0.
* **G2 (budget)**: confirmed with the download gate.  Without the gate the literal
  Algorithm-1 trigger can exceed the budget (mean 245 307 vs `B_max` 191 128).
* **G3 (regret)**: the surrogate regret is below `sqrt(N) l_max sqrt(T)`; the literal
  realized-latency version is 3-10x the bound and grows with `T`.
* **Large Sequence (Theorem 1)**: the updated randomized DRL is flat (`gamma ≈ 3-5` for
  `k=3..24`, linear-fit slope 0.070, `R²=0.22`) while deterministic Online-DRL is exactly
  linear (`gamma = 2k`, `R²=1`).  The linear-in-`k` blow-up is removed on that family.
* **Zipf traffic**: the updated repair is more conservative than the stateless repair:
  higher latency, fewer hits, lower utilization.  On the 30-service Google slice it is
  +13.5 % latency and +7.8 % cost vs LL-RC; on the 50 000-request Google trace it is
  +7.1 % latency and -56.3 % cost.
* **Paper's headline**: the paper's `+0.016..+7.41 %` latency / `-77.57..-92.54 %` cost
  split is **not reproducible from the bundled traces** because `M_n < l_n` for every
  service, so Online-DRL and LL-RC coincide.  On synthetic Zipf the deterministic
  Online-DRL is -2.41 % latency / -50.5 % cost vs LL-RC.

## Phase 5-6 update

* **Proposition 2 (surrogate regret)**: re-derived and confirmed by exp35.  The surrogate
  regret is ~0.33 of the `sqrt(N) l_max sqrt(T)` bound; the realized regret is ~0.15 of
  the `2 sqrt(N) l_max sqrt(T) + ζ T` bound.  The G3b download-charged bound is removed
  (repair-induced fetches are not covered by Lemma 3).
* **Real traces (exp36/exp27)**: the stateless repair reduces download cost in every
  regime (-30 to -42,658).  Its latency sign is trace-dependent: +309/+443 on the 10k
  Google slice, -2,448/-2,215 on the full 50k Google trace.  The Landlord repair has a
  larger latency penalty.  The K sensitivity (exp37) shows the 10k-slice latency penalty
  is stable for K=2/4/8.
* **M/l crossing**: the raw Google disk column has all `M_n < l_n`; the crossing regime
  uses the documented `disk_scale=1e7`, with the fraction `M<l` reported for every row.

## Bottom line

The updated Landlord-Repair is **feasible and budget-safe**, and it preserves the
randomized algorithm's flat competitive ratio on the adversarial Large Sequence.  Its
credit-based eviction is **download-cost aware but popularity-blind**, so on Zipf traffic it
evicts services that the stateless `p_n / max_r a_{n,r}` rule would keep; the result is a
large latency penalty and fewer hits.  The updated repair therefore trades latency for a
stronger feasibility/cost story, and is not an improvement over the stateless repair on
stationary Zipf popularity.