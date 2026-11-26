# Result — R3 + Marking-Gated Landlord on Online-DRL

Generated from the exp-02 run (`a2ffa49`).  Thresholds were frozen in
`ACCEPTANCE/` before the runs.  Full details: [`docs/REPORT.md`](docs/REPORT.md);
machine-readable: [`results/acceptance.json`](results/acceptance.json).

## Short answer

**Yes — some outcomes passed, but no end-to-end claim is accepted.**  The
precondition “baseline reproduced” fails, so every claim that depends on it is
formally **BLOCKED**, even where the isolated evidence is positive.

## Expected outcomes

| # | Expected outcome | Evidence | Formal status |
|---|---|---|---|
| 0 | **Baseline reproduced** (Google latency 0.016–7.41% above LL-RC, cost 77.57–92.54% lower; SHARCNet 9.34–26.62% above OPT) | Paper recipe: 0.00% / 0.00% (M<l for all services). Natural crossing: −0.76% / −41.83% (dataset2), −0.11% / −57.82% (50k). SHARCNet trace unavailable. | **FAIL** (SHARCNet **BLOCKED**) |
| 1 | **Reduction holds** (randomization off = deterministic; `lam=0` = LL-RC) | Seed-independent; `lam=0` and `theta=0` exactly equal LL-RC. | **PASS** |
| 2 | **Invariants hold** (zero resource/cardinality/budget/marking violations) | 0 violations across all runs; budget respected. | **PASS** |
| 3 | **R3 improves the trigger** (pure ski-rental ratio 1.58 ± 0.02) | 1.5928 (deterministic 1.9900). | **ACCEPT** (isolated trigger only) |
| 4 | **R3 beats deterministic at equal cost** (≥3/4 matched levels, CI excludes 0, ≥2/3 datasets) | Synthetic 0/4; Google 50k 0/4; dataset2 1/4 (not Holm-significant). | **FAIL** → **BLOCKED** |
| 5 | **MG breaks eviction lockstep** (≥20% lower, p<0.05, ≥30 seeds) | Cyclic scan 59.3%, 71.8%, 78.9%; burst 27.2–48.3%; all p<0.05 over 30 seeds. | **PASS in isolation** → **BLOCKED** by #0 |
| 6 | **MG does no harm on real traces** (≤2% latency and cost) | Latency +0.10% / +0.23%; cost **+3.23% / +2.80%** on natural-crossing Google. | **FAIL** → **BLOCKED** |
| 7 | **R3+MG lowers E[ALG]/OPT_t** (every k∈{6,12,24,48}, flatter slope) | 9.889→9.226, 19.889→18.718, 39.889→37.319, 79.889→74.996; slope 1.667→1.565. | **PASS in isolation** → **BLOCKED** by #0 |
| 8 | **Improves toward O(log k)** | Log-fit R²=0.920 but slope only 31.15 vs 33.18; ratio still grows roughly linearly. | **UNSUPPORTED** (conjecture) |
| 9 | **Overhead stays small** (≤2× deterministic) | Max 1.16× (7.37 → 8.25 µs/request). | **ACCEPT** (measurement) |
| 10 | **Statistical standard** (≥30 seeds, bootstrap CI, Wilcoxon, Holm) | 30 seeds; 2000-resample bootstrap; paired Wilcoxon; Holm correction applied. | **PASS** |

## What passed

* H0 reduction and `lam=0` = LL-RC.
* Resource, cardinality, budget and marking invariants.
* Threshold mean `1/(e−1)`.
* Pure ski-rental ratio 1.5928.
* MG lockstep on the cyclic family (59–79% lower latency, p<0.05).
* R3+MG Large Sequence ratio at every k, with a flatter slope.
* Overhead within 2× deterministic.
* Statistical reporting standard.

## What did not pass

* **Baseline reproduction** — the bundled Google traces do not reproduce the
  paper’s published split; SHARCNet is unavailable.
* **R3 at matched cost** — R3 loses at 7 of 8 matched levels and is not
  Holm-significant at the one win.
* **MG no-harm on real traces** — cost regression above 2%.
* **O(log k)** — still a conjecture.

## Bottom line

**NO-GO for “R3+MG is better than Online-DRL”.**  The evidence is mixed:
MG’s lockstep result and the Large Sequence improvement are real in isolation,
but the baseline precondition fails and R3 does not win at matched cost, so the
end-to-end claim is not accepted.