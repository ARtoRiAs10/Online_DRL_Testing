# Phase 0 — status

**Not committed.**

## Completed

- [x] **References.** 21/21 checked via OpenAlex/venue pages. See `REFERENCE_CHECK.md`.
  - 19 have DOIs; #18 verified on USENIX ATC 2018 page; #20 verified on Google cluster-data repo.
  - OpenAlex returned wrong top matches for #18 and #20; manual checks documented.
- [x] **Code bug candidate.**
  - Added `test_landlord_variants_actually_change_the_eviction_path` — the three eviction
    rules produce different outputs on a constructed case.
  - `FracPolicy.diagnostics()` now reports `evict_mode` and `repair_mode`.
  - The earlier identical full-run numbers are a property of the rent/evict loop converging
    to the same sample on that world, not an alias bug.
- [x] **Repair-rule metadata.**
  - `exp18`, `exp19`, `exp27` result JSONs now record `repair_mode`.
  - Code for `exp20`, `exp21`, `exp22`, `exp23`, `exp25`, `exp26` also records
    `repair_mode`; those JSONs will be refreshed in the next full run.
- [x] **Landlord repair diagnosis.**
  - `exp28_landlord_evictions.py` logs the popularity of evicted services.
  - 5-seed result (`N=20`, `T=2000`):
    - Landlord: evicted `p` mean **0.731**, max **1.000**, top-1 fraction **0.256**;
      hits **81**, utilization **0.069**.
    - Stateless: evicted `p` mean **0.427**, max **0.885**, top-1 fraction **0.074**;
      hits **702**, utilization **0.524**.
  - This confirms the Landlord repair is popularity-blind and evicts high-`p` services.
- [x] **exp19 labels.**
  - Large Sequence rows now record `det_vs_opt`, `r3_ratio_opt`, `rf_ratio_opt`.
  - The old “R3/det = 34.546” entry was actually R3/OPT; the report now labels both.
- [x] **AI-use statement.**
  - Corrected text is in `REFERENCE_CHECK.md` and ready to replace the PDF placeholder.

## Remaining Phase 0 items

- [ ] Refresh `exp20`, `exp21`, `exp22`, `exp23`, `exp25`, `exp26` JSONs with the new
      `repair_mode` fields.
- [ ] Decide whether to fix the Landlord repair or report it as a negative result.
- [ ] Update the PDF’s AI-use statement and reference list (#18, #20 formatting).

## Decision

The Landlord repair is feasible and budget-safe but popularity-blind. The stateless repair
is the strong baseline. The paper should present the Landlord rule as a conservative
feasibility heuristic, not as the main improvement.