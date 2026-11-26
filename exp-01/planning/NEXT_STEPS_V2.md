# Next steps (version 2, after Phases 5–7)

**Paper's single claim:** adding randomization to the original Online-DRL improves it.
**Current evidence:** cost drops on every real Google-trace regime (stateless repair); latency is unchanged or slightly better on the full 50k trace, slightly worse on smaller ones; the ratio is flat in k on constructed adversarial families. Not shown yet: that it beats Online-DRL's own cost–latency frontier.

## Where things stand

| Item | Status |
|---|---|
| Regret proposition re-derived (pre-update iterates), ζ and δ measured | Done (bounds hold; ≈0.33 of the bound) |
| G3b removed as a theorem | Done |
| Real-trace multi-seed CIs; stateless vs Landlord | Done |
| Second real trace | Only a 1k-row Google slice |
| Related work with checked references | **Not done** (see 0.5) |
| Cost–latency frontier (θ sweep) | **Not done** |
| Horizon sweep (MILP and trace prefixes) | **Not done** |
| `peek` and cost-accounting audit | **Not done** |
| Full manuscript | The PDF is a working draft; fix the red `[CHECK]` items |

---

## Step 0 — Validity checks (do first, ~3–5 days)

These can invalidate results, and each is cheap.

1. **AI-use statement.** The Phase 7 draft says the AI "did not contribute to the proof design or the writing of the claims". That is not what happened. Write what was actually done (the paper PDF has a template). Check the venue's policy. AI cannot be an author.
2. **`peek` mode.** Find every reported number that used `peek=True` (the default, called "optimistic"). If it lets the method use the current request before choosing its state, Online-DRL does not have that information. Rerun headline results with `peek=False`, and report which numbers changed. Only 3 regret numbers were checked so far.
3. **Cost accounting.** Hits come from the rounded cache state; downloads are charged when the randomized trigger fires. Show that every cache entry (including 0→1 rounding transitions and repair re-adds) is paid in the reported cost. Quick test: count 0→1 transitions per run and compare with the number of charged downloads. If they differ, charge every transition and report both versions.
4. **Define the "download gate" precisely.** The reports say it is necessary for the budget and makes repair refetches zero "by construction". State exactly what it does, and what counts toward `B_consumed`.
5. **References.** "First OpenAlex match" proves a paper exists, not that authors, venue, volume and pages are right. Open each of the 21 on the publisher or DOI page. Reconcile the two lists: the project's list has Rausch et al. 2020 and Alam et al. 2016, which the paper PDF does not cite. Cite only what you actually use (the Grid Workloads Archive is cited in Phase 7 but not used). Cite Borodin and Zinkevich from the published versions.
6. **Identical-results anomaly.** Earlier reports showed identical numbers to the last digit for three Landlord tie-break variants (and for `dl_ema` vs `base`). Log the size of the bankrupt set at every eviction. If it is always 1, the deficit-alignment rule never runs, so do not present it as a feature.
7. **Clean rerun.** Rerun every experiment from a clean checkout with the current code, record seed lists, and regenerate every table from the JSON files (not by hand).
8. **Abstract numbers.** Use percentages, name the baseline (Online-DRL vs LL-RC) in every sentence, and recompute the old "3–10×" literal-G3 figure (earlier exp09 data give 2.3–7.8×).

## Step 1 — The decisive experiment: the deterministic frontier (~1 week)

Online-DRL already trades cost for latency through θ. A cost reduction is an improvement only if it lies below that curve.

- Sweep θ ∈ {0, 0.25, 0.5, 1, 2, 4, 8, 16} for Online-DRL with **≥10 seeds** on (a) the synthetic world and (b) the full 50k Google trace (both regimes).
- Plot latency vs download cost with 95% CIs, with randomized (stateless) on the same axes. Also vary its own knobs (η multiplier, α, K) to trace its curve.
- Budget-matched table: at B_max = 20, 50, 100% of Online-DRL's cost, report paired latency differences with CIs.
- **Decision rule:**
  - Randomized is on or below the frontier over a meaningful range → claim "improves the cost–latency tradeoff".
  - It is on or above the frontier → claim "robustness on adversarial inputs, with a trace-dependent cost–latency effect", and make theory plus adversarial families the core.

## Step 2 — Horizon and consistency (~1 week)

- **Trace-length effect.** Run prefixes of the 50k trace (10k, 20k, 30k, 50k) at the same K and seeds. This tests the "warm-up" hypothesis for the latency sign change.
- **MILP.** Compute ALG/OPT for T = 120, 300, 600, 1200 (LP lower bound if the MILP is too slow). This tests whether the 1.23 vs 1.11 gap is a warm-up effect.
- **Define the MILP objective** (latency only, or latency plus weighted download cost).

## Step 3 — Adversarial families and ablation (~1–2 weeks)

- Explain the loss on the shuffled cyclic scan (+46% latency). Which component causes it?
- Ablate R1+R2 vs R1+R2+R3 on the Large Sequence, and add 2–3 more families not built against Online-DRL.
- Report medians and quantiles (the spread exceeds the mean in some cells).
- Check that `OPT=(k+2n)ℓ` is valid for every M/ℓ used.

## Step 4 — Real data (~1–2 weeks)

- A second real trace (an open serverless trace, e.g. Azure Functions or Alibaba, or a mirror of SHARCNet). Document preprocessing.
- A setting where M and ℓ cross naturally, or an argument for the `disk_scale=1e7` rescaling.
- Keep ≥5 seeds and CIs everywhere. Report all traces together, including the small slices where latency is worse (+1.4% to +22%).

## Step 5 — Theory (~1–2 weeks)

- Replace the churn lemma's independence assumption with a rigorous argument (the thresholds and the learned probabilities are dependent through the download indicator), or state it as an approximation.
- Bound or keep measuring repair-induced fetches separately.
- Test whether the EMA drives ζ toward 0 over longer horizons.
- Only attempt a competitive-ratio statement (even restricted) if Step 1 succeeds.
- Do not reintroduce a download-charged "regret theorem" without a corrected comparator.

## Step 6 — Writing (~1 week, after Steps 1–3)

- Extend related work to 20–40 verified references (edge service caching, online knapsack, vector bin packing, caching with fetch delay, serverless cold starts).
- Rewrite the abstract and introduction around whichever claim Step 1 supports.
- Threats to validity: synthetic data, constructed adversary, one real trace family, rescaling, hyperparameters chosen on synthetic data.
- Pick the venue and template.

---

## What I would not do now

- **Prove a competitive ratio first.** It may be false (the Zipf ratio is worse) and is wasted if the frontier test fails.
- **Tune the Landlord repair.** Stateless is better throughout, and Landlord is the original paper's component, not a contribution.
- **Chase more datasets before the frontier test.** It does not settle the central question.
- **Reframe as "robustness only" now.** Premature until Step 1 is done.

## Realistic targets

| Target | Needs |
|---|---|
| arXiv / technical report | Step 0 only |
| Workshop paper | Steps 0, 1, 2, and a narrow claim |
| Conference / journal | All steps, a second real trace, and a proved or well-bounded ratio statement |

Time estimates are rough guesses, not derived from a work breakdown.
