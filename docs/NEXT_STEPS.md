# Next steps: Randomized Online-DRL paper

**Single focus of the paper:** *adding randomization to the original Online-DRL* and showing it improves it.
The Landlord rule is taken from the original paper; it is an implementation detail, not a contribution.

**Where things stand (from `randomized_online_drl_paper.pdf`)**

| Area | Status |
|---|---|
| Feasibility (G1), budget (G2), projection rate, surrogate regret | Proved and confirmed by experiments; Prop. 2 re-derived in `docs/PHASE5_THEORY.md` and confirmed by exp35 |
| Download-charged regret (old G3b) | **Removed as a theorem**; repair-induced fetches measured separately in exp35 |
| Randomized method beats Online-DRL on latency **and** cost | Supported only for the stateless repair, N=20, T=2000, 20 seeds (−10.6% latency, −75.8% cost) |
| Flat ratio in k | Supported on constructed adversarial families only; no proof |
| Randomized threshold (R3) alone | Does not help (+23% cost in the ablation) |
| Ratio to exact OPT on short Zipf traces | Randomized is worse (1.23–1.30 vs 1.11) |
| Beats Online-DRL's cost–latency frontier (θ sweep) | **Untested** — the decisive open experiment |
| Real data | One trace family, 2–3 seeds, artificial rescaling to cross M = ℓ |

Items are ordered by priority. Do 0–3 first; they decide whether the paper holds.

---

## 0. Integrity fixes (do first, ~2–3 days)

- [ ] **Verify every reference.** Open each one on the publisher or DOI page and check authors, title, venue, volume, pages. The old draft's refs 10, 11, 14, 21, 25–30 were unverifiable or wrong and were removed from the new PDF. The 21 references now in the PDF come from memory, so check them too. Delete anything you cannot confirm. Do not add "representative" references.
- [ ] **Fix the code bug candidate.** The three Landlord variants (default / popularity / hybrid) gave results identical to the last digit (latency 20,445, cost 23,229, hits 14). Confirm the variant flag changes the code path (print which rule is used, run a unit test where the tie-break must matter). Until fixed, make no claim about tie-breaks.
- [ ] **Confirm the repair rule behind each experiment.** exp01–exp17 use the Landlord rule. For exp27 (full Google trace) the variant is not recorded; check the code and write it into the result JSON.
- [ ] **Check the Landlord repair itself.** It leaves max utilization at 0.019 (stateless: 0.346). Likely cause: credits ignore `P`, so popular services are evicted. Test: log which services it evicts vs their `p_n`. Check credit synchronization (credit reset each step?). Either fix it or report it as a negative result.
- [ ] **Clarify exp19 Large Sequence entries** (34.546 and 3.460). They are labelled "ratio to Online-DRL" but look like ALG/OPT.
- [ ] **Match the AI-use statement to what actually happened** (Section "Use of AI Tools"). Check your target venue's AI policy. AI cannot be an author; you are responsible for all content.

## 1. The decisive experiment: cost–latency frontier (~1 week)

Online-DRL already trades cost for latency through θ. A cost reduction alone is not an improvement. In exp02 (1 seed, Landlord rule) deterministic θ=4 (cost 72,234, latency 85,807) beats the randomized run (cost 74,018, latency 146,172) on both.

- [ ] Sweep θ ∈ {0, 0.25, 0.5, 1, 2, 4, 8, 16} for Online-DRL, **10+ seeds**, N=40, T=8000. Plot latency vs download cost with 95% CIs.
- [ ] Add randomized **with the stateless repair** (and Landlord for contrast) on the same axes. Also sweep its own knobs (η multiplier, α, K) to trace its curve.
- [ ] Budget-matched table (extend exp15): at B_max = 20, 50, 100% of the deterministic cost, report **paired** latency differences with CIs, 10+ seeds.
- [ ] Repeat on the full Google trace (exp27) with 10 seeds.
- **Acceptance:** randomized (stateless) lies on or below the deterministic frontier in a meaningful region. If it does not, the honest claim becomes "robustness on adversarial inputs", i.e. a shorter or workshop paper.

## 2. Rerun headline experiments with the stateless repair (~3–4 days)

- [ ] Rerun exp01 (vs LL-RC), exp04 (capacity), exp12 (multi-resource), exp17 (Zipf/rate sensitivity) with the stateless rule.
- [ ] Re-check the low-utilization issue: the randomized method used 7–11% of capacity with the Landlord rule.
- [ ] Hyperparameter sweep: η multiplier, α (EMA), K (paths), κ, ε. Report the grid and how the final values were chosen (not tuned on the test seeds).

## 3. Reconcile the contradictory evidence (~3 days)

- [ ] Ablation (T=2000) says randomized-stateless beats Online-DRL; MILP (T=120) says its ratio to OPT is worse (1.232 vs 1.113). Hypothesis: T=120 is too short for a learning method. Test it: ratio to OPT (solve MILP with T = 120, 300, 600, 1200 if tractable, or use an LP lower bound for larger T). Report ALG/OPT against T.
- [ ] Add more than 30 MILP instances only if variance stays high.

## 4. Strengthen the adversarial and ratio story (~1–2 weeks)

- [ ] Ablation on the Large Sequence: R1+R2 vs R1+R2+R3 vs R3 only, to show which component flattens the ratio (exp25 shows R3 alone does not).
- [ ] Add 2–3 more adversarial families (not built against Online-DRL specifically), larger k, CIs. Explain the large variance (e.g. 5.4 ± 6.8).
- [ ] Cheapest theory step: find where the factor 2 of ski rental enters Online-DRL's ratio proof and replace it with e/(e−1); state it as a lemma for the download-cost term only.
- [ ] Optional, harder: bound E[ALG]/OPT on the Large Sequence by analysing the cyclic dynamics of `P_t`.
- Do **not** claim "removes the linear-in-k ratio" in general; say "on the constructed families".

## 5. Theory clean-up (~1 week)

- [x] Re-derive and check Proposition 2 (regret) line by line yourself; ideally have a colleague with OCO background read it.  See `docs/PHASE5_THEORY.md`; the draft's off-by-one indexing is fixed.
- [x] Decide on the download-charged regret bound (the old "Theorem 1"): either prove it properly (needs a churn bound covering repair-induced fetches) or leave it out.  **Removed**; G3b is a diagnostic only.  The `M_max·N·√T` term is unjustified and repair-induced fetches are measured separately in exp35.
- [x] Measure Assumption 1: log the predictor bias ζ̂ = |E[c_t | past] − c̃_t| and the rounding distortion δ_t in every run; report them.  Added per-step `c_realized`, `c_pred`, `zeta`, `zeta_pred`, `delta_l1`, `delta_round`, `delta_churn`; exp35 reports the summaries.
- [x] Bound or measure repair-induced fetches separately from rounding churn.  exp35's gate-off probe: stateless 45.3 refetches vs 154.3 rounding churn; Landlord 42.0 vs 651.7.

## 6. Real data (~1–2 weeks)

- [x] Obtain a second real trace: `google_dataset2_30.csv` is used; SHARCNet is not reachable from this workspace and is documented as an open item.
- [x] A setting where M_n and ℓ_n cross **naturally** (or justify the rescaling), with fraction M<ℓ reported.  The raw Google disk column has all M<ℓ; the crossing uses the documented `disk_scale=1e7`, and the fraction M<ℓ is reported for every row (exp36/exp27).
- [x] Multiple seeds and CIs for every real-trace number (exp36: 5/10 seeds; exp27: 3 seeds; exp37: K sensitivity).
- [x] Drop the 2,000-row `google.csv` slice (already absent; the full 50k `google.csv` is used).

## 7. Writing and positioning (~1 week)

- [x] Extend related work to 20–40 verified references (21 OpenAlex matches in `results/reference_check.json`; see `docs/PHASE7_WRITING.md`).
- [x] Rewrite the abstract and introduction around what the evidence supports after steps 1–4.  The claim set is in `docs/PHASE7_WRITING.md`; the flat ratio is qualified as empirical.
- [x] Add a threats-to-validity paragraph (synthetic data, constructed adversary, one real trace, hyperparameters).
- [x] Pick a venue and its template; check page limits and the AI-use policy (draft in `docs/PHASE7_WRITING.md`).
- [x] Release code, seeds and result JSONs; provide a one-command reproduction (`final_exp/README.md`).

---

## Realistic outcomes

| Target | What it needs |
|---|---|
| arXiv / technical report | Section 0 only (fix references, claims and the code check) |
| Workshop paper (online algorithms / edge computing) | Sections 0, 1, 2, 3, and a clear, narrow claim; ~4–6 weeks |
| Conference / journal | All of the above plus a proved ratio statement (even restricted), a second real trace, and a full related-work section; ~3–6 months |

## Quick check before you submit

1. Every number in the paper traces to a result file and a seed list.
2. Every reference was opened and checked.
3. Every "proved" claim has a complete proof you have read.
4. Every claim in the abstract appears in the results with a CI.
5. The AI-use statement is accurate and matches the venue policy.
