# Plan — Randomized Online-DRL paper

**Source of truth:** `NEXT_STEPS.md` and `randomized_online_drl_paper.pdf`.

**Single focus:** adding randomization to Online-DRL and showing it improves it. The
Landlord rule is an implementation detail, not a contribution.

## Phase 0 — Integrity fixes (do first)

1. Verify all 21 references against publisher/DOI/OpenAlex pages.
2. Fix the code bug candidate: Landlord variants must exercise different code paths.
3. Record the repair rule in every result JSON (exp27 especially).
4. Check the Landlord repair: log evicted services vs `p_n`; decide fix or negative result.
5. Clarify exp19 Large Sequence labels (vs Online-DRL vs vs OPT).
6. Match the AI-use statement to what actually happened.

## Phase 1 — Decisive experiment: cost–latency frontier

- Sweep deterministic `θ ∈ {0, 0.25, 0.5, 1, 2, 4, 8, 16}` with 10+ seeds, `N=40`, `T=8000`.
- Add randomized stateless (and Landlord) on the same axes.
- Budget-matched paired table at 20/50/100% of deterministic cost.
- Repeat on full Google trace.
- Acceptance: randomized stateless lies on/below the deterministic frontier in a meaningful region.

## Phase 2 — Rerun headline experiments with stateless repair

- Rerun exp01, exp04, exp12, exp17 with stateless.
- Re-check low utilization.
- Hyperparameter sweep: `η`, `α`, `K`, `κ`, `ε`.

## Phase 3 — Reconcile contradictory evidence

- ALG/OPT vs `T` (120, 300, 600, 1200) to test warm-up hypothesis.
- Add more MILP instances only if variance stays high.

## Phase 4 — Strengthen adversarial/ratio story

- Large Sequence ablation: R1+R2 vs R1+R2+R3 vs R3 only.
- 2–3 more adversarial families, larger `k`, CIs.
- Prove the `e/(e−1)` download-cost lemma.
- Optional: bound `E[ALG]/OPT` on Large Sequence.

## Phase 5 — Theory clean-up

- Re-derive Proposition 2.
- Decide on download-charged regret bound; prove or remove.
- Measure `ζ` and `δ_t` in every run.
- Bound repair-induced fetches separately.

## Phase 6 — Real data

- Obtain second real trace (SHARCNet/Azure/Alibaba).
- Natural `M_n` vs `ℓ_n` crossing or justify rescaling.
- Multiple seeds and CIs.
- Drop the 2,000-row `google.csv` slice.

## Phase 7 — Writing and positioning

- 20–40 verified references.
- Rewrite abstract/intro to match evidence.
- Threats to validity.
- Venue template and AI policy.
- Release code, seeds, JSONs, one-command reproduction.

## Decision gates

- **After Phase 1:** if randomized stateless does not beat the deterministic frontier, reframe
  as robustness/adversarial result.
- **After Phase 3:** if ALG/OPT gap persists at long `T`, the average-case claim is negative.
- **After Phase 6:** if no second real trace, claim only Google traces.