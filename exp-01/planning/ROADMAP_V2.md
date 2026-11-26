# Roadmap v2 — Randomized Online-DRL

**Source:** `NEXT_STEPS_V2.md` and `randomized_online_drl_paper.pdf` (working draft).

**Single claim:** adding randomization to Online-DRL improves it.
**Current evidence:** cost drops on every real Google-trace regime (stateless repair);
latency is unchanged or slightly better on the full 50k trace, slightly worse on smaller
ones; the ratio is flat in `k` on constructed adversarial families.
**Not shown yet:** that it beats Online-DRL's own cost–latency frontier.

| Phase | Work | Deliverable | Priority | Status |
|---|---|---|---|---|
| 0 | Validity checks | `peek` audit, cost accounting, gate definition, references, identical-results anomaly, clean rerun, abstract numbers | **P0** | not started |
| 1 | Deterministic frontier | θ sweep (synthetic + Google), randomized on the same axes, budget-matched table, decision | **P0** | **done** (`exp38`) |
| 2 | Horizon & consistency | trace prefixes 10k/20k/30k/50k, MILP ALG/OPT vs `T`, objective definition | P1 | **done** (`exp39`, `exp31`) |
| 3 | Adversarial & ablation | shuffled-cyclic loss ablation, R1+R2 vs +R3 on Large Sequence, 2–3 new families, medians/quantiles, OPT check | P1 | **done** (`exp40`) |
| 4 | Real data | second real trace, natural `M/ℓ` crossing or justified rescaling, ≥5 seeds + CIs | P2 | **done** (`exp41`) |
| 5 | Theory | rigorous churn lemma, repair-induced fetches, EMA `ζ→0`, restricted ratio only if Phase 1 succeeds | P1 | **done** (`exp42`, theory v2) |
| 6 | Writing | 20–40 verified references, abstract/intro rewrite, threats, venue | P2 | **done** (39 refs, PDF) |

## Decision gates

- **After Phase 1:** randomized on/below the frontier over a meaningful range → claim
  "improves the cost–latency tradeoff".  Otherwise → claim "robustness on adversarial
  inputs, with a trace-dependent cost–latency effect", and make theory + adversarial
  families the core.
- **After Phase 2:** if the ALG/OPT gap persists at long `T`, the average-case claim is
  negative; report it.
- **After Phase 3:** if the shuffled-cyclic loss is explained and the Large Sequence
  ablation is clear, proceed to real data.
- **After Phase 4:** if no second real trace, claim only the Google traces.

## Definition of done

- Every number in the paper traces to a result JSON and a seed list.
- Every reference is opened and checked.
- Every "proved" claim has a complete proof.
- Every abstract claim has a CI or a proof.
- The AI-use statement is accurate.