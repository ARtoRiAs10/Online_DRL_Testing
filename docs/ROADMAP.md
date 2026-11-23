# Roadmap — Randomized Online-DRL

| Phase | Work | Deliverable | Priority |
|---|---|---|---|
| 0 | Integrity fixes | verified refs, code-path proof, repair-rule metadata, eviction log, exp19 labels, AI statement | P0 |
| 1 | Cost–latency frontier | θ sweep with CIs, budget-matched table, full Google repeat | P0 |
| 2 | Stateless reruns | exp01/04/12/17 with stateless, hyperparameter grid | **done** |
| 3 | Reconcile evidence | ALG/OPT vs T, extra MILP if needed | **done** |
| 4 | Adversarial/ratio | Large Sequence ablation, more families, `e/(e−1)` lemma | **done** |
| 5 | Theory clean-up | Prop. 2 re-derivation, churn bound, ζ/δ measurements | **done** |
| 6 | Real data | second trace, natural crossing, multi-seed CIs | **done** (Google 10k/50k + dataset2_30; SHARCNet open) |
| 7 | Writing | verified references, claims table, threats, venue | **draft done** (`docs/PHASE7_WRITING.md`) |

## Definition of done

- Every number in the paper traces to a result JSON and a seed list.
- Every reference is verified.
- Every “proved” claim has a complete proof.
- Every abstract claim has a CI or a proof.
- The AI-use statement is accurate.