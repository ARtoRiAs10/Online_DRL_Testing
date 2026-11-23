# Schedule — Phase 0 and Phase 1

Legend: `[x]` done, `[~]` in progress, `[ ]` not started.

## Phase 0 — Integrity fixes (days 1–3)

- [x] **Day 1 — references.**
  - [x] Query OpenAlex/Crossref for all 21 references.
  - [x] Flag unverifiable entries; delete or replace.
- [x] **Day 2 — code paths and repair metadata.**
  - [x] Prove Landlord variants exercise different branches (unit test).
  - [x] Record `repair_mode`/`evict_mode` in exp18/19/27 JSON (code for exp20–26).
  - [x] Fix exp19 labels (vs Online-DRL and vs OPT).
- [x] **Day 3 — Landlord diagnosis and AI statement.**
  - [x] Log evicted services vs `p_n`; report mean/max/top-1 fraction (`exp28`).
  - [x] Decide: report Landlord as conservative heuristic; stateless is the strong baseline.
  - [x] Write accurate AI-use statement (`REFERENCE_CHECK.md`).

## Phase 1 — Cost–latency frontier (week 2)

- [ ] Deterministic θ sweep, 10+ seeds, `N=40`, `T=8000`, 95% CIs.
- [ ] Randomized stateless and Landlord on the same axes.
- [ ] Hyperparameter sweep for randomized stateless.
- [ ] Budget-matched paired table at 20/50/100%.
- [ ] Full Google trace repeat with CIs.
- [ ] Decision: does randomized stateless lie on/below the frontier?

## Phase 2 — Stateless reruns (week 3)

- [x] exp01, exp04, exp12, exp17 with stateless (`exp29`).
- [x] Re-check utilization and hits (stateless util ~0.6–0.8; Landlord ~0.1).
- [x] Hyperparameter grid and selection rule (`exp30`).

## Phase 3 — Reconcile evidence (week 4)

- [x] ALG/OPT vs `T ∈ {120, 300, 600, 1200}` (`exp31`).
- [x] Variance moderate; no additional MILP instances added.

## Phase 4 — Adversarial/ratio (weeks 5–6)

- [x] Large Sequence ablation R1+R2 / +R3 / R3 (`exp32`).
- [x] 3 new families, larger `k`, CIs (`exp33`).
- [x] `e/(e−1)` download-cost lemma (`docs/PHASE4_THEORY.md`).

## Phase 5 — Theory (week 7)

- [x] Re-derive Proposition 2.
- [x] Prove or remove G3b.
- [x] Measure ζ and δ in every run.
- [x] Bound/measure repair-induced fetches separately (exp35 probe).

## Phase 6 — Real data (weeks 8–9)

- [x] Second real trace (`google_dataset2_30.csv`; SHARCNet unreachable, documented).
- [x] Natural `M_n` vs `ℓ_n` crossing or justify the rescaling, with fraction `M<ℓ` reported (exp36/exp27, `disk_scale=1e7`).
- [x] Multi-seed CIs (exp36: 5/10 seeds; exp27: 3 seeds; exp37: K sensitivity).
- [x] Drop the 2,000-row `google.csv` slice (already absent; the full 50k `google.csv` is used).

## Phase 7 — Writing (weeks 10–11)

- [x] Verified references (21 OpenAlex matches in `results/reference_check.json`).
- [x] Abstract/intro rewrite (`docs/PHASE7_WRITING.md`).
- [x] Threats to validity.
- [x] Venue template and AI policy (draft; check the target venue).
- [x] Release code, seeds, JSONs, one-command reproduction (`final_exp/README.md`).