# Schedule v2 — Randomized Online-DRL

Legend: `[x]` done, `[~]` in progress, `[ ]` not started.

## Phase 0 — Validity checks (days 1–5)

- [ ] **AI-use statement** — write what actually happened; check venue policy.
- [ ] **`peek` mode audit** — find every reported number that used `peek=True`; rerun
  headline results with `peek=False`; report which numbers changed.
- [ ] **Cost accounting** — show every cache entry (0→1 rounding transitions, repair
  re-adds) is paid in the reported cost; count 0→1 transitions vs charged downloads.
- [ ] **Download gate** — state exactly what it does and what counts toward `B_consumed`.
- [ ] **References** — open each of the 21 on the publisher/DOI page; reconcile the two
  lists; cite only what is used.
- [ ] **Identical-results anomaly** — log the bankrupt-set size at every eviction; if it
  is always 1, do not present deficit alignment as a feature.
- [ ] **Clean rerun** — rerun every experiment from a clean checkout, record seed lists,
  regenerate every table from the JSON files.
- [ ] **Abstract numbers** — use percentages, name the baseline (Online-DRL vs LL-RC),
  recompute the literal-G3 figure.

## Phase 1 — Deterministic frontier (week 2)

- [x] θ ∈ {0, 0.25, 0.5, 1, 2, 4, 8, 16} for Online-DRL, 10 seeds, synthetic (`exp38`).
- [x] Randomized stateless (3 knob settings) on the same axes; 10k Google (3 seeds) and 50k Google (3 seeds, K=2).
- [x] Budget-matched table at 20/50/100% of Online-DRL cost, paired CIs.
- [x] Decision: randomized is on/below the frontier over a meaningful range (synthetic and moderate-cost real traces).

## Phase 2 — Horizon and consistency (week 3)

- [x] Trace prefixes 10k/20k/30k/50k at the same K and seeds (`exp39`).
- [x] MILP ALG/OPT for T = 120, 300, 600, 1200 (`exp31`; latency-only objective, budget-constrained).
- [x] Defined the MILP objective: latency only, subject to the budget and per-step capacities.

## Phase 3 — Adversarial families and ablation (weeks 4–5)

- [x] Shuffled cyclic loss explained: it is caused by R1+R2 (rounding/repair), not R3 or the gate (`exp40`).
- [x] Ablated R1+R2 vs R1+R2+R3 on the Large Sequence; R1+R2 flattens the ratio, +R3 slightly improves it.
- [x] Added hotspot, random-walk Zipf, periodic-burst families (plus Markov-Zipf and bursty).
- [x] Reported medians and quartiles.
- [x] OPT check: the closed form `(k+2n)l` is an upper bound; the MILP is 1 unit lower (14 vs 15 at k=3; 29 vs 30 at k=6).

## Phase 4 — Real data (weeks 6–7)

- [x] Second real trace: `google_dataset2_30.csv` (1k/30); SHARCNet/Azure/Alibaba unreachable, documented.
- [x] Natural `M/ℓ` crossing: `D=0.1`, disk column as MB; fraction `M<ℓ` reported (0.653 google 10k, 0.664 google 50k, 0.067 dataset2_30).
- [x] ≥5 seeds and CIs: 10k (5 seeds), dataset2 (10 seeds), 50k (3 seeds, K=2, documented); all traces reported together.

## Phase 5 — Theory (week 8)

- [x] Rigorous churn lemma: clean lemma is exact under independence; Algorithm 2 violates it, so it is stated as an approximation (`PHASE5_THEORY_V2.md`).
- [x] Repair-induced fetches measured separately (exp35/exp42).
- [x] EMA `ζ` test: `ζ_mae` grows from 0.0335 (T=8k) to 0.0523 (T=32k); it does **not** go to 0.
- [x] Restricted ratio attempted: Phase 1 succeeded; empirical ratio ≤5.8 on the Large Sequence, but no formal proof.

## Phase 6 — Writing (week 9)

- [x] 39 verified references (`results/reference_check.json`).
- [x] Abstract/intro rewrite around the Phase-1 claim (`PHASE6_WRITING_V2.md`).
- [x] Threats to validity.
- [x] Venue and template (IEEE INFOCOM short / IEEE TCC).
- [x] Final PDF: `planning/randomized_online_drl_paper_v2.pdf` (9 pages).