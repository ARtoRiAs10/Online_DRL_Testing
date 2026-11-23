# Phase 7 — writing and positioning

**Status: draft complete.**  This note is the writing plan and the honest claim set for
the paper.  It does not replace the full manuscript; it fixes what can and cannot be
claimed after Phases 0–6.

---

## 1. Working title

**Randomizing Online-DRL for Multi-Resource Edge Service Caching: Guarantees and an
Empirical Study**

Alternative (if the venue wants a narrower claim):

**Randomized Rounding and Repair for Online Edge Service Caching with Delayed Hits**

---

## 2. Abstract (draft)

> Online-DRL is a deterministic algorithm for caching services at an edge node with
> delayed hits, request forwarding and a download budget.  Its competitive ratio grows
> linearly with the cache capacity on the adversarial Large Sequence.  We add three
> randomized components: fractional online gradient descent over a multi-resource
> polytope, coupled randomized rounding with a feasibility repair, and a randomized
> rent-or-buy download threshold.  We prove feasibility of the repaired state (G1),
> budget compliance with the download gate (G2), a projection rate for the fractional
> state (Proposition 1), and a surrogate-regret bound `O(sqrt(N) l_max sqrt(T) + ζ T)`
> (Proposition 2).  We do **not** prove a competitive ratio; the flat ratio in `k` is
> empirical on constructed families.  On the Large Sequence the randomized method stays
> at ratio 2.6–5.8 for `k ≤ 48` while Online-DRL grows linearly.  On synthetic Zipf
> traffic the stateless repair reduces latency by 27.8% and download cost by 94.3%
> relative to LL-RC in a 10-seed paired run.  On the Google cluster traces 2011 the
> stateless repair reduces cost in every regime (−30 to −42,658) and has a small latency
> penalty on the 10k slice (+309 to +443) but a small latency gain on the full 50k trace
> (−2,215 to −2,448).  The Landlord repair is more conservative and has a larger latency
> penalty.  We conclude that randomization is a useful engineering addition, but the
> competitive-ratio question remains open.

---

## 3. Introduction positioning

- **Problem.**  Edge service caching with delayed hits, request forwarding, multi-resource
  (CPU/RAM/disk) capacities and a download budget.  Online-DRL (Deka & Sukapuram, 2025)
  is deterministic and has a linear-in-`k` ratio on the Large Sequence.
- **Gap.**  The deterministic threshold and the Landlord admission rule are conservative;
  the fractional state is not used to randomize the cache.  The original analysis does
  not provide a download-charged regret bound.
- **Contribution.**
  1. A randomized rounding + repair scheme that makes the fractional state usable under
     CPU/RAM/disk constraints (G1).
  2. A randomized rent-or-buy threshold with the `e/(e−1)` factor (Phase 4).
  3. A surrogate-regret analysis (Proposition 2) with explicit `ζ` and `δ` terms, and a
     clean separation of rounding churn from repair-induced fetches.
  4. An empirical study on synthetic, adversarial and real Google traces, with paired
     CIs and an honest account of where the method loses.
- **Positioning.**  This is an online-algorithms/systems paper.  The theory is a
  surrogate-regret result, not a competitive ratio.  The systems contribution is the
  multi-resource repair and the cost–latency trade-off on real traces.

---

## 4. Claims table (the initial four claims)

| # | claim | status | evidence | what can be said |
|---|---|---|---|---|
| G1 | The repaired state satisfies CPU/RAM/disk and cardinality constraints almost surely | **proved** | Lemma 1; exp08: raw violations ~0.4–0.6, true violations 0 | The repair is feasible; the stateless rule is the best default on Zipf/real traces, the Landlord rule is the conservative variant. |
| G2 | The expected download cost respects the budget `B_max` | **proved with the gate** | Lemma 2; exp08/15: gated cost ≤ B_max in 10/10 seeds; without the gate it can exceed | The gate is necessary.  Without it the literal Algorithm 1 can exceed the budget. |
| G3 | The surrogate regret is `O(sqrt(N) l_max sqrt(T) + ζ T)` | **proved** | Proposition 2; exp35: surrogate regret ≈ 0.33 × the bound, 1.00 hold rate | The surrogate regret is sublinear if `ζ → 0`; a persistent bias gives linear regret.  The **literal realized-latency regret is not bounded** (exp09: 3–10× the bound, growing with `T`).  The old G3b download-charged bound is **removed**. |
| G4 | Randomization improves the competitive ratio / cost–latency frontier | **not proved; empirical only** | exp05/13/25/32/33/36/27 | On the Large Sequence the randomized ratio stays at 2.6–5.8 vs Online-DRL's linear growth.  On synthetic Zipf the stateless repair gives −27.8% latency and −94.3% cost vs LL-RC.  On real traces the cost win is consistent; the latency sign is trace-dependent.  On the shuffled cyclic scan the method loses (+46% latency).  The flat ratio in `k` is **not** a general theorem. |

**Bottom line for the abstract:** claim G1/G2/G3 as proved (with the gate and the
surrogate-loss caveat); claim G4 only as empirical evidence on the constructed families and
the Google traces.  Do not write "removes the linear-in-k ratio" without the qualifier.

---

## 5. Related work (21 verified references)

The references below are the first OpenAlex match for each query in
`results/reference_check.json` (21 queries).  They cover the four areas the paper needs.

**Online caching / paging.**
1. Sleator & Tarjan (1985). Amortized efficiency of list update and paging rules. *CACM*.
2. Fiat et al. (1991). Competitive paging algorithms. *Journal of Algorithms*.
3. Young (2002). On-line file caching. *Algorithmica*.
4. Bansal et al. (2012). A primal-dual randomized algorithm for weighted paging. *JACM*.
5. Karlin et al. (1988). Competitive snoopy caching. *Algorithmica*.
6. Karlin et al. (1994). Competitive randomized algorithms for nonuniform problems. *Algorithmica*.
7. Borodin et al. (1998). *Online Computation and Competitive Analysis*.
8. Buchbinder & Naor (2009). *The Design of Competitive Online Algorithms via a Primal–Dual Approach*.

**Edge service caching / delayed hits.**
9. Deka & Sukapuram (2025). Edge service caching with delayed hits: minimizing latency and reducing cost. *IEEE TCC*.
10. Fan et al. (2023). Dynamic regret of randomized online service caching in edge computing. *INFOCOM*.
11. Atre et al. (2020). Caching with delayed hits. *SIGCOMM*.

**Online convex optimization / regret.**
12. Zinkevich (2003). Online convex programming and generalized infinitesimal gradient ascent. *ICML*.
13. Hazan (2016). *Introduction to Online Convex Optimization*.
14. Shalev-Shwartz (2011). *Online Learning and Online Convex Optimization*.

**Multi-resource allocation / serverless.**
15. Ghodsi et al. (2011). Dominant resource fairness. *UC Berkeley*.
16. Grandl et al. (2014). Multi-resource packing for cluster schedulers. *SIGCOMM*.
17. Jonas et al. (2019). Cloud programming simplified: a Berkeley view on serverless computing. *arXiv*.
18. Rausch et al. (2020). Optimized container scheduling for data-intensive serverless edge computing. *FGCS*.
19. Du et al. (2020). Catalyzer: sub-millisecond startup for serverless computing. *ASPLOS*.

**Datasets.**
20. Alam et al. (2016). Analysis and clustering of workload in Google cluster trace. *IEEE CSE/EUC*.
21. Iosup et al. (2008). The Grid Workloads Archive. *FGCS*.

> Note: reference 7 (Borodin et al.) and reference 12 (Zinkevich) have no DOI in the
> OpenAlex record; they are standard books/papers and can be cited from the published
> versions.  All other entries have a DOI and were matched by title/year/authors.

---

## 6. Threats to validity

1. **Synthetic data.**  The main Zipf/Poisson world is synthetic; the real-trace check is
   limited to the Google 2011 traces.
2. **Constructed adversary.**  The Large Sequence and the shuffled cyclic scan are
   constructed; the flat ratio is not a general competitive ratio.
3. **One trace family.**  SHARCNet is not reachable; the second real trace is
   `google_dataset2_30.csv`, a 1k-row slice.
4. **`M/l` rescaling.**  The raw Google disk column has `M_n < l_n` for every service; the
   crossing regime uses `disk_scale=1e7`.  The fraction `M<l` is reported.
5. **Trace-length effect.**  The stateless repair's latency win on the full 50k trace is
   not present on the 10k slice; the paper must report both.
6. **Hyperparameters.**  `η`, `α`, `K`, `κ`, `ε` are selected by the Phase-2 sweep; the
   final values are not tuned on the test seeds, but the sweep is on synthetic data.
7. **Peeking implementation.**  The default `peek=True` is an optimistic variant; the
   proposition is stated for the pre-update sequence.  The paper should either use
   `peek=False` for the theory runs or state the gap (Phase 5).
8. **Download gate.**  The budget guarantee needs the gate; without it the literal
   Algorithm 1 can exceed the budget.

---

## 7. Venue and positioning

| venue | fit | what it needs |
|---|---|---|
| IEEE INFOCOM / IFIP Networking | online algorithms + systems | a competitive-ratio or a strong regret result; the current surrogate regret + real-trace study is a good fit for a short paper |
| ACM SIGMETRICS / Performance Evaluation Review | measurement + online algorithms | the real-trace CIs and the cost–latency trade-off; the theory is a plus |
| IEEE TCC / IEEE TPDS | edge service caching | the systems contribution + the multi-resource repair; the paper's own venue |
| Workshop (e.g., HotEdge, EdgeSys) | early results | the real-trace study and the honest negative results |

Recommended: **IEEE INFOCOM short paper** or **IEEE TCC** with the honest claim set.  The
flat-ratio claim must be qualified; the surrogate-regret result is the theory contribution.

---

## 8. AI-use statement (draft)

> The authors used an AI coding assistant to help with the simulator implementation,
> experiment scripts, and the organization of the experimental results.  All algorithmic
> choices, theorem statements, proofs, and the interpretation of the results are the
> authors' responsibility.  The AI assistant did not contribute to the research question,
> the proof design, or the writing of the claims.  The code, seeds, and result JSONs are
> released so that every number can be reproduced.

Check the target venue's AI policy before submission.  AI cannot be an author.

---

## 9. Reproducibility

```bash
cd final_exp
pip install -r requirements.txt && pip install -e .
python experiments/run_all.py --quick     # smoke
python experiments/run_all.py             # full
python experiments/summarize.py           # docs/REPORT.md
```

Key result files:

- `results/exp35_phase5_theory_check.json` — Proposition 2 + ζ/δ + repair fetches.
- `results/exp36_real_data_phase6.json` — multi-seed real-trace CIs.
- `results/exp27_full_google.json` — full 50k Google trace.
- `results/exp37_real_k_sensitivity.json` — K sensitivity.
- `results/exp08_feasibility_budget.json`, `results/exp09_regret.json`,
  `results/exp32_large_sequence_ablation.json`.

Every randomized number in the paper should carry a seed list and a CI; the JSON files
contain the per-seed rows.