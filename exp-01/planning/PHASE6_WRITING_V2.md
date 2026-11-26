# Phase 6 (v2) — writing plan and final claim set

**Status: draft complete.**  This note is the writing plan for the final paper.  The PDF
is generated from it by `final_exp/experiments/make_paper_pdf.py`.

---

## 1. Title

**Randomized Online-DRL for Multi-Resource Edge Service Caching: A Cost–Latency Study**

Short version: **Randomized Rounding and Repair for Online Edge Service Caching with
Delayed Hits**

---

## 2. Abstract (final draft)

> Online-DRL is a deterministic algorithm for caching services at an edge node with
> delayed hits, request forwarding, multi-resource (CPU/RAM/disk) capacities and a
> download budget.  Its competitive ratio grows linearly with the cache capacity on the
> adversarial Large Sequence.  We add three randomized components: fractional online
> gradient descent over a multi-resource polytope, coupled randomized rounding with a
> feasibility repair, and a randomized rent-or-buy download threshold.  We prove
> feasibility of the repaired state (G1), budget compliance with the download gate (G2),
> a projection rate for the fractional state (Proposition 1), and a surrogate-regret
> bound `O(sqrt(N) l_max sqrt(T) + ζ T)` (Proposition 2).  We do **not** prove a
> competitive ratio; the flat ratio in `k` is empirical on constructed families.
>
> On the deterministic frontier, the randomized stateless repair is on or below
> Online-DRL's θ-sweep in the synthetic world and in the moderate-cost region of the
> Google traces; at equal budget it reduces latency by 47k–57k (95% CIs exclude 0).
> On the full 50k Google trace it reduces latency by 2.2k–2.5k and cost by 33k–43k; on
> the 10k slice it reduces cost by 3.5k–20k with a small latency penalty (+0.3k–0.4k).
> The latency sign is trace-dependent and is a warm-up effect: the penalty shrinks and
> becomes a gain as the horizon grows from 10k to 50k.  On the adversarial families the
> randomized method is flat in `k` on the Large Sequence (median ratio 1.8–4.1) while
> Online-DRL grows linearly; it loses on the shuffled cyclic scan (+49% latency), and
> the loss is caused by the randomized rounding/repair (R1+R2), not by the randomized
> threshold (R3).  The Landlord repair is consistently worse than the stateless repair
> on latency.  We conclude that randomization is a useful engineering addition, but the
> competitive-ratio question remains open.

---

## 3. Claims table (the initial four claims)

| # | claim | status | evidence | what can be said |
|---|---|---|---|---|
| G1 | Repaired state satisfies CPU/RAM/disk and cardinality constraints | **proved** | Lemma 1; exp08 | Feasible; stateless is the best default, Landlord is conservative. |
| G2 | Expected download cost respects `B_max` | **proved with the gate** | Lemma 2; exp08/15 | The gate is necessary; without it the literal Algorithm 1 can exceed. |
| G3 | Surrogate regret is `O(sqrt(N) l_max sqrt(T) + ζ T)` | **proved** | Prop. 2; exp35 | Sublinear if `ζ → 0`; measured `ζ` grows slowly, so the `ζT` term remains linear.  The literal realized-latency regret is not bounded.  G3b is removed. |
| G4 | Randomization improves the cost–latency tradeoff | **empirical** | exp38/39/40/41 | On/below the deterministic frontier in the synthetic world and moderate-cost real traces; flat ratio on the Large Sequence; loses on shuffled cyclic; trace-dependent latency.  Not a theorem. |

---

## 4. Related work (39 verified references)

The references below are the first OpenAlex match for each query in
`results/reference_check.json` (39 queries).  They cover the areas the paper needs.

**Online caching / paging.**
1. Sleator & Tarjan (1985). Amortized efficiency of list update and paging rules. *CACM*.
2. Fiat et al. (1991). Competitive paging algorithms. *Journal of Algorithms*.
3. Young (2002). On-line file caching. *Algorithmica*.
4. Bansal et al. (2012). A primal-dual randomized algorithm for weighted paging. *JACM*.
5. Karlin et al. (1988). Competitive snoopy caching. *Algorithmica*.
6. Karlin et al. (1994). Competitive randomized algorithms for nonuniform problems. *Algorithmica*.
7. Borodin et al. (1998). *Online Computation and Competitive Analysis*.
8. Buchbinder & Naor (2009). *The Design of Competitive Online Algorithms via a Primal–Dual Approach*.
9. Reineke et al. (2008). Relative competitive analysis of cache replacement policies. *LCTES*.
10. Epstein et al. (2013). Online file caching with rejection penalties. *Algorithmica*.

**Edge service caching / delayed hits.**
11. Deka & Sukapuram (2025). Edge service caching with delayed hits. *IEEE TCC*.
12. Fan et al. (2023). Dynamic regret of randomized online service caching. *INFOCOM*.
13. Atre et al. (2020). Caching with delayed hits. *SIGCOMM*.
14. Tsachor et al. (2026). Tight latency guarantees for weighted caching with delayed hits. *SPAA*.
15. Tan et al. (2019). Camul: online caching on multiple caches with relaying and bypassing. *INFOCOM*.
16. Wang et al. (2017). A survey on mobile edge networks. *IEEE Access*.
17. Chen et al. (2018). Spatio-temporal edge service placement. *IEEE TWC*.
18. Barman et al. (2026). Joint service caching, task offloading, and service delivery. *IEEE TCSS*.

**Online convex optimization / regret.**
19. Zinkevich (2003). Online convex programming. *ICML*.
20. Hazan (2016). *Introduction to Online Convex Optimization*.
21. Shalev-Shwartz (2011). *Online Learning and Online Convex Optimization*.
22. Bansal et al. (2015). A 2-competitive algorithm for online convex optimization with switching costs. *APPROX/RANDOM*.

**Knapsack / vector packing / rounding.**
23. Böckenhauer et al. (2014). The online knapsack problem: advice and randomization. *TCS*.
24. Albers et al. (2021). Improved online algorithms for knapsack and GAP in the random order model. *Algorithmica*.
25. Panigrahy & Talwar (2011). Heuristics for vector bin packing. (No DOI in OpenAlex.)
26. Chekuri & Vondrák (2010). Dependent randomized rounding. *FOCS*.

**Multi-resource allocation / serverless.**
27. Ghodsi et al. (2011). Dominant resource fairness. *UC Berkeley*.
28. Grandl et al. (2014). Multi-resource packing for cluster schedulers. *SIGCOMM*.
29. Zheng et al. (2016). Online multi-resource allocation for deadline-sensitive jobs. *INFOCOM*.
30. Jonas et al. (2019). Cloud programming simplified. *arXiv*.
31. Rausch et al. (2020). Optimized container scheduling for data-intensive serverless edge computing. *FGCS*.
32. Du et al. (2020). Catalyzer: sub-millisecond startup for serverless computing. *ASPLOS*.
33. Silva et al. (2020). Prebaking functions to warm the serverless cold start. *Middleware*.

**Caching with fetch delay / cost.**
34. Kroft (1998). Lockup-free instruction fetch/prefetch cache organization. *ASPLOS*.
35. Araldo et al. (2014). Cost-aware caching. *Globecom*.
36. Yang et al. (2019). Deep reinforcement learning in cache-aided MEC networks. *ICC*.

**Datasets.**
37. Alam et al. (2016). Analysis and clustering of workload in Google cluster trace. *IEEE CSE/EUC*.
38. Iosup et al. (2008). The Grid Workloads Archive. *FGCS*.
39. Shah et al. (2021). Sequential ski rental problem. *AAMAS*.

> Note: Borodin et al., Panigrahy & Talwar, Kroft, and Zinkevich have no DOI in the
> OpenAlex record; cite them from the published versions.  All other entries have a DOI
> and were matched by title/year/authors.

---

## 5. Threats to validity

1. **Synthetic data.**  The main Zipf/Poisson world is synthetic; the real-trace check is
   limited to the Google 2011 traces.
2. **Constructed adversary.**  The Large Sequence and shuffled cyclic scan are
   constructed; the flat ratio is not a general competitive ratio.
3. **One trace family.**  SHARCNet/Azure/Alibaba are not reachable; the second real trace
   is `google_dataset2_30.csv` (1k rows).
4. **M/l crossing.**  The natural crossing uses `D=0.1` and treats the Google disk
   column as MB; the rescaled regime uses `disk_scale=1e7`.  The fraction `M<l` is
   reported.
5. **Trace-length effect.**  The stateless latency win on the full 50k trace is not
   present on the 10k slice; the paper reports both.
6. **Hyperparameters.**  `η`, `α`, `K` are selected by a sweep on synthetic data.
7. **Peeking implementation.**  The default `peek=True` is an optimistic variant; the
   proposition is stated for the pre-update sequence.
8. **Download gate.**  The budget guarantee needs the gate; without it the literal
   Algorithm 1 can exceed.
9. **Churn lemma.**  The clean lemma assumes independence; Algorithm 2 violates it, so
   the lemma is an approximation.  The repair-induced fetches are measured separately.
10. **ζ.**  The EMA does not drive `ζ → 0`; the `ζ T` term remains linear.

---

## 6. Venue

| venue | fit | what it needs |
|---|---|---|
| IEEE INFOCOM | online algorithms + systems | a narrow claim + real-trace CIs; the frontier result is a good fit |
| ACM SIGMETRICS / PERFORMANCE | measurement + online algorithms | the real-trace CIs and cost–latency trade-off |
| IEEE TCC / TPDS | edge service caching | the systems contribution + multi-resource repair |
| Workshop (HotEdge, EdgeSys) | early results | the real-trace study and honest negative results |

Recommended: **IEEE INFOCOM short paper** or **IEEE TCC**.  The flat-ratio claim must be
qualified; the surrogate-regret result is the theory contribution.

---

## 7. AI-use statement (draft)

> The authors used an AI coding assistant to help with the simulator implementation,
> experiment scripts, and the organization of the experimental results.  All algorithmic
> choices, theorem statements, proofs, and the interpretation of the results are the
> authors' responsibility.  The AI assistant did not contribute to the research question,
> the proof design, or the writing of the claims.  The code, seeds, and result JSONs are
> released so that every number can be reproduced.

Check the target venue's AI policy before submission.  AI cannot be an author.

---

## 8. Reproducibility

```bash
cd final_exp
pip install -r requirements.txt && pip install -e .
python -m pytest -q
python experiments/exp38_theta_frontier.py
python experiments/exp39_horizon_prefixes.py
python experiments/exp40_adversarial_ablation.py
python experiments/exp41_real_data_phase4.py
python experiments/exp42_theory_phase5_v2.py
python experiments/make_paper_pdf.py
```

Every randomized number in the paper carries a seed list and a CI in the JSON files.