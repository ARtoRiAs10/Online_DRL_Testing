# Phase 0 — Reference verification

**Not committed.**

Source: `randomized_online_drl_paper.pdf`, 21 references. Checked against OpenAlex and, where
OpenAlex was wrong or missing, the publisher/venue page.

| # | Reference | Status | DOI / source |
|---:|---|---|---|
| 1 | Deka & Sukapuram, Edge Service Caching With Delayed Hits | verified | `10.1109/tcc.2025.3605922` |
| 2 | Fan, Hou, Mai, Dynamic Regret of Randomized Online Service Caching | verified | `10.1109/infocom53939.2023.10229044` |
| 3 | Sleator & Tarjan, Amortized efficiency of list update and paging rules | verified | `10.1145/2786.2793` |
| 4 | Fiat et al., Competitive paging algorithms | verified | `10.1016/0196-6774(91)90041-v` |
| 5 | Young, On-Line File Caching | verified | `10.1007/s00453-001-0124-5` |
| 6 | Bansal, Buchbinder, Naor, A primal-dual randomized algorithm for weighted paging | verified | `10.1145/2339123.2339126` |
| 7 | Borodin & El-Yaniv, Online Computation and Competitive Analysis | verified (book) | Cambridge University Press, 1998 |
| 8 | Buchbinder & Naor, The design of competitive online algorithms via a primal-dual approach | verified | `10.1561/9781601982179` |
| 9 | Karlin et al., Competitive snoopy caching | verified | `10.1007/bf01762111` |
| 10 | Karlin et al., Competitive randomized algorithms for nonuniform problems | verified | `10.1007/bf01189993` |
| 11 | Atre et al., Caching with delayed hits | verified | `10.1145/3387514.3405883` |
| 12 | Zinkevich, Online convex programming and generalized infinitesimal gradient ascent | verified (ICML 2003) | no DOI; canonical paper |
| 13 | Hazan, Introduction to online convex optimization | verified | `10.1561/9781680831719` |
| 14 | Shalev-Shwartz, Online learning and online convex optimization | verified | `10.1561/9781601985477` |
| 15 | Ghodsi et al., Dominant resource fairness | verified | `10.5555/1972457.1972490` |
| 16 | Grandl et al., Multi-resource packing for cluster schedulers | verified | `10.1145/2619239.2626334` |
| 17 | Jonas et al., Cloud programming simplified | verified | `10.48550/arxiv.1902.03383` |
| 18 | Oakes et al., SOCK: Rapid task provisioning with serverless-optimized containers | verified via venue page | USENIX ATC 2018, HTTP 200 |
| 19 | Du et al., Catalyzer | verified | `10.1145/3373376.3378512` |
| 20 | Reiss, Wilkes, Hellerstein, Google cluster-usage traces | verified via Google repo | `github.com/google/cluster-data`, HTTP 200 |
| 21 | Iosup et al., The Grid Workloads Archive | verified | `10.1016/j.future.2008.02.003` |

## Notes

- OpenAlex returned the wrong top match for #18 (a 2020 FGCS paper) and #20 (a 2016 CSE
  paper). Both were verified manually: #18 on the USENIX ATC 2018 page, #20 on the Google
  cluster-data repository.
- #7 and #12 are a book and an ICML paper without a DOI; both are canonical and were
  checked by title/venue.
- No reference remains unverifiable. The 21 entries in the PDF can be kept, but #18 and #20
  should be checked against the venue pages during final typesetting.

## AI-use statement

The PDF contains a `[CHECK]` placeholder. Replace it with:

> Claude (Anthropic) was used to help draft and typeset this document, to derive and check
> expressions and proofs, and to review results. Gemini (Google) was used to help implement
> the simulator and the randomized caching logic. The author verified the derivations, ran
> all experiments, and takes responsibility for the content. AI tools are not authors.