# Online DRL testing

Simulator, experiment suite and results used to test and extend the algorithms of
*"Edge Service Caching With Delayed Hits: Minimizing Latency and Reducing Cost"*
(S. Deka, R. Sukapuram, IEEE Trans. Cloud Computing 13(4), 2025) and the randomized
multi-resource DRL proposal.

The canonical, self-contained suite is [`final_exp/`](final_exp/).  It implements the
updated Landlord-Repair randomized DRL (Algorithm 3) and compares it with Online-DRL,
LL-RC and R3-only on synthetic Zipf/Poisson traffic, the adversarial Large Sequence and
the paper's Google cluster traces 2011.

## Layout

| path | content |
|---|---|
| `final_exp/sim/` | simulator: engine, catalog, traces, projection, OPT, algorithms, Landlord-Repair |
| `final_exp/experiments/` | the 17 experiments, `paper_claims.py`, `summarize.py` and `run_all.py` |
| `final_exp/data/` | Google cluster traces prepared with the paper's recipe (`prepare_google.py`) |
| `final_exp/results/` | generated JSON results and PNG figures from the last full run |
| `final_exp/docs/` | generated `REPORT.md`, `PAPER_COMPARISON.md` and `RESULTS_AND_DATASETS.md` |
| `final_exp/tests/` | unit tests for the core and the Landlord-Repair |

## Quick start

```bash
cd final_exp
pip install -r requirements.txt && pip install -e .
make test
python experiments/run_all.py --quick   # smoke run
python experiments/run_all.py           # full run
```

## Results

The short comparison is in [`final_exp/docs/PAPER_COMPARISON.md`](final_exp/docs/PAPER_COMPARISON.md);
the full tables are generated in [`final_exp/docs/REPORT.md`](final_exp/docs/REPORT.md).
[`final_exp/RESULTS_AND_DATASETS.md`](final_exp/RESULTS_AND_DATASETS.md) records every
experiment, the dataset used for each one and the exact run conditions.

Headline findings:

* deterministic Online-DRL is exactly linear in the cache size on the adversarial Large
  Sequence (`gamma = 2.00*k + 0.33`, `R^2 = 1.000`);
* randomizing only the download threshold (R3) does not remove that linear behaviour;
* the full randomized DRL with the updated Landlord-Repair stays at `gamma ~ 3-5` for
  `k = 3..24` and removes every true-resource violation in the feasibility experiment;
* the cost reduction is real (about 72% vs LL-RC in the synthetic full run), while the
  latency penalty is the main empirical cost of the updated repair.

## Documents

Research papers, slide decks, manuscript drafts and their text extractions are
intentionally **not** committed.  They are ignored by [`.gitignore`](.gitignore); the
reproducible code, data, results and generated documentation live in `final_exp/`.