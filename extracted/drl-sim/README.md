# drl-sim: testing Online-DRL and the proposed Randomized DRL

Simulator for *Edge Service Caching With Delayed Hits* (Deka & Sukapuram, IEEE TCC 2025) and for the
Randomized DRL / multi-resource proposal. Works in a GitHub Codespace (devcontainer included).

```bash
pip install -r requirements.txt && pip install -e .
make test                      # unit tests (~5 s)
python experiments/run_all.py --quick   # smoke run of every experiment (~1 min)
python experiments/run_all.py           # full run (tens of minutes; exp10/exp11 dominate)
python experiments/exp06_ski_rental.py  # or any single experiment; outputs go to results/
```

**Documentation** — [`docs/README.md`](docs/README.md) is the index:
[`docs/simulation_methodology.md`](docs/simulation_methodology.md) explains the model, the
algorithms, the experimental conditions, how every number is computed and how the code follows the
paper and the proposal to the letter; [`docs/results.md`](docs/results.md) holds the measured
results with the figures and an honest discussion of what they do and do not support.

## Layout
| file | role |
|---|---|
| `sim/engine.py` | event-driven edge node: hit / delayed hit / forward, miss epochs (T_i, L_i), download bookkeeping, latency + cost + utilization accounting |
| `sim/landlord.py` | Landlord with resource checks (Algorithm 2) |
| `sim/algos.py` | `online_drl` (theta param), `ll_rc` (theta=0), `rand_landlord` (R3 only), `rand_drl` (R1+R2+R3) |
| `sim/rand_drl.py` | fractional state, OGD step, rounding with repair (R1, R2) + the **v2 revisions**: learned download probability `Q_n` (N1), expectation gradient (N2), active-set projection, served-path rounding, vectorised repair (N7) |
| `sim/projection.py` | dual-ascent projection onto D_vec (eqs. 22-25) + cvxpy reference |
| `sim/opt.py` | offline OPT as a MILP (HiGHS, no Gurobi) |
| `sim/trace.py` | synthetic Zipf/Poisson, CSV loader, the paper's adversarial *Large Sequence* |
| `sim/slots.py`, `sim/regret.py`, `sim/metrics.py` | slot reductions + Lambda, regret LP, resource metrics |

## Claim -> experiment
| claim | script |
|---|---|
| Online-DRL: slightly more latency, far less cost than LL-RC | `exp01` |
| theta trades cost vs latency; theta=0 == LL-RC | `exp02` |
| delayed hits vs uplink / downlink | `exp03` |
| latency vs C and CPU/RAM/disk limits | `exp04` |
| Theorem 1: ratio grows linearly in k (and randomized variants on the same sequence) | `exp05` |
| R3 ski-rental: 2 -> e/(e-1) | `exp06` |
| projection (22)-(25) equals the exact QP | `exp07` |
| G1 feasibility with/without repair, G2 budget | `exp08` |
| G3 regret: exogenous, endogenous, and literal | `exp09` |
| G4 expected ratio vs exact OPT on small traces | `exp10` |
| G5 slot reductions: Lambda, stranded capacity, violations | `exp11` |
| multi-resource comparison of the four algorithms (proposal Sec. 4) | `exp12` |
| does randomization reach/improve the paper's `gamma in O(k(1+3 beta_up M_x))`? | `exp13` |
| computational cost of each algorithm (running time vs N and k) | `exp14` |
| G2 budget + G3 regret with the multi-resource projection | `exp15` |
| the paper's real traces (Google / SHARCNet) - runs as soon as a CSV is dropped in `data/`, otherwise writes the gap report | `exp16` |
| the paper's synthetic experiment (Exps. 15-16): sweep of the popularity skew and the request rate | `exp17` |
| the proposed revisions: horizon-free step size, the corrected goal G3, the serving-path policy | `exp18` |
| **Randomized DRL v2** (the implemented revisions) vs Online-DRL and v1: ratio, cost, runtime, corrected goals | `exp19` |

## Assumptions made where the documents are silent (change them in `sim/rand_drl.py`)
* **A1** q_{n,t} = 1 if the node has a download of n in flight, else 0 (the documents give no update rule).
* **A2** Rounding: K persistent uniform vectors, s = 1[U < p]; marginals are exact and E|s_t - s_{t-1}| = |p_t - p_{t-1}|_1.
* **A3** The serving path k* is fixed per run (`resample_every` redraws it). The real cache = (downloaded) AND (sampled on); a sampled "cached" service that was never downloaded is not a hit.
* **A4** Downloads are triggered by the (random) threshold and, with `gate_downloads=True`, only if the sample currently wants the service. Eviction comes from the sample/projection, not Landlord.
* `peek=True` follows the proposal (update with request t's gradient, then serve). `peek=False` is textbook OGD.
* The paper's constraints (4)-(6) write (1 - x); this repo uses x (sensible reading, as in the proposal).
* OPT detail: the paper's OPT_t = (k+2n)l is off by one `l` at the end of the trace, because in the last repetition OPT can download MS, flush the cache and hit MS's 2nd request. `tests/test_core.py` checks the MILP against (k+2n)l - l.

## Reading the results honestly
* Simulation gives empirical evidence, not proofs. exp05 only shows how *one* adversarial family scales; it cannot confirm or refute O(log k).
* In the quick smoke run on synthetic Zipf data: R3 behaves as theory says (exp06: 2.000 vs 1.582 = e/(e-1)); the projection matches the QP to ~1e-9 (exp07); OGD regret stays below sqrt(N)*l_max*sqrt(T) on the exogenous surrogate (exp09); the repair step removes all resource violations that the un-repaired sample produces (exp08).
* Results that do NOT match the paper's headline: on this synthetic workload Online-DRL had *lower* latency than LL-RC (exp01), whereas the paper reports slightly higher on Google traces. Run on the real traces before drawing conclusions.
* exp09: "literal G3" (realized latency minus the best fixed fractional cache) is several times larger than sqrt(N)*l_max*sqrt(T). The fractional comparator caches instantly while the real system must download first, so G3 as literally written is not supported by this implementation. This is the gap hidden by Assumption 1.
* exp08: Randomized DRL used far fewer downloads than the deterministic algorithm on Zipf traffic; a stationary popularity favours a frequency-like fractional state. Test non-stationary traces before reading anything into that.
* exp10 uses small T (OPT is a MILP). If `opt_status` is 1 (time limit) OPT is only an upper bound, so the reported ratios are lower bounds.
* The full (non-quick) run and the complete discussion are in `docs/results.md`; the headline numbers there are: Online-DRL is exactly linear in k on the adversarial Large Sequence (`gamma = 2k`, R^2 = 1.000), R3 alone does not remove that (`1.88k`), R1+R2+R3 does (`gamma ~ 3`, flat in k) at 13-60x lower download cost, and the randomized policy costs ~35x more CPU per request (exp14).
* **The real datasets used by the paper (Google cluster traces, SHARCNet) are NOT bundled here**, so every number above comes from synthetic Zipf/Poisson traffic or from the paper's adversarial Large Sequence. `exp16` runs the same pipeline on a preprocessed Google/SHARCNet CSV as soon as one is placed in `data/`, and writes an explicit gap report otherwise; `exp17` shows that the sign of the Online-DRL vs LL-RC latency difference does not flip anywhere in the synthetic parameter space (0 of 12 configurations), i.e. the paper's headline latency result can only be checked on the paper's traces. See `docs/Does_the_proposal_hold.pdf`.
* **Proposed revisions to the randomized DRL** (`docs/Proposed_revisions_randomized_DRL.pdf`, measured in `exp18`): the horizon-free step size `eta_t = 1/(l_max sqrt(t))` is *better* than the horizon-dependent one (surrogate regret 696 vs 815 at T = 500); the serving path must be **sticky** - redrawing it per miss or per step costs 20-22 % latency and doubles the download cost; the literal goal G3 is false (3.7-9.6x the bound) but the **download-charged** version `E[sum d] - min_P sum F <= Reg_Prop1 + D_T` holds at every horizon, and the realized regret against the *offline-optimal* static cache is a constant 3.1-3.6 ms per request, so G3c must be restated against an *online* comparator.
* **The revisions are now implemented (v2)** (`docs/Updated_randomized_DRL_v2.pdf`, `exp19`): the download probability becomes a learned variable, the gradient is taken in expectation, the projection runs on the active set and only the served path is rounded and repaired.  On the multi-resource workload v2 matches v1's latency (+0.1 %) with 7 % less cost and **3.4x less computing time** (821 -> 243 us/request), and its surrogate regret is negative at every horizon.  All v1 behaviour remains the default, so the previous results are unchanged (14 unit tests pass).
