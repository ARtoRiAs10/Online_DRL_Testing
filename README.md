# Online DRL testing — repository guide

This repository contains the material used to test and extend the algorithms of

* the paper **"Edge Service Caching With Delayed Hits: Minimizing Latency and Reducing Cost"**
  (S. Deka, R. Sukapuram, IEEE Trans. Cloud Computing 13(4), 2025)
  → `Edge_Service_Caching_With_Delayed_Hits_Minimizing_Latency_and_Reducing_Cost.pdf` (+ `.txt`
  extraction),
* the proposal **"Randomized DRL with Multi-Resource (CPU, RAM, Disk) Capacity: From Slots to
  Resource Vectors"** → `randomized_drl_multiresource.pdf` (+ `.txt` extraction),
* the (corrupted) problem statement → `randomized_drl_problem_statement.pdf`.

and the simulator, experiments and documentation built on top of them.

## Layout

| path | content |
|---|---|
| `extracted/drl-sim/` | the simulator (`sim/`), the experiments (`experiments/`), the results (`results/`), the tests (`tests/`) |
| `extracted/drl-sim/docs/` | **the documentation**: [methodology](extracted/drl-sim/docs/simulation_methodology.md) and [results](extracted/drl-sim/docs/results.md) |
| `Randomized_vs_Online_DRL_comparison.pdf` | **PDF summary** (6 pages, 5 figures): how Online-DRL and Randomized Online-DRL work, and whether the randomized one improves or only equalizes the deterministic one |
| `Randomized_vs_Online_DRL_slides.pptx` | **slide deck** (17 slides): the difference between the two algorithms, what improves, where the randomized one loses, and how each metric is calculated |
| `Contexts_Randomized_vs_Online_DRL.pdf` | **contexts PDF**: what &ldquo;the price of randomization&rdquo; means, the full head-to-head (where Randomized Online-DRL loses), and what &ldquo;surrogate only&rdquo; (goal G3) and &ldquo;unsafe reduction&rdquo; (goal G5) mean in context |
| `Does_the_proposal_hold.pdf` | **validation status**: does the randomized proposal hold w.r.t. the results, and were the experiments run on the paper's datasets (Google / SHARCNet)? - **no, they were not**; the PDF lists the claim-by-claim status, the one disagreement and how to close the gap |
| `Proposed_revisions_randomized_DRL.pdf` | **the non-concrete parts of the proposal and the revised formulas**: the seven gaps (N1-N7) with the replacement expressions, three of them already implemented and measured, the revised algorithm and the revised goals G1-G5 |
| `Updated_randomized_DRL_v2.pdf` | **the updated method (v2)** with the four implemented revisions (learned download probability, expectation gradient, active-set projection, served-path rounding): the algorithm, the parameter sensitivity and the comparison with Online-DRL and Randomized DRL v1 |
| `extracted/drl-sim/results/` | the raw numbers (JSON) and the figures (PNG) of the last full run |
| `*.txt` | text extractions of the three PDFs (`randomized_drl_problem_statement.txt` is empty: that PDF is corrupted) |

## Quick start

```bash
cd extracted/drl-sim
pip install -r requirements.txt && pip install -e .
make test                       # unit tests
python experiments/run_all.py   # full run, ~30 min, writes results/*.json + results/*.png
```

## Headline results (full run, synthetic Zipf traffic + the paper's adversarial Large Sequence)

* **Online-DRL is exactly linear in the cache size on the adversarial family**:
  `γ = 2.00·k + 0.33`, `R² = 1.000` (paper's Theorem 1 allows `O(k(1+3β_up M_x))`).
* **Randomizing only the download threshold (R3) does not help**: `γ = 1.88k`, still linear —
  even though R3 alone is exactly optimal in the rent-or-buy sense (`2 → e/(e−1) = 1.582`).
* **The full randomized DRL (R1 fractional OGD + R2 randomized rounding with repair + R3)
  removes the linear behaviour**: `γ ≈ 3`, flat in `k` from 3 to 24, with 5–60× less download
  cost — at 25–70× more CPU per request.
* **The repair step of R2 is mandatory**: the un-repaired rounding violates the true
  CPU/RAM/disk constraints in 40 % of the (path, step) pairs and overshoots the capacity by
  up to 1.73×; with the repair, the violation rate is 0.
* **Unsafe slot reductions are the dominant failure mode of the reduction**: weights on the
  simplex (entropy, AHP) or uniform shadow prices give `Λ_φ < 1` and a *lower* latency while
  violating the true constraints in 55–85 % of the steps. The exact vector formulation dominates
  every safe reduction.
* **The paper's cost claim reproduces** (Online-DRL costs ≈ 50 % less than LL-RC here, 78–93 %
  less in the paper); **its latency claim does not** on synthetic traffic (Online-DRL is 2.4 %
  *faster* here, the paper reports 0.02–7.4 % slower on Google traces). The real traces are not in
  this repository — that check is still open.
* **G3 as literally written is not supported**: the *surrogate* regret is always below the
  Proposition-1 bound `√N·l_max·√T`, but the realized-latency version of G3 is 3.7–9.6× the
  bound and grows with `T` (the fractional comparator caches instantly, the real system must
  download first).

Everything above, with the exact experimental conditions and the figures, is in
[`extracted/drl-sim/docs/results.md`](extracted/drl-sim/docs/results.md); the model, the
algorithms and the claim → code mapping are in
[`extracted/drl-sim/docs/simulation_methodology.md`](extracted/drl-sim/docs/simulation_methodology.md).
