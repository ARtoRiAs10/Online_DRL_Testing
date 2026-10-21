# Documentation

| document | content |
|---|---|
| [`Updated_randomized_DRL_v2.pdf`](Updated_randomized_DRL_v2.pdf) | **the updated method (v2)**: the four revisions that could be implemented are now in the simulator (N1 the download probability as a decision variable, N2 the expectation gradient, N7 the active-set projection, the served-path rounding and the vectorised repair) - the algorithm, the parameter sensitivity and the comparison with Online-DRL and Randomized DRL v1 |
| [`Proposed_revisions_randomized_DRL.pdf`](Proposed_revisions_randomized_DRL.pdf) | **the non-concrete parts and the revised formulas**: the seven gaps of the proposal (N1-N7) with the replacement expressions, the revised algorithm and the revised goals G1-G5 |
| [`Does_the_proposal_hold.pdf`](Does_the_proposal_hold.pdf) | **validation status**: does the randomized proposal hold w.r.t. the experimental results, and **were the experiments run on the paper's datasets?** (they were not - the Google/SHARCNet traces are not bundled), with the claim-by-claim matrix, the one clear disagreement and the exact procedure to close the gap |
| [`Randomized_vs_Online_DRL_slides.pptx`](Randomized_vs_Online_DRL_slides.pptx) | **slide deck (17 slides)**: how the two algorithms differ, what the randomized one improves and where it loses, and how every metric is calculated (charts regenerated from `results/*.json`) - also as [`PDF`](Randomized_vs_Online_DRL_slides.pdf) |
| [`Contexts_Randomized_vs_Online_DRL.pdf`](Contexts_Randomized_vs_Online_DRL.pdf) | **the contexts**: what &ldquo;the price of randomization&rdquo; (section 7 of the comparison PDF) means, the complete head-to-head showing where Randomized Online-DRL **loses**, and what &ldquo;<b>surrogate only</b>&rdquo; (goal G3) and &ldquo;<b>unsafe reduction</b>&rdquo; (goal G5) mean, plus a glossary and the calculation of every metric |
| [`Randomized_vs_Online_DRL_comparison.pdf`](Randomized_vs_Online_DRL_comparison.pdf) | **PDF summary**: how Online-DRL and Randomized Online-DRL work, and whether the randomized one improves or only equalizes the deterministic one (6 pages, 5 figures, all numbers taken from `results/*.json`) |
| [`simulation_methodology.md`](simulation_methodology.md) | the model, the algorithms (paper + proposal), the experimental conditions, how every number is computed, the claim → code fidelity table, and the assumptions made where the documents are silent |
| [`results.md`](results.md) | the measured results, with tables, figures and an honest discussion of what the data do and do not support |
| [`figures/`](figures/) | the figures used in `results.md` and in the slides (copies of `results/*.png` and the regenerated charts) |

The PDF/PPTX are generated from the JSON results, so they can never drift from the data:

```bash
pip install reportlab python-pptx
python docs/make_comparison_pdf.py   # 6-page comparison PDF
python docs/make_contexts_pdf.py      # contexts PDF (surrogate, unsafe reduction, head-to-head)
python docs/make_slides.py            # 17-slide deck
python docs/make_validation_pdf.py    # validation status vs the paper's datasets
python docs/make_revisions_pdf.py     # proposed formula revisions (N1-N7)
python docs/make_v2_pdf.py            # the updated method (v2) and its performance
```

## Source documents

* **Paper** — *Edge Service Caching With Delayed Hits: Minimizing Latency and Reducing Cost*,
  S. Deka, R. Sukapuram, IEEE Transactions on Cloud Computing 13(4), 2025.
  (`../../Edge_Service_Caching_With_Delayed_Hits_Minimizing_Latency_and_Reducing_Cost.txt` is the
  extracted text.)
* **Proposal** — *Randomized DRL with Multi-Resource (CPU, RAM, Disk) Capacity: From Slots to
  Resource Vectors* (`../../randomized_drl_multiresource.txt`).
* **Problem statement** — `../../randomized_drl_problem_statement.pdf`: **corrupted**, the binary
  streams are destroyed and no reader can open it. The parts of it that are implemented were
  reconstructed from the proposal and from the code; the assumptions that this forced are listed
  in §6 of the methodology document.

## Reproducing the results

```bash
pip install -r requirements.txt && pip install -e .
make test
python experiments/run_all.py           # ~20 min, writes results/*.json + results/*.png
```

See the [methodology](simulation_methodology.md) for the exact conditions of every experiment and
[results](results.md) for the outcome.
