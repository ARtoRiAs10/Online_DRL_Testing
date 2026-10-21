"""Build the PDF "Does the randomized proposal hold?" - the validation status of the proposal
against the paper's datasets, and an explicit account of what was (and was not) run on the
paper's own traces.

    python docs/make_validation_pdf.py [--out ../Does_the_proposal_hold.pdf]
"""
from __future__ import annotations
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_comparison_pdf import (P, table, fig, load, f2, H1, H2, BODY, SMALL, CAP, ACCENT,
                                 GOOD, BAD, GREY, LIGHT, CELL, CELLB)

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer

H3 = ParagraphStyle("H3", parent=H2, fontSize=10, leading=13, textColor=colors.HexColor("#33456b"),
                    spaceBefore=7, spaceAfter=2)
KEY = ParagraphStyle("KEY", parent=BODY, fontSize=9, leading=12.4, leftIndent=4 * mm,
                     rightIndent=2 * mm, borderPadding=5, backColor=colors.HexColor("#f3f6fb"),
                     borderColor=colors.HexColor("#c9d2e0"), borderWidth=0.5)
RED = ParagraphStyle("RED", parent=KEY, backColor=colors.HexColor("#fbeaea"),
                     borderColor=colors.HexColor("#d8a0a0"))
GRN = ParagraphStyle("GRN", parent=KEY, backColor=colors.HexColor("#e9f5ee"),
                     borderColor=colors.HexColor("#9ccbb0"))


def build(out_path):
    d01 = load("exp01_baseline_vs_llrc")
    d13 = load("exp13_gamma_complexity")
    d17 = load("exp17_paper_synthetic_params")
    d16 = load("exp16_real_traces")
    d10 = load("exp10_expected_ratio")

    doc = BaseDocTemplate(out_path, pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm,
                          topMargin=18 * mm, bottomMargin=16 * mm,
                          title="Does the randomized proposal hold? (validation status)",
                          author="drl-sim")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")

    def deco(canv, d):
        canv.saveState()
        canv.setFont("Helvetica", 7.5); canv.setFillColor(GREY)
        canv.drawString(22 * mm, 10 * mm,
                        "Does the randomized proposal hold?  -  validation status  -  drl-sim")
        canv.drawRightString(A4[0] - 22 * mm, 10 * mm, "page %d" % d.page)
        canv.setStrokeColor(colors.HexColor("#c9d2e0"))
        canv.line(22 * mm, 13 * mm, A4[0] - 22 * mm, 13 * mm)
        canv.restoreState()

    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=deco)])
    S = []

    # ------------------------------------------------------------------ title
    S.append(P("Does the randomized proposal hold?", ParagraphStyle("T", parent=H1, fontSize=17,
                                                                    leading=21)))
    S.append(P("A validation status: which claims of the proposal are confirmed by the "
               "experiments in this repository, on <b>which data</b>, and what is still untested "
               "because the paper's own datasets (Google cluster traces, SHARCNet) are not "
               "bundled here.", SMALL))
    S.append(Spacer(1, 3 * mm))
    S.append(P("A.  <b>On the workloads that could be run, the proposal holds</b> - with three "
               "exceptions (the literal form of G3, the warm-up on short traces, and the "
               "computing cost).", GRN))
    S.append(P("<b>B.  None of the experiments was run on the datasets the paper used.</b>  The "
               "Google cluster traces and the SHARCNet trace are not present in this repository, "
               "so every number here comes from synthetic Zipf/Poisson traffic or from the "
               "paper's adversarial Large Sequence.  The paper's headline numbers "
               "(latency +0.016 % - +7.41 %, cost -77.57 % - -92.54 % on Google) are therefore "
               "<b>not</b> confirmed by this work.", RED))
    S.append(P("C.  Where the paper and this repository disagree, the disagreement is exactly "
               "where the trace structure matters: on synthetic Poisson traffic Online-DRL is "
               "<i>faster</i> than LL-RC in 12 of 12 configurations, while on the paper's Google "
               "trace it is always slightly <i>slower</i>.  A conclusion about the paper's "
               "numbers can only be drawn on the paper's data.", KEY))
    S.append(P("D.  <b>The gap is now explicit and closable</b>: <b>experiments/exp16_real_traces.py</b> "
               "runs the whole comparison (Online-DRL, LL-RC, R3-only, Randomized DRL) on a "
               "preprocessed Google/SHARCNet CSV as soon as one is dropped into <b>data/</b>, and "
               "<b>experiments/exp17_paper_synthetic_params.py</b> already probes how much the "
               "answer moves with the popularity skew and the request rate (the axes of the "
               "paper's own synthetic experiments, Exps. 15-16).", GRN))

    # ------------------------------------------------------------ 1 paper data
    S.append(P("1. What the paper used", H1))
    S.append(P("Sec. IV of the paper (Implementation and Evaluation) describes three datasets, "
               "their exact preprocessing, the metrics and the sixteen experiments."))
    S.append(table(
        ["dataset", "how it is built (paper, Sec. IV)", "experiments", "what the paper reports"],
        [["<b>Google cluster traces 2011</b>",
          "Job and Task event files; every JobID occurrence with a distinct timestamp is a "
          "service request; only records with event type SUBMIT are kept; 11 preprocessed files "
          "dataset0..dataset10 with columns JobID, Timestamp, CPU, RAM, Disk, Task Index, "
          "Status; zeros in CPU/RAM/Disk replaced by the column median; the disk size is in "
          "<b>KB</b>; <b>dataset2</b> is used unless stated otherwise",
          "Exp. 1-8 (Figs. 1-8): cache size C, horizon T, uplink, downlink, request size, "
          "resource limits, theta, delayed hits",
          "Online-DRL has <b>0.016 % (best) to 7.41 % (worst) more latency</b> and "
          "<b>77.57 % to 92.54 % less cost</b> than LL-RC"],
         ["<b>SHARCNet</b> (Grid Workloads Archive)",
          "columns ExecutableID, SubmitTime, UsedCPUTime, UsedMemory, JobID, Status; "
          "UsedLocalDiskSpace was -1 everywhere, so the UsedMemory values are copied into it; "
          "SubmitTime is shifted by the first value and equal timestamps are incremented; zeros "
          "-> column median; if the resources of an executable vary, the values of its <b>first</b> "
          "occurrence are used; CPU/RAM/disk in <b>MB</b>, compared against the MAXLIMIT values; "
          "the OPT program is run here only, limited to 1 000 rows (420 unique services, "
          "interspersed)",
          "Exp. 9-14 (Figs. 9-14): the same axes plus the MILP optimum OPT",
          "Online-DRL has <b>9.34 % (best) to 26.62 % (worst) more latency</b> and "
          "<b>41.31 % to 87.30 % more cost</b> than OPT"],
         ["<b>synthetic</b>",
          "the <b>unique services of Google dataset2</b>, with a Zipf popularity (parameter "
          "rho, default 1.6) and Poisson arrivals (parameter xi, default 3e6)",
          "Exp. 15-16 (Figs. 15-16): C together with rho and with xi",
          "as the skew increases, more downloads and lower latency; the cache saturates near "
          "C = 5"]],
        [30 * mm, 63 * mm, 33 * mm, 39 * mm]))
    S.append(Spacer(1, 2 * mm))
    S.append(P("Metrics in the paper: total latency, total cost, number of hits and number of "
               "delayed hits over a horizon T, compared against the offline optimum OPT (a MILP, "
               "Gurobi), the theoretical optimum OPT_t = (k+2n)l, and the baseline LL-RC.  "
               "Parameters seen in the text: uplink 40 -> 80 Mbps, C = 50 000 for the "
               "resource-limit experiment, C = 10 for the theta experiment, horizons of "
               "25 000 / 50 000 / 75 000 / 100 000 rows (Google) and 41 347 rows (SHARCNet).  "
               "Because the algorithms are deterministic, the paper runs each experiment once.",
               SMALL))

    # ------------------------------------------------------------ 2 our data
    S.append(P("2. What this repository actually ran on", H1))
    S.append(table(
        ["", "the paper", "this repository"],
        [["real traces", "Google dataset2 and SHARCNet", "<b>none</b> - they are not bundled "
          "(<b>data/README.md</b> documents the format and the download recipe)"],
         ["synthetic data", "the unique services <i>of Google dataset2</i>, Zipf rho = 1.6, "
          "Poisson xi = 3e6", "a synthetic catalogue of N = 40 services with log-normal "
          "CPU/RAM/disk demands, Zipf popularity (exponent 1.2 by default), Poisson arrivals "
          "(0.5 req/ms by default), capacities = max(cap_mult x mean demand, max demand)"],
         ["adversarial data", "the Large Sequence of the proof of Theorem 1 (OPT = (k+2n)l)",
          "<b>the same</b> Large Sequence, reconstructed in <b>sim/trace.py::large_sequence</b> "
          "and checked against the MILP in <b>tests/test_core.py</b>"],
         ["offline optimum", "a MILP in Gurobi, run only on SHARCNet (1 000 rows)",
          "an independent MILP in HiGHS (<b>sim/opt.py</b>), with delayed hits, forwarding, the "
          "three resource limits, the cardinality and the budget; solved to optimality in every "
          "instance reported"],
         ["repetitions", "one run per experiment (deterministic algorithms)",
          "10 seeds per configuration (the randomized algorithms are stochastic), and the "
          "per-seed spread is reported"]],
        [26 * mm, 62 * mm, 77 * mm]))
    S.append(Spacer(1, 2 * mm))
    S.append(P("The two synthetic workloads are <b>not</b> the same workload.  The paper's "
               "synthetic dataset inherits the <i>real</i> demand vectors of the Google jobs and "
               "the real service set; ours invents log-normal demands.  This matters: the "
               "resource-aware behaviour (the multi-resource part of the proposal) depends "
               "directly on the demand vectors, and the paper's Google dataset is extremely "
               "skewed (only 30 unique services in 1 000 rows, and they are "
               "&ldquo;mostly grouped together&rdquo;, so that the cache size C is "
               "&ldquo;less meaningful&rdquo; in the paper's own words).", KEY))

    # ------------------------------------------------------------ 3 matrix
    S.append(P("3. Claim-by-claim validation status", H1))
    S.append(table(
        ["claim", "paper's evidence", "evidence in this repository", "data used here", "status"],
        [["Online-DRL costs much less than LL-RC",
          "-77.57 % to -92.54 % (Google); -41 % to -87 % vs OPT (SHARCNet)",
          "-50.52 % +- 0.55 % (exp01); -50 % in 12/12 sweep configurations (exp17); the "
          "randomized DRL is at -90 % to -95 %",
          "synthetic", "<font color='#1b6b3a'><b>consistent</b></font> (magnitude differs)"],
         ["Online-DRL is slightly slower than LL-RC",
          "+0.016 % to +7.41 % on Google (always slower)",
          "-2.41 % +- 3.70 % (exp01); slower in <b>0 of 12</b> sweep configurations, mean "
          "-4.22 % (exp17)",
          "synthetic", "<font color='#9c2b2b'><b>NOT reproduced</b></font> - the sign is "
          "opposite"],
         ["theta trades cost against latency", "Exp. 7 (Google), Exp. 14 (SHARCNet)",
          "monotone in both directions, minimum latency near theta = 0.5 (exp02)",
          "synthetic", "<font color='#1b6b3a'><b>reproduced</b></font>"],
         ["delayed hits fall with the uplink and rise with the downlink",
          "Exp. 8 (Google)", "764 -> 173 with the uplink, 94 -> 577 with the downlink (exp03)",
          "synthetic", "<font color='#1b6b3a'><b>reproduced</b></font>"],
         ["latency falls with C and with the resource limits", "Exp. 1, Exp. 6, Exp. 13",
          "monotone in both (exp04)", "synthetic",
          "<font color='#1b6b3a'><b>reproduced</b></font>"],
         ["gamma = Theta(k) on the adversarial family (Thm. 1)", "proof",
          "gamma = 2.00k + 0.33, R^2 = 1.000 (exp05, exp13)",
          "Large Sequence (the paper's own)", "<font color='#1b6b3a'><b>reproduced</b></font>"],
         ["R1+R2+R3 removes the linear-in-k blow-up", "proposal, no experiment in the paper",
          "gamma ~ 3 and flat for k = 3..24, 5-60x less cost (exp05, exp13)",
          "Large Sequence (the paper's own)", "<font color='#1b6b3a'><b>supported</b></font>"],
         ["G1: the repair step makes the constraints exact", "proposal",
          "40 % raw violations -> 0 % after the repair, max utilisation 1.0 vs 1.73 (exp08)",
          "synthetic", "<font color='#1b6b3a'><b>supported</b></font> (data-independent)"],
         ["G2: the expected download budget is respected", "proposal",
          "respected at 20/50/100 % of the deterministic cost, every seed (exp08, exp15)",
          "synthetic", "<font color='#1b6b3a'><b>supported</b></font> (data-independent)"],
         ["G3: the regret bound sqrt(N) l_max sqrt(T)", "proposal",
          "holds for the surrogate (exogenous and endogenous, peek and no-peek); the literal "
          "form is 3.7-9.6x the bound (exp09, exp15)",
          "synthetic", "<font color='#c07a00'><b>surrogate only</b></font>"],
         ["G4: the expected ratio against the exact optimum", "proposal",
          "1.102 (R3), 1.145 (Online-DRL), 1.279 (Randomized) on short traces (exp10)",
          "synthetic + MILP", "<font color='#c07a00'><b>partial</b></font>"],
         ["G5: slot reductions must not break the true constraints", "proposal",
          "entropy / AHP / shadow prices violate the true limits in 55-85 % of the steps; the "
          "exact vector formulation dominates every safe reduction (exp11, exp12)",
          "synthetic", "<font color='#1b6b3a'><b>supported</b></font>"],
         ["the randomized DRL is affordable", "-",
          "25-70x the deterministic CPU cost per request (exp14)",
          "synthetic", "<font color='#9c2b2b'><b>not free</b></font>"]],
        [34 * mm, 30 * mm, 45 * mm, 24 * mm, 32 * mm]))
    S.append(Spacer(1, 2 * mm))
    S.append(P("<b>Read the last column carefully:</b> &ldquo;reproduced on synthetic data&rdquo; is "
               "not &ldquo;reproduced on the paper's data&rdquo;.  Of the twelve rows above, "
               "none has been tested on the Google or the SHARCNet trace, because those traces "
               "are not in this repository.", RED))

    # ------------------------------------------------------------ 4 the disagreement
    S.append(P("4. The one clear disagreement: the sign of the latency difference", H1))
    S.append(P("The paper reports that Online-DRL is <i>always</i> slightly slower than LL-RC on "
               "the Google traces (+0.016 % to +7.41 %) and that LL-RC achieves more hits at a "
               "high cost.  On our synthetic traffic the opposite holds: Online-DRL is faster "
               "in 8 of 10 seeds and in 12 of 12 sweep configurations."))
    rows = [[f"{r['rho']:.1f}", f"{r['rate']:.2f}", f"{r['lat_drl_pct'][0]:+.2f} %",
             f"{r['lat_drl_pct'][1]:.2f}", f"{r['cost_drl_pct'][0]:+.2f} %",
             f"{r['lat_rand_pct'][0]:+.2f} %", f"{r['cost_rand_pct'][0]:+.2f} %"] for r in d17["rows"]]
    S.append(P("Sweep over the popularity skew rho and the request rate (exp17, the axes of the "
               "paper's synthetic Exps. 15-16; the paper's default is rho = 1.6):", SMALL))
    S.append(table(["rho (skew)", "rate (req/ms)", "Online-DRL vs LL-RC: latency", "sd",
                    "Online-DRL vs LL-RC: cost", "Randomized vs LL-RC: latency",
                    "Randomized vs LL-RC: cost"], rows,
                   [19 * mm, 24 * mm, 30 * mm, 12 * mm, 28 * mm, 30 * mm, 22 * mm]))
    S.append(Spacer(1, 2 * mm))
    S.append(P("The sign of the latency effect does <b>not</b> flip inside the synthetic "
               "parameter space: Online-DRL is faster in every configuration, and the gap grows "
               "with the skew (rho = 1.0: -2.5 %; rho = 2.0: -7.9 %) and with the request rate.  "
               "The magnitude of the cost effect, on the other hand, is stable at about -50 %, "
               "while the paper measures -78 % to -93 % on Google.  Both facts point at the same "
               "conclusion: <b>the trace structure, not the algorithm, decides the sign of the "
               "latency effect</b>.", KEY))
    S.append(P("Why the Google trace behaves differently", H3))
    S.append(P("In the Google dataset only <b>30 unique services appear in 1 000 rows and they "
               "are mostly grouped together</b> (the paper's own observation, Sec. IV).  A trace "
               "with long consecutive runs of the same service changes the trade-off in two ways:"))
    S.append(table(
        ["effect of grouped requests", "consequence for the comparison"],
        [["a request that arrives while the previous download of the same service is still in "
          "flight is a <b>delayed hit</b> or a forwarding, and LL-RC always has a download in "
          "flight (it downloads on every miss)",
          "LL-RC pays a forwarding latency l on every such request, while Online-DRL may have "
          "the service cached - this is where LL-RC loses latency"],
         ["with few services and long runs, the <b>cache</b> keeps the service for a long time, "
          "so the extra hits of LL-RC are worth more",
          "the paper's &ldquo;LL-RC achieves a higher number of hits, but at a high cost&rdquo;"],
         ["a service can be requested again shortly after its download finished, so the "
          "<b>delayed hit</b> mechanism (which our simulator reproduces exactly) is exercised far "
          "more often than with independent Poisson draws per request",
          "the latency of a forward is partially replaced by a buffered wait, which changes the "
          "sign of the Online-DRL - LL-RC difference"]],
        [80 * mm, 85 * mm]))
    S.append(Spacer(1, 2 * mm))
    S.append(P("Our synthetic generator draws the service of every request "
               "<b>independently</b> (Zipf + Poisson), so consecutive requests rarely concern the "
               "same service; the Google trace is strongly autocorrelated.  This is a property of "
               "the <i>workload generator</i>, not of the simulator - and it is exactly why the "
               "paper's numbers cannot be checked here.", RED))
    S.append(fig("exp01_baseline_vs_llrc.png",
                 "Figure 1 - per-seed latency and cost difference of Online-DRL with respect to "
                 "LL-RC on synthetic Zipf traffic (exp01). Eight of the ten seeds are negative, "
                 "i.e. Online-DRL is faster - the opposite of the paper's Google result."))

    # ------------------------------------------------------------ 5 dependency
    S.append(P("5. Which conclusions survive without the real traces?", H1))
    S.append(table(
        ["result", "does it depend on the trace?", "why"],
        [["G1: the repair step removes every violation of the true constraints",
          "<b>no</b>", "it is a property of the rounding and of the repair rule, not of the "
          "traffic: the repair is applied to every violating sample path"],
         ["G2: the expected download budget is respected", "<b>no</b>",
          "it follows from the trigger probability budget_left/(M_n x requests_left) and the "
          "engine's hard budget check"],
         ["G3 (surrogate): the regret stays below sqrt(N) l_max sqrt(T)",
          "<b>almost no</b>", "the bound is a property of projected OGD on D_vec; it was tested "
          "on exogenous, endogenous, stationary and popularity-shifting gradient sequences, and "
          "on two different workloads"],
         ["R2/R3 mechanics: the projection equals the QP; e/(e-1) = 1.582",
          "<b>no</b>", "both are mathematical identities, verified to 2e-9 and to 4 decimals"],
         ["gamma = 2k for the deterministic algorithm", "<b>no</b>",
          "the Large Sequence is the paper's own adversarial family and is reproduced exactly"],
         ["R1+R2+R3 makes gamma flat on that family", "<b>partly</b>",
          "the family is the paper's, but 'flat' is an empirical observation over k = 3..24; an "
          "adversarial family designed against the randomized algorithm does not exist yet"],
         ["the ratio on ordinary traffic (1.43 vs 1.43)", "<b>yes</b>",
          "it is a workload statistic; the value of the Zipf skew, the request rate and the "
          "autocorrelation of the trace all move it (exp17)"],
         ["the sign of the Online-DRL - LL-RC latency difference", "<b>yes - and it flips</b>",
          "12/12 synthetic configurations disagree with the paper's Google result"],
         ["G5: unsafe slot reductions violate the true limits", "<b>partly</b>",
          "the safety analysis is algebraic, but the numerical damage (55-85 % of the steps) "
          "depends on the demand vectors, which on Google are real job footprints"],
         ["the multi-resource advantage of the randomized DRL", "<b>yes</b>",
          "on a stationary popularity part of the gain is simply 'download less'; a "
          "non-stationary resource mix is needed to isolate the multi-resource effect"]],
        [56 * mm, 30 * mm, 79 * mm]))
    S.append(Spacer(1, 2 * mm))
    S.append(P("<b>In short:</b> the <b>mechanism-level</b> results (the repair, the budget, the "
               "surrogate bound, the ski-rental constant, the projection) are established; the "
               "<b>workload-level</b> results (the ratio, the sign of the latency difference, the "
               "size of the multi-resource gain) still need the paper's traces.", GRN))

    # ------------------------------------------------------------ 6 closing the gap
    S.append(P("6. How to close the gap", H1))
    S.append(P("The pipeline is already in place - only the data are missing."))
    S.append(table(
        ["step", "what to do", "where"],
        [["1", "download the Google cluster traces 2011 (clusterdata-2011-2, Job and Task event "
          "files) and/or the SHARCNet trace from the Grid Workloads Archive",
          "not redistributable, therefore not bundled here"],
         ["2", "preprocess exactly as Sec. IV of the paper describes: one row per request with "
          "the columns service id, timestamp, CPU, RAM, disk; keep the records with event type "
          "SUBMIT; replace zeros by the column median; take the resources of a service from its "
          "first occurrence; disk in KB for Google, in MB for SHARCNet",
          "the recipe is also written down in <b>data/README.md</b> and implemented in "
          "<b>sim/trace.py::load_trace_csv</b>"],
         ["3", "put the file in <b>data/</b> (e.g. data/google.csv) and run "
          "<b>python experiments/exp16_real_traces.py</b>",
          "runs Online-DRL, LL-RC, R3-only and Randomized DRL on the trace and reports latency, "
          "cost, hits, delayed hits and the difference with respect to LL-RC; the paper's "
          "metrics exactly"],
         ["4", "for the paper's synthetic dataset (Exps. 15-16), rebuild it from the unique "
          "services of Google dataset2 with Zipf rho = 1.6 and Poisson arrivals, then sweep rho, "
          "xi and C",
          "<b>experiments/exp17_paper_synthetic_params.py</b> already does the sweep on a "
          "synthetic catalogue, and the loader accepts the real services"],
         ["5", "run the OPT program on the SHARCNet rows (the paper limits it to 1 000 rows) "
          "and compare",
          "<b>sim/opt.py</b> is an independent MILP (HiGHS) with the same model; it is verified "
          "against the paper's OPT_t = (k+2n)l in <b>tests/test_core.py</b>"]],
        [10 * mm, 82 * mm, 73 * mm]))
    S.append(Spacer(1, 2 * mm))
    S.append(P("Until step 2 is done, <b>experiments/exp16_real_traces.py</b> writes an explicit "
               "gap report into <b>results/exp16_real_traces.json</b> (status "
               "&ldquo;NO REAL TRACE AVAILABLE&rdquo;) and lists every claim that is still "
               "tested only on synthetic data, so the experiment suite itself documents the "
               "limitation instead of hiding it.", SMALL))
    S.append(fig("exp17_paper_synthetic_params.png",
                 "Figure 2 - the sweep of exp17: the latency difference of Online-DRL with "
                 "respect to LL-RC (green = Online-DRL faster) never changes sign over 12 "
                 "configurations, and the cost difference stays at about -50 %."))

    # ------------------------------------------------------------ 7 bottom line
    S.append(P("7. Bottom line", H1))
    S.append(P("<b>Does the randomized proposal hold with respect to the experimental results?</b> "
               "On everything that could be measured without the paper's datasets, yes: the "
               "repair step is necessary and sufficient for the resource constraints, the "
               "download budget is respected in expectation and per seed, the surrogate regret "
               "stays under the Proposition-1 bound, the ski-rental constant is exactly "
               "e/(e-1), the projection is exact, and on the paper's own adversarial family the "
               "randomized algorithm removes the linear-in-k blow-up at 5-60x less download cost.",
               GRN))
    S.append(P("<b>But the experiments were NOT run against the datasets the paper used.</b>  The "
               "Google cluster traces and the SHARCNet trace are absent from this repository; "
               "every number above comes from synthetic traffic or from the paper's adversarial "
               "sequence.  Consequently the paper's own headline results are <b>neither "
               "confirmed nor refuted</b> here - with one exception that matters: on synthetic "
               "traffic the latency comparison has the <b>opposite sign</b> to the paper's Google "
               "result, and the sweep in exp17 shows that no setting of the synthetic axes "
               "reproduces it.  That is a statement about the <i>workload</i>, not about the "
               "algorithm, and it can only be settled by running the code on the paper's traces, "
               "which the pipeline is ready for.", RED))
    S.append(P("The three claims that would change if the real traces were used: (i) the sign and "
               "the size of the Online-DRL - LL-RC latency difference; (ii) the size of the "
               "cost reduction (ours -50 %, the paper -78 % to -93 %); (iii) whether the "
               "multi-resource advantage of the randomized algorithm survives when the request "
               "pattern is autocorrelated and the demand vectors are real job footprints.",
               SMALL))

    doc.build(S)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                                                   "Does_the_proposal_hold.pdf"))
    a = ap.parse_args()
    print("written:", build(a.out))
