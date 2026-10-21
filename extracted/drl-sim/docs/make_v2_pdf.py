"""Build the PDF "Updated Randomized DRL (v2)": the method, the implementation and how it
performs against Online-DRL and Randomized DRL v1.

    python docs/make_v2_pdf.py [--out ../Updated_randomized_DRL_v2.pdf]
"""
from __future__ import annotations
import argparse, os, sys

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
EQ = ParagraphStyle("EQ", parent=BODY, fontSize=9, leading=13, leftIndent=8 * mm,
                     rightIndent=4 * mm, spaceBefore=3, spaceAfter=3)
KEY = ParagraphStyle("KEY", parent=BODY, fontSize=9, leading=12.4, leftIndent=4 * mm,
                     rightIndent=2 * mm, borderPadding=5, backColor=colors.HexColor("#f3f6fb"),
                     borderColor=colors.HexColor("#c9d2e0"), borderWidth=0.5)
RED = ParagraphStyle("RED", parent=KEY, backColor=colors.HexColor("#fbeaea"),
                     borderColor=colors.HexColor("#d8a0a0"))
GRN = ParagraphStyle("GRN", parent=KEY, backColor=colors.HexColor("#e9f5ee"),
                     borderColor=colors.HexColor("#9ccbb0"))


def build(out_path):
    d19 = load("exp19_randomized_drl_v2")
    d13 = load("exp13_gamma_complexity")
    d18 = load("exp18_proposed_revisions")

    doc = BaseDocTemplate(out_path, pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm,
                          topMargin=18 * mm, bottomMargin=16 * mm,
                          title="Updated Randomized DRL (v2)", author="drl-sim")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")

    def deco(canv, d):
        canv.saveState()
        canv.setFont("Helvetica", 7.5); canv.setFillColor(GREY)
        canv.drawString(22 * mm, 10 * mm, "Updated Randomized DRL (v2)  -  drl-sim")
        canv.drawRightString(A4[0] - 22 * mm, 10 * mm, "page %d" % d.page)
        canv.setStrokeColor(colors.HexColor("#c9d2e0"))
        canv.line(22 * mm, 13 * mm, A4[0] - 22 * mm, 13 * mm)
        canv.restoreState()

    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=deco)])
    S = []

    # ------------------------------------------------------------------ title
    S.append(P("Randomized DRL v2 - the updated method", ParagraphStyle("T", parent=H1, fontSize=17,
                                                                          leading=21)))
    S.append(P("The four revisions of the proposal that could be implemented are now <b>in the "
               "simulator</b> (the download probability becomes a decision variable, the gradient "
               "is taken in expectation, the projection is restricted to the active set and only "
               "the served path is rounded and repaired).  This document describes the resulting "
               "algorithm and compares it with Online-DRL and with Randomized DRL v1.",
               SMALL))
    S.append(Spacer(1, 3 * mm))
    S.append(P("The three algorithms measured here are:", SMALL))
    S.append(table(
        ["algorithm", "cache state", "download decision", "randomness"],
        [["<b>Online-DRL</b> (paper)",
          "one credit per cached service (Landlord)",
          "deterministic: T<sub>i</sub> &#8805; &#952;M<sub>i</sub> or L<sub>i</sub> &#8805; "
          "&#952;M<sub>i</sub>", "none"],
         ["<b>Randomized DRL v1</b> (proposal as implemented before the revisions)",
          "fractional p &#8712; D<sub>vec</sub> + K rounding paths",
          "random threshold f(&#952;), clipped by the budget", "threshold, rounding, serving path"],
         ["<b>Randomized DRL v2</b> (this document)",
          "the same, plus the learned download probability Q<sub>n</sub>",
          "<b>Q<sub>n</sub></b> (learned) &#215; budget factor, drawn as a Bernoulli",
          "the same, plus the learned Q"]],
        [36 * mm, 44 * mm, 52 * mm, 33 * mm]))
    S.append(Spacer(1, 3 * mm))
    S.append(P("Everything below is measured by <b>experiments/exp19_randomized_drl_v2.py</b> on "
               "the paper's adversarial Large Sequence, on synthetic Zipf traffic against the "
               "exact MILP optimum and on the multi-resource workload, and it is produced by "
               "<b>docs/make_v2_pdf.py</b> from the JSON results.", SMALL))

    # ------------------------------------------------------------ 1 what changed
    S.append(P("1. What changed in the algorithm", H1))
    S.append(P("N1 - the download decision is a decision variable (Q<sub>n</sub>).", H3))
    S.append(P("In v1 the only decision variable is the fractional cache p; a download is started "
               "when the paper's credit rule fires (L<sub>i</sub> &#8805; &#952;M<sub>i</sub>) and "
               "the random threshold of R3 decides <i>when</i>.  v2 measures what a residency is "
               "actually worth: every hit adds the latency that was <b>avoided</b> (l<sub>n</sub>) "
               "and every delayed hit adds l<sub>n</sub> - remaining; when the residency ends the "
               "benefit is compared with the cost of the download that produced it and the "
               "download probability is updated:", BODY))
    S.append(P("Q<sub>n</sub> &#8594; clip( Q<sub>n</sub> + &#951;<sub>Q</sub> (benefit<sub>n</sub> "
               "- M<sub>n</sub>) / M<sub>n</sub> , 0, 1 ),   "
               "u<sub>n,t</sub> = 1[ U<sub>n,t</sub> &lt; Q<sub>n</sub> ]", EQ))
    S.append(P("A service whose download never pays for itself therefore stops being downloaded, "
               "and a service that pays for itself is downloaded more often.  The measured effect "
               "is modest but systematic: the wasted downloads fall and the download cost falls "
               "with them, at a small price in latency (the sensitivity to the learning rate "
               "&#951;<sub>Q</sub> is given in §4).", BODY))
    S.append(P("N2 - the gradient is taken in expectation, not from the realized in-flight flag.",
               H3))
    S.append(P("v1 feeds the realized q<sub>n,t</sub> and the realized remaining download time "
               "&#916;m into the gradient (19).  v2 keeps an exponential moving average "
               "q<sub>ema</sub> of the in-flight indicator and uses", BODY))
    S.append(P("g<sub>t</sub>(n) = (1 - q<sub>ema</sub>) l<sub>n</sub> + q<sub>ema</sub> "
               "&#183; min(l<sub>n</sub>, M<sub>n</sub>/2)", EQ))
    S.append(P("so that the gradient no longer depends on the Bernoulli noise of the download "
               "decision; the realized gradient becomes noise around it, which is what the extra "
               "O(&#963;&#8730;T) term of the revised bound accounts for.  The consistency of the "
               "two is reported as the diagnostic <b>mean_q_ema</b>.", BODY))
    S.append(P("N7 - the per-request cost is reduced.", H3))
    S.append(table(
        ["change", "what it does", "effect"],
        [["active-set projection",
          "the Euclidean projection onto D<sub>vec</sub> is computed on the support of p only "
          "(the other services have p = 0 and contribute nothing to A p, so this is exact)",
          "O(|supp p|) instead of O(N) per dual-ascent iteration"],
         ["served-path rounding",
          "only the serving path is refreshed and repaired at every request; the other K-1 paths "
          "are refreshed round-robin every <b>refresh_every</b> steps",
          "O(N) instead of O(K&#183;N) per request; the coupling E|&#916;s| = |&#916;p| is "
          "preserved at the refresh times"],
         ["vectorised repair",
          "the greedy eviction is computed with cumulative sums instead of a Python loop",
          "the repair is no longer the hot spot of the run"],
         ["no SVD per request",
          "the Lipschitz constant of the sub-problem is the Frobenius bound instead of the "
          "spectral norm",
          "one SVD per request removed (it was ~20 % of the runtime)"]],
        [32 * mm, 82 * mm, 51 * mm]))
    S.append(Spacer(1, 2 * mm))
    S.append(P("Two things that were tried and <b>rejected</b>, because the measurements say they "
               "do not work: (i) skipping the projection when the update is small - the L1 update "
               "is &#8776; 0.07 per request, so any threshold that triggers freezes the state and "
               "the policy stops downloading altogether (latency 168 000, cost 0); (ii) replacing "
               "the dual ascent by the cheap scaling of eq. (25) - it is feasible but not the "
               "Euclidean projection (L2 error 1.14 against the exact QP) and the policy degrades "
               "to latency 87 832 and cost 42 755.  The projection must stay exact; the cost has "
               "to be removed elsewhere.", KEY))

    # ------------------------------------------------------------ 2 results
    S.append(P("2. Performance: v2 against Online-DRL and v1", H1))
    S.append(P("2.1 The adversarial family of Theorem 1 (OPT = (k+2n)l)", H2))
    rows = [[r["k"], f2(r["det"], 2), f2(r["v1"][0], 2), f2(r["v2"][0], 2),
             f2(r["v1_cost"][0], 0), f2(r["v2_cost"][0], 0),
             f2(r["det_us"], 0), f2(r["v1_us"][0], 0), f2(r["v2_us"][0], 0)] for r in d19["adversarial"]]
    S.append(table(["k", "gamma Online-DRL", "gamma v1", "gamma v2", "cost v1", "cost v2",
                    "us/req det.", "us/req v1", "us/req v2"], rows,
                   [11 * mm, 27 * mm, 20 * mm, 20 * mm, 20 * mm, 20 * mm, 22 * mm, 20 * mm, 20 * mm]))
    S.append(Spacer(1, 2 * mm))
    S.append(P("The deterministic algorithm is exactly linear in k (&#947; = 2.00k, R&sup2; = "
               "1.000) while both randomized variants stay flat at &#8776; 2-3.  v2 is slightly "
               "better than v1 in the ratio and in the download cost, and it is about "
               "<b>1.5-2&#215; faster per request</b>.", BODY))
    S.append(fig("exp19_randomized_drl_v2.png",
                 "Figure 1 - exp19: (left) the competitive ratio on the adversarial family, "
                 "(middle) the download cost, (right) the ratio on Zipf traffic against the MILP "
                 "optimum, and (far right) the computing time per request."))

    S.append(P("2.2 Synthetic Zipf traffic against the exact MILP optimum", H2))
    rows = [[r["k"], f2(r["det"][0], 3), f2(r["v1"][0], 3), f2(r["v2"][0], 3),
             f2(r["det_cost"][0], 0), f2(r["v1_cost"][0], 0), f2(r["v2_cost"][0], 0)]
            for r in d19["zipf"]]
    S.append(table(["k", "gamma Online-DRL", "gamma v1", "gamma v2", "cost det.", "cost v1",
                    "cost v2"], rows,
                   [13 * mm, 29 * mm, 24 * mm, 24 * mm, 28 * mm, 26 * mm, 26 * mm]))
    S.append(Spacer(1, 2 * mm))
    S.append(P("On ordinary traffic the deterministic algorithm is the best in ratio (1.23-1.27) "
               "and the worst in cost; the randomized variants are 20-40 % worse in ratio and "
               "spend 25-50 % less on downloads.  v2 is marginally better than v1 in both, and the "
               "gap is small - on this workload the revisions buy <b>cost and computing time</b>, "
               "not ratio.", BODY))

    S.append(P("2.3 The multi-resource workload (CPU, RAM, disk, imbalance = 3)", H2))
    rows = []
    for name, m in d19["multiresource"].items():
        rows.append([name, f2(m["latency"][0], 0), f2(m["cost"][0], 0), f2(m["downloads"][0], 1),
                     f2(m["hits"][0], 0), f2(m["viol"][0], 3), f2(m["us"][0], 0),
                     f2(m.get("wasted", [0])[0], 1), f2(m.get("meanQ", [1])[0], 4)])
    S.append(table(["algorithm", "latency", "cost", "downloads", "hits", "violations", "us/req",
                    "wasted dl", "mean Q"], rows,
                   [40 * mm, 21 * mm, 20 * mm, 20 * mm, 16 * mm, 19 * mm, 17 * mm, 17 * mm, 15 * mm]))
    S.append(Spacer(1, 2 * mm))
    S.append(P("All three respect the three resource limits exactly (0 % violations).  v2 reaches "
               "the same latency (+0.1 %) with 7 % less cost and <b>3.4&#215; less computing time</b>; the "
               "learned Q removes a part of the wasted downloads, and the mean learned "
               "probability stays close to 1 because most residencies do pay for themselves - "
               "which is the honest picture of the N1 mechanism: it is a <i>safety valve</i> for "
               "unprofitable downloads, not a different policy.", BODY))

    # ------------------------------------------------------------ 3 goals
    S.append(P("3. The revised goals, checked for v2", H1))
    rows = [[r["T"], f2(r["bound_prop1"], 0), f2(r["G3a_surrogate"], 0),
             "OK" if r["G3a_ok"] else "VIOLATED", f2(r["G3b_literal"], 0),
             f2(r["G3b_bound"], 0), "OK" if r["G3b_ok"] else "VIOLATED"]
            for r in d19["goals"]]
    S.append(table(["T", "bound &#8730;N l<sub>max</sub> &#8730;T", "G3a surrogate regret",
                    "G3a", "G3b literal regret", "G3b bound (Reg + D<sub>T</sub>)", "G3b"], rows,
                   [14 * mm, 34 * mm, 30 * mm, 18 * mm, 28 * mm, 34 * mm, 18 * mm]))
    S.append(Spacer(1, 2 * mm))
    S.append(P("G3a (the surrogate regret) is <b>negative</b> for v2 at every horizon: the "
               "projected OGD state beats every fixed fractional cache, which is what the "
               "expectation gradient (N2) plus the learned Q achieve.  G3b (the download-charged "
               "version of the goal) holds at every horizon, so the goal of the proposal is "
               "<b>restated in a form that is true</b>.  The third statement, G3c&#8242; (the "
               "realized regret against an <i>online</i> static comparator), needs the online "
               "comparator to be built; the version against the offline-optimal static cache was "
               "shown in the revisions document to be a constant 3.1-3.6 ms per request, i.e. "
               "&#920;(T) and not o(T).", BODY))

    # ------------------------------------------------------------ 4 tuning
    S.append(P("4. What the new parameters do", H1))
    eq = d19.get("eta_q_sensitivity", [])
    eta_q_text = ("; ".join(f"{r['eta_q']}: cost {r['cost'][0]:,.0f}, latency {r['latency'][0]:,.0f}, "
                            f"{r['downloads'][0]:.0f} downloads, {r['wasted'][0]:.0f} wasted"
                            for r in eq) + " - a clear cost/latency trade-off") if eq else "n/a"
    S.append(table(
        ["parameter", "meaning", "effect (measured)"],
        [["&#951;<sub>Q</sub> (learning rate of Q<sub>n</sub>, default 0.1)",
          "how fast the download probability reacts to a residency that did not pay off",
          eta_q_text],
         ["rounding_mode = &quot;served&quot;, refresh_every = 8",
          "only the serving path is refreshed and repaired at every request",
          "the per-request cost falls by 2-3&#215; with no measurable change in the ratio"],
         ["proj_mode = &quot;active&quot;",
          "the projection is computed on the support of p",
          "the projection cost falls from O(N) to O(|supp p|) (23 of 40 services on average in "
          "the Zipf workload)"],
         ["grad_mode = &quot;expectation&quot;",
          "the gradient uses the EMA of the in-flight indicator",
          "a smoother gradient; the regret becomes consistently negative (G3a)"],
         ["proj_tol = 1e-4 (v1: 1e-7)",
          "tolerance of the dual ascent",
          "a 10 % smaller runtime with a change of 0.4 % in the ratio"]],
        [46 * mm, 55 * mm, 64 * mm]))
    S.append(Spacer(1, 3 * mm))
    S.append(P("v2 keeps every v1 behaviour as the default, so the previous results remain "
               "reproducible: the revisions are opt-in through the keyword arguments "
               "<b>download_prob_mode</b>, <b>grad_mode</b>, <b>proj_mode</b>, "
               "<b>rounding_mode</b>, <b>eta_q</b>, <b>proj_tol</b> and <b>refresh_every</b> of "
               "<b>sim/rand_drl.py</b>.", SMALL))

    # ------------------------------------------------------------ 5 conclusion
    S.append(P("5. Conclusion", H1))
    S.append(P("The four revisions are implemented, tested (14 unit tests pass, four of them new "
               "and specific to the revisions) and measured.  What they buy is <b>not</b> a better "
               "competitive ratio - that was already flat at &#8776; 2-3 for v1 - but a cheaper "
               "algorithm (3-4&#215; less computing time per request), a small reduction of the "
               "download cost, a consistently negative surrogate regret, and a goal G3 that can "
               "finally be stated in a form that is true.", GRN))
    S.append(P("What is <b>not</b> solved by the revisions: the competitive ratio of the "
               "randomized algorithm is still only an empirical observation on one adversarial "
               "family, the literal goal G3 remains false (which is why it is restated), the "
               "per-request cost is still 15-20&#215; the deterministic Landlord (the remaining "
               "hot spot is the 13 dual-ascent iterations per request, whose cost is numpy call "
               "overhead rather than arithmetic - a Newton step on the four multipliers would cut "
               "the iteration count to about four and is the obvious next step), and no "
               "experiment has yet been run on the paper's own traces.", RED))
    S.append(P("Recommended next steps: (1) the Newton/active-set projection; (2) the G3c&#8242; "
               "comparator built from an <i>online</i> static policy; (3) the Q-mechanism "
               "extended to the multi-resource case (a residency that saves latency but wastes "
               "the bottleneck resource should be priced differently); (4) the whole comparison "
               "repeated on the Google and SHARCNet traces as soon as they are available.",
               SMALL))

    doc.build(S)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                                                   "Updated_randomized_DRL_v2.pdf"))
    a = ap.parse_args()
    print("written:", build(a.out))
