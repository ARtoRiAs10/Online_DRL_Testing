"""Build the PDF "Contexts: what the terms mean and where the randomized DRL wins or loses".

Companion to Randomized_vs_Online_DRL_comparison.pdf.  It explains
  * section 7 of that document ("the price of randomization"),
  * a full head-to-head: where Randomized Online-DRL LOSES against the other methods,
  * the two verdicts "surrogate only" (goal G3) and "unsafe reduction" (goal G5),
  * a glossary of the terms used in both documents,
  * how every metric in the comparison is computed.

    python docs/make_contexts_pdf.py [--out ../Contexts_Randomized_vs_Online_DRL.pdf]
"""
from __future__ import annotations
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_comparison_pdf import (P, table, fig, load, f2, H1, H2, BODY, SMALL, CAP, ACCENT,
                                 GOOD, BAD, GREY, CELL, CELLB, LIGHT)

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer

H3 = ParagraphStyle("H3", parent=H2, fontSize=10, leading=13, textColor=colors.HexColor("#33456b"),
                    spaceBefore=7, spaceAfter=2)
KEY = ParagraphStyle("KEY", parent=BODY, fontSize=9, leading=12.4,
                     leftIndent=4 * mm, rightIndent=2 * mm,
                     borderPadding=4, backColor=colors.HexColor("#f3f6fb"),
                     borderColor=colors.HexColor("#c9d2e0"), borderWidth=0.5)


def build(out_path):
    d08, d09 = load("exp08_feasibility_budget"), load("exp09_regret")
    d10, d11 = load("exp10_expected_ratio"), load("exp11_slot_reduction")
    d12b, d12i = load("exp12_multiresource_balanced"), load("exp12_multiresource_imbalanced")
    d13, d13z = load("exp13_gamma_complexity"), load("exp13_gamma_complexity_zipf")
    d14, d15 = load("exp14_time_complexity"), load("exp15_budget_regret")
    d06, d01 = load("exp06_ski_rental"), load("exp01_baseline_vs_llrc")

    doc = BaseDocTemplate(out_path, pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm,
                          topMargin=18 * mm, bottomMargin=16 * mm,
                          title="Contexts: Randomized Online-DRL vs Online-DRL", author="drl-sim")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")

    def deco(canv, d):
        canv.saveState()
        canv.setFont("Helvetica", 7.5)
        canv.setFillColor(GREY)
        canv.drawString(22 * mm, 10 * mm, "Contexts: Randomized Online-DRL vs Online-DRL  -  drl-sim")
        canv.drawRightString(A4[0] - 22 * mm, 10 * mm, "page %d" % d.page)
        canv.setStrokeColor(colors.HexColor("#c9d2e0"))
        canv.line(22 * mm, 13 * mm, A4[0] - 22 * mm, 13 * mm)
        canv.restoreState()

    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=deco)])
    S = []

    # ------------------------------------------------------------------ title
    S.append(P("Contexts and verdicts", ParagraphStyle("T", parent=H1, fontSize=17, leading=21)))
    S.append(P("What &ldquo;the price of randomization&rdquo; means, where Randomized Online-DRL "
               "<b>loses</b> against the other methods, and what the two verdicts "
               "&ldquo;<b>surrogate only</b>&rdquo; and &ldquo;<b>unsafe reduction</b>&rdquo; mean in "
               "the context in which they are used. Companion to "
               "<i>Randomized_vs_Online_DRL_comparison.pdf</i>.", SMALL))
    S.append(Spacer(1, 3 * mm))

    # ------------------------------------------------------------- 1. section 7
    S.append(P("1. What section 7 of the comparison PDF explains", H1))
    S.append(P("Section 7 is titled <b>&ldquo;The price of randomization&rdquo;</b>. It does not "
               "compare the <i>quality</i> of the two algorithms; it quantifies the <b>computing "
               "time</b> that the randomized policy needs, because the randomized policy does far "
               "more work per request than the deterministic one."))
    S.append(P("Why the randomized policy is more expensive", H3))
    S.append(table(
        ["step", "Online-DRL", "Randomized Online-DRL", "cost per request"],
        [["decide to download", "two integer comparisons (T<sub>i</sub>, L<sub>i</sub> against "
          "&#952;M<sub>i</sub>)", "same counters, but one random draw from "
          "f(&#952;) = e<sup>&#952;/M</sup>/(M(e-1))", "O(1)"],
         ["update the cache state", "credit arithmetic on the cached services only "
          "(O(|cache|))", "one gradient step on the fractional vector p, then the Euclidean "
          "projection onto D<sub>vec</sub> (dual ascent on 4 multipliers)", "**O(N)**"],
         ["choose the cache content", "deterministic eviction order", "rounding of K paths "
          "(s<sub>k</sub> = 1[U<sub>k</sub> &lt; p]) plus a feasibility repair of every violating "
          "path", "**O(K&#183;N)**"],
         ["cost per request", "13&#8211;26 &#181;s", "440&#8211;1 665 &#181;s", "**25&#8211;70&#215;**"]],
        [30 * mm, 42 * mm, 62 * mm, 31 * mm]))
    S.append(Spacer(1, 2 * mm))
    rows = [[r["N"], f2(r["Online-DRL"][0], 1), f2(r["LL-RC"][0], 1), f2(r["R3 only"][0], 1),
             f"<b>{f2(r['Randomized DRL'][0], 1)}</b>",
             f2(r["Randomized DRL"][0] / r["Online-DRL"][0], 1) + "&#215;"]
            for r in d14["by_N"]]
    S.append(P("Measured (exp14, microseconds per request, wall clock of the request loop only):",
               SMALL))
    S.append(table(["N (services)", "Online-DRL", "LL-RC", "R3 only", "Randomized", "factor"], rows,
                   [24 * mm, 32 * mm, 22 * mm, 22 * mm, 26 * mm, 22 * mm]))
    S.append(Spacer(1, 2 * mm))
    S.append(P("The power-law fit gives &#181;s/req ~ N<sup>0.52</sup> (R&sup2; = 0.97) for the "
               "randomized policy against N<sup>0.17&#8211;0.32</sup> for the deterministic ones, "
               "and the cost is essentially <b>independent of the cache size k</b> "
               "(fit C<sup>-0.07</sup>). In practice: an edge node serving 1 000 requests/s spends "
               "&#8776; 35 ms of CPU per second with the randomized policy against &#8776; 1 ms "
               "with the deterministic one. This is a real but acceptable cost &ndash; and it is the "
               "price paid for the competitive-ratio improvement of section 5 of the comparison "
               "document.", KEY))
    S.append(fig("exp14_time_complexity.png",
                 "Figure 1 - running time per request against the catalogue size N and the cache "
                 "size k (exp14). The red curve is the randomized policy; it is ~35x above the "
                 "deterministic ones and does not depend on k."))

    # ------------------------------------------------------------- 2. head-to-head
    S.append(P("2. Does Randomized Online-DRL lose against the other methods?", H1))
    S.append(P("Yes, in three clearly identifiable situations, and in none of the others. The "
               "table below is the complete head-to-head of the four algorithms (plus the exact "
               "optimum) over every metric that was measured."))
    m10, m12i = d10["mean"], d12i["algos"]
    rows = [
        ["competitive ratio, adversarial family (exp13)",
         "&#947; = 2.00k", "&#947; = 1.88k", "<font color='#1b6b3a'><b>&#947; &#8776; 3 (best)</b></font>",
         "&#947; = 0.40k", "<b>win</b>"],
        ["competitive ratio, Zipf traffic vs MILP (exp13)",
         "1.443", "1.457", "<font color='#1b6b3a'><b>1.430 (best)</b></font>", "-", "<b>win</b>"],
        ["competitive ratio, short traces T = 120 (exp10)",
         "1.145 &#177; 0.072", "<font color='#1b6b3a'><b>1.102 &#177; 0.058 (best)</b></font>",
         "<font color='#9c2b2b'><b>1.279 &#177; 0.138 (worst)</b></font>", "-", "<b>loss</b>"],
        ["download cost, same trace (exp13/exp12)",
         "reference", "2x det.", "<font color='#1b6b3a'><b>3&#8211;13&#215; lower</b></font>",
         "2x det.", "<b>win</b>"],
        ["latency, multi-resource workload (exp12)",
         "136 226", "139 029", "<font color='#1b6b3a'><b>87 333 (-36 %)</b></font>", "-", "<b>win</b>"],
        ["resource violations of the true constraints",
         "0 %", "0 %", "0 % (after the repair)", "-", "<b>tie</b>"],
        ["download budget G2", "clipped (loses 11.6 % latency)", "not enforced",
         "<font color='#1b6b3a'><b>respected in expectation and per seed</b></font>", "-", "<b>win</b>"],
        ["regret G3, surrogate form (exp09)",
         "-", "-", "<font color='#1b6b3a'><b>always below &#8730;N l<sub>max</sub> &#8730;T</b></font>",
         "-", "<b>win</b>"],
        ["regret G3, literal form (exp09)",
         "-", "-", "<font color='#9c2b2b'><b>3.7&#8211;9.6&#215; the bound, grows with T</b></font>",
         "-", "<b>loss</b>"],
        ["computing time per request (exp14)",
         "13&#8211;26 &#181;s", "12&#8211;29 &#181;s",
         "<font color='#9c2b2b'><b>440&#8211;1 665 &#181;s (25&#8211;70&#215;)</b></font>", "-",
         "<b>loss</b>"],
        ["warm-up needed before the state is useful",
         "none", "none", "<font color='#9c2b2b'><b>yes (visible at T = 120)</b></font>", "-",
         "<b>loss</b>"]]
    S.append(table(["metric", "Online-DRL", "LL-RC", "Randomized DRL", "offline OPT",
                    "verdict for the randomized DRL"], rows,
                   [40 * mm, 25 * mm, 22 * mm, 40 * mm, 17 * mm, 21 * mm],
                   hl={3: (colors.HexColor("#fbeaea"), colors.black),
                       1: (colors.HexColor("#e6f4ea"), colors.black)}))
    S.append(Spacer(1, 2 * mm))
    S.append(P("<b>Summary of the head-to-head.</b> The randomized algorithm wins on the "
               "competitive ratio in the regime that matters (adversarial patterns, large caches), "
               "wins by a factor 3&#8211;13 on download cost, wins on latency on the "
               "multi-resource workload, and is the only one of the three that respects a download "
               "budget. It loses in exactly three places: (i) on <i>short</i> traces it is worse "
               "than both deterministic variants (the fractional state is not learned yet), "
               "(ii) it needs 25&#8211;70&#215; more CPU per request, and (iii) the "
               "<i>literal</i> form of the proposal's goal G3 is not satisfied. Note also that it "
               "never loses on resource feasibility: the repair step makes the constraints exact.",
               KEY))

    # ------------------------------------------------------------- 3. surrogate
    S.append(P("3. What &ldquo;surrogate only&rdquo; means (goal G3)", H1))
    S.append(P("The proposal states a regret goal G3: the algorithm should be close to the best "
               "<i>fixed</i> fractional cache, with a bound of "
               "&#8730;N&#183;l<sub>max</sub>&#183;&#8730;T. The measurement in exp09 has to "
               "decide what is compared with what, and this is where the word "
               "<b>&ldquo;surrogate&rdquo;</b> comes in."))
    S.append(P("The real quantity we would like to bound", H3))
    S.append(P("The natural regret is <b>realized latency</b>:"))
    S.append(P("&#931;<sub>t</sub> l<sub>n<sub>t</sub></sub> &#8722; min<sub>P &#8712; "
               "D<sub>vec</sub></sub> &#931;<sub>t</sub> l<sub>n<sub>t</sub></sub> "
               "1[n<sub>t</sub> &#8712; P] &#8804; Reg<sub>T</sub>", BODY))
    S.append(P("i.e. the latency the system really paid minus the latency the best possible "
               "<i>fixed</i> cache would have paid. The problem is that this expression is "
               "<b>not a function of p alone</b>: whether a request is a hit, a delayed hit or a "
               "forward also depends on the download that may be in flight (the indicator q<sub>t</sub>) "
               "and on the remaining download time dm<sub>t</sub>, both of which are decided by the "
               "discrete download dynamics and not by the fractional vector. The bound of "
               "Proposition 1 cannot be derived for that expression."))
    S.append(P("The surrogate that the bound applies to", H3))
    S.append(P("The proposal replaces the latency by a <b>surrogate loss</b> that <i>is</i> a "
               "function of p:"))
    S.append(P("F<sub>t</sub>(P) = (1 &#8722; p<sub>n<sub>t</sub></sub>)&#183;g<sub>t</sub>, "
               "with g<sub>t</sub> = (1 &#8722; q<sub>t</sub>)&#183;l<sub>n<sub>t</sub></sub> + "
               "q<sub>t</sub>&#183;min(l<sub>n<sub>t</sub></sub>, dm<sub>t</sub>)", BODY))
    S.append(P("It says: <i>a request that is not covered by the fractional cache costs "
               "g<sub>t</sub></i>. The regret that is actually bounded is therefore"))
    S.append(P("&#931;<sub>t</sub> F<sub>t</sub>(P<sub>t</sub>) &#8722; min<sub>P &#8712; "
               "D<sub>vec</sub></sub> &#931;<sub>t</sub> F<sub>t</sub>(P) &#8804; "
               "&#8730;N&#183;l<sub>max</sub>&#183;&#8730;T", BODY))
    S.append(P("because D<sub>vec</sub> has diameter &#8804; &#8730;N and the step size is "
               "&#951; = &#8730;N/(l<sub>max</sub>&#8730;T). The comparator "
               "min<sub>P</sub> &#931;F<sub>t</sub>(P) is the best <b>fixed</b> fractional cache, "
               "computed exactly by a linear program (not an approximation).", SMALL))
    rows = [[r["T"], f2(r["bound"], 0), f2(r["regret_exog_stationary"], 0),
             f2(r["regret_exog_peek"], 0), f2(r["regret_endog"][0], 0),
             "<font color='#9c2b2b'><b>" + f2(r["literal_G3"][0], 0) + "</b></font>"]
            for r in d09["rows"]]
    S.append(P("Measured (exp09):", SMALL))
    S.append(table(["T", "bound &#8730;N l<sub>max</sub> &#8730;T", "surrogate, stationary",
                    "surrogate, peek", "surrogate, endogenous", "<b>literal G3</b>"], rows,
                   [14 * mm, 32 * mm, 30 * mm, 28 * mm, 30 * mm, 31 * mm]))
    S.append(Spacer(1, 2 * mm))
    S.append(P("<b>Reading the table.</b> Every surrogate regret is below the bound (the column "
               "&ldquo;surrogate&rdquo;), for exogenous and for endogenous gradients, with and "
               "without the peek update; the negative values simply mean that the OGD state beats "
               "every fixed fractional cache. The last column is the regret computed with the "
               "<i>realized latency</i> instead of the surrogate: it is 3.7&#8211;9.6&#215; the "
               "bound and it grows with T. <b>&ldquo;Surrogate only&rdquo;</b> is the verdict that "
               "says: <i>the statement of G3 holds for the surrogate loss and only for the "
               "surrogate loss</i> - the literal version of the goal is not supported by this "
               "implementation.", KEY))
    S.append(P("Why the literal version fails, and what would fix it", H3))
    S.append(P("The fractional comparator is allowed to <b>cache a service instantly</b>, while "
               "the real system must first <b>download</b> it (paying M<sub>n</sub> of cost and of "
               "latency) and only then serve it. The comparator is not charged for those "
               "downloads, so the gap between the two is at least the cost of the downloads the "
               "real algorithm was forced to make. Three possible corrections of the goal: "
               "(i) define G3 on the surrogate loss (what the bound actually proves); "
               "(ii) charge the comparator for the downloads its fixed cache implies; "
               "(iii) compare against a &ldquo;download-aware&rdquo; comparator that also has to "
               "wait for the service.", SMALL))

    # ------------------------------------------------------------- 4. unsafe
    S.append(P("4. What &ldquo;unsafe reduction&rdquo; means (goal G5)", H1))
    S.append(P("The proposal replaces the three resource constraints of D<sub>vec</sub> by a "
               "<b>single</b> scalar constraint, the so-called <b>slot reduction</b>:"))
    S.append(P("&#931;<sub>n</sub> &#966;(a<sub>n</sub>)&#183;p<sub>n</sub> &#8804; 1 "
               "&#160;&#160;instead of&#160;&#160; &#931;<sub>n</sub> a<sub>n,r</sub>&#183;"
               "p<sub>n</sub> &#8804; 1 for every r", BODY))
    S.append(P("with a weight function &#966; (dominant resource, a linear or <i>l</i><sub>p</sub> "
               "combination of the resources, shadow prices, AHP or entropy weights). The weight "
               "function is supposed to be a <b>safe</b> surrogate of the true constraints, and "
               "the proposal defines the capacity loss it causes as"))
    S.append(P("&#923;<sub>&#966;</sub> = max<sub>n</sub> &#966;(a<sub>n</sub>) / "
               "max<sub>r</sub> a<sub>n,r</sub> &#160;&#160;(eq. 18)", BODY))
    S.append(P("&#923;<sub>&#966;</sub> &lt; 1 means &ldquo;the reduction charges a service "
               "<i>less</i> than its true bottleneck demand&rdquo;, i.e. the reduced constraint is "
               "<b>weaker</b> than the true one, and the cache it produces can exceed the real "
               "CPU/RAM/disk limits. Such a reduction is called <b>unsafe</b>. The proposal itself "
               "warns about it: with weights on the simplex (&#8721;<sub>r</sub> "
               "&#945;<sub>r</sub> = 1) a service that uses only disk looks cheap, and the "
               "guarantee is safe only if &#945;<sub>r</sub> &#8805; 1 for every r (at the price of "
               "wasting up to a factor R of the capacity).", SMALL))
    rows = []
    for name, v in d11.items():
        rows.append([name, "-" if v["Lambda"] is None else f2(v["Lambda"], 2),
                     f2(v["latency"][0], 0), f2(v["Umax"][0], 3), f2(v["stranded"][0], 3),
                     f2(v["max_util"], 2),
                     ("<font color='#9c2b2b'><b>" + f2(v["violation_rate"][0], 3) + "</b></font>")
                     if v["violation_rate"][0] > 0 else f2(v["violation_rate"][0], 3)])
    S.append(P("Measured on the same randomized algorithm, same trace, only the reduction "
               "changes (exp11, imbalanced workload):", SMALL))
    S.append(table(["reduction &#966;", "&#923;<sub>&#966;</sub>", "latency", "U^max", "stranded W",
                    "max utilisation", "<b>violations of the true resources</b>"], rows,
                   [40 * mm, 20 * mm, 22 * mm, 16 * mm, 20 * mm, 22 * mm, 25 * mm],
                   hl={1: (colors.HexColor("#e6f4ea"), colors.black)}))
    S.append(Spacer(1, 2 * mm))
    S.append(P("<b>Reading the table.</b> The three reductions that <i>look</i> attractive - "
               "entropy weights, AHP weights and uniform shadow prices - all have "
               "&#923;<sub>&#966;</sub> &lt; 1 and give a <i>lower</i> latency than the exact "
               "vector formulation, and they are exactly the three that break the real resource "
               "limits in 55&#8211;85 % of the steps (up to 2.37&#215; the capacity). "
               "The safe reductions (dominant resource, <i>l</i><sub>2</sub>-norm, linear with "
               "&#945; = 1) never violate anything, but they waste capacity: the linear "
               "&#945; = 1 reduction has &#923;<sub>&#966;</sub> = 2.52, leaves the bottleneck "
               "resource at 11 % utilization and costs 71 % more latency. <b>&ldquo;Unsafe "
               "reduction&rdquo;</b> is therefore the warning that a slot reduction is not a free "
               "simplification: unless the weights dominate the true demands, it produces a cache "
               "that the edge node physically cannot hold - and the simulation would show the "
               "violation as a utilization above 1 on the true resources, not as a slower run.", KEY))
    S.append(fig("exp11_slot_reduction.png",
                 "Figure 2 - the same randomized algorithm under different slot reductions "
                 "(exp11). The unsafe ones (entropy, AHP, shadow price) have a lower latency and "
                 "violate the true resource limits."))

    # ------------------------------------------------------------- 5. glossary
    S.append(P("5. Glossary of the terms used in both documents", H1))
    S.append(table(
        ["term", "meaning in this context"],
        [["hit / delayed hit / forwarding", "the three ways a request is served: from the cache "
          "(latency 0), buffered while a download finishes (latency min(l, remaining)), or "
          "forwarded to the cloud (latency l)"],
         ["latency vs. cost", "latency is what the user waits; cost is the download time the "
          "operator pays. They are optimized separately and never mixed in a metric"],
         ["miss epoch, T<sub>i</sub>, L<sub>i</sub>", "the period since the service was last "
          "cached; T<sub>i</sub> = elapsed time, L<sub>i</sub> = l<sub>n</sub> &#215; number of "
          "requests of the service in the epoch. The download trigger compares them with "
          "&#952;M<sub>i</sub>"],
         ["credit (Landlord)", "a real-valued budget attached to every cached service, initialised "
          "to M<sub>f</sub> and reset on every hit; it reaches 0 and the service becomes evictable"],
         ["D<sub>vec</sub>", "the polytope of feasible fractional caches: p &#8712; [0,1]<sup>N</sup> "
          "with &#8721;p<sub>n</sub> &#8804; C and &#8721;a<sub>n,r</sub>p<sub>n</sub> &#8804; 1 for "
          "every resource r"],
         ["projected OGD", "one gradient step on the requested service followed by a projection back "
          "onto D<sub>vec</sub>; the gradient is the surrogate gradient g<sub>t</sub>"],
         ["rounding / repair", "the fractional p is turned into a boolean cache by thresholding K "
          "persistent uniform vectors; a path that violates the constraints is repaired by "
          "evicting services until it fits"],
         ["q<sub>t</sub>, dm<sub>t</sub>", "q<sub>t</sub> = 1 if a download of the requested service "
          "is in flight; dm<sub>t</sub> = its remaining time. Together they decide hit vs. delayed "
          "hit vs. forwarding and they enter the surrogate gradient"],
         ["peek", "update the state with the gradient of request t <i>before</i> serving it "
          "(the proposal's variant) as opposed to after (textbook OGD)"],
         ["&#947; (competitive ratio)", "ALG(I)/OPT(I): the latency of the algorithm divided by the "
          "latency of the offline optimum (the MILP, or the paper's (k+2n)l on the Large "
          "Sequence)"],
         ["&#946;<sub>up</sub>, &#946;<sub>down</sub>", "the fraction of the download slots during "
          "which a request is forwarded instead of buffered; &#946;<sub>up</sub> = max over "
          "services (used with M<sub>x</sub>), &#946;<sub>down</sub> = mean (used with M<sub>min</sub>)"],
         ["U<sub>r,t</sub>, U^max, imbalance, stranded W", "per-resource occupancy, its maximum over "
          "the resources, the spread between the resources, and the capacity of the bottleneck "
          "resource that the other resources cannot use"],
         ["&#923;<sub>&#966;</sub>, &#923;&#770;", "the theoretical capacity loss of a reduction "
          "(eq. 18) and its empirical estimate E[U^max]/E[mean<sub>r</sub>U<sub>r</sub>]"],
         ["surrogate", "a differentiable, p-dependent stand-in for the discrete latency; the "
          "quantity the regret bound applies to"],
         ["unsafe reduction", "a slot reduction whose weights under-charge the true demands "
          "(&#923;<sub>&#966;</sub> &lt; 1), so the resulting cache can exceed the real resource "
          "limits"]],
        [40 * mm, 125 * mm]))

    # ------------------------------------------------------------- 6. metrics
    S.append(P("6. How each metric in the comparison is calculated", H1))
    S.append(table(
        ["metric", "formula / procedure", "why it is the right one"],
        [["latency", "&#931;<sub>t</sub> lat<sub>t</sub> with lat<sub>t</sub> = 0 (hit), "
          "min(l<sub>n</sub>, remaining) (delayed hit) or l<sub>n</sub> (forward)",
          "exactly the objective of the paper"],
         ["cost", "&#931; M<sub>n</sub> over the <i>started</i> downloads",
          "the operator's cost; forwarding is charged through the latency, not the cost"],
         ["downloads", "number of started downloads", "the budget is spent per download"],
         ["competitive ratio &#947;", "ALG(I)/OPT(I), OPT from the MILP (HiGHS) or "
          "(k+2n)l on the Large Sequence", "the paper's definition; status = 1 would mean the "
          "MILP hit the time limit, then the ratio is only a lower bound"],
         ["fit of &#947;(k)", "least squares: linear, logarithmic, power law (log-log), with R&sup2;",
          "decides empirically whether the linear behaviour of Theorem 1 disappears; it is not a "
          "proof"],
         ["U<sub>r,t</sub>", "&#931;<sub>n cached at t</sub> a<sub>n,r</sub>",
          "the proposal's per-resource occupancy"],
         ["imbalance, stranded W", "U^max &#8722; min<sub>r</sub>U<sub>r</sub> and "
          "&#931;<sub>r</sub>(U^max &#8722; U<sub>r</sub>)", "proposal eqs. (30)-(33)"],
         ["violation rate", "fraction of steps with max<sub>r</sub>U<sub>r,t</sub> &gt; 1 + 1e-9",
          "goal G1 is about the <i>true</i> constraints, not the reduced ones"],
         ["&#923;<sub>&#966;</sub>", "max<sub>n</sub> &#966;(a<sub>n</sub>)/max<sub>r</sub>"
          "a<sub>n,r</sub>", "proposal eq. (18)"],
         ["surrogate regret", "&#931;<sub>t</sub>F<sub>t</sub>(P<sub>t</sub>) &#8722; min<sub>P</sub>"
          "&#931;<sub>t</sub>F<sub>t</sub>(P), the minimum computed exactly by an LP",
          "the comparator of Proposition 1, solved to optimality"],
         ["&#946;<sub>up</sub>, &#946;<sub>down</sub>", "per service fwd<sub>i</sub>/M<sub>i</sub>, "
          "where fwd<sub>i</sub> counts the requests forwarded while a download of i was in flight",
          "the paper's beta is exactly that fraction"],
         ["running time", "10<sup>6</sup>&#183;wall-clock time of the request loop / T",
          "measures the algorithm only, not the catalogue or the trace"],
         ["latency per unit resource", "L / &#931;<sub>t</sub>U^max<sub>t</sub>",
          "proposal eq. (34): latency normalized by the bottleneck resource actually used"]],
        [30 * mm, 72 * mm, 63 * mm]))
    S.append(Spacer(1, 3 * mm))
    S.append(P("All numbers are reproducible with <b>python experiments/run_all.py</b>; the raw "
               "values are in <b>results/*.json</b> and the figures in <b>results/*.png</b>. The "
               "companion document <i>Randomized_vs_Online_DRL_comparison.pdf</i> contains the "
               "same data in a shorter, more graphical form, and "
               "<i>docs/simulation_methodology.md</i> the full description of the model.", SMALL))

    doc.build(S)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                                                   "Contexts_Randomized_vs_Online_DRL.pdf"))
    a = ap.parse_args()
    print("written:", build(a.out))
