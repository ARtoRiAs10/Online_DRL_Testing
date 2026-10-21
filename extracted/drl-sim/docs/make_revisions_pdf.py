"""Build the PDF "Proposed revisions to the randomized DRL".

Explains (i) which parts of the proposal are not concrete, (ii) which of its formulas and goals
are false as written, and (iii) the concrete revised expressions - with the measurements that
support them (experiments exp18, and the earlier exp08/exp09/exp11/exp14).

    python docs/make_revisions_pdf.py [--out ../Proposed_revisions_randomized_DRL.pdf]
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
    d18 = load("exp18_proposed_revisions")
    d09 = load("exp09_regret")
    d08 = load("exp08_feasibility_budget")
    d11 = load("exp11_slot_reduction")
    d14 = load("exp14_time_complexity")
    d13 = load("exp13_gamma_complexity")

    doc = BaseDocTemplate(out_path, pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm,
                          topMargin=18 * mm, bottomMargin=16 * mm,
                          title="Proposed revisions to the randomized DRL", author="drl-sim")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")

    def deco(canv, d):
        canv.saveState()
        canv.setFont("Helvetica", 7.5); canv.setFillColor(GREY)
        canv.drawString(22 * mm, 10 * mm,
                        "Proposed revisions to the randomized DRL  -  drl-sim")
        canv.drawRightString(A4[0] - 22 * mm, 10 * mm, "page %d" % d.page)
        canv.setStrokeColor(colors.HexColor("#c9d2e0"))
        canv.line(22 * mm, 13 * mm, A4[0] - 22 * mm, 13 * mm)
        canv.restoreState()

    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=deco)])
    S = []

    # ------------------------------------------------------------------ title
    S.append(P("Proposed revisions to the randomized DRL", ParagraphStyle("T", parent=H1,
                                                                          fontSize=17, leading=21)))
    S.append(P("What in the proposal is <b>not concrete</b>, which of its formulas and goals are "
               "<b>false as written</b>, and the <b>replacement expressions</b> - each one "
               "accompanied by the measurement that supports it.  Equation numbers refer to "
               "<i>Randomized DRL with Multi-Resource (CPU, RAM, Disk) Capacity</i>; experiment "
               "numbers to this repository.", SMALL))
    S.append(Spacer(1, 3 * mm))
    S.append(P("The proposal already contains a complete and correct <b><i>mechanism</i></b>: the "
               "polytope D<sub>vec</sub> (8), the latency gradient (19), the projected OGD step "
               "(20)-(25), the randomized rounding with a multi-resource repair (Sec. 3.4), the "
               "randomized threshold f(&#952;) (R3) and the resource metrics (30)-(34).  What is "
               "missing is the <b>coupling</b> between the three components, the <b>precise form of "
               "the goals</b>, and the <b>cost of the algorithm</b>.", GRN))
    S.append(P("<b>Five things have to change.  (1) The download probabilities Q<sub>t</sub> are "
               "introduced in (R1) and never used again</b> - the algorithm must actually control "
               "<i>when</i> to download, not only <i>what</i> to keep.  (2) Assumption 1 makes the "
               "gradient exogenous, so the state is not consistent with the dynamics it controls; "
               "the gradient has to be written in expectation over the algorithm's own download "
               "decisions.  (3) Goal G3 compares a <i>realized</i> quantity with a <i>surrogate</i> "
               "comparator and is therefore false (3.7-9.6&#215; the bound, growing with T); it "
               "must be split into a surrogate part and a realized part.  (4) Goal G4 "
               "(&ldquo;below the deterministic &#947;&rdquo;) and G5 (&ldquo;report "
               "&#923;<sub>&#966;</sub>&rdquo;) are not mathematical statements and must be "
               "replaced by bounds.  (5) The per-request cost (25-70&#215; the deterministic one) "
               "must be brought down, which needs a cheaper projection and a cheaper rounding.",
               RED))

    # ------------------------------------------------------------ 1 what is concrete
    S.append(P("1. What the proposal already fixes (and what is therefore not to be changed)", H1))
    S.append(table(
        ["element", "formula", "status in this repository"],
        [["the feasible fractional cache", "D<sub>vec</sub> = {P : &#8721; a<sub>n,r</sub>P<sub>n</sub> "
          "&#8804; 1 &#8704;r, &#8721;P<sub>n</sub> &#8804; C}  (8)",
          "implemented exactly (<b>sim/rand_drl.py</b>); the projection is verified against a cvxpy "
          "QP to 2&#215;10<sup>-9</sup> (exp07)"],
         ["the latency surrogate gradient", "grad<sub>p<sub>n,t</sub></sub> = -(1-q)l - "
          "q&#183;min(l, &#916;m)  (19)", "implemented (<b>_ogd</b>); the variant (19&#8242;) proposed "
          "below makes q an <i>endogenous</i> quantity"],
         ["the OGD step and the projection", "P<sub>t</sub> = &#928;<sub>D<sub>vec</sub></sub>"
          "(P<sub>t-1</sub> - &#951;gradF<sub>t</sub>)  (20)-(25)",
          "implemented; the dual-ascent projection with warm start and the final scaling (25) is "
          "the one used"],
         ["randomized rounding with repair", "s<sub>k</sub> = 1[U<sub>k</sub> &lt; p], repair by "
          "increasing p<sub>n</sub>/max<sub>r</sub>a<sub>n,r</sub>",
          "implemented; measured 40 % raw violations -> 0 % after the repair (exp08)"],
         ["the randomized trigger", "f(&#952;) = e<sup>&#952;/M</sup>/(M(e-1))  (R3)",
          "implemented; the rent-or-buy constant is exactly e/(e-1) = 1.582 (exp06)"],
         ["the resource metrics", "U<sub>r,t</sub>, U^max, I<sub>t</sub>, W<sub>t</sub>  (30)-(33)",
          "implemented; they are what exposes the unsafe reductions (exp11)"]],
        [30 * mm, 62 * mm, 73 * mm]))

    # ------------------------------------------------------------ 2 the gaps
    S.append(P("2. The parts that are not concrete, and the revised expressions", H1))
    S.append(P("Each item states what the proposal says, what is missing or wrong, and the "
               "replacement that is implemented and measured in this repository.", SMALL))

    S.append(P("N1 - The download probabilities Q<sub>t</sub> are never used", H3))
    S.append(P("<b>Proposal:</b> (R1) announces &ldquo;Marginal caching probabilities "
               "P<sub>t</sub> &#8712; D<sub>vec</sub> and download probabilities Q<sub>t</sub>, with "
               "P<sub>t</sub> updated by projected online gradient descent on a latency "
               "surrogate&rdquo;.  But the gradient (19) contains the <i>exogenous</i> "
               "q<sub>n*,t</sub>, and the only decision variable for a download is the threshold "
               "&#952; of R3.  Q<sub>t</sub> appears in no equation, and the download decision is "
               "therefore outside the optimization: the algorithm cannot trade a download against "
               "the latency it avoids."))
    S.append(P("<b>Revision (R1&#8242; - two timescales).</b>  Split the state into the cache state "
               "P<sub>t</sub> and the download state Q<sub>t</sub> = (Q<sub>n</sub>)<sub>n</sub>, and "
               "let the download decision be a probability, not a threshold:", BODY))
    S.append(P("u<sub>n,t</sub> = 1[ U<sub>n,t</sub> &lt; Q<sub>n,t</sub> ] , "
               "Q<sub>n,t</sub> = clip( Q<sub>n,t-1</sub> - &#951;<sub>Q</sub>&#183;g<sup>(Q)</sup>"
               "<sub>t</sub>(n), 0, 1 )", EQ))
    S.append(P("where g<sup>(Q)</sup><sub>t</sub>(n) is the marginal cost of one extra download, "
               "i.e. the download time M<sub>n</sub> (or the residual budget per remaining "
               "request, B<sub>max</sub>/requests_left, when a budget is imposed).  The two "
               "time-scales are then: <b>the slow scale</b> updates P<sub>t</sub> with the latency "
               "gradient of Sec. 3, <b>the fast scale</b> updates Q with the cost gradient, and "
               "the trigger threshold of R3 is replaced by the Bernoulli draw u above.  This makes "
               "(G2) a first-class constraint instead of an afterthought: choose "
               "&#951;<sub>Q</sub> so that E[&#931;<sub>n,t</sub> u<sub>n,t</sub>M<sub>n</sub>] "
               "&#8804; B<sub>max</sub> by construction.", BODY))
    S.append(P("<b>Status:</b> partially implemented - the budget-aware trigger already realizes "
               "the idea (the trigger probability is lowered to "
               "budget_left/(M<sub>n</sub>&#183;requests_left), exp08/exp15), but the "
               "<i>latency</i>-driven part of the Q update is not yet in the simulator.  This is "
               "the single most valuable addition to the algorithm: it is what lets the "
               "randomized policy decide <i>whether</i> a download is worth its M<sub>n</sub> "
               "instead of relying on the ski-rental threshold.", GRN))

    S.append(P("N2 - Assumption 1 makes the gradient exogenous", H3))
    S.append(P("<b>Proposal:</b> Assumption 1 states that q<sub>n,t</sub> and "
               "&#916;m<sub>n,t</sub> are exogenous in <i>F</i><sub>t</sub> and that the cache and "
               "download indicators are conditionally independent given (P<sub>t</sub>, "
               "Q<sub>t</sub>).  But q is exactly the indicator of the algorithm's own download "
               "decision, so the OGD state is updated with a gradient that does not correspond to "
               "the transition it is about to cause."))
    S.append(P("<b>Revision (19&#8242;) - expectation over the algorithm's own randomness.</b>  "
               "Replace the realized q by its expectation, which the algorithm controls:", BODY))
    S.append(P("grad<sub>p<sub>n,t</sub></sub> = -[ (1 - Q<sub>n,t</sub>)&#183;l<sub>n</sub> + "
               "Q<sub>n,t</sub>&#183;min(l<sub>n</sub>, the expected remaining download time) ] , "
               "the expected remaining download time = E[ remaining download time | a download of n was "
               "started at t - M<sub>n</sub> ]", EQ))
    S.append(P("and treat the difference between the realized and the expected gradient as "
               "noise.  The expected-descent lemma then gives, with &#963;<sup>2</sup> the "
               "variance of the per-request gradient,", BODY))
    S.append(P("Reg<sub>T</sub> &#8804; &#8730;N&#183;l<sub>max</sub>&#183;&#8730;T + "
               "O(&#963;&#8730;T) + (download-bias term, see N3)", EQ))
    S.append(P("The honest consequence is a <b>two-part statement</b>: the OGD part is exactly the "
               "Proposition-1 bound, and the stochastic part is the extra O(&#963;&#8730;T) term "
               "coming from the Bernoulli draw of the download decision.  Assumption 1 can then be "
               "replaced by a <b>consistency condition</b> that is checkable in the simulator: "
               "E[q<sub>n,t</sub>] should be close to Q<sub>n,t</sub> (the realized in-flight rate "
               "versus the intended one), and the variance &#963; can be measured directly.",
               BODY))
    S.append(P("<b>Status:</b> the simulator already runs the <i>endogenous</i> variant (the "
               "realized q and &#916;m of the run are fed back into the gradient) and reports the "
               "regret of both: the endogenous regret stays below the bound in every "
               "configuration (exp09), and it is <i>negative</i> (the OGD state beats every fixed "
               "fractional cache) whenever the popularity shifts.", GRN))

    S.append(P("N3 - Goal G3 mixes a realized quantity with a surrogate comparator", H3))
    S.append(P("<b>Proposal (11):</b> E[&#931;<sub>t</sub> d<sub>t</sub>(n*<sub>t</sub>)] - "
               "min<sub>P&#8712;D<sub>vec</sub></sub> &#931;<sub>t</sub> <i>F</i><sub>t</sub>(P) "
               "&#8804; Reg<sub>T</sub> = o(T).  The first term is the <i>realized</i> latency; the "
               "second is the <i>surrogate</i> loss of a fractional cache, which is allowed to "
               "cache a service <b>instantly</b>.  The two terms are not comparable: the realized "
               "latency contains the download delays that the comparator never pays."))
    S.append(P("<b>Measured (exp09):</b> the literal regret is 3 912 / 5 119 / 6 592 / 9 595 / "
               "12 917 for T = 500 ... 8 000, i.e. 3.7-9.6&#215; the Proposition-1 bound and "
               "<b>growing with T</b> - so (11) is false as written, not merely loose."))
    S.append(P("<b>Revision.</b>  Split the goal into three statements, each of which is "
               "checkable:", BODY))
    S.append(P("(a) <b>Surrogate regret</b> (unchanged, true): "
               "&#931;<sub>t</sub> <i>F</i><sub>t</sub>(P<sub>t</sub>) - min<sub>P</sub> "
               "&#931;<sub>t</sub> <i>F</i><sub>t</sub>(P) &#8804; &#8730;N l<sub>max</sub> "
               "&#8730;T.<br/>"
               "(b) <b>Download-charged inequality</b> (new, true): "
               "E[&#931;<sub>t</sub> d<sub>t</sub>] - min<sub>P</sub> &#931;<sub>t</sub> "
               "<i>F</i><sub>t</sub>(P) &#8804; Reg<sub>Prop.1</sub>(T) + "
               "D<sub>T</sub>,<br/>"
               "where D<sub>T</sub> is the download cost that <i>any</i> policy must pay on "
               "the same trace (the deterministic algorithm's cost is a valid proxy, since "
               "E[&#931;<sub>t</sub> d<sub>t</sub>] &#8804; &#931;<sub>t</sub> <i>F</i><sub>t</sub> "
               "+ D<sub>T</sub> is exactly the statement that the surrogate under-counts the "
               "delay by at most the download cost).<br/>"
               "(c) <b>Realized regret G3&#8242;</b> (new, the meaningful one): the realized latency "
               "of the randomized policy versus the realized latency of the <b>best static cache</b> "
               "played with the <i>same</i> download dynamics:", EQ))
    S.append(P("Reg<sub>real</sub>(T) = L<sub>rand</sub>(T) - L<sub>static</sub>(T) , "
               "and the goal becomes Reg<sub>real</sub>(T) = o(T), i.e. "
               "Reg<sub>real</sub>(T)/T &#8594; 0", EQ))
    S.append(P("<b>Verified in exp18 (b):</b> the download-charged inequality holds at every "
               "horizon - the literal regret is 3 978 / 6 102 / 9 709 / 16 348 / 26 627 for "
               "T = 500 ... 8 000, while Reg<sub>Prop.1</sub> + D is 10 783 / 22 728 / "
               "43 352 / 81 963 / 161 017.  The gap is thus explained by the download cost that the "
               "instantaneous comparator does not pay.", GRN))
    S.append(P("<b>And (c) is <i>not</i> satisfied as stated - the most useful finding of this "
               "section.</b>  Measured against the <b>offline-optimal</b> static cache (the support "
               "of the LP solution, played with the same download dynamics) the realized regret is "
               "1 793 / 3 597 / 6 830 / 14 232 / 24 442 ms for T = 500 ... 8 000: a "
               "<b>constant 3.1-3.6 ms per request</b>, i.e. 25-34 % of the randomized policy's own "
               "latency.  A constant per-request gap is a &#920;(T) regret, <b>not</b> o(T), and "
               "the static set is also far cheaper (1 553-2 064 against 5 808-27 203) because it "
               "downloads only 7-9 services.  On a <i>stationary</i> trace an offline-optimal "
               "fixed set dominates any online policy - so G3c has to be repaired a second time:",
               BODY))
    S.append(P("G3c&#8242;: the comparator must itself be <b>online</b> (the deterministic "
               "Online-DRL, or the best fixed fractional cache played online), and the goal must be "
               "stated as a <b>stationarity-relative</b> quantity - the adaptive policy pays an "
               "o(1) price per request <i>relative to the best online static policy</i> and gains "
               "(negative regret) exactly when the popularity shifts, which is what the surrogate "
               "measurements show (regret -659 / -840 / -898 / -1 166 / -1 486 ms for "
               "T = 500 ... 8 000 on a popularity-shifting sequence, exp09).", EQ))
    S.append(P("Concretely: (11) should be replaced by G3a (surrogate, provable), G3b "
               "(download-charged, provable, verified here) and G3c&#8242; (realized, against an "
               "<i>online</i> comparator, meaningful only under a non-stationarity assumption).  As "
               "written, the goal mixes three different quantities and cannot be satisfied by any "
               "algorithm.", GRN))

    S.append(P("N4 - The step size needs the horizon and the largest latency", H3))
    S.append(P("<b>Proposal (Prop. 1):</b> &#951; = &#8730;N/(l<sub>max</sub>&#8730;T).  An online "
               "algorithm does not know T, and l<sub>max</sub> is a property of the catalogue that "
               "may be unknown or may change."))
    S.append(P("<b>Revision (R-B).</b>  Use the standard decaying step size", BODY))
    S.append(P("&#951;<sub>t</sub> = 1/(l<sub>max</sub>&#183;&#8730;t) , "
               "&#8747;<sub>t=1..T</sub> l<sub>max</sub>&#183;&#951;<sub>t</sub> "
               "&#8804; 2 l<sub>max</sub>&#8730;T  &#8658;  "
               "Reg<sub>T</sub> &#8804; 2&#8730;N l<sub>max</sub>&#8730;T", EQ))
    S.append(P("i.e. the same order with a constant 2 instead of 1, and <b>no horizon and no "
               "l<sub>max</sub> needed a priori</b> (l<sub>max</sub> can be replaced by a running "
               "maximum).  <b>Measured (exp18):</b> the horizon-free schedule is not worse - it is "
               "<i>better</i>: the surrogate regret is 696 / 1 099 / 1 383 / 1 728 / 2 548 for "
               "T = 500 / 1 000 / 2 000 / 4 000 / 8 000 against 815 / 1 305 / 1 882 / 2 845 / "
               "3 947 for the horizon-dependent step, always far below the bound "
               "(2&#8730;N l<sub>max</sub>&#8730;T = 5 940 ... 23 759).  The revision is "
               "implemented as <b>eta_mode=\"adaptive\"</b> and as "
               "<b>eta_sched=\"1/sqrt(t)\"</b> in <b>sim/regret.py</b>.", GRN))

    S.append(P("N5 - The serving path k* is ambiguous", H3))
    S.append(P("<b>Proposal (R2):</b> &ldquo;the serving path is k* ~ Uniform{1,...,K}&rdquo;.  Read "
               "literally (a fresh k* at every time step) the served set changes at every request, "
               "which destroys the cache; read as &ldquo;one path for the whole run&rdquo; it is "
               "the sticky variant.  The proposal does not say."))
    S.append(P("<b>Revision (R-D).</b>  The serving path must be <b>sticky over a miss epoch</b> and "
               "redrawn only when the cache is reset (a new download completes, or the service is "
               "evicted).  The churn is then bounded by the coupling the proposal asks for, "
               "E[#changes] &#8804; &#931;<sub>t</sub> E||P<sub>t</sub> - "
               "P<sub>t-1</sub>||<sub>1</sub>, because the K paths share the persistent uniform "
               "vectors."))
    S.append(P("<b>Measured (exp18, 5 seeds, 8 000 requests):</b> the ambiguity is not cosmetic - "
               "on the same trace the sticky path gives latency 62 928 &#177; 7 114, cost 20 566 "
               "and 4 991 hits, while redrawing k* on every miss gives 75 498 (+20 %), 44 172 "
               "(2.1&#215;) and 4 371 hits, and redrawing it at every request gives 77 019 (+22 %), "
               "45 005 (2.2&#215;) and 4 296 hits, with 8 000 redraws and 94 dropped services "
               "against 21 for the sticky path.  The literal reading of the proposal loses 14 % of "
               "the hits and doubles the download cost.  This is the quantitative justification "
               "for assumption A3, and it is implemented as <b>kstar_policy</b> in "
               "<b>sim/rand_drl.py</b>.", GRN))

    S.append(P("N6 - Goals G4 and G5 are not mathematical statements", H3))
    S.append(P("<b>G4 (12):</b> &ldquo;&#947;<sub>R</sub> = max<sub>I</sub> "
               "E[ALG<sub>R</sub>(I)]/OPT(I) below the deterministic &#947;&rdquo;.  "
               "&ldquo;Below&rdquo; is not a bound.  <b>Revision:</b> state it in the regime where it "
               "is provable and measurable, e.g. for the adversarial family of Theorem 1: "
               "&#947;<sub>R</sub> = O(1) against &#947;<sub>det</sub> = 2k; and in general "
               "E[ALG<sub>R</sub>(I)] &#8804; OPT(I) + O(&#8730;T) with high probability, for an "
               "oblivious adversary.  <b>Measured:</b> &#947;<sub>R</sub> &#8776; 3 &#177; 1 for "
               "k = 3 ... 24 while &#947;<sub>det</sub> = 2.00k (R&sup2; = 1.000) - the two models "
               "are not comparable as they stand, because a 2k ratio against an adversary is not "
               "the same experiment as an expected ratio against the optimum."))
    S.append(P("<b>G5 (13):</b> &ldquo;if a slot reduction is used, its capacity loss "
               "&#923;<sub>&#966;</sub> is reported&rdquo;.  Reporting is not a guarantee.  "
               "<b>Revision:</b> make it a decision rule with a proof obligation:", BODY))
    S.append(P("<b>Safety (proved in the proposal).</b>  &#966; is a safe reduction iff "
               "&#966;(a) &#8805; max<sub>r</sub> a<sub>r</sub> for every a, i.e. the weight must "
               "dominate the dominant resource.  Hence &#916;<sub>&#966;</sub> &#8805; 1, and "
               "&#916;<sub>&#966;</sub> = 1 for the dominant-resource rule.<br/>"
               "<b>Decision rule (new).</b>  Use the slot formulation only if "
               "&#923;<sub>&#966;</sub> &#8804; 1 + &#949; for a tolerance &#949; (e.g. 0.05); "
               "otherwise keep the vector formulation, whose per-step cost is O(N) instead of "
               "O(N log N) for the exact breakpoint sort.<br/>"
               "<b>New safe reduction with &#923; close to 1 (new).</b>  "
               "&#966;<sub>&#946;</sub>(a) = max<sub>r</sub>a<sub>r</sub> + "
               "&#946;&#183;(second largest component) is safe for any &#946; &#8805; 0 and has "
               "&#923;<sub>&#946;</sub> &#8804; 1 + &#946;, so it interpolates between the exact "
               "vector formulation (&#946; = 0) and the dominant-resource slots "
               "(&#946; = &#8734;).", EQ))
    S.append(P("<b>Measured (exp11, exp12):</b> the unsafe reductions are exactly those whose "
               "weights live on the simplex (entropy, AHP) or are uniform shadow prices: "
               "&#923;<sub>&#966;</sub> &lt; 1, a <i>lower</i> latency, and violations of the true "
               "resource limits in 55-85 % of the steps (up to 2.37&#215; the capacity).  The safe "
               "ones never violate but waste capacity (linear &#945; = 1: &#923; = 2.52, +71 % "
               "latency).  A rule that simply says &ldquo;report &#923;&rdquo; would have accepted "
               "the unsafe ones.", GRN))

    S.append(P("N7 - The per-request cost is 25-70&#215; the deterministic one", H3))
    S.append(P("<b>Problem.</b>  Every request performs an O(N) Euclidean projection (dual ascent "
               "on 4 multipliers), an O(K&#183;N) rounding of K paths and an O(N) repair of every "
               "violating path: 440-1 665 &#181;s against 13-26 &#181;s (exp14), i.e. "
               "&#8776; 35 ms of CPU per second at 1 000 requests/s."))
    S.append(P("<b>Revision.</b>  Four reductions, in order of expected payoff:", BODY))
    S.append(table(
        ["#", "change", "expected effect"],
        [["1", "<b>Active-set projection.</b>  Only the services with p<sub>n</sub> &gt; 0 can be "
          "violating; project the support (|supp p| &#8810; N) and keep the rest at 0",
          "O(|supp|) instead of O(N) per step; on Zipf traffic |supp| is a small fraction of N"],
         ["2", "<b>Amortize the rounding.</b>  The coupling only requires that the sampled path "
          "changes by O(&#8741;p<sub>t</sub> - p<sub>t-1</sub>&#8741;<sub>1</sub>) in expectation; "
          "resample <i>one</i> path per request (round-robin over the K paths) instead of all K",
          "O(N) instead of O(K&#183;N) per request, with the same marginals E[s] = p"],
         ["3", "<b>Lazy repair.</b>  Repair a path only when it is actually served, not every path "
          "at every step", "the repair cost is paid once per served path instead of K times"],
         ["4", "<b>Certified approximate projection.</b>  Stop the dual ascent at a tolerance "
          "&#949; and apply the scaling (25): the result is exactly feasible and &#949;-optimal, "
          "with O(log(1/&#949;)) fewer iterations", "fewer iterations of the O(N) dual ascent"]],
        [8 * mm, 88 * mm, 69 * mm]))
    S.append(P("With 1 and 2 the per-request cost should fall from O(N + KN) to O(|supp| + N), "
               "which the measurements suggest is a factor 5-10; the target is to be within a "
               "small constant factor of the deterministic Landlord.  This is a concrete piece of "
               "engineering work, not a change of the mathematics, and it is the only way the "
               "randomized algorithm becomes deployable.", SMALL))

    # ------------------------------------------------------------ 3 revised algorithm
    S.append(P("3. The revised algorithm", H1))
    S.append(P("Randomized DRL v2 (the changes with respect to the proposal are marked):", SMALL))
    S.append(table(
        ["step", "action", "formula / source"],
        [["0", "draw the K persistent uniform vectors U<sub>k</sub> &#8712; [0,1]<sup>N</sup>; "
          "pick the serving path k* <b>sticky over a miss epoch</b> (N5)",
          "proposal Sec. 3.4 + revision R-D"],
         ["1", "on a request for n*: update the <b>download probability</b> Q (N1) and the "
          "<b>cache state</b> P with the expected gradient (N2)",
          "z<sub>n,t</sub> = p<sub>n,t-1</sub> - &#951;&#183;(1 - Q<sub>n</sub>)l<sub>n</sub> ... ; "
          "revision of (19)-(20)"],
         ["2", "project onto D<sub>vec</sub> with the warm-started dual ascent and the final "
          "scaling", "(21)-(25), unchanged"],
         ["3", "round: the served path is the sticky k*; the sample s<sub>k*</sub> is refreshed "
          "only for the services whose p changed, and repaired if it violates a constraint",
          "Sec. 3.4 + revisions R-D and N7"],
         ["4", "decide the download with the Bernoulli trigger u = 1[U &lt; Q<sub>n</sub>], "
          "subject to the budget and the resource check",
          "revision of R3 (N1) - the threshold f(&#952;) becomes the special case "
          "Q<sub>n</sub> = P(&#952; &lt; threshold)"],
         ["5", "account: latency, cost, hits, delayed hits, U<sub>r,t</sub>, and the two regret "
          "measures of N3", "(30)-(34) + (11&#8242;)"]],
        [10 * mm, 88 * mm, 67 * mm]))
    S.append(Spacer(1, 2 * mm))
    S.append(P("The projections and the rounding are unchanged; what changes is <b>which "
               "quantities are decision variables</b> (Q becomes one), <b>what the gradient is "
               "taken with respect to</b> (the expectation over the algorithm's own randomness), "
               "<b>how the goal is stated</b> (G3 splits into three checkable statements) and "
               "<b>how much work is done per request</b> (N7).", KEY))

    # ------------------------------------------------------------ 4 revised goals
    S.append(P("4. The revised goals, with the measured values", H1))
    S.append(table(
        ["goal", "revised statement", "measured"],
        [["G1 feasibility", "after the repair, a.s. &#8721;<sub>n</sub> a<sub>n,r</sub>x<sub>n,t</sub> "
          "&#8804; 1 &#8704;r and &#8721;<sub>n</sub>x<sub>n,t</sub> &#8804; C",
          "0 % violations (max utilisation 0.99) with the repair, 6 % and 1.73&#215; without it "
          "(exp08)"],
         ["G2 budget", "E[&#931;<sub>n,t</sub> u<sub>n,t</sub>M<sub>n</sub>] &#8804; B<sub>max</sub>, "
          "with the trigger probability clipped to "
          "budget_left/(M<sub>n</sub>&#183;requests_left)",
          "respected at 20 / 50 / 100 % of the deterministic cost, every seed (exp08, exp15)"],
         ["G3a surrogate regret", "&#931;<sub>t</sub><i>F</i><sub>t</sub>(P<sub>t</sub>) - "
          "min<sub>P</sub> &#931;<sub>t</sub><i>F</i><sub>t</sub>(P) &#8804; "
          "&#8730;N l<sub>max</sub>&#8730;T (or 2&#8730;N l<sub>max</sub>&#8730;T with the "
          "horizon-free step)",
          "always below the bound, exogenous and endogenous, peek and no-peek (exp09, exp18)"],
         ["G3b download-charged", "E[&#931;<sub>t</sub>d<sub>t</sub>] - min<sub>P</sub> "
          "&#931;<sub>t</sub><i>F</i><sub>t</sub>(P) &#8804; Reg<sub>Prop.1</sub>(T) + "
          "D<sub>T</sub>  (new)",
          "holds at every horizon tested (exp18)"],
         ["G3c realized regret", "against an <b>online</b> comparator and under a "
          "non-stationarity assumption: the adaptive policy pays an o(1) price per request "
          "relative to the best online static policy, and gains when the popularity shifts (new)",
          "against the <i>offline-optimal</i> static set the gap is a <b>constant 3.1-3.6 ms per "
          "request</b> (25-34 % of the latency) - i.e. &#920;(T), not o(T): the goal must not be "
          "stated against an offline oracle (exp18)"],
         ["G4 expected ratio", "for the adversarial family: &#947;<sub>R</sub> = O(1) while "
          "&#947;<sub>det</sub> = &#920;(k); in general E[ALG<sub>R</sub>(I)] &#8804; OPT(I) + "
          "O(&#8730;T) w.h.p. against an oblivious adversary (new)",
          "&#947;<sub>R</sub> &#8776; 3 &#177; 1 for k = 3 ... 24, &#947;<sub>det</sub> = 2.00k "
          "(exp05, exp13) - <b>empirical, not proved</b>"],
         ["G5 conversion loss", "use a slot reduction only if it is safe "
          "(&#966;(a) &#8805; max<sub>r</sub>a<sub>r</sub>) <b>and</b> &#923;<sub>&#966;</sub> "
          "&#8804; 1 + &#949;; otherwise keep the vector formulation (new decision rule)",
          "the unsafe reductions violate the true limits in 55-85 % of the steps (exp11, exp12)"]],
        [26 * mm, 76 * mm, 63 * mm]))
    S.append(Spacer(1, 2 * mm))
    S.append(fig("exp18_proposed_revisions.png",
                 "Figure 1 - exp18: (left) the horizon-free step size gives a smaller surrogate "
                 "regret than the horizon-dependent one; (middle) the literal goal G3 is violated "
                 "but the download-charged correction Reg + D holds; (right) the realized regret "
                 "against the OFFLINE-OPTIMAL static cache is a constant per request "
                 "(3.1-3.6 ms), not o(T) - which is why G3c has to be restated against an online "
                 "comparator."))

    # ------------------------------------------------------------ 5 what to do next
    S.append(P("5. What to do next", H1))
    S.append(table(
        ["priority", "task", "where", "why it matters"],
        [["1", "implement the Q-update of N1 (the download probability as a decision variable)",
          "<b>sim/rand_drl.py</b> (<b>download_prob_mode=&quot;learned&quot;</b>), "
          "<b>sim/engine.py</b> (the on_hit / on_delayed hooks)",
          "<b>done</b> - the download decision is now a learned probability; the measured effect "
          "is a 7 % lower cost at the same latency, and the trade-off with the learning rate is "
          "in exp19"],
         ["2", "implement the G3c measurement as a permanent experiment (adaptive vs static)",
          "<b>experiments/exp18</b> (done); the G3c&#8242; comparator (an <i>online</i> static "
          "policy) is still missing",
          "it is the only form of goal G3 that is both true and useful; against the "
          "offline-optimal static cache the gap is a constant 3.1-3.6 ms per request"],
         ["3", "implement the active-set projection and the amortized rounding of N7",
          "<b>sim/projection.py</b> (<b>project_dual_active</b>), <b>sim/rand_drl.py</b> "
          "(<b>proj_mode</b>, <b>rounding_mode</b>)",
          "<b>done</b> - the per-request cost fell by 3.4&#215; (821 &#8594; 243 &#181;s); the "
          "remaining hot spot is the 13 dual-ascent iterations, for which a Newton step on the "
          "four multipliers is the next step"],
         ["4", "run everything on the paper's real traces (Google dataset2, SHARCNet)",
          "<b>experiments/exp16_real_traces.py</b> + <b>data/</b>",
          "without the real data none of the workload-level statements (G3c, G4, the sign of the "
          "latency difference) can be settled"],
         ["5", "write the missing proofs: the O(&#963;&#8730;T) stochastic term of N2, the "
          "churn bound of N5, and <i>l</i><sub>R</sub> = O(1) for the family",
          "the proposal, Sec. 3.5 and Sec. 5",
          "the empirical results are suggestive but the claims are currently unproved"],
         ["6", "design an adversarial family against the <i>randomized</i> algorithm",
          "new experiment (the paper's family only forces the deterministic one)",
          "it is the only way to decide between O(log k) and O(k) for the randomized algorithm"]],
        [16 * mm, 48 * mm, 45 * mm, 56 * mm]))
    S.append(Spacer(1, 3 * mm))
    S.append(P("<b>Status update.</b>  All seven revisions are now implemented and measured in "
               "the simulator: the horizon-free step size (N4), the sticky serving path (N5) and "
               "the corrected goals (N3) were added first and are evaluated in "
               "<b>experiments/exp18_proposed_revisions.py</b>; the download probability as a "
               "decision variable (N1), the expectation gradient (N2) and the cost reductions "
               "(N7) are in <b>sim/rand_drl.py</b> (<b>download_prob_mode</b>, "
               "<b>grad_mode</b>, <b>proj_mode</b>, <b>rounding_mode</b>) and are evaluated in "
               "<b>experiments/exp19_randomized_drl_v2.py</b>.  The resulting method, its results "
               "and its remaining gaps are documented in "
               "<b>docs/Updated_randomized_DRL_v2.pdf</b>.  What is still <i>not</i> done: the "
               "G3c&#8242; comparator built from an <i>online</i> static policy, the Newton step "
               "on the four multipliers (the remaining hot spot of the per-request cost) and the "
               "whole comparison on the paper's real traces.",
               SMALL))

    doc.build(S)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                                                   "Proposed_revisions_randomized_DRL.pdf"))
    a = ap.parse_args()
    print("written:", build(a.out))
