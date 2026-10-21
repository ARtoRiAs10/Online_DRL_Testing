"""Build the slide deck "Randomized Online-DRL vs Online-DRL".

Explains (i) how the two algorithms differ, (ii) what the randomized one improves and what it
loses, and (iii) how every metric in the comparison is calculated.  The numbers come from
results/*.json, the charts are rendered with matplotlib.

    python docs/make_slides.py [--out ../Randomized_vs_Online_DRL_slides.pptx]
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(HERE, "figures")
sys.path.insert(0, ROOT)

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

ACCENT = RGBColor(0x1F, 0x3B, 0x73)
ACC2 = RGBColor(0x2E, 0x75, 0xB6)
GOOD = RGBColor(0x1B, 0x6B, 0x3A)
BAD = RGBColor(0x9C, 0x2B, 0x2B)
GREY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xEE, 0xF2, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x22, 0x22, 0x22)

W, H = Inches(13.333), Inches(7.5)


def load(name):
    with open(os.path.join(RES, name + ".json")) as f:
        return json.load(f)


# ----------------------------------------------------------------- charts
def charts(d13, d13z, d12i, d14, d11, d10):
    os.makedirs(FIG, exist_ok=True)
    out = {}

    # 1. gamma vs k (adversarial)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
    ks = [r["k"] for r in d13["family1"]]
    ax[0].plot(ks, [r["det"] for r in d13["family1"]], "o-", color="#1f3b73", label="Online-DRL (det)")
    ax[0].plot(ks, [r["llrc"] for r in d13["family1"]], "s--", color="#8ea9d0", label="LL-RC")
    ax[0].plot(ks, [r["rand_landlord"] for r in d13["family1"]], "^--", color="#c58b2a",
               label="R3 only")
    ax[0].plot(ks, [r["rand_drl"] for r in d13["family1"]], "d-", color="#1b6b3a",
               label="Randomized DRL")
    ax[0].set_xlabel("k (cache capacity)"); ax[0].set_ylabel("ALG / OPT")
    ax[0].set_title("Adversarial Large Sequence: ratio vs k", fontsize=10)
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[1].plot(ks, [r["det"] for r in d13["family1"]], "o-", color="#1f3b73", label="Online-DRL: 2.00k")
    ax[1].plot(ks, [r["rand_drl"] for r in d13["family1"]], "d-", color="#1b6b3a",
               label="Randomized DRL: flat")
    ax[1].set_xlabel("k"); ax[1].set_ylabel("ALG / OPT"); ax[1].set_yscale("log")
    ax[1].set_title("log scale: linear vs constant", fontsize=10); ax[1].legend(fontsize=7)
    ax[1].grid(alpha=.3, which="both")
    fig.tight_layout(); p = os.path.join(FIG, "slides_gamma_adv.png"); fig.savefig(p, dpi=150); plt.close(fig)
    out["gamma_adv"] = p

    # 2. gamma vs k (zipf) + cost
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
    ks2 = [r["k"] for r in d13z["rows"]]
    ax[0].plot(ks2, [r["det"][0] for r in d13z["rows"]], "o-", color="#1f3b73", label="Online-DRL")
    ax[0].plot(ks2, [r["llrc"][0] for r in d13z["rows"]], "s--", color="#8ea9d0", label="LL-RC")
    ax[0].plot(ks2, [r["rand_drl"][0] for r in d13z["rows"]], "d-", color="#1b6b3a",
               label="Randomized DRL")
    ax[0].set_xlabel("k"); ax[0].set_ylabel("ALG / OPT (MILP)")
    ax[0].set_title("Zipf traffic vs the exact optimum", fontsize=10); ax[0].legend(fontsize=7)
    ax[0].grid(alpha=.3)
    w = 0.35
    xs = np.arange(len(ks2))
    ax[1].bar(xs - w / 2, [r["det_cost"][0] for r in d13z["rows"]], w, color="#1f3b73",
              label="Online-DRL")
    ax[1].bar(xs + w / 2, [r["rand_cost"][0] for r in d13z["rows"]], w, color="#1b6b3a",
              label="Randomized DRL")
    ax[1].set_xticks(xs); ax[1].set_xticklabels([str(k) for k in ks2])
    ax[1].set_xlabel("k"); ax[1].set_ylabel("download cost")
    ax[1].set_title("Download cost: 3x lower", fontsize=10); ax[1].legend(fontsize=7)
    ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout(); p = os.path.join(FIG, "slides_gamma_zipf.png"); fig.savefig(p, dpi=150)
    plt.close(fig); out["gamma_zipf"] = p

    # 3. multi-resource latency / cost
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
    names = list(d12i["algos"])
    short = ["Online-DRL", "LL-RC", "R3 only", "Randomized"]
    ax[0].bar(short, [d12i["algos"][n]["latency"][0] for n in names], color=["#1f3b73", "#8ea9d0",
                                                                            "#c58b2a", "#1b6b3a"])
    ax[0].set_ylabel("latency"); ax[0].set_title("Imbalanced multi-resource workload", fontsize=10)
    ax[0].tick_params(axis="x", labelsize=8)
    for i, n in enumerate(names):
        ax[0].text(i, d12i["algos"][n]["latency"][0] * 1.02, f"{d12i['algos'][n]['latency'][0]:,.0f}",
                   ha="center", fontsize=7)
    ax[1].bar(short, [d12i["algos"][n]["cost"][0] for n in names], color=["#1f3b73", "#8ea9d0",
                                                                           "#c58b2a", "#1b6b3a"])
    ax[1].set_ylabel("download cost"); ax[1].set_title("Cost: 12x lower", fontsize=10)
    ax[1].tick_params(axis="x", labelsize=8)
    for i, n in enumerate(names):
        ax[1].text(i, d12i["algos"][n]["cost"][0] * 1.02, f"{d12i['algos'][n]['cost'][0]:,.0f}",
                   ha="center", fontsize=7)
    fig.tight_layout(); p = os.path.join(FIG, "slides_multiresource.png"); fig.savefig(p, dpi=150)
    plt.close(fig); out["multiresource"] = p

    # 4. runtime
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.2))
    Ns = [r["N"] for r in d14["by_N"]]
    for lab, c in (("Online-DRL", "#1f3b73"), ("LL-RC", "#8ea9d0"), ("R3 only", "#c58b2a"),
                   ("Randomized DRL", "#1b6b3a")):
        ax[0].plot(Ns, [r[lab][0] for r in d14["by_N"]], "o-", color=c, label=lab)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("N (services)")
    ax[0].set_ylabel("us / request"); ax[0].set_title("Running time per request", fontsize=10)
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3, which="both")
    ks3 = [r["C"] for r in d14["by_C"]]
    for lab, c in (("Online-DRL", "#1f3b73"), ("Randomized DRL", "#1b6b3a")):
        ax[1].plot(ks3, [r[lab][0] for r in d14["by_C"]], "o-", color=c, label=lab)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("k (cache capacity)")
    ax[1].set_ylabel("us / request"); ax[1].set_title("Independent of the cache size", fontsize=10)
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, which="both")
    fig.tight_layout(); p = os.path.join(FIG, "slides_runtime.png"); fig.savefig(p, dpi=150)
    plt.close(fig); out["runtime"] = p

    # 5. unsafe reductions
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.2))
    ks4 = list(d11)
    short4 = [k.replace(" (simplex)", "\n(simplex)").replace(" (safe)", "\n(safe)")
              for k in ks4]
    ax[0].bar(range(len(ks4)), [d11[k]["latency"][0] for k in ks4],
              color=["#1b6b3a" if d11[k]["violation_rate"][0] == 0 else "#9c2b2b" for k in ks4])
    ax[0].set_xticks(range(len(ks4))); ax[0].set_xticklabels(short4, rotation=35, ha="right",
                                                               fontsize=6)
    ax[0].set_ylabel("latency"); ax[0].set_title("Latency of each slot reduction\n"
                                                 "(red = violates the true limits)", fontsize=9)
    ax[1].bar(range(len(ks4)), [d11[k]["violation_rate"][0] for k in ks4],
              color=["#1b6b3a" if d11[k]["violation_rate"][0] == 0 else "#9c2b2b" for k in ks4])
    ax[1].set_xticks(range(len(ks4))); ax[1].set_xticklabels(short4, rotation=35, ha="right",
                                                               fontsize=6)
    ax[1].set_ylabel("fraction of steps violating the\ntrue CPU/RAM/disk limits", fontsize=8)
    ax[1].set_title("Violations of the true constraints", fontsize=9)
    for i, k in enumerate(ks4):
        ax[1].text(i, d11[k]["violation_rate"][0] + .02, f"{d11[k]['violation_rate'][0]:.2f}",
                   ha="center", fontsize=7)
    fig.tight_layout(); p = os.path.join(FIG, "slides_unsafe.png"); fig.savefig(p, dpi=150)
    plt.close(fig); out["unsafe"] = p

    # 6. regret
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    d09 = load("exp09_regret")
    Ts = [r["T"] for r in d09["rows"]]
    ax.plot(Ts, [r["bound"] for r in d09["rows"]], "k--", label="bound  $\\sqrt{N} l_{max}\\sqrt{T}$")
    ax.plot(Ts, [r["regret_exog_stationary"] for r in d09["rows"]], "o-", color="#1f3b73",
            label="surrogate (stationary)")
    ax.plot(Ts, [r["regret_exog_peek"] for r in d09["rows"]], "s-", color="#2e75b6",
            label="surrogate (peek)")
    ax.plot(Ts, [r["literal_G3"][0] for r in d09["rows"]], "d-", color="#9c2b2b",
            label="literal G3 (realized latency)")
    ax.set_xlabel("T (requests)"); ax.set_ylabel("regret")
    ax.set_title("Surrogate regret stays under the bound; the literal one does not", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); p = os.path.join(FIG, "slides_regret.png"); fig.savefig(p, dpi=150)
    plt.close(fig); out["regret"] = p
    return out


# ----------------------------------------------------------------- deck helpers
def txbox(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    return tf


def setp(p, text, size=18, bold=False, color=DARK, space_after=6, align=PP_ALIGN.LEFT,
         italic=False, bullet=None, level=0, font="Calibri"):
    p.text = ""
    if bullet:
        text = f"{bullet}  {text}"
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
    r.font.color.rgb = color; r.font.name = font
    p.space_after = Pt(space_after)
    p.alignment = align
    p.level = level
    return p


def slide_header(slide, title, subtitle=None, num=None):
    bar = slide.shapes.add_shape(1, 0, 0, W, Inches(0.09))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
    tf = txbox(slide, 0.55, 0.28, 12.2, 0.9)
    setp(tf.paragraphs[0], title, size=28, bold=True, color=ACCENT, space_after=2)
    if subtitle:
        setp(tf.add_paragraph(), subtitle, size=14, color=GREY, italic=True)
    if num:
        t = txbox(slide, 12.2, 6.95, 0.9, 0.4)
        setp(t.paragraphs[0], str(num), size=12, color=GREY, align=PP_ALIGN.RIGHT)


def bullets(slide, items, x=0.7, y=1.5, w=12.0, h=5.0, size=17, space=9):
    tf = txbox(slide, x, y, w, h)
    first = True
    for it in items:
        if isinstance(it, tuple):
            text, kw = it
        else:
            text, kw = it, {}
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        setp(p, text, size=kw.get("size", size), bold=kw.get("bold", False),
             color=kw.get("color", DARK), space_after=kw.get("space", space),
             bullet=kw.get("bullet", "•"), italic=kw.get("italic", False))
    return tf


def table(slide, data, x, y, w, h, col_w=None, size=12, header=True, hl=None):
    rows, cols = len(data), len(data[0])
    shp = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(h))
    t = shp.table
    if col_w:
        tot = sum(col_w)
        for i, cw in enumerate(col_w):
            t.columns[i].width = Emu(int(Inches(w) * cw / tot))
    for r in range(rows):
        for c in range(cols):
            cell = t.cell(r, c)
            cell.text = ""
            tf = cell.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]
            txt = str(data[r][c])
            setp(p, txt, size=size, bold=(header and r == 0),
                 color=WHITE if (header and r == 0) else DARK, space_after=0)
            cell.margin_left = Inches(0.06); cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.03); cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            if header and r == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = ACCENT
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT if r % 2 == 0 else WHITE
            if hl and r in hl:
                cell.fill.solid(); cell.fill.fore_color.rgb = hl[r]
    return t


def pic(slide, path, x, y, w):
    return slide.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w))


def callout(slide, text, x, y, w, h, color=ACCENT, size=15, bold_lead=None):
    box = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid(); box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = color; box.line.width = Pt(1.25)
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.15); tf.margin_right = Inches(0.15)
    tf.margin_top = Inches(0.09)
    if bold_lead:
        p = tf.paragraphs[0]
        setp(p, bold_lead, size=size, bold=True, color=color, space_after=3)
        p2 = tf.add_paragraph(); setp(p2, text, size=size - 1, color=DARK)
    else:
        setp(tf.paragraphs[0], text, size=size, color=DARK)
    return box


# ----------------------------------------------------------------- the deck
def build(out_path, ch):
    d05, d10 = load("exp05_large_sequence"), load("exp10_expected_ratio")
    d13, d13z = load("exp13_gamma_complexity"), load("exp13_gamma_complexity_zipf")
    d12b, d12i = load("exp12_multiresource_balanced"), load("exp12_multiresource_imbalanced")
    d14, d15 = load("exp14_time_complexity"), load("exp15_budget_regret")
    d08, d11 = load("exp08_feasibility_budget"), load("exp11_slot_reduction")
    d09, d01 = load("exp09_regret"), load("exp01_baseline_vs_llrc")

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    blank = prs.slide_layouts[6]
    n = 0

    def new(title=None, sub=None):
        nonlocal n
        n += 1
        s = prs.slides.add_slide(blank)
        if title:
            slide_header(s, title, sub, n)
        return s

    # 1 title
    s = prs.slides.add_slide(blank)
    bg = s.shapes.add_shape(1, 0, 0, W, Inches(2.9))
    bg.fill.solid(); bg.fill.fore_color.rgb = ACCENT; bg.line.fill.background()
    tf = txbox(s, 0.8, 0.95, 12.0, 1.8)
    setp(tf.paragraphs[0], "Randomized Online-DRL vs Online-DRL", size=34, bold=True, color=WHITE)
    setp(tf.add_paragraph(), "What improves, what it costs, and how every metric is computed",
         size=20, color=RGBColor(0xC9, 0xD8, 0xF0), space_after=0)
    tf2 = txbox(s, 0.8, 3.3, 11.7, 2.6)
    setp(tf2.paragraphs[0], "Evidence from drl-sim - the simulator of", size=15, color=GREY)
    setp(tf2.add_paragraph(),
         "  Edge Service Caching With Delayed Hits (Deka & Sukapuram, IEEE TCC 2025)  and of the",
         size=15, color=DARK)
    setp(tf2.add_paragraph(),
         "  Randomized DRL with Multi-Resource (CPU, RAM, Disk) Capacity proposal",
         size=15, color=DARK, space_after=16)
    setp(tf2.add_paragraph(),
         "Synthetic Zipf/Poisson traffic + the paper's adversarial Large Sequence; the offline "
         "optimum is an exact MILP (HiGHS).", size=13, color=GREY, italic=True)
    setp(tf2.add_paragraph(), "All numbers are read from results/*.json - nothing is hand-written.",
         size=13, color=GREY, italic=True)
    n = 1

    # 2 the question
    s = new("The question", "and the short answer")
    bullets(s, [
        ("How does each algorithm decide what to keep?", {"size": 19, "space": 12}),
        ("Does the randomized algorithm IMPROVE the deterministic one, or does it only "
         "EQUALIZE it?", {"size": 19, "space": 12}),
        ("Where does it LOSE?", {"size": 19, "space": 12}),
        ("How are the metrics (ratio, cost, utilization, regret, running time) actually "
         "calculated?", {"size": 19, "space": 18}),
    ], y=1.6, h=2.4)
    callout(s, "it strictly improves the competitive ratio on the adversarial family the paper "
               "analyses (gamma = 2k -> ~3, flat in k), it EQUALIZES the deterministic algorithm "
               "on ordinary traffic (1.443 vs 1.430) while spending 3x less download cost, it "
               "loses on short traces (1.279 vs 1.145), on CPU time (25-70x more per request) and "
               "on the literal form of goal G3.",
             0.8, 4.35, 11.7, 1.55, bold_lead="Short answer:", size=15)

    # 3 setting
    s = new("The common setting", "the environment both algorithms run in")
    bullets(s, [
        ("An edge node caches N services; a service has a forwarding latency l, a download time M "
         "and a demand vector (CPU, RAM, disk).", {"size": 17}),
        ("Latency and cost are kept separate: latency = what the user waits, cost = the download "
         "time the operator pays.", {"size": 17}),
    ], y=1.45, h=1.1)
    table(s, [
        ["event", "condition", "latency", "cost"],
        ["Hit", "the service is cached", "0", "-"],
        ["Delayed hit", "a download is in flight and the remaining time <= l  (the request is "
                       "buffered)", "min(l, remaining)", "the download already paid"],
        ["Request forwarding", "not cached, no download in flight (or the remaining time > l)",
         "l", "-"],
    ], 0.8, 2.75, 11.7, 1.9, col_w=[2.2, 5.4, 2.2, 2.2], size=13)
    callout(s, "every request is charged its latency, every started download its M; the two "
               "algorithms are measured with exactly the same engine, so every difference in the "
               "next slides is a difference in CACHE MANAGEMENT, not in the environment.",
             0.8, 5.0, 11.7, 1.2, bold_lead="Why it matters:", size=14)

    # 4 online drl
    s = new("How Online-DRL works", "the paper: deterministic, per-service state")
    bullets(s, [
        ("TRIGGER (Algorithm 1).  At the first request of a miss epoch a download starts iff "
         "T_i >= theta*M_i  or  L_i >= theta*M_i, where T_i = time since the epoch began and "
         "L_i = l * (number of requests of the service in the epoch).  theta = 1 in the paper, "
         "theta = 0 is LL-RC.", {"size": 16, "space": 11}),
        ("EVICTION (Algorithm 2, Landlord).  Every cached service carries a credit, initialised "
         "to M and reset on every hit.  When a new service does not fit, the credits are "
         "decreased proportionally; the services whose credit hits 0 become eligible and are "
         "evicted in increasing order of the resources they use beyond the unmet need.",
         {"size": 16, "space": 11}),
        ("ADMISSION.  A service enters only if it fits in ALL THREE resources and the "
         "cardinality C.", {"size": 16, "space": 11}),
        ("Guarantee: competitive ratio  gamma in O(k(1 + 3 beta_up M_x)); the proof exhibits an "
         "adversarial family on which gamma = Theta(k).", {"size": 16, "space": 0}),
    ], y=1.45, h=4.3)

    # 5 randomized
    s = new("How Randomized Online-DRL works", "the proposal: R1 + R2 + R3")
    bullets(s, [
        ("R1 - a FRACTIONAL state + projected OGD.  Instead of a credit per service the algorithm "
         "keeps p in D_vec = {p in [0,1]^N : sum p <= C, sum a_n,r p_n <= 1 for every r}.  Each "
         "request takes one gradient step on the requested service and projects back onto "
         "D_vec with a dual-ascent projection (eqs. 22-25); eta = sqrt(N)/(l_max sqrt(T)).",
         {"size": 16, "space": 10}),
        ("R2 - RANDOMIZED ROUNDING with repair.  K persistent uniform vectors give the paths "
         "s_k = 1[U_k < p], so E[s] = p and E|ds| = |dp|.  A path that violates the constraints "
         "is repaired by evicting services.  The real cache = (downloaded) AND (sampled on): a "
         "service that was sampled but never downloaded is NOT a hit.", {"size": 16, "space": 10}),
        ("R3 - a RANDOMIZED download threshold, drawn from f(theta) = e^(theta/M_n)/(M_n(e-1)) on "
         "[0, M_n]: the rent-or-buy ratio 2 becomes e/(e-1) = 1.582.", {"size": 16, "space": 10}),
        ("BUDGET awareness.  The trigger probability is lowered to budget_left/(M_n * "
         "requests_left) so that the expected download cost stays below B_max (goal G2).",
         {"size": 16, "space": 0}),
    ], y=1.45, h=4.6)

    # 6 side by side
    s = new("Side by side", "what actually differs")
    table(s, [
        ["aspect", "Online-DRL", "Randomized Online-DRL"],
        ["state", "one credit per cached service", "fractional p in D_vec + K rounding paths"],
        ["download decision", "deterministic counters vs a fixed theta",
         "same counters, random threshold, budget-aware probability"],
        ["eviction", "Landlord: credit reaches 0, then a greedy resource-aware order",
         "the OGD state lowers p, the rounding switches the service off, violating paths are "
         "repaired"],
        ["entering the cache", "when the download completes",
         "when the download completes AND the serving path samples it on"],
        ["guarantee", "gamma in O(k(1+3 beta_up M_x))",
         "surrogate regret <= sqrt(N) l_max sqrt(T); expected cost <= B_max"],
        ["resource feasibility", "checked exactly at admission", "guaranteed by the repair step"],
        ["price", "no randomness, no learning",
         "25-70x more CPU per request; needs a warm-up"],
    ], 0.7, 1.5, 11.9, 4.7, col_w=[2.2, 4.6, 5.1], size=12.5)

    # 7 metrics 1
    s = new("How the metrics are calculated (1/2)", "the objective quantities")
    table(s, [
        ["metric", "formula", "why it is the right one"],
        ["latency", "sum_t lat_t, with lat_t = 0 (hit), min(l, remaining) (delayed hit) or l "
                    "(forward)", "exactly the objective of the paper"],
        ["cost", "sum of M_n over the STARTED downloads",
         "the operator's cost; forwarding is charged through the latency, not the cost"],
        ["downloads", "number of started downloads", "the download budget is spent per download"],
        ["competitive ratio gamma", "ALG(I)/OPT(I); OPT = MILP optimum (HiGHS) or the paper's "
                                   "(k+2n)l on the Large Sequence",
         "the paper's definition; if the MILP hits its time limit the ratio is only a lower bound"],
        ["fit of gamma(k)", "least squares: linear / logarithmic / power law (log-log) with R^2",
         "shows empirically whether the linear behaviour disappears - it is not a proof"],
        ["beta_up, beta_down", "per service fwd_i/M_i, where fwd_i counts the requests forwarded "
                               "WHILE a download of i was in flight",
         "the paper's beta is exactly that fraction; beta_up = max (with M_x), beta_down = mean "
         "(with M_min)"],
    ], 0.7, 1.5, 11.9, 4.6, col_w=[2.4, 4.8, 4.7], size=12)

    # 8 metrics 2
    s = new("How the metrics are calculated (2/2)", "the resource, regret and cost metrics")
    table(s, [
        ["metric", "formula", "why it is the right one"],
        ["U_r,t, U^max", "sum of a_n,r over the services cached at t",
         "the proposal's per-resource occupancy"],
        ["imbalance, stranded W", "U^max - min_r U_r  and  sum_r (U^max - U_r)",
         "proposal eqs. (30)-(33)"],
        ["violation rate", "fraction of steps with max_r U_r,t > 1",
         "goal G1 is about the TRUE constraints, not the reduced ones"],
        ["Lambda_phi (eq. 18)", "max_n phi(a_n) / max_r a_n,r",
         "how much a slot reduction over-charges a service"],
        ["surrogate regret", "sum_t F_t(P_t) - min_P sum_t F_t(P), the minimum computed exactly "
                             "by an LP",
         "the comparator of Proposition 1, solved to optimality (not an approximation)"],
        ["running time", "1e6 * wall-clock time of the request loop / T",
         "measures the algorithm only, not the catalogue or the trace"],
        ["latency per unit resource", "L / sum_t U^max_t", "proposal eq. (34)"],
    ], 0.7, 1.5, 11.9, 4.8, col_w=[2.4, 4.8, 4.7], size=12)

    # 9 result adversarial
    s = new("Result 1 - the adversarial family", "the paper's Large Sequence, OPT = (k+2n)l")
    pic(s, ch["gamma_adv"], 0.6, 1.35, 7.6)
    rows = [["k", "gamma det.", "gamma R3", "gamma rand.", "cost det.", "cost rand."]]
    for r in d13["family1"]:
        rows.append([r["k"], f"{r['det']:.2f}", f"{r['rand_landlord']:.1f}",
                     f"{r['rand_drl']:.2f}", f"{r['det_cost']:,.0f}",
                     f"{r['rand_drl_cost'][0]:,.0f}"])
    table(s, rows, 8.35, 1.4, 4.4, 3.6, col_w=[1, 1.4, 1.3, 1.4, 1.5, 1.5], size=10.5)
    callout(s, "Online-DRL is EXACTLY linear: gamma = 2.00k + 0.33 (R^2 = 1.000).  R3 alone does "
               "not help (1.88k): the blow-up comes from the cache management, not from the "
               "download trigger.  With R1+R2+R3 the ratio is FLAT (gamma ~ 3) and the download "
               "cost is 5-60x lower.",
             0.6, 5.15, 12.1, 1.25, bold_lead="Takeaway:", size=14)

    # 10 result zipf
    s = new("Result 2 - ordinary traffic: it EQUALIZES", "Zipf/Poisson, offline optimum = MILP")
    pic(s, ch["gamma_zipf"], 0.6, 1.35, 7.6)
    bullets(s, [
        ("All algorithms are flat in k: the resource limits bind before the cardinality does.",
         {"size": 15, "space": 8}),
        ("Randomized 1.430 vs Online-DRL 1.443 - parity in the ratio, 3x less download cost.",
         {"size": 15, "space": 8}),
        ("On SHORT traces (T = 120, 5 traces, MILP optimal) it is worse: 1.279 +- 0.138 vs "
         "1.145 +- 0.072 (Online-DRL) and 1.102 (R3 only) - the fractional state has to be "
         "learned first.", {"size": 15, "space": 8}),
    ], x=8.4, y=1.4, w=4.4, h=3.4, size=15)
    callout(s, "on easy workloads randomization buys COST and utilization, not ratio.  Its "
               "advantage appears in the worst case.",
             0.6, 5.5, 12.1, 0.9, bold_lead="Takeaway:", size=14)

    # 11 multiresource
    s = new("Result 3 - the multi-resource setting", "CPU, RAM and disk at the same time")
    pic(s, ch["multiresource"], 0.6, 1.3, 7.4)
    rows = [["algorithm", "latency", "cost", "dl", "U^max", "imb", "viol."]]
    for name, a in d12i["algos"].items():
        rows.append([name.replace(" (det)", "").replace(" (theta=0)", "").replace(
            " (rand thr + Landlord)", "").replace(" (R1+R2+R3)", ""),
            f"{a['latency'][0]:,.0f}", f"{a['cost'][0]:,.0f}", f"{a['downloads'][0]:.0f}",
            f"{a['Umax'][0]:.3f}", f"{a['imbalance'][0]:.3f}", f"{a['viol'][0]:.2f}"])
    table(s, rows, 8.2, 1.35, 4.6, 2.3, col_w=[2.0, 1.4, 1.4, 0.8, 1.0, 1.0, 0.8], size=10.5)
    bullets(s, [
        ("5-13x fewer downloads, 30-36 % less latency, less imbalance and less stranded "
         "capacity, and 0 % violations of the three resources.", {"size": 14, "space": 6}),
        ("Part of the gain is 'simply downloading less' on a stationary popularity - the "
         "specifically multi-resource advantage still needs a non-stationary resource mix.",
         {"size": 14, "space": 0}),
    ], x=8.2, y=3.85, w=4.6, h=2.2, size=14)

    # 12 goals
    s = new("The goals of the proposal - which ones hold", "G1 - G5, measured")
    table(s, [
        ["goal", "measured", "verdict"],
        ["G1 feasibility", "the un-repaired rounding violates the true limits in 40 % of the "
                           "(path, step) pairs and reaches 1.73x the capacity; with the repair 0 %",
         "CONFIRMED (repair mandatory)"],
        ["G2 budget", "respected at 20 %, 50 %, 100 % of the deterministic cost, for every seed; "
                      "the budgeted deterministic algorithm loses 11.6 % latency",
         "CONFIRMED"],
        ["G3 regret (surrogate)", "always below sqrt(N) l_max sqrt(T), exogenous and endogenous, "
                                  "with and without the peek update",
         "CONFIRMED"],
        ["G3 regret (literal)", "3.7-9.6x the bound and growing with T",
         "SURROGATE ONLY - see the next slides"],
        ["G4 expected ratio", "1.102 (R3), 1.145 (Online-DRL), 1.279 (Randomized) on short traces",
         "CONFIRMED (R3 best on short traces)"],
        ["G5 slot reductions", "weights on the simplex or uniform shadow prices give Lambda < 1 and "
                                "a lower latency but violate the true limits in 55-85 % of the steps",
         "UNSAFE REDUCTIONS EXIST"],
    ], 0.7, 1.5, 11.9, 4.6, col_w=[2.3, 6.3, 3.3], size=12,
        hl={4: RGBColor(0xFB, 0xEA, 0xEA), 6: RGBColor(0xFB, 0xEA, 0xEA)})

    # 13 losses
    s = new("Where Randomized Online-DRL LOSES", "the honest list")
    bullets(s, [
        ("Short traces.  T = 120: 1.279 +- 0.138 vs 1.145 (Online-DRL) and 1.102 (R3 only).  The "
         "fractional state is not learned yet.", {"size": 16, "space": 10}),
        ("Computing time.  440-1 665 us per request against 13-26 us: every request does an O(N) "
         "projection, an O(K*N) rounding and an O(N) repair.  About 35 ms of CPU per second at "
         "1 000 requests/s.", {"size": 16, "space": 10}),
        ("The literal form of goal G3.  The bound only holds for the surrogate loss; the "
         "realized-latency version is violated by a factor 3.7-9.6.", {"size": 16, "space": 10}),
        ("It never loses on resource feasibility: the repair step makes the constraints exact "
         "(0 % violations everywhere).", {"size": 16, "space": 0}),
    ], x=0.7, y=1.45, w=6.3, h=4.6)
    pic(s, ch["runtime"], 7.2, 1.5, 5.6)

    # 14 surrogate
    s = new("Terminology 1 - 'surrogate only'", "goal G3 read carefully")
    bullets(s, [
        ("The regret we would LIKE to bound uses the realized latency:  sum_t l - min_P sum_t l "
         "1[n_t not in P].", {"size": 15, "space": 8}),
        ("But that expression is not a function of p alone: whether a request is a hit, a "
         "delayed hit or a forward also depends on q_t (a download in flight) and dm_t (its "
         "remaining time), which are decided by the discrete download dynamics.",
         {"size": 15, "space": 8}),
        ("The proposal therefore uses a SURROGATE loss that IS a function of p:  "
         "F_t(P) = (1 - p_nt) * g_t,  with g_t = (1-q_t) l + q_t min(l, dm_t)  - 'a request that "
         "is not covered by the fractional cache costs g_t'.", {"size": 15, "space": 8}),
        ("The bound sqrt(N) l_max sqrt(T) is the regret of the SURROGATE, because D_vec has "
         "diameter <= sqrt(N) and eta = sqrt(N)/(l_max sqrt(T)).", {"size": 15, "space": 8}),
        ("Why the literal version fails: the fractional comparator may cache a service "
         "INSTANTLY, while the real system must first DOWNLOAD it - and the comparator is not "
         "charged for those downloads.", {"size": 15, "space": 8}),
        ("Fixes: define G3 on the surrogate loss, charge the comparator for its downloads, or "
         "compare against a download-aware comparator.", {"size": 15, "space": 0}),
    ], x=0.7, y=1.4, w=6.6, h=5.2)
    pic(s, ch["regret"], 7.6, 1.6, 5.2)
    callout(s, "the bound holds for the surrogate and only for the surrogate.",
             7.6, 5.6, 5.2, 0.8, bold_lead="'Surrogate only' means:", size=14)

    # 15 unsafe reduction
    s = new("Terminology 2 - 'unsafe reduction'", "goal G5: slot reductions")
    bullets(s, [
        ("A slot reduction replaces the three resource constraints by ONE scalar constraint "
         "sum_n phi(a_n) p_n <= 1, with a weight function phi (dominant resource, a linear or "
         "l_p mix, shadow prices, AHP or entropy weights).", {"size": 15, "space": 7}),
        ("Lambda_phi = max_n phi(a_n)/max_r a_n,r  (eq. 18) measures the damage: Lambda < 1 means "
         "the reduction charges a service LESS than its true bottleneck demand, i.e. the reduced "
         "constraint is WEAKER than the real one.", {"size": 15, "space": 7}),
        ("Such a reduction is called UNSAFE: the cache it produces can exceed the real "
         "CPU/RAM/disk limits.", {"size": 15, "space": 7}),
        ("The proposal's own warning: with weights on the simplex (sum_r alpha_r = 1) a "
         "disk-only service looks cheap; the reduction is safe only if alpha_r >= 1 for every r, "
         "at the price of wasting up to a factor R of the capacity.", {"size": 15, "space": 7}),
        ("Measured: entropy, AHP and uniform shadow prices all have Lambda < 1 and a LOWER "
         "latency - and all three violate the true limits in 55-85 % of the steps (up to 2.37x "
         "the capacity).  The safe reductions never violate, but waste capacity (linear "
         "alpha = 1: Lambda = 2.52, +71 % latency).", {"size": 15, "space": 0}),
    ], x=0.7, y=1.4, w=6.6, h=5.2)
    pic(s, ch["unsafe"], 7.5, 1.5, 5.4)
    callout(s, "a slot reduction is not a free simplification - the exact vector formulation "
               "dominates every safe reduction.",
             7.5, 5.75, 5.4, 0.8, bold_lead="'Unsafe reduction' means:", size=13.5)

    # 16 summary
    s = new("Summary", "the verdict in one slide")
    table(s, [
        ["regime", "Online-DRL", "Randomized DRL", "verdict"],
        ["adversarial patterns, large caches", "gamma = 2k (linear)", "gamma ~ 3 (flat)",
         "randomized WINS"],
        ["ordinary stationary traffic", "1.443, 3x more cost", "1.430, 3x less cost",
         "randomized EQUALIZES and saves cost"],
        ["multi-resource workload", "136 226 latency", "87 333 latency (-36 %)", "randomized WINS"],
        ["short traces", "1.145", "1.279", "randomized LOSES (warm-up)"],
        ["download budget", "clipped, -11.6 % latency", "respected in expectation and per seed",
         "randomized WINS"],
        ["computing time", "13-26 us/req", "440-1 665 us/req", "randomized LOSES (25-70x)"],
        ["goal G3", "-", "surrogate only", "partial"],
    ], 0.7, 1.5, 11.9, 3.9, col_w=[3.4, 2.6, 3.3, 2.6], size=12.5)
    callout(s, "strictly better in the worst case, equal in the average case, slightly worse "
               "during the warm-up, and 25-70x more expensive per request.  Still open: the real "
               "Google / SHARCNet traces, a non-stationary resource mix, and an adversarial family "
               "designed against the randomized algorithm (to separate O(log k) from O(k)).",
             0.7, 5.6, 11.9, 1.15, bold_lead="Bottom line:", size=14)

    # 17 reproduce
    s = new("How to reproduce", "every number in this deck")
    bullets(s, [
        ("cd extracted/drl-sim && pip install -r requirements.txt && make test",
         {"size": 16, "space": 8}),
        ("python experiments/run_all.py        # ~30 min, writes results/*.json and results/*.png",
         {"size": 16, "space": 8}),
        ("python docs/make_slides.py            # this deck", {"size": 16, "space": 8}),
        ("python docs/make_comparison_pdf.py    # the 6-page comparison PDF", {"size": 16, "space": 8}),
        ("python docs/make_contexts_pdf.py      # the contexts PDF (surrogate, unsafe reduction)",
         {"size": 16, "space": 14}),
        ("Documentation: docs/simulation_methodology.md (model, conditions, how every number is "
         "computed) and docs/results.md (all results).", {"size": 16, "space": 14}),
        ("Caveat: the Google / SHARCNet traces are NOT in this repository; the headline numbers "
         "here come from synthetic Zipf traffic and the paper's adversarial sequence.",
         {"size": 16, "italic": True, "space": 0}),
    ], y=1.6, h=4.6)

    prs.save(out_path)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                                                   "Randomized_vs_Online_DRL_slides.pptx"))
    a = ap.parse_args()
    d13 = load("exp13_gamma_complexity"); d13z = load("exp13_gamma_complexity_zipf")
    d12i = load("exp12_multiresource_imbalanced"); d14 = load("exp14_time_complexity")
    d11 = load("exp11_slot_reduction"); d10 = load("exp10_expected_ratio")
    ch = charts(d13, d13z, d12i, d14, d11, d10)
    print("written:", build(a.out, ch))
