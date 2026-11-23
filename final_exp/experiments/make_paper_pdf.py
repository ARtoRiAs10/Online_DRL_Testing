"""Generate the final PDF paper from the experiment JSONs.

Uses reportlab (installed) and matplotlib (installed).  The output is
``planning/randomized_online_drl_paper_v2.pdf`` (uncommitted).
"""
from __future__ import annotations
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, Preformatted)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(ROOT, "planning", "randomized_online_drl_paper_v2.pdf")


def load(name):
    with open(os.path.join(RES, name + ".json")) as f:
        return json.load(f)


def fmt(x, d=0):
    try:
        return f"{x:,.{d}f}"
    except Exception:
        return str(x)


# ------------------------------------------------------------------ figures
def fig_frontier(path):
    d = load("exp38_theta_frontier")
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for family, color in (("synthetic", "#1f3b73"), ("google10k", "#1b6b3a"),
                          ("google50k", "#c58b2a")):
        fr = [r for r in d["rows"] if r["family"] == family and r["algo"] == "online_drl"]
        if fr:
            thetas = sorted(set(r["theta"] for r in fr))
            lat = [np.mean([r["latency"] for r in fr if r["theta"] == t]) for t in thetas]
            cost = [np.mean([r["cost"] for r in fr if r["theta"] == t]) for t in thetas]
            ax.plot(cost, lat, "o-", color=color, label=f"{family} Online-DRL")
        fr = [r for r in d["rows"] if r["family"] == family and r["algo"] == "rand_stateless"
              and r["variant"] == "default"]
        if fr:
            ax.scatter(np.mean([r["cost"] for r in fr]),
                       np.mean([r["latency"] for r in fr]),
                       marker="*", s=150, color=color, zorder=5)
    ax.set_xlabel("download cost"); ax.set_ylabel("latency")
    ax.set_title("Cost–latency frontier (stars = randomized stateless)")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_budget(path):
    d = load("exp38_theta_frontier")
    fr = [r for r in d["rows"] if r["family"] == "budget"]
    fracs = [0.2, 0.5, 1.0]
    det = [np.mean([r["det_lat"] for r in fr if r["frac"] == f]) for f in fracs]
    rnd = [np.mean([r["rand_lat"] for r in fr if r["frac"] == f]) for f in fracs]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(fracs)); w = 0.36
    ax.bar(x - w/2, det, w, label="Online-DRL", color="#8ea9d0")
    ax.bar(x + w/2, rnd, w, label="Randomized stateless", color="#1f3b73")
    ax.set_xticks(x); ax.set_xticklabels([f"{f:.0%}" for f in fracs])
    ax.set_xlabel("budget as fraction of Online-DRL cost"); ax.set_ylabel("latency")
    ax.set_title("Budget-matched latency"); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_horizon(path):
    d = load("exp39_horizon_prefixes")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for regime, color in (("raw", "#1f3b73"), ("crossing", "#c58b2a")):
        fr = [r for r in d["rows"] if r["disk_scale"] == (1.0 if regime == "raw" else 1e7)]
        fr = sorted(fr, key=lambda r: r["prefix"])
        x = [r["prefix"] / 1000 for r in fr]
        lat = [r["paired_lat"]["mean"] for r in fr]
        cost = [r["paired_cost"]["mean"] for r in fr]
        ax.plot(x, lat, "o-", color=color, label=f"{regime} Δlat")
        ax.plot(x, cost, "s--", color=color, alpha=.7, label=f"{regime} Δcost")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("prefix length (k requests)"); ax.set_ylabel("Δ vs Online-DRL")
    ax.set_title("Warm-up: latency and cost differences vs horizon")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_large_sequence(path):
    d = load("exp40_adversarial_ablation")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ks = sorted(set(r["k"] for r in d["rows"] if r["family"] == "large_sequence"))
    for name, color in (("r3_only", "#c58b2a"), ("r1r2_stateless", "#1f3b73"),
                        ("r1r2r3_stateless", "#1b6b3a"), ("r1r2r3_landlord", "#8a1c1c")):
        fr = [r for r in d["rows"] if r["family"] == "large_sequence"
              and r["M_over_l"] == 4.0 and r["variant"] == name]
        if not fr:
            continue
        xs, ys = [], []
        for k in ks:
            vals = [r["lat_ratio_opt"] for r in fr if r["k"] == k]
            if vals:
                xs.append(k); ys.append(np.median(vals))
        ax.plot(xs, ys, "o-", color=color, label=name)
    ax.set_xscale("log", base=2); ax.set_xlabel("k (cache capacity)")
    ax.set_ylabel("median latency ratio to closed-form OPT")
    ax.set_title("Large Sequence: R1+R2 flattens the ratio (M/l=4)")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


# ------------------------------------------------------------------ styles
styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica",
                      fontSize=9.2, leading=12.4, alignment=TA_JUSTIFY, spaceAfter=4)
small = ParagraphStyle("small", parent=body, fontSize=7.8, leading=9.8)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold",
                    fontSize=13, spaceBefore=10, spaceAfter=4)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                    fontSize=10.5, spaceBefore=7, spaceAfter=3)
title_style = ParagraphStyle("title", parent=styles["Title"], fontName="Helvetica-Bold",
                             fontSize=17, leading=21, alignment=TA_CENTER)
abstract_style = ParagraphStyle("abstract", parent=body, fontSize=8.8, leading=11.6,
                                leftIndent=10, rightIndent=10)


def P(text, style=body):
    return Paragraph(text, style)


def make_table(data, widths=None, font=7.6):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b73")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f7")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


# ------------------------------------------------------------------ content
def build():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = os.path.join(os.path.dirname(OUT), "_figs")
    os.makedirs(tmp, exist_ok=True)
    f_front = os.path.join(tmp, "frontier.png"); fig_frontier(f_front)
    f_budget = os.path.join(tmp, "budget.png"); fig_budget(f_budget)
    f_horizon = os.path.join(tmp, "horizon.png"); fig_horizon(f_horizon)
    f_ls = os.path.join(tmp, "ls.png"); fig_large_sequence(f_ls)

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=1.9*cm, rightMargin=1.9*cm,
                            topMargin=1.7*cm, bottomMargin=1.7*cm,
                            title="Randomized Online-DRL for Multi-Resource Edge Service Caching",
                            author="Gaurav Kumar")
    story = []

    # ---------------- title / abstract
    story.append(Spacer(1, 2.5*cm))
    story.append(P("Randomized Online-DRL for Multi-Resource<br/>Edge Service Caching: A Cost–Latency Study", title_style))
    story.append(Spacer(1, 0.5*cm))
    story.append(P("Gaurav Kumar", ParagraphStyle("author", parent=body, alignment=TA_CENTER, fontSize=10)))
    story.append(P("Working draft — generated from the experiment suite in <i>final_exp/</i>", ParagraphStyle("aff", parent=body, alignment=TA_CENTER, fontSize=8, textColor=colors.grey)))
    story.append(Spacer(1, 0.7*cm))
    abstract = (
        "Online-DRL is a deterministic algorithm for caching services at an edge node with "
        "delayed hits, request forwarding, multi-resource (CPU/RAM/disk) capacities and a "
        "download budget. Its competitive ratio grows linearly with the cache capacity on the "
        "adversarial Large Sequence. We add three randomized components: fractional online "
        "gradient descent over a multi-resource polytope, coupled randomized rounding with a "
        "feasibility repair, and a randomized rent-or-buy download threshold. We prove "
        "feasibility of the repaired state (G1), budget compliance with the download gate (G2), "
        "a projection rate for the fractional state (Proposition 1), and a surrogate-regret "
        "bound O(sqrt(N) l_max sqrt(T) + zeta T) (Proposition 2). We do not prove a competitive "
        "ratio; the flat ratio in k is empirical on constructed families. On the deterministic "
        "frontier, the randomized stateless repair is on or below Online-DRL's theta-sweep in the "
        "synthetic world and in the moderate-cost region of the Google traces; at equal budget it "
        "reduces latency by 47k-57k (95% CIs exclude 0). On the full 50k Google trace it reduces "
        "latency by 2.2k-2.5k and cost by 33k-43k; on the 10k slice it reduces cost by 3.5k-20k "
        "with a small latency penalty. The latency sign is trace-dependent and is a warm-up effect. "
        "On the adversarial families the randomized method is flat in k on the Large Sequence "
        "(median ratio 1.8-4.1) while Online-DRL grows linearly; it loses on the shuffled cyclic "
        "scan (+49% latency), and the loss is caused by the randomized rounding/repair (R1+R2), "
        "not by the randomized threshold (R3). The Landlord repair is consistently worse than the "
        "stateless repair on latency. We conclude that randomization is a useful engineering "
        "addition, but the competitive-ratio question remains open."
    )
    story.append(P("<b>Abstract.</b> " + abstract, abstract_style))
    story.append(PageBreak())

    # ---------------- 1 introduction
    story.append(P("1. Introduction", h1))
    story.append(P(
        "Edge service caching places latency-sensitive services close to users while keeping a "
        "cloud copy for cache misses. Online-DRL (Deka and Sukapuram, 2025) is a deterministic "
        "online algorithm for this setting: it maintains a fractional cache state, forwards "
        "requests while a download is in flight, and triggers downloads with a deterministic "
        "rent-or-buy threshold. Its competitive ratio grows linearly with the cache capacity on "
        "the adversarial Large Sequence, and the deterministic threshold is conservative."))
    story.append(P(
        "We add three randomized components. R1 replaces the deterministic cache state by "
        "fractional online gradient descent over a multi-resource polytope. R2 rounds the "
        "fractional state with persistent coupled uniforms and repairs the rounded sample to "
        "satisfy CPU/RAM/disk and cardinality constraints. R3 replaces the deterministic "
        "download threshold by the randomized rent-or-buy threshold, which improves the "
        "worst-case factor from 2 to e/(e-1). The Landlord repair from the original paper is "
        "kept as a conservative variant; the stateless repair is the better default on "
        "popularity-driven and nonstationary workloads."))
    story.append(P("Our contributions are:"))
    for b in [
        "A randomized rounding and repair scheme that makes the fractional state usable under CPU/RAM/disk constraints (G1).",
        "A randomized rent-or-buy threshold with the e/(e-1) factor, and a component-level download-cost lemma.",
        "A surrogate-regret analysis (Proposition 2) with explicit zeta and delta terms, and a clean separation of rounding churn from repair-induced fetches.",
        "An empirical study on synthetic, adversarial and real Google traces, with paired confidence intervals, a deterministic frontier comparison, and an honest account of where the method loses.",
    ]:
        story.append(P("&bull; " + b, small))
    story.append(P(
        "The paper is an online-algorithms/systems paper. The theory is a surrogate-regret "
        "result, not a competitive ratio. The systems contribution is the multi-resource repair "
        "and the cost-latency trade-off on real traces."))

    # ---------------- 2 model
    story.append(P("2. Model and algorithms", h1))
    story.append(P("2.1 System model", h2))
    story.append(P(
        "There are N services and a resource matrix A in R^{m x N} with entries a_{n,r} in [0,1], "
        "and a capacity vector b in R^m_{>0}. The fractional cache polytope is "
        "D_vec = {p in [0,1]^N : A p &lt;= b}. At round t a request n*_t arrives. The learner holds "
        "a state p_{t-1} in D_vec before the round and plays it. The realized latency coefficient "
        "is c_t = (1 - q_t) l_{n*_t} + q_t min(l_{n*_t}, dm_t), where q_t indicates whether a "
        "download is already in flight and dm_t is the remaining download time. The forwarding "
        "latency is l_n and the download time is M_n."))
    story.append(P("2.2 Online-DRL", h2))
    story.append(P(
        "Online-DRL uses the Landlord admission rule and a deterministic threshold theta_n. It "
        "caches a service when the accumulated latency or forwarding latency reaches theta_n M_n, "
        "and evicts according to the Landlord credits. The deterministic threshold is the source "
        "of the factor-2 download-cost term in the competitive-ratio analysis."))
    story.append(P("2.3 Randomized DRL", h2))
    story.append(P(
        "R1 updates the fractional state with projected online gradient descent: "
        "p_t = Pi_{D_vec}(p_{t-1} + eta c~_t e_{n*_t}), where c~_t is the surrogate cost "
        "coefficient (the realized cost in the default mode, or an EMA estimate in the expected "
        "mode). R2 draws K persistent uniform vectors U_k and sets x_{k,n,t} = 1[U_{k,n} &lt; p_{n,t}]. "
        "The serving path k* is kept fixed for the run. If the rounded sample violates the "
        "constraints, a repair evicts services. The stateless repair evicts in increasing order of "
        "p_n / max_r a_{n,r}; the Landlord repair uses persistent credits and rent proportional to "
        "the bottleneck size. R3 draws a threshold theta_n from the density "
        "f(theta) = e^{theta/M_n} / (M_n (e-1)) on [0, M_n]. The download gate starts a download "
        "only if the serving sample currently contains the service; this is what makes the budget "
        "guarantee hold and makes repair-induced fetches zero by construction."))

    # ---------------- 3 theory
    story.append(P("3. Theoretical results", h1))
    story.append(P("Lemma 1 (feasibility, G1).", h2))
    story.append(P(
        "Any repair that evicts at least one cached service in every pass and stops as soon as the "
        "state is feasible returns a state with A x &lt;= b + eps after at most N passes. This holds "
        "for the stateless and Landlord rules."))
    story.append(P("Lemma 2 (budget, G2).", h2))
    story.append(P(
        "Let B_consumed be the running sum of M_n for downloads triggered by R3. With the budget "
        "check of Algorithm 1 and the download gate, B_consumed &lt;= B_max at all times on every "
        "sample path. Without the gate the budget can be exceeded."))
    story.append(P("Proposition 1 (projection).", h2))
    story.append(P(
        "Projected dual ascent with step 1/L satisfies the same first-order and second-order "
        "conditions as the exact Euclidean projection; the error is O(1/t) and the feasibility "
        "residual is zero. The implementation matches the exact QP to about 1e-9."))
    story.append(P("Proposition 2 (surrogate regret).", h2))
    story.append(P(
        "Let eta = sqrt(N)/(l_max sqrt(T)) and L_t(P) = (1 - p_{n*_t}) c_t. (i) For every "
        "realization, the pre-update projected-OGD sequence satisfies "
        "sum_t &lt;grad F_t(p_{t-1}), p_{t-1} - P*&gt; &lt;= sqrt(N) l_max sqrt(T). (ii) Under the "
        "consistency assumption |E[c_t | F_{t-1}] - c~_t| &lt;= zeta, "
        "E[sum_t L_t(p_{t-1})] - E[min_P sum_t L_t(P)] &lt;= 2 sqrt(N) l_max sqrt(T) + zeta T. "
        "The regret is o(T) if the predictor bias vanishes; a persistent bias yields linear "
        "regret."))
    story.append(P("Lemma 3 (rounding churn, approximation).", h2))
    story.append(P(
        "If the persistent uniforms are independent of the fractional state, "
        "Pr[x_{k,n,t-1}=0, x_{k,n,t}=1] = (p_{n,t} - p_{n,t-1})_+, and the expected rounding "
        "churn is at most M_max N sqrt(T). In Algorithm 2 the fractional state depends on the "
        "in-flight indicator, which depends on the sample states, so the independence assumption "
        "fails. The lemma is therefore an approximation; the repair-induced fetches are measured "
        "separately (Section 5.5)."))
    story.append(P("Remark (G3b removed).", h2))
    story.append(P(
        "An earlier draft claimed a download-charged regret bound with an M_max N sqrt(T) term. "
        "That term does not cover repair-induced fetches, which are not bounded by the rounding "
        "churn. We do not claim G3b; the download-charged regret is open."))

    # ---------------- 4 setup
    story.append(P("4. Experimental setup", h1))
    story.append(P(
        "Synthetic world: N=40, T=8000, Zipf rho=1.2, Poisson arrivals at 0.5 requests/ms, "
        "lognormal demands, U=100 Mbps, D=1000 Mbps, RTT=20 ms. Large Sequence: k in {9,12,24,48}, "
        "M/l in {2,4,8}, OPT=(k+2n)l (an upper bound; the MILP is 1 unit lower on small k). "
        "Adversarial families: shuffled cyclic scan, Markov-modulated Zipf, bursty, hotspot, "
        "random-walk Zipf, periodic burst. Real traces: Google cluster traces 2011 (50k requests / "
        "4,619 services; dataset2_30 1k/30; dataset2_paper 1k/6). MILP: HiGHS via scipy, "
        "latency-only objective subject to the budget and per-step capacities. All randomized "
        "results are means over seeds with 95% paired CIs."))

    # ---------------- 5 results
    story.append(P("5. Results", h1))
    story.append(P("5.1 Cost-latency frontier (Phase 1)", h2))
    story.append(P(
        "Online-DRL trades cost for latency through theta. The randomized stateless repair is on "
        "or below the deterministic frontier in the synthetic world and in the moderate-cost "
        "region of the Google traces. On the synthetic world (N=40, T=8000, 10 seeds), "
        "Online-DRL ranges from (latency 144,537, cost 366,644) at theta=0 to (143,712, 24,691) "
        "at theta=16, while randomized stateless is (93,237, 12,011) - below the entire curve. "
        "On the 10k Google slice (3 seeds), Online-DRL ranges from (161,818, 82,510) to "
        "(163,253, 2,830), while randomized is (161,669, 34,002). On the 50k Google trace "
        "(3 seeds, K=2), Online-DRL ranges from (689,898, 107,107) to (697,240, 12,687), while "
        "randomized is (680,384, 24,831). At very low download budgets the deterministic method "
        "is cheaper at essentially the same latency."))
    story.append(Image(f_front, width=12.5*cm, height=7.2*cm))
    story.append(P("<b>Figure 1.</b> Cost-latency frontier. Stars are the randomized stateless "
                   "repair; lines are Online-DRL theta sweeps.", small))
    story.append(Spacer(1, 0.3*cm))
    story.append(P(
        "The budget-matched table uses the theta=1 Online-DRL cost as the reference and runs both "
        "algorithms with the same bmax. The paired latency differences are large and their 95% "
        "CIs exclude zero."))
    bt = [["B_max", "Online-DRL lat", "Randomized lat", "paired Δlat", "paired Δcost"]]
    for frac in ("0.2", "0.5", "1.0"):
        s = d_budget[frac]
        bt.append([frac, fmt(s["det_lat"][0]), fmt(s["rand_lat"][0]),
                   f"{s['paired_lat']['mean']:+,.0f} ±{s['paired_lat']['ci_half']:,.0f}",
                   f"{s['paired_cost']['mean']:+,.0f} ±{s['paired_cost']['ci_half']:,.0f}"])
    story.append(make_table(bt))
    story.append(Image(f_budget, width=10.5*cm, height=6.0*cm))
    story.append(P("<b>Figure 2.</b> Budget-matched latency at 20/50/100% of the Online-DRL cost.", small))

    story.append(P("5.2 Horizon and consistency (Phase 2)", h2))
    story.append(P(
        "The latency sign change between the 10k slice and the full 50k trace is a warm-up effect. "
        "On the raw regime the paired latency difference vs Online-DRL is +221 (10k), +50 (20k), "
        "+159 (30k), -2,448 (50k); on the crossing regime it is +353, -4, -54, -2,215. The cost "
        "reduction grows monotonically from -18,498 to -42,658 (raw) and from -1,676 to -33,867 "
        "(crossing). The MILP ALG/OPT gap to Online-DRL closes with T: the randomized stateless "
        "ratio is 1.301 at T=120, 1.631 at T=300, 1.675 at T=600, 1.622 at T=1200, while "
        "Online-DRL is 1.168, 1.456, 1.717, 1.753. The gap changes from +11% worse to -7% "
        "better."))
    story.append(Image(f_horizon, width=12.5*cm, height=6.8*cm))
    story.append(P("<b>Figure 3.</b> Paired latency and cost differences vs Online-DRL as the "
                   "trace prefix grows.", small))
    ht = [["T", "Online-DRL/OPT", "R3/OPT", "Landlord/OPT", "Stateless/OPT"]]
    for T in ("120", "300", "600", "1200"):
        s = d_milp["summary"][T]
        ht.append([T, fmt(s["det_ratio"][0], 3), fmt(s["r3_ratio"][0], 3),
                   fmt(s["rand_landlord_ratio"][0], 3), fmt(s["rand_stateless_ratio"][0], 3)])
    story.append(make_table(ht))

    story.append(P("5.3 Adversarial families and ablation (Phase 3)", h2))
    story.append(P(
        "On the shuffled cyclic scan, Online-DRL and R3-only both have latency 90, while R1+R2 "
        "has 128 (1.49x), R1+R2+R3 has 128 (1.49x), and the gate on/off makes no difference. "
        "The loss is caused by the randomized rounding/repair (R1+R2), not by R3 or the gate. "
        "On the Large Sequence, R3 alone remains linear in k (median ratio 7.9 at k=9 to 116.0 at "
        "k=48 for M/l=8), while R1+R2 is flat (2.4-4.0), R1+R2+R3 is slightly better, and "
        "Landlord is similar or slightly worse. On the additional families, the stateless repair "
        "is better on hotspot (0.862), Markov-Zipf (0.709), periodic burst (0.633) and "
        "random-walk Zipf (0.739), slightly worse on bursty (1.034), and worse on the shuffled "
        "cyclic scan. The Landlord repair is worse on all five."))
    story.append(Image(f_ls, width=12.0*cm, height=6.6*cm))
    story.append(P("<b>Figure 4.</b> Large Sequence median latency ratio to the closed-form OPT "
                   "(M/l=4). R1+R2 flattens the ratio; R3 alone is linear.", small))
    at = [["family", "Online-DRL", "R3 only", "R1+R2", "R1+R2+R3", "Landlord"]]
    for fam in ("hotspot", "markov_zipf", "periodic_burst", "random_walk_zipf", "bursty"):
        s = d_adv["summary"][fam]
        row = [fam]
        for name in ("online_drl", "r3_only", "r1r2_stateless", "r1r2r3_stateless", "r1r2r3_landlord"):
            row.append(fmt(s[name]["latency"]["median"]))
        at.append(row)
    story.append(make_table(at))
    story.append(P(
        "The closed-form OPT=(k+2n)l is an upper bound, not the exact optimum: on small "
        "instances the MILP is 1 unit lower (14 vs 15 at k=3; 29 vs 30 at k=6). Ratios using "
        "the closed form are therefore slightly overestimated (about 3-7%)."))

    story.append(P("5.4 Real data (Phase 4)", h2))
    story.append(P(
        "The raw Google disk column has M_n &lt; l_n for every service. A natural crossing is "
        "obtained with a slow downlink (D=0.1) and treating the disk column as MB: the fraction "
        "M&lt;l is 0.653 on the 10k Google slice, 0.664 on the 50k trace, 0.067 on dataset2_30 and "
        "0.500 on dataset2_paper. The randomized stateless repair reduces cost in every regime. "
        "The latency sign is trace-dependent: on the 10k slice it is +309 (raw) and +437 "
        "(crossing); on the 50k trace it is -2,448 (raw) and -2,505 (crossing); on dataset2_30 "
        "it is +1,327 (raw) and +256 (crossing). The Landlord repair has a larger latency "
        "penalty throughout."))
    rt = [["trace", "regime", "M<l", "Online-DRL lat", "stateless Δlat", "stateless Δcost"]]
    for r in d_real["rows"]:
        if r["trace"] in ("google.csv", "google_dataset2_30.csv"):
            rt.append([r["trace"], r["regime"], fmt(r["frac_M_lt_l"], 3),
                       fmt(r["det_lat"]),
                       f"{r['stateless_vs_det_lat']['mean']:+,.0f} ±{r['stateless_vs_det_lat']['ci_half']:,.0f}",
                       f"{r['stateless_vs_det_cost']['mean']:+,.0f} ±{r['stateless_vs_det_cost']['ci_half']:,.0f}"])
    story.append(make_table(rt, font=7.0))

    story.append(P("5.5 Theory measurements (Phase 5)", h2))
    story.append(P(
        "The surrogate regret is about 0.33 of the Proposition-2(i) bound in all four "
        "configurations of exp35, and the realized regret is about 0.15 of the Proposition-2(ii) "
        "bound. The EMA does not drive zeta to zero: on N=20 synthetic traces, zeta_mae grows "
        "from 0.0335 at T=8k to 0.0523 at T=32k for the expected gradient. The repair-induced "
        "fetches (gate off) are 0.2-0.7% of the rounding churn on the N=20 horizons and 6-29% on "
        "the N=40 horizons. A formal restricted competitive ratio remains open; the empirical "
        "ratio on the Large Sequence is at most 5.8 (median 1.8-4.1)."))
    tt = [["T", "zeta_mae (realized)", "zeta_mae (expected)", "repair fetches / rounding churn"]]
    for i, T in enumerate(("8,000", "16,000", "32,000")):
        rr = d_zeta["summary"]["realized"]
        ee = d_zeta["summary"]["expectation"]
        tt.append([T, fmt(rr["zeta_mae"][i], 4), fmt(ee["zeta_mae"][i], 4),
                   f"{rr['refetch'][i]}/{rr['churn'][i]:.0f}"])
    story.append(make_table(tt))

    # ---------------- 6 discussion
    story.append(P("6. Discussion", h1))
    story.append(P(
        "The evidence supports G1, G2 and G3 as proved (with the gate and the surrogate-loss "
        "caveat). G4 is empirical: the randomized stateless repair is on or below the deterministic "
        "frontier over a meaningful range, but it loses on the shuffled cyclic scan and its "
        "latency effect is trace-dependent. The flat ratio in k on the Large Sequence is not a "
        "general competitive ratio. The paper should claim the cost-latency improvement and the "
        "surrogate-regret result, and should state the open problems: a rigorous churn lemma, a "
        "download-charged regret bound, and a restricted competitive ratio."))
    story.append(P("Threats to validity.", h2))
    for b in [
        "Synthetic data and constructed adversaries; the real-trace check is limited to the Google 2011 traces.",
        "One trace family; SHARCNet/Azure/Alibaba are not reachable from this workspace.",
        "The M/l crossing uses a slow downlink (D=0.1) or the documented disk rescaling; the fraction M&lt;l is reported.",
        "The latency sign is trace-dependent; both the 10k and 50k results are reported.",
        "The churn lemma is an approximation; the repair-induced fetches are measured separately.",
        "The EMA does not drive zeta to zero; the zeta T term remains linear.",
        "The peeking implementation is an optimistic variant; the proposition is stated for the pre-update sequence.",
    ]:
        story.append(P("&bull; " + b, small))

    # ---------------- 7 related work
    story.append(P("7. Related work", h1))
    story.append(P(
        "Online caching and paging (Sleator and Tarjan, 1985; Fiat et al., 1991; Young, 2002; "
        "Bansal et al., 2012; Karlin et al., 1988, 1994; Borodin et al., 1998; Buchbinder and "
        "Naor, 2009; Reineke et al., 2008; Epstein et al., 2013). Edge service caching with "
        "delayed hits (Deka and Sukapuram, 2025; Fan et al., 2023; Atre et al., 2020; Tsachor "
        "et al., 2026; Tan et al., 2019; Wang et al., 2017; Chen et al., 2018; Barman et al., "
        "2026). Online convex optimization and regret (Zinkevich, 2003; Hazan, 2016; "
        "Shalev-Shwartz, 2011; Bansal et al., 2015). Knapsack, vector packing and rounding "
        "(Bockenhauer et al., 2014; Albers et al., 2021; Panigrahy and Talwar, 2011; Chekuri "
        "and Vondrak, 2010). Multi-resource allocation and serverless (Ghodsi et al., 2011; "
        "Grandl et al., 2014; Zheng et al., 2016; Jonas et al., 2019; Rausch et al., 2020; Du "
        "et al., 2020; Silva et al., 2020). Caching with fetch delay and cost (Kroft, 1998; "
        "Araldo et al., 2014; Yang et al., 2019). Datasets (Alam et al., 2016; Iosup et al., "
        "2008; Shah et al., 2021)."))

    # ---------------- 8 conclusion
    story.append(P("8. Conclusion", h1))
    story.append(P(
        "Randomization is a useful engineering addition to Online-DRL. The stateless repair "
        "reduces download cost on synthetic and real traces and is on or below the deterministic "
        "frontier over a meaningful range; R1+R2 is the component that flattens the Large "
        "Sequence ratio. The method is not a universal improvement: it loses on the shuffled "
        "cyclic scan, and the latency effect is trace-dependent. The theory proves feasibility, "
        "budget compliance and a surrogate-regret bound, but not a competitive ratio. The open "
        "problems are a rigorous churn lemma, a download-charged regret bound, and a restricted "
        "competitive ratio."))

    # ---------------- references
    story.append(PageBreak())
    story.append(P("References", h1))
    try:
        refs = json.load(open(os.path.join(RES, "reference_check.json")))
    except Exception:
        refs = []
    for i, r in enumerate(refs, 1):
        ms = r.get("matches", [])
        if not ms:
            continue
        m = ms[0]
        authors = m.get("authors", [])
        a0 = authors[0] if authors else "?"
        year = m.get("year", "")
        title = m.get("title", "")
        venue = m.get("venue") or ""
        doi = m.get("doi") or ""
        line = f"[{i}] {a0} et al. ({year}). {title}. {venue}."
        if doi:
            line += f" {doi}"
        story.append(P(line, small))

    story.append(Spacer(1, 0.4*cm))
    story.append(P("Appendix: reproducibility", h2))
    story.append(P(
        "The code, seeds and result JSONs are in final_exp/. Run `python experiments/run_all.py` "
        "for the full suite, or the individual experiments exp38-exp42 for the Phase 1-5 v2 "
        "results. Every randomized number in the paper carries a seed list and a CI in the JSON "
        "files.", small))

    doc.build(story)
    print("wrote", OUT)


# load data
d_budget = load("exp38_theta_frontier")["budget"]
d_milp = load("exp31_alg_opt_vs_T")
d_adv = load("exp40_adversarial_ablation")
d_real = load("exp41_real_data_phase4")
d_zeta = load("exp42_theory_phase5_v2")

if __name__ == "__main__":
    build()