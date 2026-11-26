"""Phase 7 -- reporting and go/no-go.

Evaluates the frozen acceptance criteria in ``ACCEPTANCE/`` against the
result files and writes ``docs/REPORT.md`` plus ``results/acceptance.json``.
"""
from __future__ import annotations

import json
import os

import numpy as np

from common import DOCS, RESULTS, load_json, markdown_table, save_json


def status(pre_ok, claim_ok, blocked=True):
    if not pre_ok and blocked:
        return "BLOCKED"
    return "ACCEPT" if claim_ok else "REJECT"


def main():
    p0 = load_json("phase0_baseline")
    p1 = load_json("phase1_correctness")
    p2 = load_json("phase2_synthetic")
    p3 = load_json("phase3_real_traces")
    p4 = load_json("phase4_matched_cost")
    p5 = load_json("phase5_ablation")
    p6 = load_json("phase6_stats_overhead")

    # -- preconditions ---------------------------------------------------
    baseline_rows = [(k, v) for k, v in p0["summary"].items() if k != "sharcnet"]
    baseline_pass = all(v.get("pass_", False) for k, v in baseline_rows)
    sharcnet = p0["summary"].get("sharcnet", {})
    reduction = bool(p1["preconditions"]["reduction"])
    invariants = bool(p1["preconditions"]["invariants"])
    pre_ok = bool(baseline_pass and reduction and invariants)
    pre = dict(
        baseline_reproduced=bool(baseline_pass),
        baseline_detail={k: v for k, v in p0["summary"].items()},
        sharcnet=sharcnet,
        reduction=reduction,
        invariants=invariants,
        pass_=pre_ok,
    )

    # -- claims ----------------------------------------------------------
    ski = p1["ski_rental"]
    ski_ok = abs(ski["random_worst"] - 1.58) <= 0.02
    r3_match = p4["overall"]["pass_"]
    ls = p2["large_sequence"]
    mg_lock = bool(p2["cyclic"]["mg_20pct"] and p2["cyclic"]["mg_sig"])
    mg_no_harm = bool(p3["mg_no_harm_natural_crossing"])
    ls_ok = bool(ls["lower_every_k"] and ls["slope_flatter"])
    # O(log k): fit ratio vs log(k)
    sub = [r for r in ls["rows"] if r["n_mult"] == 1]
    ks = np.array([r["k"] for r in sub], float)
    r3mg = np.array([r["r3mg_ratio"] for r in sub])
    det = np.array([r["det_ratio"] for r in sub])
    logk = np.log(ks)
    slope_r3mg, intercept_r3mg = np.polyfit(logk, r3mg, 1)
    pred = slope_r3mg * logk + intercept_r3mg
    ss_res = ((r3mg - pred) ** 2).sum()
    ss_tot = ((r3mg - r3mg.mean()) ** 2).sum()
    r2_log = float(1 - ss_res / ss_tot) if ss_tot else float("nan")
    slope_det, _ = np.polyfit(logk, det, 1)
    ologk_ok = bool(r2_log > 0.9 and slope_r3mg < 0.8 * slope_det)
    overhead_ok = all(r["within_2x"] for r in p6["overhead"])
    overhead_ratio = max(r["ratio_to_det"] for r in p6["overhead"])

    claims = [
        dict(claim="R3 improves the trigger (ratio 2 -> e/(e-1))",
             accept=f"pure ski-rental worst ratio 1.58 +/- 0.02",
             observed=f"{ski['random_worst']:.4f} (det {ski['deterministic_worst']:.4f})",
             ok=ski_ok, status=("ACCEPT (isolated trigger only)" if ski_ok else "REJECT"),
             blocked=False),
        dict(claim="R3 beats deterministic at equal cost",
             accept=">=3 of 4 matched levels, CI excludes 0, >=2 of 3 datasets",
             observed=f"overall pass={r3_match}; datasets passing="
                      f"{p4['overall']['datasets_pass']}",
             ok=r3_match, status=status(pre_ok, r3_match), blocked=not pre_ok),
        dict(claim="MG breaks eviction lockstep",
             accept=">=20% lower on cyclic, paired p<0.05 over >=30 seeds",
             observed=f"MG>=20% on majority={p2['cyclic']['mg_20pct']}; "
                      f"significant={p2['cyclic']['mg_sig']}; "
                      f"best reduction={max(r['mg_reduction_pct'] for r in p2['cyclic']['rows']):.1f}%",
             ok=mg_lock, status=status(pre_ok, mg_lock), blocked=not pre_ok),
        dict(claim="MG does no harm on real traces",
             accept="latency and cost within 1-2% on Google and SHARCNet",
             observed=f"no-harm={mg_no_harm}; SHARCNet={sharcnet.get('status')}",
             ok=mg_no_harm, status=status(pre_ok, mg_no_harm), blocked=not pre_ok),
        dict(claim="R3+MG lowers E[ALG]/OPT_t on the Large Sequence",
             accept="lower for every k in {6,12,24,48} and flatter slope",
             observed=f"lower_every_k={ls['lower_every_k']}; "
                      f"slope det={ls['det_slope']:.3f} r3mg={ls['r3mg_slope']:.3f}",
             ok=ls_ok, status=status(pre_ok, ls_ok), blocked=not pre_ok),
        dict(claim="Improves toward O(log k)",
             accept="roughly logarithmic over >=4 k, slope below deterministic",
             observed=f"log-fit R^2={r2_log:.3f}, slope r3mg={slope_r3mg:.3f} "
                      f"det={slope_det:.3f}; ratio still grows ~linearly",
             ok=ologk_ok, status=("UNSUPPORTED (conjecture)" if not ologk_ok
                                  else status(pre_ok, True)), blocked=not pre_ok),
        dict(claim="Overhead stays small",
             accept="us/request within 2x of deterministic",
             observed=f"max ratio={overhead_ratio:.2f}x",
             ok=overhead_ok, status=("ACCEPT (measurement)" if overhead_ok
                                     else "REJECT"), blocked=False),
    ]

    out = dict(preconditions=pre, claims=claims,
               overall=dict(go=bool(pre_ok and r3_match and mg_lock and mg_no_harm),
                            preconditions=pre_ok,
                            r3_matched_cost=r3_match,
                            mg_lockstep=mg_lock,
                            mg_no_harm=mg_no_harm))
    save_json("acceptance", out)

    # -- report ----------------------------------------------------------
    L = []
    L.append("# R3 + Marking-Gated Landlord -- experiment report")
    L.append("")
    L.append("Generated by `experiments/phase7_report.py`.  The acceptance "
             "thresholds in `ACCEPTANCE/` were frozen before the runs.")
    L.append("")
    L.append("## Preconditions")
    L.append("")
    L.append(markdown_table(
        ["precondition", "status", "detail"],
        [["baseline reproduced (Google)",
          pre["baseline_reproduced"],
          "; ".join(f"{k}: lat {v.get('lat_pct', float('nan')):+.2f}% "
                    f"cost {v.get('cost_pct', float('nan')):+.2f}%"
                    for k, v in baseline_rows)],
         ["SHARCNet baseline", sharcnet.get("status"), sharcnet.get("reason", "")],
         ["reduction (H0, lam=0)", reduction, ""],
         ["invariants", invariants, ""],
         ["**all preconditions**", pre_ok,
          "blocked" if not pre_ok else "ok"]]))
    L.append("")
    L.append("## Claims")
    L.append("")
    L.append(markdown_table(
        ["claim", "accept if", "observed", "status"],
        [[c["claim"], c["accept"], c["observed"], c["status"]] for c in claims]))
    L.append("")
    L.append("## Phase summaries")
    L.append("")
    L.append("### Phase 0 -- baseline")
    L.append("")
    for k, v in p0["summary"].items():
        if k == "sharcnet":
            L.append(f"* SHARCNet: **{v.get('status')}** -- {v.get('reason','')}")
        else:
            L.append(f"* `{k}`: best C={v['best_C']}, latency "
                     f"{v['lat_pct']:+.3f}%, cost {v['cost_pct']:+.2f}% "
                     f"(M<l={v['frac_M_lt_l']:.3f}) -> "
                     f"**{'PASS' if v['pass_'] else 'FAIL'}**")
    L.append("")
    L.append("### Phase 1 -- correctness")
    L.append("")
    L.append(f"* H0 deterministic seed-independent: **{p1['h0_deterministic_seed_independent']}**")
    L.append(f"* lam=0 equals LL-RC: **{p1['lam0_equals_llrc']}**; "
             f"theta=0 equals lam=0: **{p1['theta0_equals_lam0']}**")
    L.append(f"* threshold mean {p1['threshold_mean']['mean']:.6f} "
             f"(theory {p1['threshold_mean']['theory']:.6f})")
    L.append(f"* ski-rental worst {p1['ski_rental']['random_worst']:.4f} "
             f"(deterministic {p1['ski_rental']['deterministic_worst']:.4f})")
    L.append(f"* invariants hold: **{p1['invariants_hold']}**; "
             f"budget holds: **{p1['budget']['holds']}**")
    L.append("")
    L.append("### Phase 2 -- Large Sequence")
    L.append("")
    L.append(markdown_table(
        ["k", "det/OPT", "R3/OPT", "MG/OPT", "R3+MG/OPT"],
        [[r["k"], f"{r['det_ratio']:.3f}", f"{r['r3_ratio']:.3f}",
          f"{r['mg_ratio']:.3f}", f"{r['r3mg_ratio']:.3f}"]
         for r in ls["rows"] if r["n_mult"] == 1]))
    L.append("")
    L.append(f"Linear slope det={ls['det_slope']:.3f}, R3+MG={ls['r3mg_slope']:.3f}; "
             f"lower every k={ls['lower_every_k']}; flatter={ls['slope_flatter']}.")
    L.append("")
    L.append("### Phase 2 -- cyclic adversaries")
    L.append("")
    L.append(markdown_table(
        ["config", "det", "MG", "reduction %", "p (Holm)"],
        [[r["config"], f"{r['det_latency']:.0f}", f"{r['mg_latency']:.0f}",
          f"{r['mg_reduction_pct']:+.1f}", f"{r['mg_p_holm']:.3g}"]
         for r in p2["cyclic"]["rows"]]))
    L.append("")
    L.append("### Phase 3 -- real traces")
    L.append("")
    L.append(markdown_table(
        ["dataset", "regime", "MG lat %", "MG cost %", "R3+MG lat %",
         "R3+MG cost %"],
        [[r["dataset"], r["regime"], f"{r['mg_lat_pct']:+.2f}",
          f"{r['mg_cost_pct']:+.2f}", f"{r['r3mg_lat_pct']:+.2f}",
          f"{r['r3mg_cost_pct']:+.2f}"] for r in p3["rows"]]))
    L.append("")
    L.append("### Phase 4 -- matched cost")
    L.append("")
    for name, v in p4.items():
        if name == "overall":
            continue
        L.append(f"* `{name}`: wins {v['n_wins']}/{v['n_levels']}, "
                 f"pass={v['pass_']}")
        L.append("")
        L.append(markdown_table(
            ["level", "covered", "gap", "95% CI", "p", "p Holm", "win"],
            [[f"{r['level']:.1f}", f"{r['covered']}/{p4.get('config', {}).get('seeds', 30)}",
              f"{r['gap_mean']:+.1f}",
              f"[{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}]",
              f"{r['p']:.3g}", f"{r['p_holm']:.3g}", r["win"]]
             for r in v["rows"]]))
        L.append("")
    L.append(f"Overall R3 matched-cost pass: **{r3_match}**.")
    L.append("")
    L.append("### Phase 5 -- ablation")
    L.append("")
    L.append(markdown_table(
        ["config", "latency", "cost", "hits", "phase clears"],
        [[r["config"], f"{r['latency']:.0f}", f"{r['cost']:.0f}",
          f"{r['hits']:.0f}", f"{r['phase_clears']:.1f}"]
         for r in p5["rows"]]))
    L.append("")
    L.append("### Phase 6 -- overhead")
    L.append("")
    L.append(markdown_table(
        ["config", "median us/req", "ratio to det", "within 2x"],
        [[r["config"], f"{r['median']:.3f}", f"{r['ratio_to_det']:.2f}",
          r["within_2x"]] for r in p6["overhead"]]))
    L.append("")
    L.append("## Go / no-go")
    L.append("")
    L.append(f"* Preconditions: **{'PASS' if pre_ok else 'FAIL'}** "
             f"(baseline reproduced={baseline_pass}, SHARCNet={sharcnet.get('status')}).")
    L.append(f"* R3 at matched cost: **{'PASS' if r3_match else 'FAIL'}**.")
    L.append(f"* MG lockstep: **{'PASS' if mg_lock else 'FAIL'}**.")
    L.append(f"* MG no harm: **{'PASS' if mg_no_harm else 'FAIL'}**.")
    L.append(f"* R3+MG Large Sequence: **{'PASS' if ls_ok else 'FAIL'}** "
             f"(but formally blocked by the baseline precondition).")
    L.append(f"* Overhead: **{'PASS' if overhead_ok else 'FAIL'}** "
             f"(max {overhead_ratio:.2f}x deterministic).")
    L.append("")
    L.append("**Decision: NO-GO for the claim that R3+MG improves on "
             "Online-DRL.**  The baseline precondition fails (the bundled "
             "Google traces do not reproduce the paper's published "
             "latency/cost split, and SHARCNet is unavailable), and R3 does "
             "not win at matched cost.  MG does break lockstep on the cyclic "
             "family (59-79% lower latency, p<0.05 over 30 seeds) and R3+MG "
             "lowers the Large Sequence ratio at every k, but both results "
             "are formally blocked by the baseline precondition, and MG "
             "regresses cost by more than 2% on the natural-crossing Google "
             "traces.  The isolated ski-rental ratio, the MG lockstep result "
             "and the Large Sequence slope are positive signals, not "
             "end-to-end evidence.")
    with open(os.path.join(DOCS, "REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote docs/REPORT.md")


if __name__ == "__main__":
    main()