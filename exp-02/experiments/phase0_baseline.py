"""Phase 0 -- reproduce the paper's baseline (precondition 1).

Runs Online-DRL and LL-RC on the bundled Google traces in the paper's recipe
and in the documented natural-crossing regime, sweeps the cache size C, and
checks the published intervals with the frozen 2-percentage-point tolerance.

The SHARCNet trace is not bundled and its host is unreachable from this
environment; that precondition is recorded as BLOCKED, not as a pass.
"""
from __future__ import annotations

import os

import numpy as np

from common import DATA, markdown_table, save_fig, save_json
import online_drl_r3 as m

PAPER_LAT = (0.016, 7.41)
PAPER_COST = (-92.54, -77.57)
TOL = 2.0

REGIMES = {
    # name: kwargs for load_trace_csv
    "paper_recipe": dict(time_scale_ms=1e-3, disk_unit_mb=1.0 / 1024.0,
                         D=1000.0, U=100.0),
    "natural_crossing": dict(time_scale_ms=1e-3, disk_unit_mb=1.0,
                             D=0.1, U=100.0),
}

DATASETS = {
    "google_dataset2_30": "google_dataset2_30.csv",
    "google_dataset2_paper": "google_dataset2_paper.csv",
    "google_50k": "google.csv",
}


def in_range(x, interval, tol=TOL):
    return interval[0] - tol <= x <= interval[1] + tol


def run_one(path, regime, C, rows=None):
    cfg = dict(id_col="JobID", time_col="Timestamp", cpu_col="CPU",
               ram_col="RAM", disk_col="Disk", max_rows=rows, **REGIMES[regime])
    cat, tr = m.load_trace_csv(path, **cfg)
    d = m.online_drl(cat, tr, C)
    b = m.ll_rc(cat, tr, C)
    lat = 100.0 * (d.latency - b.latency) / b.latency if b.latency else float("nan")
    cost = 100.0 * (d.cost - b.cost) / b.cost if b.cost else float("nan")
    return dict(C=int(C), n_services=int(cat.N), n_requests=int(len(tr)),
                latency=float(d.latency), llrc_latency=float(b.latency),
                cost=float(d.cost), llrc_cost=float(b.cost),
                lat_pct=float(lat), cost_pct=float(cost),
                downloads=int(d.downloads), llrc_downloads=int(b.downloads),
                hits=int(d.hits), llrc_hits=int(b.hits),
                delayed_hits=int(d.delayed_hits),
                frac_M_lt_l=float(np.mean(cat.M < cat.l)),
                violations=d.extra.get("violations", {}))


def main():
    rows = []
    summary = {}
    for ds, fname in DATASETS.items():
        path = os.path.join(DATA, fname)
        rows_ds = 50000 if ds == "google_50k" else 1000
        for regime in REGIMES:
            # C grid: small values matter for the baseline; the 50k trace gets
            # a wider grid because it has 4619 services.
            if ds == "google_50k":
                Cs = [1, 2, 5, 10, 20, 50]
            else:
                Cs = [1, 2, 3, 5, 8, 10, 15, 20]
            best = None
            for C in Cs:
                if C > 0:
                    r = run_one(path, regime, C, rows=rows_ds)
                    r.update(dataset=ds, regime=regime)
                    rows.append(r)
                    # distance to the paper rectangle (0 if inside)
                    dx = max(PAPER_LAT[0] - TOL - r["lat_pct"], 0,
                             r["lat_pct"] - (PAPER_LAT[1] + TOL))
                    dy = max(PAPER_COST[0] - TOL - r["cost_pct"], 0,
                             r["cost_pct"] - (PAPER_COST[1] + TOL))
                    dist = dx + dy
                    if best is None or dist < best[0]:
                        best = (dist, r)
            b = best[1]
            ok = (in_range(b["lat_pct"], PAPER_LAT)
                  and in_range(b["cost_pct"], PAPER_COST))
            summary[f"{ds}/{regime}"] = dict(
                best_C=b["C"], lat_pct=b["lat_pct"], cost_pct=b["cost_pct"],
                latency_in_range=in_range(b["lat_pct"], PAPER_LAT),
                cost_in_range=in_range(b["cost_pct"], PAPER_COST),
                pass_=bool(ok),
                frac_M_lt_l=b["frac_M_lt_l"])
            print(f"{ds:24s} {regime:16s} best C={b['C']:3d} "
                  f"lat={b['lat_pct']:+.3f}% cost={b['cost_pct']:+.2f}% "
                  f"M<l={b['frac_M_lt_l']:.3f} pass={ok}")

    # SHARCNet: only evaluate if a preprocessed CSV is present.
    sharcnet = os.path.join(DATA, "sharcnet.csv")
    if os.path.exists(sharcnet):
        # The paper's SHARCNet comparison is against OPT, not LL-RC.  Without
        # the Gurobi OPT here we can only record the raw Online-DRL numbers.
        cfg = dict(id_col="ExecutableID", time_col="SubmitTime",
                   cpu_col="UsedCPUTime", ram_col="UsedMemory",
                   disk_col="UsedLocalDiskSpace", time_scale_ms=1000.0,
                   disk_unit_mb=1.0, D=2000.0, U=200.0, max_rows=1000)
        cat, tr = m.load_trace_csv(sharcnet, **cfg)
        d = m.online_drl(cat, tr, min(50, max(1, cat.N // 2)))
        summary["sharcnet"] = dict(status="loaded",
                                   latency=float(d.latency), cost=float(d.cost),
                                   note="OPT comparison not run")
    else:
        summary["sharcnet"] = dict(status="BLOCKED",
                                   reason="sharcnet.csv not bundled; "
                                          "gwa.unc.edu unreachable")
        print("SHARCNet: BLOCKED (no trace)")

    out = dict(paper=dict(latency_pct=PAPER_LAT, cost_pct=PAPER_COST,
                          tolerance=TOL),
               rows=rows, summary=summary)
    save_json("phase0_baseline", out)

    # markdown table
    lines = ["# Phase 0 -- baseline reproduction", "",
             f"Published Google intervals: latency {PAPER_LAT[0]:+.3f}% .. "
             f"{PAPER_LAT[1]:+.2f}%, cost {PAPER_COST[0]:.2f}% .. "
             f"{PAPER_COST[1]:.2f}%; tolerance +/-{TOL} pp.", ""]
    table = []
    for key, s in summary.items():
        if key == "sharcnet":
            table.append(["SHARCNet", s.get("status", ""), "", "", "",
                          s.get("reason", "")])
        else:
            table.append([key, s["best_C"], f"{s['lat_pct']:+.3f}",
                          f"{s['cost_pct']:+.2f}",
                          "PASS" if s["pass_"] else "FAIL",
                          f"M<l={s['frac_M_lt_l']:.3f}"])
    lines.append(markdown_table(["dataset/regime", "best C", "lat %", "cost %",
                                 "verdict", "note"], table))
    lines += ["", "## Verdict", "",
              "The Google baseline is **not reproduced** in either regime: "
              "the paper-recipe regime has M < l for every service (so "
              "Online-DRL and LL-RC coincide), and the natural-crossing regime "
              "gives a much smaller cost reduction than the published "
              "77.57--92.54%.  The SHARCNet precondition is blocked because "
              "the trace is unavailable.  Per the frozen preconditions, all "
              "downstream claims are therefore **blocked**."]
    with open(os.path.join(os.path.dirname(DATA), "docs", "phase0_baseline.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote docs/phase0_baseline.md")


if __name__ == "__main__":
    main()