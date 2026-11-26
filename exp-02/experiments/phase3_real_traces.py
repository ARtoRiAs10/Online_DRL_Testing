"""Phase 3 -- real traces (paper Experiments 1-14).

Google dataset2 (1k/30) and the 50k Google trace, in three regimes:

* ``paper_recipe`` -- the literal KB recipe (M < l for every service);
* ``natural_crossing`` -- disk column as MB, D=0.1;
* ``calibrated`` -- D=1e-4, the M_i > l_i calibration that reproduces the
  paper's baseline (row #0).

Hard MG and Soft-MG (tau=0.15) are compared with deterministic Online-DRL.
SHARCNet is BLOCKED when the trace is absent.
"""
from __future__ import annotations

import os

import numpy as np

from common import (DATA, markdown_table, paired_ci, save_fig, save_json,
                    wilcoxon)
import online_drl_r3 as m

SEEDS = 30
SOFT_TAU = 0.15
REGIMES = {
    "paper_recipe": dict(time_scale_ms=1e-3, disk_unit_mb=1.0 / 1024.0,
                         D=1000.0, U=100.0),
    "natural_crossing": dict(time_scale_ms=1e-3, disk_unit_mb=1.0,
                             D=0.1, U=100.0),
    "calibrated": dict(time_scale_ms=1e-3, disk_unit_mb=1.0,
                       D=1e-4, U=100.0),
}


def pct(a, b):
    return 100.0 * (a - b) / b if b else float("nan")


def run_dataset(name, path, rows, C, seeds=SEEDS):
    out = []
    for regime, cfg0 in REGIMES.items():
        cfg = dict(id_col="JobID", time_col="Timestamp", cpu_col="CPU",
                   ram_col="RAM", disk_col="Disk", max_rows=rows, **cfg0)
        cat, tr = m.load_trace_csv(path, **cfg)
        det = m.online_drl(cat, tr, C)
        ll = m.ll_rc(cat, tr, C)
        r3 = [m.drl_r3(cat, tr, C, seed=s) for s in range(seeds)]
        mg = [m.drl_mg(cat, tr, C, seed=s) for s in range(seeds)]
        mgs = [m.drl_mg(cat, tr, C, seed=s, soft_tau=SOFT_TAU)
               for s in range(seeds)]
        r3mg = [m.drl_r3_mg(cat, tr, C, seed=s) for s in range(seeds)]
        r3mgs = [m.drl_r3_mg(cat, tr, C, seed=s, soft_tau=SOFT_TAU)
                 for s in range(seeds)]
        base = np.full(seeds, det.latency)
        row = dict(
            dataset=name, regime=regime, C=int(C), n_services=int(cat.N),
            n_requests=int(len(tr)), seeds=seeds, soft_tau=SOFT_TAU,
            frac_M_lt_l=float(np.mean(cat.M < cat.l)),
            frac_M_gt_l=float(np.mean(cat.M > cat.l)),
            det_latency=float(det.latency), det_cost=float(det.cost),
            llrc_latency=float(ll.latency), llrc_cost=float(ll.cost),
            det_vs_llrc_lat_pct=pct(det.latency, ll.latency),
            det_vs_llrc_cost_pct=pct(det.cost, ll.cost),
            r3_latency=float(np.mean([r.latency for r in r3])),
            r3_cost=float(np.mean([r.cost for r in r3])),
            mg_latency=float(np.mean([r.latency for r in mg])),
            mg_cost=float(np.mean([r.cost for r in mg])),
            mgs_latency=float(np.mean([r.latency for r in mgs])),
            mgs_cost=float(np.mean([r.cost for r in mgs])),
            r3mg_latency=float(np.mean([r.latency for r in r3mg])),
            r3mg_cost=float(np.mean([r.cost for r in r3mg])),
            r3mgs_latency=float(np.mean([r.latency for r in r3mgs])),
            r3mgs_cost=float(np.mean([r.cost for r in r3mgs])),
            mg_lat_pct=pct(np.mean([r.latency for r in mg]), det.latency),
            mg_cost_pct=pct(np.mean([r.cost for r in mg]), det.cost),
            mgs_lat_pct=pct(np.mean([r.latency for r in mgs]), det.latency),
            mgs_cost_pct=pct(np.mean([r.cost for r in mgs]), det.cost),
            r3mg_lat_pct=pct(np.mean([r.latency for r in r3mg]), det.latency),
            r3mg_cost_pct=pct(np.mean([r.cost for r in r3mg]), det.cost),
            r3mgs_lat_pct=pct(np.mean([r.latency for r in r3mgs]), det.latency),
            r3mgs_cost_pct=pct(np.mean([r.cost for r in r3mgs]), det.cost),
            r3_lat_pct=pct(np.mean([r.latency for r in r3]), det.latency),
            r3_cost_pct=pct(np.mean([r.cost for r in r3]), det.cost),
            mg_ci=paired_ci([r.latency for r in mg], base),
            mgs_ci=paired_ci([r.latency for r in mgs], base),
            mg_p=wilcoxon([r.latency for r in mg], base)[1],
            mgs_p=wilcoxon([r.latency for r in mgs], base)[1],
            violations=sum(len(r.extra.get("violations", {})) for r in r3mgs),
        )
        out.append(row)
        print(f"{name:20s} {regime:16s} C={C:3d} M>l={row['frac_M_gt_l']:.3f} "
              f"det={det.latency:.0f}/{det.cost:.0f} "
              f"MG={row['mg_lat_pct']:+.2f}%/{row['mg_cost_pct']:+.2f}% "
              f"SoftMG={row['mgs_lat_pct']:+.2f}%/{row['mgs_cost_pct']:+.2f}% "
              f"R3+SoftMG={row['r3mgs_lat_pct']:+.2f}%/{row['r3mgs_cost_pct']:+.2f}%")
    return out


def main():
    rows = []
    rows += run_dataset("google_dataset2_30",
                        os.path.join(DATA, "google_dataset2_30.csv"),
                        1000, C=10)
    rows += run_dataset("google_50k", os.path.join(DATA, "google.csv"),
                        50000, C=50)

    sharcnet = os.path.join(DATA, "sharcnet.csv")
    sh_status = "BLOCKED"
    sh_reason = "sharcnet.csv not bundled; gwa.unc.edu unreachable"
    if os.path.exists(sharcnet):
        sh_status = "loaded"
        sh_reason = "OPT comparison not run"

    # #6 is evaluated on Soft-MG in the non-degenerate regimes.
    nontrivial = [r for r in rows if r["regime"] in ("natural_crossing", "calibrated")]
    no_harm_soft = all(r["mgs_lat_pct"] <= 2.0 and r["mgs_cost_pct"] <= 2.0
                       for r in nontrivial)
    no_harm_hard = all(r["mg_lat_pct"] <= 2.0 and r["mg_cost_pct"] <= 2.0
                       for r in nontrivial)
    out = dict(config=dict(seeds=SEEDS, soft_tau=SOFT_TAU), rows=rows,
               sharcnet=dict(status=sh_status, reason=sh_reason),
               mg_no_harm_hard=bool(no_harm_hard),
               mg_no_harm_soft=bool(no_harm_soft))
    save_json("phase3_real_traces", out)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    labels = [f"{r['dataset']}\n{r['regime']}" for r in rows]
    x = np.arange(len(rows))
    ax[0].bar(x - 0.2, [r["mg_lat_pct"] for r in rows], 0.4, label="MG")
    ax[0].bar(x + 0.2, [r["mgs_lat_pct"] for r in rows], 0.4, label="Soft-MG")
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].axhline(2, color="r", ls="--", lw=.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=6, rotation=30)
    ax[0].set_ylabel("latency change vs Online-DRL (%)")
    ax[0].legend(fontsize=7)
    ax[1].bar(x - 0.2, [r["mg_cost_pct"] for r in rows], 0.4, label="MG")
    ax[1].bar(x + 0.2, [r["mgs_cost_pct"] for r in rows], 0.4, label="Soft-MG")
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].axhline(2, color="r", ls="--", lw=.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=6, rotation=30)
    ax[1].set_ylabel("cost change vs Online-DRL (%)")
    ax[1].legend(fontsize=7)
    save_fig(fig, "phase3_real_traces")

    lines = ["# Phase 3 -- real traces", "",
             f"Seeds: {SEEDS}.  Soft-MG tau={SOFT_TAU}.  "
             f"SHARCNet: **{sh_status}** ({sh_reason}).", "",
             markdown_table(
                 ["dataset", "regime", "C", "M>l", "MG lat %", "MG cost %",
                  "SoftMG lat %", "SoftMG cost %", "R3+SoftMG lat %",
                  "R3+SoftMG cost %"],
                 [[r["dataset"], r["regime"], r["C"],
                   f"{r['frac_M_gt_l']:.3f}",
                   f"{r['mg_lat_pct']:+.2f}", f"{r['mg_cost_pct']:+.2f}",
                   f"{r['mgs_lat_pct']:+.2f}", f"{r['mgs_cost_pct']:+.2f}",
                   f"{r['r3mgs_lat_pct']:+.2f}",
                   f"{r['r3mgs_cost_pct']:+.2f}"] for r in rows]),
             "", f"MG no harm (<=2%) hard: **{no_harm_hard}**; "
             f"Soft-MG no harm (<=2%): **{no_harm_soft}**"]
    with open("docs/phase3_real_traces.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote docs/phase3_real_traces.md")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main()