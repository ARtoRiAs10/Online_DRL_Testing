"""Phase 3 -- real traces (paper Experiments 1-14).

Google dataset2 (1k/30), the 50k Google trace and, if present, SHARCNet.
The paper-recipe regime has M < l for every Google service, so Online-DRL and
LL-RC coincide; the natural-crossing regime (disk column as MB, D=0.1) is
reported as well because it is the only regime in which the download decision
is non-trivial.

SHARCNet is BLOCKED when the trace is absent.
"""
from __future__ import annotations

import os

import numpy as np

from common import (DATA, bootstrap_ci, describe, markdown_table, paired_ci,
                    save_fig, save_json, wilcoxon)
import online_drl_r3 as m

SEEDS = 30
REGIMES = {
    "paper_recipe": dict(time_scale_ms=1e-3, disk_unit_mb=1.0 / 1024.0,
                         D=1000.0, U=100.0),
    "natural_crossing": dict(time_scale_ms=1e-3, disk_unit_mb=1.0,
                             D=0.1, U=100.0),
}


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
        r3mg = [m.drl_r3_mg(cat, tr, C, seed=s) for s in range(seeds)]
        base = np.full(seeds, det.latency)
        row = dict(dataset=name, regime=regime, C=int(C), n_services=int(cat.N),
                   n_requests=int(len(tr)), seeds=seeds,
                   frac_M_lt_l=float(np.mean(cat.M < cat.l)),
                   det_latency=float(det.latency), det_cost=float(det.cost),
                   llrc_latency=float(ll.latency), llrc_cost=float(ll.cost),
                   det_vs_llrc_lat_pct=float(100 * (det.latency - ll.latency) / ll.latency),
                   det_vs_llrc_cost_pct=float(100 * (det.cost - ll.cost) / ll.cost),
                   r3_latency=float(np.mean([r.latency for r in r3])),
                   r3_cost=float(np.mean([r.cost for r in r3])),
                   mg_latency=float(np.mean([r.latency for r in mg])),
                   mg_cost=float(np.mean([r.cost for r in mg])),
                   r3mg_latency=float(np.mean([r.latency for r in r3mg])),
                   r3mg_cost=float(np.mean([r.cost for r in r3mg])),
                   mg_lat_pct=float(100 * (np.mean([r.latency for r in mg]) - det.latency) / det.latency),
                   mg_cost_pct=float(100 * (np.mean([r.cost for r in mg]) - det.cost) / det.cost),
                   r3mg_lat_pct=float(100 * (np.mean([r.latency for r in r3mg]) - det.latency) / det.latency),
                   r3mg_cost_pct=float(100 * (np.mean([r.cost for r in r3mg]) - det.cost) / det.cost),
                   r3_lat_pct=float(100 * (np.mean([r.latency for r in r3]) - det.latency) / det.latency),
                   r3_cost_pct=float(100 * (np.mean([r.cost for r in r3]) - det.cost) / det.cost),
                   mg_ci=paired_ci([r.latency for r in mg], base),
                   r3mg_ci=paired_ci([r.latency for r in r3mg], base),
                   mg_p=wilcoxon([r.latency for r in mg], base)[1],
                   r3mg_p=wilcoxon([r.latency for r in r3mg], base)[1],
                   violations=sum(len(r.extra.get("violations", {})) for r in r3mg))
        out.append(row)
        print(f"{name:22s} {regime:16s} C={C:3d} M<l={row['frac_M_lt_l']:.3f} "
              f"det={det.latency:.0f}/{det.cost:.0f} "
              f"mg={row['mg_lat_pct']:+.2f}%/{row['mg_cost_pct']:+.2f}% "
              f"r3mg={row['r3mg_lat_pct']:+.2f}%/{row['r3mg_cost_pct']:+.2f}%")
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

    # no-harm check: use natural_crossing rows only (the non-trivial regime)
    nat = [r for r in rows if r["regime"] == "natural_crossing"]
    no_harm = all(r["mg_lat_pct"] <= 2.0 and r["mg_cost_pct"] <= 2.0 for r in nat)
    out = dict(config=dict(seeds=SEEDS), rows=rows,
               sharcnet=dict(status=sh_status, reason=sh_reason),
               mg_no_harm_natural_crossing=bool(no_harm))
    save_json("phase3_real_traces", out)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    labels = [f"{r['dataset']}\n{r['regime']}" for r in rows]
    x = np.arange(len(rows))
    ax[0].bar(x - 0.2, [r["mg_lat_pct"] for r in rows], 0.4, label="MG")
    ax[0].bar(x + 0.2, [r["r3mg_lat_pct"] for r in rows], 0.4, label="R3+MG")
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].axhline(2, color="r", ls="--", lw=.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=6, rotation=30)
    ax[0].set_ylabel("latency change vs Online-DRL (%)")
    ax[0].legend(fontsize=7)
    ax[1].bar(x - 0.2, [r["mg_cost_pct"] for r in rows], 0.4, label="MG")
    ax[1].bar(x + 0.2, [r["r3mg_cost_pct"] for r in rows], 0.4, label="R3+MG")
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=6, rotation=30)
    ax[1].set_ylabel("cost change vs Online-DRL (%)")
    ax[1].legend(fontsize=7)
    save_fig(fig, "phase3_real_traces")

    lines = ["# Phase 3 -- real traces", "",
             f"Seeds: {SEEDS}.  SHARCNet: **{sh_status}** ({sh_reason}).", "",
             markdown_table(["dataset", "regime", "C", "M<l", "MG lat %",
                             "MG cost %", "R3+MG lat %", "R3+MG cost %",
                             "MG p", "R3+MG p"],
                            [[r["dataset"], r["regime"], r["C"],
                              f"{r['frac_M_lt_l']:.3f}",
                              f"{r['mg_lat_pct']:+.2f}",
                              f"{r['mg_cost_pct']:+.2f}",
                              f"{r['r3mg_lat_pct']:+.2f}",
                              f"{r['r3mg_cost_pct']:+.2f}",
                              f"{r['mg_p']:.3g}",
                              f"{r['r3mg_p']:.3g}"] for r in rows]),
             "", f"MG no harm (<=2%) on natural-crossing rows: **{no_harm}**"]
    with open("docs/phase3_real_traces.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote docs/phase3_real_traces.md")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main()