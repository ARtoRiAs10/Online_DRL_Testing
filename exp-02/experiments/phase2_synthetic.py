"""Phase 2 -- adversarial and synthetic workloads.

* Large Sequence (Theorem 1): ratio to OPT_t for k in {6, 12, 24, 48}.
* Cyclic C+1 adversaries: classic scan, burst scan and the deterministic
  eviction trace.
* Zipf/Poisson synthetic traffic: no-regression check.

Randomized variants use 30 seeds.  Paired Wilcoxon tests and Holm correction
are reported for the cyclic family.
"""
from __future__ import annotations

import numpy as np

from common import (bootstrap_ci, describe, holm, markdown_table, paired_ci,
                    save_fig, save_json, wilcoxon)
import online_drl_r3 as m

SEEDS = 30


def run_variant(fn, cat, tr, C, seeds=SEEDS):
    det = fn(cat, tr, C)
    rows = []
    for s in range(seeds):
        r = fn(cat, tr, C, seed=s) if fn is not m.online_drl else det
        rows.append(r)
    return det, rows


def large_sequence():
    rows = []
    ks = [6, 12, 24, 48]
    for k in ks:
        for n_mult in (1, 4):
            n_reps = k * n_mult
            cat, tr, opt = m.large_sequence(k, n_reps)
            det = m.online_drl(cat, tr, k)
            r3 = [m.drl_r3(cat, tr, k, seed=s) for s in range(SEEDS)]
            mg = [m.drl_mg(cat, tr, k, seed=s) for s in range(SEEDS)]
            r3mg = [m.drl_r3_mg(cat, tr, k, seed=s) for s in range(SEEDS)]
            row = dict(k=k, n_mult=n_mult, n_requests=len(tr), opt=float(opt),
                       det_ratio=float(det.latency / opt),
                       det_latency=float(det.latency),
                       r3_ratio=float(np.mean([r.latency / opt for r in r3])),
                       mg_ratio=float(np.mean([r.latency / opt for r in mg])),
                       r3mg_ratio=float(np.mean([r.latency / opt for r in r3mg])),
                       r3mg_ratio_ci=bootstrap_ci([r.latency / opt for r in r3mg]),
                       r3_latency=float(np.mean([r.latency for r in r3])),
                       mg_latency=float(np.mean([r.latency for r in mg])),
                       r3mg_latency=float(np.mean([r.latency for r in r3mg])),
                       det_cost=float(det.cost),
                       r3_cost=float(np.mean([r.cost for r in r3])),
                       mg_cost=float(np.mean([r.cost for r in mg])),
                       r3mg_cost=float(np.mean([r.cost for r in r3mg])),
                       violations=sum(len(r.extra.get("violations", {})) for r in r3mg))
            rows.append(row)
            print(f"LS k={k:2d} n={n_mult}k det={row['det_ratio']:.3f} "
                  f"r3={row['r3_ratio']:.3f} mg={row['mg_ratio']:.3f} "
                  f"r3mg={row['r3mg_ratio']:.3f}")
    # slope of ratio vs k for n_mult=1
    sub = [r for r in rows if r["n_mult"] == 1]
    ks_arr = np.array([r["k"] for r in sub], float)
    det_slope = np.polyfit(ks_arr, [r["det_ratio"] for r in sub], 1)[0]
    r3mg_slope = np.polyfit(ks_arr, [r["r3mg_ratio"] for r in sub], 1)[0]
    mg_slope = np.polyfit(ks_arr, [r["mg_ratio"] for r in sub], 1)[0]
    return dict(rows=rows, det_slope=float(det_slope),
                r3mg_slope=float(r3mg_slope), mg_slope=float(mg_slope),
                lower_every_k=bool(all(r["r3mg_ratio"] < r["det_ratio"] for r in sub)),
                slope_flatter=bool(r3mg_slope < det_slope))


def cyclic_family():
    rows = []
    configs = []
    # classic cyclic scan
    for k in (8, 16, 32):
        cat, tr = m.cyclic_scan(k, 10)
        configs.append((f"cyclic_scan_k{k}", cat, tr, k))
    # burst scan
    for k in (8, 16):
        for burst in (2, 4):
            cat, tr = m.cyclic_burst_scan(k, 10, burst=burst)
            configs.append((f"cyclic_burst_k{k}_b{burst}", cat, tr, k))
    # deterministic eviction trace
    for s in range(5):
        cat0, init = m.synthetic_world(N=20, T=500, seed=s, imbalance=2.0,
                                       cap_mult=4)
        tr, base, ev = m.deterministic_eviction_trace(cat0, init, C=8)
        configs.append((f"det_evict_s{s}", cat0, tr, 8))

    pvals = []
    for name, cat, tr, C in configs:
        det = m.online_drl(cat, tr, C)
        r3 = [m.drl_r3(cat, tr, C, seed=s) for s in range(SEEDS)]
        mg = [m.drl_mg(cat, tr, C, seed=s) for s in range(SEEDS)]
        r3mg = [m.drl_r3_mg(cat, tr, C, seed=s) for s in range(SEEDS)]
        dlat = np.array([r.latency for r in mg])
        dlat_r3mg = np.array([r.latency for r in r3mg])
        base = np.full(SEEDS, det.latency)
        red_mg = 100.0 * (det.latency - dlat.mean()) / det.latency
        red_r3mg = 100.0 * (det.latency - dlat_r3mg.mean()) / det.latency
        stat, p = wilcoxon(dlat, base)
        stat2, p2 = wilcoxon(dlat_r3mg, base)
        row = dict(config=name, n_requests=len(tr), C=C,
                   det_latency=float(det.latency),
                   r3_latency=float(np.mean([r.latency for r in r3])),
                   mg_latency=float(dlat.mean()),
                   r3mg_latency=float(dlat_r3mg.mean()),
                   mg_reduction_pct=float(red_mg),
                   r3mg_reduction_pct=float(red_r3mg),
                   mg_p=float(p), r3mg_p=float(p2),
                   mg_ci=paired_ci(dlat, base),
                   r3mg_ci=paired_ci(dlat_r3mg, base),
                   det_cost=float(det.cost),
                   mg_cost=float(np.mean([r.cost for r in mg])),
                   r3mg_cost=float(np.mean([r.cost for r in r3mg])),
                   phase_clears=float(np.mean([r.extra.get("phase_clears", 0) for r in mg])),
                   violations=sum(len(r.extra.get("violations", {})) for r in r3mg))
        rows.append(row)
        pvals.extend([p, p2])
        print(f"{name:28s} det={det.latency:.0f} mg={dlat.mean():.0f} "
              f"r3mg={dlat_r3mg.mean():.0f} red_mg={red_mg:+.1f}% "
              f"p={p:.3g}")
    adj = holm(pvals)
    for i, row in enumerate(rows):
        row["mg_p_holm"] = float(adj[2 * i])
        row["r3mg_p_holm"] = float(adj[2 * i + 1])
    return dict(rows=rows,
                mg_majority=bool(np.mean([r["mg_reduction_pct"] > 0 for r in rows]) > 0.5),
                mg_20pct=bool(np.mean([r["mg_reduction_pct"] >= 20.0 for r in rows]) > 0.5),
                mg_sig=bool(np.mean([r["mg_p_holm"] < 0.05 for r in rows]) > 0.5))


def zipf_family():
    rows = []
    for rho in (1.2, 1.4, 1.6):
        for C in (1, 3, 5, 9):
            cat, tr = m.synthetic_world(N=40, T=8000, zipf=rho, seed=0)
            det = m.online_drl(cat, tr, C)
            r3 = [m.drl_r3(cat, tr, C, seed=s) for s in range(SEEDS)]
            mg = [m.drl_mg(cat, tr, C, seed=s) for s in range(SEEDS)]
            r3mg = [m.drl_r3_mg(cat, tr, C, seed=s) for s in range(SEEDS)]
            row = dict(rho=rho, C=C,
                       det_latency=float(det.latency), det_cost=float(det.cost),
                       r3_latency=float(np.mean([r.latency for r in r3])),
                       r3_cost=float(np.mean([r.cost for r in r3])),
                       mg_latency=float(np.mean([r.latency for r in mg])),
                       mg_cost=float(np.mean([r.cost for r in mg])),
                       r3mg_latency=float(np.mean([r.latency for r in r3mg])),
                       r3mg_cost=float(np.mean([r.cost for r in r3mg])),
                       mg_lat_pct=float(100 * (np.mean([r.latency for r in mg]) - det.latency) / det.latency),
                       mg_cost_pct=float(100 * (np.mean([r.cost for r in mg]) - det.cost) / det.cost) if det.cost else float("nan"),
                       r3mg_lat_pct=float(100 * (np.mean([r.latency for r in r3mg]) - det.latency) / det.latency),
                       r3mg_cost_pct=float(100 * (np.mean([r.cost for r in r3mg]) - det.cost) / det.cost) if det.cost else float("nan"),
                       violations=sum(len(r.extra.get("violations", {})) for r in r3mg))
            rows.append(row)
            print(f"zipf rho={rho} C={C} det={det.latency:.0f} "
                  f"mg={row['mg_latency']:.0f} ({row['mg_lat_pct']:+.1f}%) "
                  f"r3mg={row['r3mg_latency']:.0f}")
    return dict(rows=rows,
                mg_no_harm=bool(all(r["mg_lat_pct"] <= 2.0 and r["mg_cost_pct"] <= 2.0
                                    for r in rows)))


def main():
    ls = large_sequence()
    cy = cyclic_family()
    zp = zipf_family()
    out = dict(config=dict(seeds=SEEDS), large_sequence=ls,
               cyclic=cy, zipf=zp)
    save_json("phase2_synthetic", out)

    # figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sub = [r for r in ls["rows"] if r["n_mult"] == 1]
    ks = [r["k"] for r in sub]
    ax[0].plot(ks, [r["det_ratio"] for r in sub], "o-", label="Online-DRL")
    ax[0].plot(ks, [r["r3_ratio"] for r in sub], "s--", label="R3")
    ax[0].plot(ks, [r["mg_ratio"] for r in sub], "^--", label="MG")
    ax[0].plot(ks, [r["r3mg_ratio"] for r in sub], "d-", label="R3+MG")
    ax[0].set_xlabel("k"); ax[0].set_ylabel("latency / OPT_t")
    ax[0].set_title("Large Sequence"); ax[0].legend(fontsize=7)
    names = [r["config"] for r in cy["rows"]]
    ax[1].barh(names, [r["mg_reduction_pct"] for r in cy["rows"]])
    ax[1].axvline(0, color="k", lw=.8)
    ax[1].axvline(20, color="r", ls="--", lw=.8)
    ax[1].set_xlabel("MG latency reduction vs Online-DRL (%)")
    ax[1].set_title("Cyclic adversaries")
    save_fig(fig, "phase2_synthetic")

    lines = ["# Phase 2 -- synthetic and adversarial", "",
             f"Seeds: {SEEDS}", "",
             "## Large Sequence", "",
             markdown_table(["k", "n", "det/OPT", "R3/OPT", "MG/OPT", "R3+MG/OPT"],
                            [[r["k"], f"{r['n_mult']}k", f"{r['det_ratio']:.3f}",
                              f"{r['r3_ratio']:.3f}", f"{r['mg_ratio']:.3f}",
                              f"{r['r3mg_ratio']:.3f}"] for r in ls["rows"]]),
             "", f"det slope {ls['det_slope']:.4f}; R3+MG slope "
             f"{ls['r3mg_slope']:.4f}; lower every k: {ls['lower_every_k']}; "
             f"flatter: {ls['slope_flatter']}", "",
             "## Cyclic adversaries", "",
             markdown_table(["config", "det", "MG", "R3+MG", "MG reduction %",
                             "MG p (Holm)", "R3+MG p (Holm)"],
                            [[r["config"], f"{r['det_latency']:.0f}",
                              f"{r['mg_latency']:.0f}", f"{r['r3mg_latency']:.0f}",
                              f"{r['mg_reduction_pct']:+.1f}",
                              f"{r['mg_p_holm']:.3g}",
                              f"{r['r3mg_p_holm']:.3g}"] for r in cy["rows"]]),
             "", f"MG >=20% on majority: {cy['mg_20pct']}; "
             f"MG significant on majority: {cy['mg_sig']}", "",
             "## Zipf no-regression", "",
             markdown_table(["rho", "C", "MG lat %", "MG cost %",
                             "R3+MG lat %", "R3+MG cost %"],
                            [[r["rho"], r["C"], f"{r['mg_lat_pct']:+.2f}",
                              f"{r['mg_cost_pct']:+.2f}",
                              f"{r['r3mg_lat_pct']:+.2f}",
                              f"{r['r3mg_cost_pct']:+.2f}"] for r in zp["rows"]]),
             "", f"MG no harm (<=2%): {zp['mg_no_harm']}"]
    with open("docs/phase2_synthetic.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote docs/phase2_synthetic.md")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main()