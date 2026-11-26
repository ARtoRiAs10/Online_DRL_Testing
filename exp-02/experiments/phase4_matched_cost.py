"""Phase 4 -- matched-cost comparison (decides the R3 claim).

Sweeps the deterministic threshold ``theta`` and the R3 scale ``lam``, then
interpolates latency at 3-4 fixed cost levels.  The comparison is paired by
seed; a level counts as a win only if the bootstrap 95% CI of the latency gap
excludes zero and the gap is negative.
"""
from __future__ import annotations

import os

import numpy as np

from common import (DATA, bootstrap_ci, holm, markdown_table, paired_ci,
                    save_fig, save_json, wilcoxon)
import online_drl_r3 as m

SEEDS = 30
THETAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
LAMS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
LEVEL_FRACS = [0.50, 0.75, 1.00, 1.25]


def curve(det, r3, level):
    """Interpolate latency at ``level`` from (cost, latency) curves."""
    dc = np.array([r.cost for r in det])
    dl = np.array([r.latency for r in det])
    rc = np.array([r.cost for r in r3])
    rl = np.array([r.latency for r in r3])
    # sort by cost; np.interp needs increasing x
    di = np.argsort(dc); ri = np.argsort(rc)
    dc, dl = dc[di], dl[di]
    rc, rl = rc[ri], rl[ri]
    # keep only the overlapping range
    lo = max(dc.min(), rc.min())
    hi = min(dc.max(), rc.max())
    if level < lo or level > hi:
        return None
    return float(np.interp(level, dc, dl)), float(np.interp(level, rc, rl))


def run_family(name, cat, tr, C, seeds=SEEDS):
    det = []
    r3 = []
    for s in range(seeds):
        det.append([m.online_drl(cat, tr, C, theta=t) for t in THETAS])
        r3.append([m.drl_r3(cat, tr, C, lam=l, seed=s) for l in LAMS])
    # fixed levels from the mean deterministic theta=1 cost
    base_cost = float(np.mean([d[THETAS.index(1.0)].cost for d in det]))
    levels = [f * base_cost for f in LEVEL_FRACS]
    rows = []
    for li, level in enumerate(levels):
        gaps = []
        covered = 0
        for s in range(seeds):
            pair = curve(det[s], r3[s], level)
            if pair is None:
                gaps.append(np.nan)
            else:
                covered += 1
                gaps.append(pair[1] - pair[0])
        gaps = np.array(gaps, float)
        ok = gaps[np.isfinite(gaps)]
        if len(ok) == 0:
            rows.append(dict(family=name, level=level, frac=LEVEL_FRACS[li],
                             covered=0, gap_mean=float("nan"),
                             ci=[float("nan")] * 2, p=1.0, win=False))
            continue
        ci = bootstrap_ci(ok)
        p = wilcoxon(ok, np.zeros_like(ok))[1]
        rows.append(dict(family=name, level=float(level),
                         frac=LEVEL_FRACS[li], covered=int(covered),
                         gap_mean=float(ok.mean()),
                         ci=[float(ci[1]), float(ci[2])],
                         p=float(p),
                         win=bool(ok.mean() < 0 and ci[1] < 0 and ci[2] < 0)))
        print(f"{name:22s} level={level:10.1f} ({LEVEL_FRACS[li]:.2f}x) "
              f"covered={covered}/{seeds} gap={ok.mean():+.1f} "
              f"CI=[{ci[1]:+.1f},{ci[2]:+.1f}] p={p:.3g} win={rows[-1]['win']}")
    adj = holm([r["p"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = float(a)
    wins = sum(r["win"] and r["p_holm"] < 0.05 for r in rows)
    return dict(rows=rows, n_wins=int(wins), n_levels=len(rows),
                base_cost=base_cost,
                pass_=bool(wins >= 3 and wins >= len(rows) - 1))


def main():
    out = {}
    # synthetic
    cat, tr = m.synthetic_world(N=40, T=8000, zipf=1.2, seed=0)
    out["synthetic"] = run_family("synthetic", cat, tr, 12)
    # Google dataset2 (natural crossing)
    cat2, tr2 = m.load_trace_csv(os.path.join(DATA, "google_dataset2_30.csv"),
                                 id_col="JobID", time_col="Timestamp",
                                 cpu_col="CPU", ram_col="RAM", disk_col="Disk",
                                 time_scale_ms=1e-3, disk_unit_mb=1.0,
                                 D=0.1, U=100.0, max_rows=1000)
    out["google_dataset2_30"] = run_family("google_dataset2_30", cat2, tr2, 10)
    # Google 50k (natural crossing)
    cat3, tr3 = m.load_trace_csv(os.path.join(DATA, "google.csv"),
                                 id_col="JobID", time_col="Timestamp",
                                 cpu_col="CPU", ram_col="RAM", disk_col="Disk",
                                 time_scale_ms=1e-3, disk_unit_mb=1.0,
                                 D=0.1, U=100.0, max_rows=50000)
    out["google_50k"] = run_family("google_50k", cat3, tr3, 50)

    # overall: at least 3 of 4 levels on at least 2 of 3 datasets
    datasets_pass = [k for k, v in out.items() if v["pass_"]]
    overall = bool(len(datasets_pass) >= 2)
    out["overall"] = dict(datasets_pass=datasets_pass, pass_=overall)
    save_json("phase4_matched_cost", out)

    # Pareto plot (mean curves)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, (name, v) in zip(ax, out.items()):
        if name == "overall":
            continue
        # reconstruct mean curves from stored summary is not possible; re-run
        # cheap deterministic mean only is skipped.  Plot the level gaps.
        rows = v["rows"]
        a.bar(range(len(rows)), [r["gap_mean"] for r in rows])
        a.axhline(0, color="k", lw=.8)
        a.set_xticks(range(len(rows)))
        a.set_xticklabels([f"{r['frac']:.2f}x" for r in rows])
        a.set_title(name)
        a.set_ylabel("R3 latency - det latency")
    save_fig(fig, "phase4_matched_cost")

    lines = ["# Phase 4 -- matched cost", "",
             "Levels are fractions of the mean deterministic theta=1 cost.", ""]
    for name, v in out.items():
        if name == "overall":
            continue
        lines += [f"## {name}", "",
                  markdown_table(["level", "frac", "covered", "gap", "95% CI",
                                  "p", "p Holm", "win"],
                                 [[f"{r['level']:.1f}", f"{r['frac']:.2f}",
                                   f"{r['covered']}/{SEEDS}",
                                   f"{r['gap_mean']:+.1f}",
                                   f"[{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}]",
                                   f"{r['p']:.3g}", f"{r['p_holm']:.3g}",
                                   r["win"]] for r in v["rows"]]),
                  "", f"wins {v['n_wins']}/{v['n_levels']}; pass {v['pass_']}", ""]
    lines += [f"Overall pass (R3 at matched cost): **{overall}**; "
              f"datasets passing: {datasets_pass}"]
    with open("docs/phase4_matched_cost.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote docs/phase4_matched_cost.md")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main()