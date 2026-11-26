"""Phase 4 -- matched-cost comparison (decides the R3 / Bayes-R3 claim).

Sweeps the deterministic threshold ``theta``, R3's scale ``lam`` and the
Bayes-R3 scale ``lam``, then interpolates latency at four fixed cost levels.
The comparison is paired by seed; a level counts as a win only if the
bootstrap 95% CI of the latency gap excludes zero and the gap is negative.

The acceptance claim #4 is evaluated for **Bayes-R3** (the user's option #2);
R3 is kept as the reference.
"""
from __future__ import annotations

import os

import numpy as np

from common import (DATA, bootstrap_ci, holm, markdown_table, save_fig,
                    save_json, wilcoxon)
import online_drl_r3 as m

SEEDS = 30
THETAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
LAMS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
LEVEL_FRACS = [0.50, 0.75, 1.00, 1.25]
BAYES_C = 0.1


def interp_pair(det, other, level):
    dc = np.array([r.cost for r in det])
    dl = np.array([r.latency for r in det])
    oc = np.array([r.cost for r in other])
    ol = np.array([r.latency for r in other])
    di = np.argsort(dc); oi = np.argsort(oc)
    dc, dl = dc[di], dl[di]
    oc, ol = oc[oi], ol[oi]
    lo = max(dc.min(), oc.min())
    hi = min(dc.max(), oc.max())
    if level < lo or level > hi:
        return None
    return float(np.interp(level, dc, dl)), float(np.interp(level, oc, ol))


def eval_curve(name, det, other, base_cost, seeds):
    levels = [f * base_cost for f in LEVEL_FRACS]
    rows = []
    for li, level in enumerate(levels):
        gaps = []
        covered = 0
        for s in range(seeds):
            pair = interp_pair(det[s], other[s], level)
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
        print(f"{name:24s} level={level:10.1f} ({LEVEL_FRACS[li]:.2f}x) "
              f"covered={covered}/{seeds} gap={ok.mean():+.1f} "
              f"CI=[{ci[1]:+.1f},{ci[2]:+.1f}] p={p:.3g} win={rows[-1]['win']}")
    adj = holm([r["p"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = float(a)
    wins = sum(r["win"] and r["p_holm"] < 0.05 for r in rows)
    return dict(rows=rows, n_wins=int(wins), n_levels=len(rows),
                pass_=bool(wins >= 3 and wins >= len(rows) - 1))


def run_family(name, cat, tr, C, seeds=SEEDS):
    det, r3, bayes = [], [], []
    for s in range(seeds):
        det.append([m.online_drl(cat, tr, C, theta=t) for t in THETAS])
        r3.append([m.drl_r3(cat, tr, C, lam=l, seed=s) for l in LAMS])
        bayes.append([m.drl_bayes_r3(cat, tr, C, lam=l, seed=s, bayes_c=BAYES_C)
                      for l in LAMS])
    base_cost = float(np.mean([d[THETAS.index(1.0)].cost for d in det]))
    return dict(base_cost=base_cost,
                r3=eval_curve("R3", det, r3, base_cost, seeds),
                bayes=eval_curve("Bayes-R3", det, bayes, base_cost, seeds))


def main():
    out = {}
    cat, tr = m.synthetic_world(N=40, T=8000, zipf=1.2, seed=0)
    out["synthetic"] = run_family("synthetic", cat, tr, 12)

    cat2, tr2 = m.load_trace_csv(os.path.join(DATA, "google_dataset2_30.csv"),
                                 id_col="JobID", time_col="Timestamp",
                                 cpu_col="CPU", ram_col="RAM", disk_col="Disk",
                                 time_scale_ms=1e-3, disk_unit_mb=1.0,
                                 D=1e-4, U=100.0, max_rows=1000)
    out["google_dataset2_30"] = run_family("google_dataset2_30", cat2, tr2, 10)

    cat3, tr3 = m.load_trace_csv(os.path.join(DATA, "google.csv"),
                                 id_col="JobID", time_col="Timestamp",
                                 cpu_col="CPU", ram_col="RAM", disk_col="Disk",
                                 time_scale_ms=1e-3, disk_unit_mb=1.0,
                                 D=1e-4, U=100.0, max_rows=50000)
    out["google_50k"] = run_family("google_50k", cat3, tr3, 5)

    bayes_pass = [k for k, v in out.items() if v["bayes"]["pass_"]]
    r3_pass = [k for k, v in out.items() if v["r3"]["pass_"]]
    overall = dict(bayes_pass=bool(len(bayes_pass) >= 2),
                   r3_pass=bool(len(r3_pass) >= 2),
                   bayes_datasets=bayes_pass, r3_datasets=r3_pass)
    out["overall"] = overall
    save_json("phase4_matched_cost", out)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, (name, v) in zip(ax, out.items()):
        if name == "overall":
            continue
        for key, label in (("r3", "R3"), ("bayes", "Bayes-R3")):
            rows = v[key]["rows"]
            a.bar(np.arange(len(rows)) + (-0.2 if key == "r3" else 0.2),
                  [r["gap_mean"] for r in rows], 0.4, label=label)
        a.axhline(0, color="k", lw=.8)
        a.set_xticks(range(len(v["r3"]["rows"])))
        a.set_xticklabels([f"{r['frac']:.2f}x" for r in v["r3"]["rows"]])
        a.set_title(name)
        a.set_ylabel("R3/Bayes latency - det latency")
        a.legend(fontsize=7)
    save_fig(fig, "phase4_matched_cost")

    lines = ["# Phase 4 -- matched cost", "",
             "Levels are fractions of the mean deterministic theta=1 cost.  "
             f"Bayes-R3 uses c={BAYES_C}.", ""]
    for name, v in out.items():
        if name == "overall":
            continue
        for key, label in (("r3", "R3"), ("bayes", "Bayes-R3")):
            lines += [f"## {name} -- {label}", "",
                      markdown_table(
                          ["level", "frac", "covered", "gap", "95% CI",
                           "p", "p Holm", "win"],
                          [[f"{r['level']:.1f}", f"{r['frac']:.2f}",
                            f"{r['covered']}/{SEEDS}",
                            f"{r['gap_mean']:+.1f}",
                            f"[{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}]",
                            f"{r['p']:.3g}", f"{r['p_holm']:.3g}",
                            r["win"]] for r in v[key]["rows"]]),
                      "", f"wins {v[key]['n_wins']}/{v[key]['n_levels']}; "
                      f"pass {v[key]['pass_']}", ""]
    lines += [f"Overall R3 matched-cost pass: **{overall['r3_pass']}** "
              f"({overall['r3_datasets']})",
              f"Overall Bayes-R3 matched-cost pass: **{overall['bayes_pass']}** "
              f"({overall['bayes_datasets']})"]
    with open("docs/phase4_matched_cost.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote docs/phase4_matched_cost.md")
    print(overall)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main()