"""Phase 6 -- statistics and overhead.

* 30 seeds per randomized configuration (100 for the small synthetic world
  when ``--many`` is passed).
* mean, bootstrap 95% CI, worst seed, 95th percentile.
* paired Wilcoxon signed-rank test against deterministic Online-DRL, Holm
  correction across configurations.
* overhead in microseconds per request, measured on identical hardware.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import (bootstrap_ci, describe, holm, markdown_table, save_fig,
                    save_json, wilcoxon)
import online_drl_r3 as m


def stats_rows(cat, tr, C, seeds):
    det = m.online_drl(cat, tr, C)
    base = np.full(seeds, det.latency)
    out = []
    pvals = []
    for name, fn in [
        ("R3", lambda s: m.drl_r3(cat, tr, C, seed=s)),
        ("MG", lambda s: m.drl_mg(cat, tr, C, seed=s)),
        ("R3+MG", lambda s: m.drl_r3_mg(cat, tr, C, seed=s)),
    ]:
        rs = [fn(s) for s in range(seeds)]
        lat = np.array([r.latency for r in rs])
        cost = np.array([r.cost for r in rs])
        d = describe(lat)
        p = wilcoxon(lat, base)[1]
        pvals.append(p)
        out.append(dict(config=name, latency=d, cost=describe(cost),
                        p=float(p),
                        win_rate=float(np.mean(lat < det.latency))))
    adj = holm(pvals)
    for r, a in zip(out, adj):
        r["p_holm"] = float(a)
    return det, out


def overhead(cat, tr, C, repeats=15):
    """Median us/request over repeats; deterministic is the reference."""
    configs = [
        ("Online-DRL", lambda: m.online_drl(cat, tr, C)),
        ("R3", lambda: m.drl_r3(cat, tr, C, seed=0)),
        ("MG", lambda: m.drl_mg(cat, tr, C, seed=0)),
        ("R3+MG", lambda: m.drl_r3_mg(cat, tr, C, seed=0)),
    ]
    rows = []
    for name, fn in configs:
        vals = []
        for _ in range(repeats):
            r = fn()
            vals.append(1e6 * r.time_s / max(1, r.n_req))
        vals = np.array(vals)
        rows.append(dict(config=name, median=float(np.median(vals)),
                         mean=float(vals.mean()), sd=float(vals.std(ddof=1)),
                         min=float(vals.min()), max=float(vals.max())))
    base = rows[0]["median"]
    for r in rows:
        r["ratio_to_det"] = r["median"] / base if base else float("nan")
        r["within_2x"] = bool(r["ratio_to_det"] <= 2.0)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--many", action="store_true", help="100 seeds")
    a = ap.parse_args()
    seeds = 100 if a.many else 30

    cat, tr = m.synthetic_world(N=20, T=2000, seed=0)
    det, syn = stats_rows(cat, tr, 10, seeds)

    cat2, tr2 = m.cyclic_scan(16, 10)
    det2, cyc = stats_rows(cat2, tr2, 16, seeds)

    oh = overhead(cat, tr, 10)
    out = dict(config=dict(seeds=seeds), synthetic=dict(deterministic=det.summary(), rows=syn),
               cyclic=dict(deterministic=det2.summary(), rows=cyc),
               overhead=oh)
    save_json("phase6_stats_overhead", out)

    lines = ["# Phase 6 -- statistics and overhead", "", f"Seeds: {seeds}", "",
             "## Synthetic Zipf (N=20, T=2000, C=10)", "",
             markdown_table(["config", "mean", "95% CI", "worst", "p95",
                             "p", "p Holm", "win rate"],
                            [[r["config"], f"{r['latency']['mean']:.0f}",
                              f"[{r['latency']['ci_low']:.0f}, {r['latency']['ci_high']:.0f}]",
                              f"{r['latency']['worst']:.0f}",
                              f"{r['latency']['p95']:.0f}",
                              f"{r['p']:.3g}", f"{r['p_holm']:.3g}",
                              f"{r['win_rate']:.2f}"] for r in syn]),
             "", "## Cyclic scan (k=16)", "",
             markdown_table(["config", "mean", "95% CI", "worst", "p95",
                             "p", "p Holm"],
                            [[r["config"], f"{r['latency']['mean']:.0f}",
                              f"[{r['latency']['ci_low']:.0f}, {r['latency']['ci_high']:.0f}]",
                              f"{r['latency']['worst']:.0f}",
                              f"{r['latency']['p95']:.0f}",
                              f"{r['p']:.3g}", f"{r['p_holm']:.3g}"] for r in cyc]),
             "", "## Overhead (us/request, median of 15)", "",
             markdown_table(["config", "median", "ratio to det", "within 2x"],
                            [[r["config"], f"{r['median']:.3f}",
                              f"{r['ratio_to_det']:.2f}", r["within_2x"]]
                             for r in oh])]
    with open("docs/phase6_stats_overhead.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote docs/phase6_stats_overhead.md")
    for r in oh:
        print(r)


if __name__ == "__main__":
    main()