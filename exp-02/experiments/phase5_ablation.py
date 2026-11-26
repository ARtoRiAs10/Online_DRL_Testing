"""Phase 5 -- ablation and sensitivity.

Five variants plus the implementation knobs named in the plan:
size weight (sum vs max), victim choice (uniform vs weighted), threshold
redraw point (download/evict vs miss epoch vs request) and deterministic
tie-break (residual-deficit vs LRU).
"""
from __future__ import annotations

import numpy as np

from common import markdown_table, save_fig, save_json
import online_drl_r3 as m

SEEDS = 20


def main():
    cat, tr = m.synthetic_world(N=20, T=2000, seed=0)
    C = 10
    configs = [
        ("LL-RC", lambda s: m.ll_rc(cat, tr, C)),
        ("Online-DRL", lambda s: m.online_drl(cat, tr, C)),
        ("Online-DRL LRU", lambda s: m.online_drl(cat, tr, C, tie_break="lru")),
        ("R3", lambda s: m.drl_r3(cat, tr, C, seed=s)),
        ("MG", lambda s: m.drl_mg(cat, tr, C, seed=s)),
        ("R3+MG", lambda s: m.drl_r3_mg(cat, tr, C, seed=s)),
        ("R3+MG size=max", lambda s: m.drl_r3_mg(cat, tr, C, seed=s, size_mode="max")),
        ("R3+MG victim=weighted", lambda s: m.drl_r3_mg(cat, tr, C, seed=s, victim_mode="weighted")),
        ("R3+MG redraw=miss", lambda s: m.drl_r3_mg(cat, tr, C, seed=s, redraw_mode="miss_epoch")),
        ("R3+MG redraw=request", lambda s: m.drl_r3_mg(cat, tr, C, seed=s, redraw_mode="request")),
    ]
    rows = []
    for name, fn in configs:
        rs = [fn(s) for s in range(SEEDS)]
        lat = np.array([r.latency for r in rs])
        cost = np.array([r.cost for r in rs])
        row = dict(config=name, latency=float(lat.mean()),
                   latency_sd=float(lat.std(ddof=1)),
                   cost=float(cost.mean()), cost_sd=float(cost.std(ddof=1)),
                   hits=float(np.mean([r.hits for r in rs])),
                   downloads=float(np.mean([r.downloads for r in rs])),
                   evictions=float(np.mean([r.extra.get("evictions", 0) for r in rs])),
                   phase_clears=float(np.mean([r.extra.get("phase_clears", 0) for r in rs])),
                   us_per_req=float(np.mean([r.extra.get("us_per_req", 0) for r in rs])),
                   violations=sum(len(r.extra.get("violations", {})) for r in rs))
        rows.append(row)
        print(f"{name:28s} lat={row['latency']:.0f} cost={row['cost']:.0f} "
              f"hits={row['hits']:.0f} ev={row['evictions']:.1f} "
              f"phase={row['phase_clears']:.1f}")
    save_json("phase5_ablation", dict(config=dict(seeds=SEEDS), rows=rows))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    names = [r["config"] for r in rows]
    ax[0].barh(names, [r["latency"] for r in rows])
    ax[0].set_xlabel("latency")
    ax[1].barh(names, [r["cost"] for r in rows])
    ax[1].set_xlabel("cost")
    save_fig(fig, "phase5_ablation")

    lines = ["# Phase 5 -- ablation", "", f"Seeds: {SEEDS}", "",
             markdown_table(["config", "latency", "cost", "hits", "downloads",
                             "evictions", "phase clears", "us/req"],
                            [[r["config"], f"{r['latency']:.0f}",
                              f"{r['cost']:.0f}", f"{r['hits']:.0f}",
                              f"{r['downloads']:.0f}", f"{r['evictions']:.1f}",
                              f"{r['phase_clears']:.1f}",
                              f"{r['us_per_req']:.2f}"] for r in rows])]
    with open("docs/phase5_ablation.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote docs/phase5_ablation.md")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main()