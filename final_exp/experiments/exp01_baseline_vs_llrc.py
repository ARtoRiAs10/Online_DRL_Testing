"""Experiment 1 -- the paper's headline: Online-DRL vs LL-RC, and where the updated
Landlord-Repair randomized DRL (R1+R2+R3) lands on the same traces.

Paper (Google cluster traces): Online-DRL has a *slightly higher* latency than LL-RC
(+0.016 % .. +7.41 %) but a *much lower* download cost (-77.57 % .. -92.54 %).

This script reproduces the comparison on synthetic Zipf/Poisson traffic (the paper's
Google traces are used in exp16) and reports the updated randomized algorithm next to
the two deterministic baselines.
"""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, ll_rc, rand_drl
from sim.metrics import pct
from paper_claims import (PAPER_EXP01_LATENCY_PCT, PAPER_EXP01_COST_PCT, in_range)

a = parse(__doc__, C=15)
rows = []
for seed in range(a.seeds):
    cat, tr = synthetic_world(N=a.N, T=a.T, seed=seed)
    d = online_drl(cat, tr, a.C)
    b = ll_rc(cat, tr, a.C)
    r = rand_drl(cat, tr, a.C, np.random.default_rng(seed), K=4)
    rows.append(dict(
        seed=seed,
        lat_drl_pct=pct(d.latency, b.latency), cost_drl_pct=pct(d.cost, b.cost),
        lat_rand_pct=pct(r.latency, b.latency), cost_rand_pct=pct(r.cost, b.cost),
        hits_drl=d.hits, hits_llrc=b.hits, hits_rand=r.hits,
        dh_drl=d.delayed_hits, dh_llrc=b.delayed_hits, dh_rand=r.delayed_hits,
        downloads_rand=r.downloads,
    ))
    print(rows[-1])

agg = {}
for k in ("lat_drl_pct", "cost_drl_pct", "lat_rand_pct", "cost_rand_pct"):
    agg[k] = ms([r[k] for r in rows])
    print(f"{k:14s} mean={agg[k][0]:+8.2f} %  sd={agg[k][1]:6.2f} %")

paper_lat_ok = in_range(agg["lat_drl_pct"][0], PAPER_EXP01_LATENCY_PCT)
paper_cost_ok = in_range(agg["cost_drl_pct"][0], PAPER_EXP01_COST_PCT)
print(f"\nPaper (Google): latency {PAPER_EXP01_LATENCY_PCT[0]:+.3f} .. "
      f"{PAPER_EXP01_LATENCY_PCT[1]:+.2f} %, cost {PAPER_EXP01_COST_PCT[0]:.2f} .. "
      f"{PAPER_EXP01_COST_PCT[1]:.2f} %")
print(f"Online-DRL latency in paper range? {paper_lat_ok}; cost in paper range? {paper_cost_ok}")
print(f"Updated randomized DRL vs LL-RC: latency {agg['lat_rand_pct'][0]:+.2f} %, "
      f"cost {agg['cost_rand_pct'][0]:+.2f} %")

fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
x = np.arange(len(rows)); w = 0.27
ax[0].bar(x - w, [r["lat_drl_pct"] for r in rows], w, label="Online-DRL")
ax[0].bar(x, [r["lat_rand_pct"] for r in rows], w, label="Randomized DRL")
ax[0].axhline(0, color="k", lw=.8); ax[0].set_title("latency vs LL-RC (%)"); ax[0].legend(fontsize=7)
ax[1].bar(x - w, [r["cost_drl_pct"] for r in rows], w, label="Online-DRL")
ax[1].bar(x, [r["cost_rand_pct"] for r in rows], w, label="Randomized DRL")
ax[1].axhline(0, color="k", lw=.8); ax[1].set_title("cost vs LL-RC (%)"); ax[1].legend(fontsize=7)
ax[2].bar(x - w, [r["dh_drl"] for r in rows], w, label="Online-DRL")
ax[2].bar(x, [r["dh_rand"] for r in rows], w, label="Randomized DRL")
ax[2].set_title("delayed hits"); ax[2].legend(fontsize=7)
save(a, "exp01_baseline_vs_llrc",
     dict(rows=rows, agg=agg, paper=dict(latency_pct=PAPER_EXP01_LATENCY_PCT,
                                         cost_pct=PAPER_EXP01_COST_PCT,
                                         latency_in_range=paper_lat_ok,
                                         cost_in_range=paper_cost_ok)), fig)