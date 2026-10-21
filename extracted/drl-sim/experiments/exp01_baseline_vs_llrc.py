"""Original paper: Online-DRL has slightly higher latency but much lower cost than LL-RC."""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, ll_rc
from sim.metrics import pct

a = parse(__doc__, C=15)
rows = []
for seed in range(a.seeds):
    cat, tr = synthetic_world(N=a.N, T=a.T, seed=seed)
    d, b = online_drl(cat, tr, a.C), ll_rc(cat, tr, a.C)
    rows.append(dict(seed=seed, lat_diff_pct=pct(d.latency, b.latency), cost_diff_pct=pct(d.cost, b.cost),
                     hits_drl=d.hits, hits_llrc=b.hits, dh_drl=d.delayed_hits))
    print(rows[-1])
lat, cost = ms([r["lat_diff_pct"] for r in rows]), ms([r["cost_diff_pct"] for r in rows])
print(f"Online-DRL vs LL-RC: latency {lat[0]:+.2f}% (sd {lat[1]:.2f}), cost {cost[0]:+.2f}% (sd {cost[1]:.2f})")
print("Paper (Google traces): latency +0.016..+7.41%, cost -77.57..-92.54%")
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].bar(range(len(rows)), [r["lat_diff_pct"] for r in rows]); ax[0].set_title("latency vs LL-RC (%)")
ax[1].bar(range(len(rows)), [r["cost_diff_pct"] for r in rows]); ax[1].set_title("cost vs LL-RC (%)")
save(a, "exp01_baseline_vs_llrc", dict(rows=rows, latency_pct=lat, cost_pct=cost), fig)
