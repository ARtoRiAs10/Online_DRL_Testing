"""Phase-0 diagnostic: which services does the Landlord repair evict?

Runs the Landlord and stateless repairs on the same synthetic world and reports the
popularity `p_n` of evicted services.  If the Landlord repair is popularity-blind, the
mean/max `p_n` of evicted services should be higher than for the stateless rule.
"""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, rand_drl

a = parse(__doc__, C=12, K=4)
N = 16 if a.quick else 20
T = 500 if a.quick else 2000
seeds = min(a.seeds, 5) if not a.quick else a.seeds
print(f"phase-0 eviction diagnostic: N={N} T={T} K={a.K} C={a.C} seeds={seeds}")

rows = []
for seed in range(seeds):
    cat, tr = synthetic_world(N=N, T=T, seed=seed, imbalance=2.0, cap_mult=4)
    det = online_drl(cat, tr, a.C)
    for mode in ("landlord", "stateless"):
        R = rand_drl(cat, tr, a.C, np.random.default_rng(seed), K=a.K,
                     log_steps=True, repair_mode=mode, gate_downloads=True)
        rows.append(dict(
            seed=seed, repair_mode=mode,
            latency=float(R.latency), cost=float(R.cost), hits=int(R.hits),
            util_max=float(R.util.mean(0).max()),
            repair_evictions=int(R.extra.get("repair_evictions", 0)),
            evicted_p_mean=float(R.extra.get("evicted_p_mean", 0.0)),
            evicted_p_max=float(R.extra.get("evicted_p_max", 0.0)),
            evicted_top1_fraction=float(R.extra.get("evicted_top1_fraction", 0.0)),
            mean_repair_evicted_fraction=float(R.extra.get("mean_repair_evicted_fraction", 0.0)),
            det_latency=float(det.latency), det_cost=float(det.cost),
        ))

summary = {}
for mode in ("landlord", "stateless"):
    fr = [r for r in rows if r["repair_mode"] == mode]
    summary[mode] = {
        "latency": ms([r["latency"] for r in fr]),
        "cost": ms([r["cost"] for r in fr]),
        "hits": ms([r["hits"] for r in fr]),
        "util_max": ms([r["util_max"] for r in fr]),
        "repair_evictions": ms([r["repair_evictions"] for r in fr]),
        "evicted_p_mean": ms([r["evicted_p_mean"] for r in fr]),
        "evicted_p_max": ms([r["evicted_p_max"] for r in fr]),
        "evicted_top1_fraction": ms([r["evicted_top1_fraction"] for r in fr]),
        "mean_repair_evicted_fraction": ms([r["mean_repair_evicted_fraction"] for r in fr]),
    }

out = dict(config=dict(N=N, T=T, K=a.K, C=a.C, seeds=seeds), rows=rows, summary=summary)
save(a, "exp28_landlord_evictions", out)

print("\n### eviction popularity")
for mode, s in summary.items():
    print(f"{mode:10s} lat={s['latency'][0]:.0f} cost={s['cost'][0]:.0f} "
          f"hits={s['hits'][0]:.0f} util={s['util_max'][0]:.3f} "
          f"evicted_p_mean={s['evicted_p_mean'][0]:.3f} "
          f"evicted_p_max={s['evicted_p_max'][0]:.3f} "
          f"top1_frac={s['evicted_top1_fraction'][0]:.3f}")