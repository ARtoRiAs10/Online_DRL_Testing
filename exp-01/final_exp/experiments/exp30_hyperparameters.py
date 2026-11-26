"""Phase-2: one-at-a-time hyperparameter sweep for randomized stateless DRL.

Sweeps:
  * eta multiplier;
  * q-EMA alpha;
  * number of paths K;
  * projection tolerance;
  * projection skip tolerance.

The final values are chosen from this grid, not tuned on the test seeds.
"""
from common import *
from sim.trace import synthetic_world
from sim.algos import rand_drl
from sim.rand_drl import FracPolicy

a = parse(__doc__, C=12, K=4)
N = 16 if a.quick else 20
T = 500 if a.quick else 2000
seeds = min(a.seeds, 3) if a.quick else 5
cat, tr = synthetic_world(N=N, T=T, seed=0, imbalance=2.0, cap_mult=4)
pol0 = FracPolicy(cat, a.C, np.random.default_rng(0), K=a.K)
pol0.set_horizon(T)
base_eta = float(pol0.eta)
print(f"phase-2 hyperparameters: N={N} T={T} base_eta={base_eta:.6g} seeds={seeds}")

BASE = dict(repair_mode="stateless", gate_downloads=True)
SWEEPS = {
    "eta_mult": [1, 4, 16, 64],
    "alpha": [0.05, 0.1, 0.2, 0.4],
    "K": [2, 4, 8],
    "proj_tol": [1e-6, 1e-7, 1e-8],
    "proj_skip_tol": [0.0, 1e-3],
}

rows = []
for group, values in SWEEPS.items():
    for value in values:
        kw = dict(BASE)
        if group == "eta_mult":
            kw["eta"] = base_eta * value
        elif group == "alpha":
            kw["q_ema_alpha"] = value
            kw["grad_mode"] = "expectation"
        elif group == "K":
            kw["K"] = value
        elif group == "proj_tol":
            kw["proj_tol"] = value
        elif group == "proj_skip_tol":
            kw["proj_skip_tol"] = value
        acc = []
        for s in range(seeds):
            R = rand_drl(cat, tr, a.C, np.random.default_rng(s), log_steps=False, **kw)
            acc.append(dict(latency=float(R.latency), cost=float(R.cost),
                            hits=int(R.hits), util_max=float(R.util.mean(0).max())))
        row = dict(group=group, value=value,
                   latency=ms([r["latency"] for r in acc]),
                   cost=ms([r["cost"] for r in acc]),
                   hits=ms([r["hits"] for r in acc]),
                   util_max=ms([r["util_max"] for r in acc]))
        rows.append(row)
        print(f"{group:16s} {str(value):>8s}  "
              f"lat={row['latency'][0]:9.0f} cost={row['cost'][0]:8.0f} "
              f"hits={row['hits'][0]:6.0f} util={row['util_max'][0]:.3f}")

out = dict(config=dict(N=N, T=T, C=a.C, seeds=seeds, base_eta=base_eta,
                       base=BASE, sweeps=SWEEPS), rows=rows)
save(a, "exp30_hyperparameters", out)

print("\n### best per group (lowest latency)")
for group in SWEEPS:
    fr = [r for r in rows if r["group"] == group]
    best = min(fr, key=lambda r: r["latency"][0])
    print(f"{group:16s} value={str(best['value']):>8s} "
          f"lat={best['latency'][0]:.0f} cost={best['cost'][0]:.0f} "
          f"hits={best['hits'][0]:.0f}")