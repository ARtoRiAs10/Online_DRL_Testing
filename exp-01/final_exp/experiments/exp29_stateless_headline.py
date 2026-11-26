"""Phase-2: rerun the headline experiments with the stateless repair.

Covers:
  * exp01 — Online-DRL vs LL-RC, with randomized stateless vs LL-RC;
  * exp04 — latency vs capacity C and resource limits;
  * exp12 — multi-resource balanced/imbalanced workloads;
  * exp17 — Zipf rho / request-rate sensitivity.

All randomized runs use ``repair_mode="stateless"``.  Results are written to
``results/exp29_stateless_headline.json``.
"""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, ll_rc, rand_landlord, rand_drl
from sim.metrics import pct, resource_metrics

a = parse(__doc__, C=15, K=4)
N = 20 if a.quick else 40
T = 2000 if a.quick else 8000
seeds = a.seeds
seeds12 = min(a.seeds, 3) if a.quick else min(a.seeds, 5)
print(f"phase-2 stateless rerun: N={N} T={T} seeds={seeds} seeds12={seeds12}")

out = {"config": dict(N=N, T=T, C=a.C, K=a.K, seeds=seeds, repair_mode="stateless")}

# ---------------------------------------------------------------- exp01
rows = []
for seed in range(seeds):
    cat, tr = synthetic_world(N=N, T=T, seed=seed)
    d = online_drl(cat, tr, a.C)
    b = ll_rc(cat, tr, a.C)
    r = rand_drl(cat, tr, a.C, np.random.default_rng(seed), K=a.K,
                 repair_mode="stateless", gate_downloads=True)
    rows.append(dict(
        seed=seed,
        lat_drl_pct=pct(d.latency, b.latency), cost_drl_pct=pct(d.cost, b.cost),
        lat_rand_pct=pct(r.latency, b.latency), cost_rand_pct=pct(r.cost, b.cost),
        hits_drl=d.hits, hits_llrc=b.hits, hits_rand=r.hits,
        delayed_drl=d.delayed_hits, delayed_rand=r.delayed_hits,
        downloads_rand=r.downloads, util_max=float(r.util.mean(0).max()),
    ))
agg = {k: ms([r[k] for r in rows]) for k in
       ("lat_drl_pct", "cost_drl_pct", "lat_rand_pct", "cost_rand_pct")}
out["exp01"] = dict(rows=rows, agg=agg)
print("\n[exp01] vs LL-RC")
for k, v in agg.items():
    print(f"  {k:16s} {v[0]:+8.2f} %  sd {v[1]:6.2f}")

# ---------------------------------------------------------------- exp04
cat, tr = synthetic_world(N=N, T=T, seed=0)
outC, outM = [], []
for C in [1, 2, 4, 8, 16, 32]:
    d = online_drl(cat, tr, C)
    b = ll_rc(cat, tr, C)
    r = rand_drl(cat, tr, C, np.random.default_rng(0), K=a.K,
                 repair_mode="stateless", gate_downloads=True)
    outC.append(dict(C=C, lat=d.latency, cost=d.cost, lat_llrc=b.latency,
                     lat_rand=r.latency, cost_rand=r.cost, hits=r.hits,
                     util_max=float(r.util.mean(0).max())))
for m in [2, 3, 4, 6, 8, 16]:
    cat2, tr2 = synthetic_world(N=N, T=T, seed=0, cap_mult=m)
    d = online_drl(cat2, tr2, 10 ** 6)
    b = ll_rc(cat2, tr2, 10 ** 6)
    r = rand_drl(cat2, tr2, 10 ** 6, np.random.default_rng(0), K=a.K,
                 repair_mode="stateless", gate_downloads=True)
    outM.append(dict(cap_mult=m, lat=d.latency, cost=d.cost, lat_llrc=b.latency,
                     lat_rand=r.latency, cost_rand=r.cost, hits=r.hits,
                     util_max=float(r.util.mean(0).max())))
out["exp04"] = dict(C=outC, resources=outM)
print("\n[exp04] capacity sweep (stateless)")
for r in outC:
    print(f"  C={r['C']:2d} det={r['lat']:.0f} rand={r['lat_rand']:.0f} "
          f"util={r['util_max']:.3f}")

# ---------------------------------------------------------------- exp12
algos = {
    "online_drl": lambda cat, tr, C, s: online_drl(cat, tr, C),
    "ll_rc": lambda cat, tr, C, s: ll_rc(cat, tr, C),
    "r3_only": lambda cat, tr, C, s: rand_landlord(cat, tr, C, np.random.default_rng(s)),
    "rand_stateless": lambda cat, tr, C, s: rand_drl(
        cat, tr, C, np.random.default_rng(s), K=a.K,
        repair_mode="stateless", gate_downloads=True),
    "rand_landlord": lambda cat, tr, C, s: rand_drl(
        cat, tr, C, np.random.default_rng(s), K=a.K,
        repair_mode="landlord", gate_downloads=True),
}
out12 = {}
for imbalance, tag in ((1.0, "balanced"), (4.0, "imbalanced")):
    acc = {}
    for seed in range(seeds12):
        cat, tr = synthetic_world(N=N, T=T, seed=seed, imbalance=imbalance, cap_mult=5)
        for name, fn in algos.items():
            R = fn(cat, tr, a.C, seed)
            m = resource_metrics(R.util, R.latency)
            acc.setdefault(name, []).append(dict(
                latency=R.latency, cost=R.cost, downloads=R.downloads,
                hits=R.hits, delayed_hits=R.delayed_hits,
                Umax=m["Umax_mean"], imbalance=m["imbalance_mean"],
                stranded=m["stranded_mean"], viol=m["violation_rate"],
                util_max=float(R.util.mean(0).max())))
    out12[tag] = {}
    for name, rows in acc.items():
        out12[tag][name] = {k: ms([r[k] for r in rows]) for k in
                            ("latency", "cost", "downloads", "hits", "Umax",
                             "imbalance", "stranded", "viol", "util_max")}
        print(f"[exp12/{tag}] {name:16s} lat={out12[tag][name]['latency'][0]:.0f} "
              f"cost={out12[tag][name]['cost'][0]:.0f} "
              f"Umax={out12[tag][name]['Umax'][0]:.3f} "
              f"util={out12[tag][name]['util_max'][0]:.3f}")
out["exp12"] = out12

# ---------------------------------------------------------------- exp17
rows17 = []
for rho in [1.0, 1.2, 1.6, 2.0]:
    for rate in [0.25, 0.5, 2.0]:
        acc = []
        for seed in range(seeds):
            cat, tr = synthetic_world(N=N, T=T, seed=seed, zipf=rho, rate=rate)
            d = online_drl(cat, tr, a.C)
            b = ll_rc(cat, tr, a.C)
            r = rand_drl(cat, tr, a.C, np.random.default_rng(seed), K=a.K,
                         repair_mode="stateless", gate_downloads=True)
            acc.append(dict(lat_drl_pct=pct(d.latency, b.latency),
                            cost_drl_pct=pct(d.cost, b.cost),
                            lat_rand_pct=pct(r.latency, b.latency),
                            cost_rand_pct=pct(r.cost, b.cost),
                            hits_rand=r.hits, util_max=float(r.util.mean(0).max())))
        row = dict(rho=rho, rate=rate, n_seeds=len(acc))
        for k in ("lat_drl_pct", "cost_drl_pct", "lat_rand_pct", "cost_rand_pct",
                  "hits_rand", "util_max"):
            v = [x[k] for x in acc]
            row[k] = (float(np.mean(v)), float(np.std(v)))
        rows17.append(row)
out["exp17"] = dict(rows=rows17)
print("\n[exp17] rho/rate sweep (stateless, vs LL-RC)")
for r in rows17:
    print(f"  rho={r['rho']:.1f} rate={r['rate']:.2f} "
          f"rand lat={r['lat_rand_pct'][0]:+7.2f}% cost={r['cost_rand_pct'][0]:+8.2f}% "
          f"util={r['util_max'][0]:.3f}")

save(a, "exp29_stateless_headline", out)
print("\n[exp29] saved results/exp29_stateless_headline.json")