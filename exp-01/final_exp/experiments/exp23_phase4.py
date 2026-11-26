"""Phase-4 experiments: EMA download-time tracking, churn charging, lazy repair.

Variants
--------
* ``base``           — `M_n/2` proxy, no churn charge, no lazy repair.
* ``dl_ema``         — EMA of the observed remaining download time.
* ``churn_all``      — charge every 0->1 sample transition to the download budget.
* ``lazy``           — keep the previous path and evict only when infeasible.
* ``dl_ema_lazy``    — both EMA and lazy repair.
* ``dl_ema_churn_lazy`` — all three.

Every variant runs on the same trace with the same seed, so differences are paired.
"""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, rand_drl
from sim.regret_bound import (regret_bound_terms, ema_tracking_error,
                              optimal_ema_alpha)

a = parse(__doc__, C=12, K=4, D=10000.0)
N = 16 if a.quick else 20
T = 500 if a.quick else 1000
seeds = min(a.seeds, 10) if not a.quick else a.seeds
print(f"phase-4: N={N} T={T} K={a.K} C={a.C} seeds={seeds}")

VARIANTS = {
    "base": dict(download_time_mode="half", charge_churn="none", lazy=False),
    "dl_ema": dict(download_time_mode="ema", charge_churn="none", lazy=False),
    "churn_all": dict(download_time_mode="half", charge_churn="all", lazy=False),
    "lazy": dict(download_time_mode="half", charge_churn="none", lazy=True),
    "dl_ema_lazy": dict(download_time_mode="ema", charge_churn="none", lazy=True),
    "dl_ema_churn_lazy": dict(download_time_mode="ema", charge_churn="all", lazy=True),
}

rows = []
for seed in range(seeds):
    cat, tr = synthetic_world(N=N, T=T, seed=seed, imbalance=2.0, cap_mult=4, D=a.D)
    det = online_drl(cat, tr, a.C)
    for name, kw in VARIANTS.items():
        R = rand_drl(cat, tr, a.C, np.random.default_rng(seed), K=a.K,
                     bmax=det.cost, log_steps=True, repair_mode="landlord",
                     gate_downloads=True, **kw)
        terms = regret_bound_terms(R.extra["step_log"], cat, a.C)
        rows.append(dict(
            seed=seed, variant=name, repair_mode="landlord",
            lazy=bool(kw.get("lazy", False)),
            download_time_mode=kw.get("download_time_mode", "half"),
            charge_churn=kw.get("charge_churn", "none"),
            latency=float(R.latency), cost=float(R.cost),
            hits=int(R.hits), downloads=int(R.downloads),
            churn_credit=float(R.extra.get("churn_credit", 0.0)),
            churn_events=int(R.extra.get("churn_events", 0)),
            zeta_q_hat=float(terms["zeta_q_hat"]),
            zeta_dl_hat=float(terms["zeta_dl_hat"]),
            zeta_hat=float(terms["zeta_hat"]),
            mean_dl_ema=float(R.extra.get("mean_dl_ema", 0.0)),
            det_latency=float(det.latency), det_cost=float(det.cost),
        ))

by_seed = {}
for r in rows:
    by_seed.setdefault(r["seed"], {})[r["variant"]] = r

metrics = ("latency", "cost", "hits", "downloads", "churn_credit",
           "churn_events", "zeta_q_hat", "zeta_dl_hat", "zeta_hat",
           "mean_dl_ema")
summary = {}
for name in VARIANTS:
    summary[name] = {m: ms([by_seed[s][name][m] for s in range(seeds)]) for m in metrics}
    summary[name]["vs_base_lat"] = paired_ci(
        [by_seed[s][name]["latency"] for s in range(seeds)],
        [by_seed[s]["base"]["latency"] for s in range(seeds)])
    summary[name]["vs_base_cost"] = paired_ci(
        [by_seed[s][name]["cost"] for s in range(seeds)],
        [by_seed[s]["base"]["cost"] for s in range(seeds)])
    summary[name]["budget_respected"] = float(np.mean([
        by_seed[s][name]["cost"] <= by_seed[s][name]["det_cost"] for s in range(seeds)]))

out = dict(config=dict(N=N, T=T, K=a.K, C=a.C, seeds=seeds, variants=VARIANTS),
           rows=rows, summary=summary)
save(a, "exp23_phase4", out)

print("\n### phase-4 summary")
print(f"{'variant':22s} {'lat':>9s} {'cost':>9s} {'hits':>6s} "
      f"{'churn':>10s} {'zeta_dl':>8s} {'dLat':>12s} {'dCost':>12s}")
for name in VARIANTS:
    s = summary[name]
    dl = s["vs_base_lat"]; dc = s["vs_base_cost"]
    print(f"{name:22s} {s['latency'][0]:9.0f} {s['cost'][0]:9.0f} "
          f"{s['hits'][0]:6.0f} {s['churn_credit'][0]:10.0f} "
          f"{s['zeta_dl_hat'][0]:8.3f} "
          f"{dl['mean']:+8.0f}±{dl['ci_half']:.0f} "
          f"{dc['mean']:+8.0f}±{dc['ci_half']:.0f}")

print("\n### EMA tracking helpers")
for drift in (0.0, 0.1, 1.0):
    alpha = optimal_ema_alpha(sigma=1.0, drift=drift) if drift > 0 else 1.0
    err = ema_tracking_error(alpha, sigma=1.0, drift=drift)
    print(f"drift={drift:.2f}  alpha*={alpha:.3f}  error={err:.3f}")