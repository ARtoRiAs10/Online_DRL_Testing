"""Phase-3 ablation table with paired differences and 95% confidence intervals.

Variants
--------
* ``online_drl``          — deterministic Online-DRL baseline.
* ``r3_only``             — randomized download threshold with the deterministic Landlord policy.
* ``r1r2_landlord``       — R1+R2 with the updated Landlord-Repair, deterministic threshold.
* ``r1r2_stateless``      — R1+R2 with the previous stateless repair.
* ``r1r2_topup``          — R1+R2 with Landlord-Repair plus the optional top-up step.
* ``r1r2r3_landlord``     — full R1+R2+R3 with Landlord-Repair and the download gate.
* ``r1r2r3_stateless``    — full R1+R2+R3 with the stateless repair.
* ``r1r2r3_topup``        — full R1+R2+R3 with Landlord-Repair plus top-up.

Every variant is run on the same trace for a given seed, so the reported differences are
paired.  The default is 20 seeds (3 with ``--quick``).
"""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, rand_landlord, rand_drl

a = parse(__doc__, C=12, K=4)
N = 16 if a.quick else 20
T = 1000 if a.quick else 2000
K = a.K
C = a.C
seeds = max(a.seeds, 20) if not a.quick else a.seeds
print(f"phase-3 ablation: N={N} T={T} K={K} C={C} seeds={seeds}")

VARIANTS = {
    "online_drl": lambda cat, tr, rng: online_drl(cat, tr, C),
    "r3_only": lambda cat, tr, rng: rand_landlord(cat, tr, C, rng),
    "r1r2_landlord": lambda cat, tr, rng: rand_drl(
        cat, tr, C, rng, K=K, thr="det", repair_mode="landlord", gate_downloads=False),
    "r1r2_stateless": lambda cat, tr, rng: rand_drl(
        cat, tr, C, rng, K=K, thr="det", repair_mode="stateless", gate_downloads=False),
    "r1r2_topup": lambda cat, tr, rng: rand_drl(
        cat, tr, C, rng, K=K, thr="det", repair_mode="landlord",
        gate_downloads=False, topup=True),
    "r1r2r3_landlord": lambda cat, tr, rng: rand_drl(
        cat, tr, C, rng, K=K, thr="random", repair_mode="landlord", gate_downloads=True),
    "r1r2r3_stateless": lambda cat, tr, rng: rand_drl(
        cat, tr, C, rng, K=K, thr="random", repair_mode="stateless", gate_downloads=True),
    "r1r2r3_topup": lambda cat, tr, rng: rand_drl(
        cat, tr, C, rng, K=K, thr="random", repair_mode="landlord",
        gate_downloads=True, topup=True),
}

rows = []
for seed in range(seeds):
    cat, tr = synthetic_world(N=N, T=T, seed=seed, imbalance=2.0, cap_mult=4)
    for name, fn in VARIANTS.items():
        R = fn(cat, tr, np.random.default_rng(seed))
        rows.append(dict(seed=seed, variant=name,
                         repair_mode=("stateless" if "stateless" in name else "landlord"),
                         latency=float(R.latency), cost=float(R.cost),
                         hits=int(R.hits), downloads=int(R.downloads),
                         delayed=int(R.delayed_hits)))

# Pivot per seed for paired comparisons.
by_seed = {}
for r in rows:
    by_seed.setdefault(r["seed"], {})[r["variant"]] = r

baseline = "online_drl"
repair_ref = "r1r2r3_landlord"
summary = {}
for name in VARIANTS:
    lat = [by_seed[s][name]["latency"] for s in range(seeds)]
    cost = [by_seed[s][name]["cost"] for s in range(seeds)]
    base_lat = [by_seed[s][baseline]["latency"] for s in range(seeds)]
    base_cost = [by_seed[s][baseline]["cost"] for s in range(seeds)]
    ref_lat = [by_seed[s][repair_ref]["latency"] for s in range(seeds)]
    ref_cost = [by_seed[s][repair_ref]["cost"] for s in range(seeds)]
    summary[name] = dict(
        latency=ms(lat), cost=ms(cost),
        hits=ms([by_seed[s][name]["hits"] for s in range(seeds)]),
        downloads=ms([by_seed[s][name]["downloads"] for s in range(seeds)]),
        vs_online_lat=paired_ci(lat, base_lat),
        vs_online_cost=paired_ci(cost, base_cost),
        vs_landlord_lat=paired_ci(lat, ref_lat),
        vs_landlord_cost=paired_ci(cost, ref_cost),
    )

out = dict(config=dict(N=N, T=T, K=K, C=C, seeds=seeds),
           variants=list(VARIANTS), rows=rows, summary=summary)
save(a, "exp18_ablation", out)

print("\n### ablation (mean [95% CI half-width], paired vs Online-DRL)")
print(f"{'variant':20s} {'latency':>12s} {'cost':>12s} {'hits':>8s} "
      f"{'dLat vs det':>18s} {'dCost vs det':>18s}")
for name in VARIANTS:
    s = summary[name]
    dl = s["vs_online_lat"]; dc = s["vs_online_cost"]
    print(f"{name:20s} {s['latency'][0]:12.0f} {s['cost'][0]:12.0f} "
          f"{s['hits'][0]:8.0f} "
          f"{dl['mean']:+10.0f} ±{dl['ci_half']:.0f} "
          f"{dc['mean']:+10.0f} ±{dc['ci_half']:.0f}")

print("\n### repair comparison (paired vs full R1+R2+R3 landlord)")
for name in ("r1r2_landlord", "r1r2_stateless", "r1r2_topup",
             "r1r2r3_stateless", "r1r2r3_topup"):
    s = summary[name]
    dl = s["vs_landlord_lat"]; dc = s["vs_landlord_cost"]
    print(f"{name:20s} dLat={dl['mean']:+10.0f} ±{dl['ci_half']:.0f} "
          f"dCost={dc['mean']:+10.0f} ±{dc['ci_half']:.0f} "
          f"win_lat={dl['win_rate']:.2f} win_cost={dc['win_rate']:.2f}")