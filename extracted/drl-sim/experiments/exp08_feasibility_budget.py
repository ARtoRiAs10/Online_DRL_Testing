"""G1 (feasibility with/without repair) and G2 (expected download budget)."""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, rand_landlord, rand_drl
from sim.metrics import resource_metrics

a = parse(__doc__, C=12)
cat, tr = synthetic_world(N=a.N, T=a.T, seed=0, imbalance=3.0, cap_mult=4)
det = online_drl(cat, tr, a.C); Bmax = det.cost
print(f"deterministic Online-DRL: latency={det.latency:.0f} cost={det.cost:.0f} (used as Bmax)")
out = {}
for name, fn in {
    "rand_landlord": lambda rng: rand_landlord(cat, tr, a.C, rng),
    "rand_drl_repair": lambda rng: rand_drl(cat, tr, a.C, rng, K=4, repair=True),
    "rand_drl_norepair": lambda rng: rand_drl(cat, tr, a.C, rng, K=4, repair=False)}.items():
    R = [fn(np.random.default_rng(s)) for s in range(a.seeds)]
    m = [resource_metrics(r.util, r.latency) for r in R]
    cost = np.array([r.cost for r in R]); lat = np.array([r.latency for r in R])
    out[name] = dict(latency=ms(lat), cost=ms(cost), mean_cost_le_Bmax=bool(cost.mean() <= Bmax), frac_seeds_cost_le_Bmax=float((cost <= Bmax).mean()),
                     violation_rate=ms([x["violation_rate"] for x in m]), max_util=float(max(x["max_util"] for x in m)),
                     raw_violation_rate=ms([r.extra.get("raw_violation_rate", 0) for r in R]),
                     repairs=ms([r.extra.get("repairs", 0) for r in R]), wasted_downloads=ms([r.extra.get("wasted_downloads", 0) for r in R]))
    print(name, json.dumps(out[name], default=float))
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ks = list(out); ax[0].bar(ks, [out[k]["cost"][0] for k in ks], yerr=[out[k]["cost"][1] for k in ks]); ax[0].axhline(Bmax, c="r", ls="--"); ax[0].set_title("download cost vs Bmax"); ax[0].tick_params(axis="x", rotation=20)
ax[1].bar(ks, [out[k]["violation_rate"][0] for k in ks]); ax[1].set_title("fraction of steps violating a resource"); ax[1].tick_params(axis="x", rotation=20)
save(a, "exp08_feasibility_budget", dict(Bmax=Bmax, det_latency=det.latency, results=out), fig)
