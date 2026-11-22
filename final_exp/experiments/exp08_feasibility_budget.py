"""Experiment 8 -- G1 (feasibility) and G2 (download budget), with the updated
Landlord-Repair of Algorithm 3.

Four randomized variants are compared on the same imbalanced Zipf trace:

  * ``landlord``     : Algorithm 3 of the updated pseudocode (persistent credits,
                       download-cost aware, deficit-aligned eviction) -- literal (no
                       download gate);
  * ``landlord+gate``: the same repair, but a download is started only when the serving
                       sample currently wants the service (proposal assumption A4);
  * ``stateless``    : the v1 repair (evict in increasing p_n / max_r a_n,r);
  * ``no repair``    : raw randomized rounding (shows why a repair is needed).

For every variant we report the true-resource violation rate, the maximum utilization,
the download cost against ``B_max`` (the deterministic Online-DRL cost) and the number of
wasted downloads.
"""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, rand_landlord, rand_drl
from sim.metrics import resource_metrics
from paper_claims import PAPER_G1, PAPER_G2

a = parse(__doc__, C=12)
cat, tr = synthetic_world(N=a.N, T=a.T, seed=0, imbalance=3.0, cap_mult=4)
det = online_drl(cat, tr, a.C)
Bmax = det.cost
print(f"deterministic Online-DRL: latency={det.latency:.0f} cost={det.cost:.0f} (used as Bmax)")

VARIANTS = {
    "rand_landlord": lambda rng: rand_landlord(cat, tr, a.C, rng),
    "landlord (literal, no gate)": lambda rng: rand_drl(
        cat, tr, a.C, rng, K=4, repair_mode="landlord", gate_downloads=False),
    "landlord + gate": lambda rng: rand_drl(
        cat, tr, a.C, rng, K=4, repair_mode="landlord", gate_downloads=True),
    "stateless + gate": lambda rng: rand_drl(
        cat, tr, a.C, rng, K=4, repair_mode="stateless", gate_downloads=True),
    "no repair + gate": lambda rng: rand_drl(
        cat, tr, a.C, rng, K=4, repair_mode="none", gate_downloads=True),
}
out = {}
for name, fn in VARIANTS.items():
    R = [fn(np.random.default_rng(s)) for s in range(a.seeds)]
    m = [resource_metrics(r.util, r.latency) for r in R]
    cost = np.array([r.cost for r in R]); lat = np.array([r.latency for r in R])
    out[name] = dict(
        latency=ms(lat), cost=ms(cost),
        mean_cost_le_Bmax=bool(cost.mean() <= Bmax),
        frac_seeds_cost_le_Bmax=float((cost <= Bmax).mean()),
        violation_rate=ms([x["violation_rate"] for x in m]),
        max_util=float(max(x["max_util"] for x in m)),
        raw_violation_rate=ms([r.extra.get("raw_violation_rate", 0) for r in R]),
        repairs=ms([r.extra.get("repairs", 0) for r in R]),
        repair_evictions=ms([r.extra.get("repair_evictions", 0) for r in R]),
        landlord_rent_rounds=ms([r.extra.get("landlord_rent_rounds", 0) for r in R]),
        wasted_downloads=ms([r.extra.get("wasted_downloads", 0) for r in R]))
    print(name, json.dumps(out[name], default=float))

print("\npaper G1:", PAPER_G1)
print("paper G2:", PAPER_G2)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ks = list(out)
ax[0].bar(ks, [out[k]["cost"][0] for k in ks], yerr=[out[k]["cost"][1] for k in ks])
ax[0].axhline(Bmax, c="r", ls="--"); ax[0].set_title("download cost vs Bmax")
ax[1].bar(ks, [out[k]["violation_rate"][0] for k in ks])
ax[1].set_title("fraction of steps violating a true resource")
ax[2].bar(ks, [out[k]["wasted_downloads"][0] for k in ks])
ax[2].set_title("wasted downloads")
for x in ax:
    x.tick_params(axis="x", rotation=25)
save(a, "exp08_feasibility_budget",
     dict(Bmax=Bmax, det_latency=det.latency, paper=dict(G1=PAPER_G1, G2=PAPER_G2),
          results=out), fig)