"""G4: expected competitive ratio on small random traces against the exact MILP OPT
(download budget Bmax = cost of the deterministic algorithm)."""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, rand_landlord, rand_drl
from sim.opt import solve_opt

a = parse(__doc__, opt_time=60.0, C=3)
T, ntr = (60, 2) if a.quick else (120, 5)
WORLD = dict(rate=1.0, req_kbit=100, D=4000, U=100)   # downloads (50-200 ms) are cheap relative to inter-arrivals, so OPT actually downloads
rows = []
for tr_seed in range(ntr):
    cat, tr = synthetic_world(N=10, T=T, seed=tr_seed, **WORLD)
    det = online_drl(cat, tr, a.C)
    o = solve_opt(cat, tr, a.C, bmax=max(det.cost, cat.M.min()), time_limit=a.opt_time)
    if not np.isfinite(o["latency"]): print("OPT failed", o["message"]); continue
    rl = [rand_landlord(cat, tr, a.C, np.random.default_rng(s)).latency for s in range(a.seeds)]
    rf = [rand_drl(cat, tr, a.C, np.random.default_rng(s), K=4).latency for s in range(a.seeds)]
    row = dict(trace=tr_seed, opt=o["latency"], opt_status=int(o["status"]), opt_gap=o["gap"], det=det.latency / o["latency"],
               rand_landlord=float(np.mean(rl)) / o["latency"], rand_drl=float(np.mean(rf)) / o["latency"]); rows.append(row); print(row)
agg = {k: ms([r[k] for r in rows]) for k in ("det", "rand_landlord", "rand_drl")}
print("mean ALG/OPT:", agg); print("(opt_status 1 = time limit: OPT is an upper bound, so ratios are lower bounds)")
fig, ax = plt.subplots(figsize=(4.5, 3.4)); ax.bar(list(agg), [v[0] for v in agg.values()], yerr=[v[1] for v in agg.values()]); ax.set_ylabel("ALG / OPT")
save(a, "exp10_expected_ratio", dict(rows=rows, mean=agg), fig)
