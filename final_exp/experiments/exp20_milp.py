"""Phase-3 MILP comparisons on 30 random traces (``N=10..16``).

For every trace we solve the offline MILP with HiGHS (time limit) and compare
Online-DRL, R3-only, randomized DRL with Landlord-Repair, and randomized DRL with the
stateless repair.  Ratios are ``ALG / OPT``; when the MILP hits the time limit, OPT is an
upper bound and the ratios are lower bounds.
"""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, rand_landlord, rand_drl
from sim.opt import solve_opt

a = parse(__doc__, opt_time=10.0, C=3, K=4)
n_traces = 5 if a.quick else 30
T = 80 if a.quick else 120
opt_time = 2.0 if a.quick else a.opt_time
n_seeds = 2
print(f"phase-3 MILP: traces={n_traces} T={T} opt_time={opt_time}s seeds={n_seeds}")

rows = []
for i in range(n_traces):
    rng = np.random.default_rng(1000 + i)
    N = int(rng.integers(10, 17))
    C = int(rng.integers(2, 6))
    cat, tr = synthetic_world(N=N, T=T, seed=i, imbalance=1.5, cap_mult=4,
                              D=4000, req_kbit=100, rate=1.0)
    det = online_drl(cat, tr, C)
    o = solve_opt(cat, tr, C, bmax=max(det.cost, cat.M.min()), time_limit=opt_time)
    if not np.isfinite(o["latency"]):
        print(f"trace {i}: OPT failed ({o['message']})")
        continue
    r3 = [rand_landlord(cat, tr, C, np.random.default_rng(s)) for s in range(n_seeds)]
    rl = [rand_drl(cat, tr, C, np.random.default_rng(s), K=a.K,
                   repair_mode="landlord", gate_downloads=True) for s in range(n_seeds)]
    rs = [rand_drl(cat, tr, C, np.random.default_rng(s), K=a.K,
                   repair_mode="stateless", gate_downloads=True) for s in range(n_seeds)]
    row = dict(
        trace=i, N=N, C=C, opt=float(o["latency"]), opt_status=int(o["status"]),
        opt_gap=float(o["gap"]) if np.isfinite(o["gap"]) else None,
        repair_mode_landlord="landlord", repair_mode_stateless="stateless",
        det_lat=float(det.latency), det_cost=float(det.cost),
        det_ratio=float(det.latency / o["latency"]),
        r3_ratio=float(np.mean([r.latency for r in r3]) / o["latency"]),
        rand_ratio=float(np.mean([r.latency for r in rl]) / o["latency"]),
        rand_stateless_ratio=float(np.mean([r.latency for r in rs]) / o["latency"]),
        r3_cost=float(np.mean([r.cost for r in r3])),
        rand_cost=float(np.mean([r.cost for r in rl])),
        rand_stateless_cost=float(np.mean([r.cost for r in rs])),
        det_hits=int(det.hits),
        rand_hits=float(np.mean([r.hits for r in rl])),
    )
    rows.append(row)
    print(f"trace {i:2d} N={N:2d} C={C} opt={row['opt']:.0f} "
          f"det={row['det_ratio']:.3f} r3={row['r3_ratio']:.3f} "
          f"rand={row['rand_ratio']:.3f} stateless={row['rand_stateless_ratio']:.3f} "
          f"status={row['opt_status']}")

metrics = ("det_ratio", "r3_ratio", "rand_ratio", "rand_stateless_ratio")
summary = {m: ms([r[m] for r in rows]) for m in metrics}
summary["paired_vs_det"] = {
    m: paired_ci([r[m] for r in rows], [r["det_ratio"] for r in rows])
    for m in ("r3_ratio", "rand_ratio", "rand_stateless_ratio")
}
summary["opt_status_counts"] = {
    str(s): sum(1 for r in rows if r["opt_status"] == s) for s in sorted(set(r["opt_status"] for r in rows))
}

out = dict(config=dict(n_traces=n_traces, T=T, opt_time=opt_time, n_seeds=n_seeds),
           rows=rows, summary=summary)
save(a, "exp20_milp", out)

print("\n### summary")
for m in metrics:
    print(f"{m:24s} {summary[m][0]:.3f} ± {summary[m][1]:.3f}")
for m, d in summary["paired_vs_det"].items():
    print(f"{m:24s} vs det: {d['mean']:+.3f} ± {d['ci_half']:.3f} "
          f"win={d['win_rate']:.2f}")
print("OPT status counts:", summary["opt_status_counts"],
      "(0 = optimal, 1 = time limit)")