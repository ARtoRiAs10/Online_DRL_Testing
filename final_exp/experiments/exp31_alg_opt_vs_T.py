"""Phase-3: reconcile the ablation vs MILP contradiction.

Measures ALG/OPT as a function of the trace length T.  If the randomized method needs
warm-up, its ratio to OPT should improve as T grows.
"""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, rand_landlord, rand_drl
from sim.opt import solve_opt

a = parse(__doc__, C=3, K=4, opt_time=20.0)
Ts = [120, 300, 600, 1200]
n_traces = {120: 30, 300: 15, 600: 10, 1200: 5}
n_seeds = 2
print(f"phase-3 ALG/OPT vs T: traces={n_traces} seeds={n_seeds}")

rows = []
for T in Ts:
    opt_time = 5.0 if T >= 600 else a.opt_time
    for i in range(n_traces[T]):
        rng = np.random.default_rng(1000 + i)
        N = int(rng.integers(10, 13))
        cat, tr = synthetic_world(N=N, T=T, seed=i, imbalance=1.5, cap_mult=4,
                                  D=4000, req_kbit=100, rate=1.0)
        det = online_drl(cat, tr, a.C)
        o = solve_opt(cat, tr, a.C, bmax=max(det.cost, cat.M.min()), time_limit=opt_time)
        if not np.isfinite(o["latency"]):
            print(f"T={T} trace {i}: OPT failed ({o['message']})")
            continue
        r3 = [rand_landlord(cat, tr, a.C, np.random.default_rng(s)) for s in range(n_seeds)]
        rl = [rand_drl(cat, tr, a.C, np.random.default_rng(s), K=a.K,
                       repair_mode="landlord", gate_downloads=True) for s in range(n_seeds)]
        rs = [rand_drl(cat, tr, a.C, np.random.default_rng(s), K=a.K,
                       repair_mode="stateless", gate_downloads=True) for s in range(n_seeds)]
        row = dict(
            T=T, trace=i, N=N, opt=float(o["latency"]), opt_status=int(o["status"]),
            opt_gap=float(o["gap"]) if np.isfinite(o["gap"]) else None,
            det_ratio=float(det.latency / o["latency"]),
            r3_ratio=float(np.mean([r.latency for r in r3]) / o["latency"]),
            rand_landlord_ratio=float(np.mean([r.latency for r in rl]) / o["latency"]),
            rand_stateless_ratio=float(np.mean([r.latency for r in rs]) / o["latency"]),
            det_cost=float(det.cost),
            rand_landlord_cost=float(np.mean([r.cost for r in rl])),
            rand_stateless_cost=float(np.mean([r.cost for r in rs])),
        )
        rows.append(row)
        print(f"T={T:4d} trace {i:2d} N={N:2d} opt={row['opt']:.0f} "
              f"det={row['det_ratio']:.3f} r3={row['r3_ratio']:.3f} "
              f"landlord={row['rand_landlord_ratio']:.3f} "
              f"stateless={row['rand_stateless_ratio']:.3f} "
              f"status={row['opt_status']}")

summary = {}
for T in Ts:
    fr = [r for r in rows if r["T"] == T]
    if not fr:
        continue
    summary[str(T)] = {}
    for m in ("det_ratio", "r3_ratio", "rand_landlord_ratio", "rand_stateless_ratio"):
        summary[str(T)][m] = ms([r[m] for r in fr])
    summary[str(T)]["opt_status_counts"] = {
        str(s): sum(1 for r in fr if r["opt_status"] == s)
        for s in sorted(set(r["opt_status"] for r in fr))
    }
    summary[str(T)]["n_traces"] = len(fr)

out = dict(config=dict(Ts=Ts, n_traces=n_traces, n_seeds=n_seeds, opt_time=a.opt_time),
           rows=rows, summary=summary)
save(a, "exp31_alg_opt_vs_T", out)

print("\n### summary")
for T in Ts:
    s = summary.get(str(T))
    if not s:
        continue
    print(f"T={T:4d} n={s['n_traces']:2d} "
          f"det={s['det_ratio'][0]:.3f} r3={s['r3_ratio'][0]:.3f} "
          f"landlord={s['rand_landlord_ratio'][0]:.3f} "
          f"stateless={s['rand_stateless_ratio'][0]:.3f} "
          f"status={s['opt_status_counts']}")