"""Phase C: prefixes/windows, exact Large-Sequence optima, MILP vs T, statistics.

Outputs results/phase_c_horizon_opt.json.
"""
from common import *

rows = []
print("C1: prefixes")
for regime, kw in (("raw", dict(D=1000.0, disk_scale=1.0)),
                   ("crossing", dict(D=1000.0, disk_scale=1e7))):
    cat, full = load_google("google.csv", max_rows=50000, **kw)
    C = min(12, max(1, cat.N // 2))
    fl = float(np.mean(cat.M < cat.l))
    for T in (10000, 20000, 30000, 50000):
        tr = full.head(T)
        d = online_drl(cat, tr, C, theta=1.0)
        nseed = 2 if T <= 20000 else 1
        rr = [rand_drl(cat, tr, C, np.random.default_rng(s), K=2,
                       repair_mode="stateless", gate_downloads=True,
                       download_time_mode="ema") for s in range(nseed)]
        rows.append(dict(kind="prefix", regime=regime, T=T, seeds=nseed,
                         frac_M_lt_l=fl, det_lat=float(d.latency), det_cost=float(d.cost),
                         rand_lat=ms([r.latency for r in rr]),
                         rand_cost=ms([r.cost for r in rr]),
                         paired_lat=paired_ci([r.latency for r in rr], [d.latency]*nseed),
                         paired_cost=paired_ci([r.cost for r in rr], [d.cost]*nseed)))
        print(f"prefix {regime} T={T} done", flush=True)

print("C1: sliding 10k windows (stride 5k, 1 seed, raw)")
cat, full = load_google("google.csv", max_rows=50000, D=1000.0, disk_scale=1.0)
C = min(12, max(1, cat.N // 2))
for start in range(0, 40001, 5000):
    tr = full.head(start + 10000) if start == 0 else None
    # build a window by slicing the trace arrays
    from sim.trace import Trace
    tr = Trace(full.t[start:start+10000], full.n[start:start+10000])
    d = online_drl(cat, tr, C, theta=1.0)
    r = rand_drl(cat, tr, C, np.random.default_rng(0), K=2,
                 repair_mode="stateless", gate_downloads=True, download_time_mode="ema")
    rows.append(dict(kind="window", regime="raw", start=start, T=10000, seeds=1,
                     det_lat=float(d.latency), det_cost=float(d.cost),
                     rand_lat=float(r.latency), rand_cost=float(r.cost),
                     paired_lat=dict(mean=float(r.latency-d.latency), ci_half=0.0),
                     paired_cost=dict(mean=float(r.cost-d.cost), ci_half=0.0)))
    print(f"window start={start} done", flush=True)

print("C2: exact MILP for Large Sequence")
opt_rows = []
for k in (3, 6, 9, 12):
    for ratio in (2.0, 4.0, 8.0):
        cat, tr, closed = large_sequence(k, 2*k, M=ratio, Ml=2.0*ratio, l=1.0)
        o = solve_opt(cat, tr, k, bmax=None, time_limit=30.0)
        det = online_drl(cat, tr, k)
        opt_rows.append(dict(k=k, M_over_l=ratio, T=len(tr), closed=float(closed),
                             milp=float(o["latency"]), status=int(o["status"]),
                             gap=float(o["gap"]) if np.isfinite(o["gap"]) else None,
                             det_lat=float(det.latency),
                             det_ratio=float(det.latency/o["latency"]),
                             closed_ratio=float(det.latency/closed)))
        print(f"OPT k={k} M/l={ratio:g}: closed={closed:.0f} milp={o['latency']:.0f} "
              f"det={det.latency:.0f} ratio={det.latency/o['latency']:.3f}", flush=True)

print("C3: MILP ALG/OPT vs T")
Ts = [120, 300, 600, 1200]
n_traces = {120: 3, 300: 2, 600: 1, 1200: 1}
trows = []
for T in Ts:
    for i in range(n_traces[T]):
        rng = np.random.default_rng(1000+i)
        N = int(rng.integers(10, 13))
        cat, tr = synthetic_world(N=N, T=T, seed=i, imbalance=1.5, cap_mult=4,
                                  D=4000, req_kbit=100, rate=1.0)
        det = online_drl(cat, tr, 3)
        o = solve_opt(cat, tr, 3, bmax=max(det.cost, cat.M.min()), time_limit=20.0)
        if not np.isfinite(o["latency"]):
            print(f"T={T} trace {i}: OPT failed", flush=True); continue
        r3 = [rand_landlord(cat, tr, 3, np.random.default_rng(s)) for s in range(2)]
        rl = [rand_drl(cat, tr, 3, np.random.default_rng(s), K=4,
                       repair_mode="landlord", gate_downloads=True) for s in range(2)]
        rs = [rand_drl(cat, tr, 3, np.random.default_rng(s), K=4,
                       repair_mode="stateless", gate_downloads=True) for s in range(2)]
        trows.append(dict(T=T, trace=i, N=N, opt=float(o["latency"]),
                          opt_status=int(o["status"]),
                          opt_gap=float(o["gap"]) if np.isfinite(o["gap"]) else None,
                          det_ratio=float(det.latency/o["latency"]),
                          r3_ratio=float(np.mean([r.latency for r in r3])/o["latency"]),
                          landlord_ratio=float(np.mean([r.latency for r in rl])/o["latency"]),
                          stateless_ratio=float(np.mean([r.latency for r in rs])/o["latency"]),
                          det_cost=float(det.cost),
                          landlord_cost=float(np.mean([r.cost for r in rl])),
                          stateless_cost=float(np.mean([r.cost for r in rs]))))
        print(f"T={T} trace={i} N={N} opt={o['latency']:.0f} "
              f"det={trows[-1]['det_ratio']:.3f} "
              f"landlord={trows[-1]['landlord_ratio']:.3f}", flush=True)

# C4: ratio statistics
ratio_summary = {}
for key in ("det_ratio", "r3_ratio", "landlord_ratio", "stateless_ratio"):
    ratio_summary[key] = qstats([r[key] for r in trows])
ls_summary = {}
for k in (3,6,9,12):
    for ratio in (2.0,4.0,8.0):
        fr=[r for r in opt_rows if r["k"]==k and r["M_over_l"]==ratio]
        if fr:
            ls_summary[f"k={k},M/l={ratio:g}"]=dict(
                det_ratio=qstats([r["det_ratio"] for r in fr]),
                closed_ratio=qstats([r["closed_ratio"] for r in fr]),
                milp=fr[0]["milp"], closed=fr[0]["closed"], status=fr[0]["status"])

out=dict(config=dict(prefixes=[10000,20000,30000,50000], window=10000, stride=5000,
                     opt_ks=[3,6,9,12], opt_ratios=[2,4,8], Ts=Ts, n_traces=n_traces),
         rows=rows, opt_rows=opt_rows, trows=trows,
         ratio_summary=ratio_summary, ls_summary=ls_summary,
         note=("C1 prefixes use 2 seeds; windows use 1 seed. C2 uses the exact MILP "
               "(latency objective, no budget). C3 uses the paper's budgeted MILP; "
               "status/gap are HiGHS status and relative MIP gap."))
save("phase_c_horizon_opt", out)

print("\n### prefixes")
for r in rows:
    if r["kind"]=="prefix":
        print(f"{r['regime']:8s} T={r['T']:5d} dlat={r['paired_lat']['mean']:+.0f} "
              f"±{r['paired_lat']['ci_half']:.0f} dcost={r['paired_cost']['mean']:+.0f}")
print("\n### windows")
for r in rows:
    if r["kind"]=="window":
        print(f"start={r['start']:5d} dlat={r['paired_lat']['mean']:+.0f} dcost={r['paired_cost']['mean']:+.0f}")
print("\n### OPT")
for r in opt_rows:
    print(f"k={r['k']:2d} M/l={r['M_over_l']:g} closed={r['closed']:.0f} milp={r['milp']:.0f} "
          f"det={r['det_lat']:.0f} ratio={r['det_ratio']:.3f} closed_ratio={r['closed_ratio']:.3f}")
print("\n### T summary")
for k,s in ratio_summary.items(): print(k,s)
