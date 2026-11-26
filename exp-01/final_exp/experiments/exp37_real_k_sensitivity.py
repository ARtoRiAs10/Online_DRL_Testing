"""Phase-6 K sensitivity on the real Google trace (10k slice).

The full 50k run uses K=2 for tractability; this experiment checks that the
stateless-repair result is not an artifact of K.  It runs the stateless and Landlord
repairs with K in {2,4,8} on the 10k slice, for both the natural and crossing
M/l regimes.
"""
from common import *
from sim.trace import load_trace_csv
from sim.algos import online_drl, rand_drl

a = parse(__doc__, C=50)
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
path = os.path.join(DATA, "google.csv")
CFG = dict(id_col="JobID", time_col="Timestamp", cpu_col="CPU", ram_col="RAM",
           disk_col="Disk", time_scale_ms=1e-3, disk_unit_mb=1.0 / 1024.0,
           U=100.0)
CONFIGS = [(1000.0, 1.0), (1000.0, 1e7)]
Ks = [2, 4, 8]
seeds = 3
print(f"phase-6 K sensitivity: rows=10000 seeds={seeds} Ks={Ks} (stateless repair)")

rows = []
for D, disk_scale in CONFIGS:
    cat, tr = load_trace_csv(path, max_rows=10000, D=D, disk_scale=disk_scale, **CFG)
    C = min(a.C, max(1, cat.N // 2))
    frac_lt = float(np.mean(cat.M < cat.l))
    det = online_drl(cat, tr, C)
    for K in Ks:
        for repair_mode in ("stateless",):
            rr = [rand_drl(cat, tr, C, np.random.default_rng(s), K=K,
                           repair_mode=repair_mode, gate_downloads=True,
                           download_time_mode="ema") for s in range(seeds)]
            rows.append(dict(
                D=D, disk_scale=disk_scale, K=K, repair_mode=repair_mode,
                n_services=int(cat.N), n_requests=int(len(tr)), C=int(C),
                frac_M_lt_l=frac_lt,
                det_lat=float(det.latency), det_cost=float(det.cost),
                lat=ms([r.latency for r in rr]), cost=ms([r.cost for r in rr]),
                hits=ms([r.hits for r in rr]),
                vs_det_lat=paired_ci([r.latency for r in rr], [det.latency] * seeds),
                vs_det_cost=paired_ci([r.cost for r in rr], [det.cost] * seeds),
            ))
            print(f"D={D:.0f} disk_x={disk_scale:.0e} K={K} {repair_mode:10s} "
                  f"lat={rows[-1]['lat'][0]:.0f}±{rows[-1]['lat'][1]:.0f} "
                  f"cost={rows[-1]['cost'][0]:.0f}±{rows[-1]['cost'][1]:.0f} "
                  f"vs det lat {rows[-1]['vs_det_lat']['mean']:+.0f}±{rows[-1]['vs_det_lat']['ci_half']:.0f} "
                  f"cost {rows[-1]['vs_det_cost']['mean']:+.0f}±{rows[-1]['vs_det_cost']['ci_half']:.0f}")

out = dict(config=dict(rows=10000, seeds=seeds, Ks=Ks, configs=CONFIGS),
           rows=rows,
           note=("Stateless repair only; the Landlord repair is dominated on latency in the "
                 "main Phase-6 runs and is not repeated for every K."))
save(a, "exp37_real_k_sensitivity", out)

print("\n### summary")
for r in rows:
    print(f"D={r['D']:.0f} disk_x={r['disk_scale']:.0e} K={r['K']} {r['repair_mode']:10s} "
          f"vs det lat {r['vs_det_lat']['mean']:+.0f}±{r['vs_det_lat']['ci_half']:.0f} "
          f"cost {r['vs_det_cost']['mean']:+.0f}±{r['vs_det_cost']['ci_half']:.0f}")