"""Phase-2 (v2): horizon and consistency.

Runs prefixes of the 50k Google trace (10k, 20k, 30k, 50k) at the same K and seeds,
for both the raw and crossing M/l regimes.  This tests the "warm-up" hypothesis for the
latency sign change between the 10k slice and the full 50k trace.

The MILP ALG/OPT vs T result is `exp31_alg_opt_vs_T.json` (T = 120, 300, 600, 1200).
The MILP objective there is **latency only**, subject to the download budget and the
per-step resource capacities; the download cost is charged in the constraint, not in the
objective.
"""
from common import *
from sim.trace import load_trace_csv
from sim.algos import online_drl, rand_drl

a = parse(__doc__, C=50)
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
path = os.path.join(DATA, "google.csv")
CFG = dict(id_col="JobID", time_col="Timestamp", cpu_col="CPU", ram_col="RAM",
           disk_col="Disk", time_scale_ms=1e-3, disk_unit_mb=1.0 / 1024.0, U=100.0)
PREFIXES = [10000, 20000, 30000, 50000]
CONFIGS = [(1000.0, 1.0), (1000.0, 1e7)]
seeds = 3
K = 2
print(f"phase-2 horizon: prefixes={PREFIXES} seeds={seeds} K={K}")

rows = []
for D, disk_scale in CONFIGS:
    cat, tr_full = load_trace_csv(path, max_rows=50000, D=D, disk_scale=disk_scale, **CFG)
    C = min(a.C, max(1, cat.N // 2))
    frac_lt = float(np.mean(cat.M < cat.l))
    for T in PREFIXES:
        tr = tr_full.head(T)
        det = online_drl(cat, tr, C, theta=1.0)
        rr = [rand_drl(cat, tr, C, np.random.default_rng(s), K=K,
                       repair_mode="stateless", gate_downloads=True,
                       download_time_mode="ema") for s in range(seeds)]
        row = dict(
            D=D, disk_scale=disk_scale, prefix=int(T), seeds=seeds, K=K,
            frac_M_lt_l=frac_lt,
            det_lat=float(det.latency), det_cost=float(det.cost), det_hits=int(det.hits),
            rand_lat=ms([r.latency for r in rr]), rand_cost=ms([r.cost for r in rr]),
            rand_hits=ms([r.hits for r in rr]),
            paired_lat=paired_ci([r.latency for r in rr], [det.latency] * seeds),
            paired_cost=paired_ci([r.cost for r in rr], [det.cost] * seeds),
        )
        rows.append(row)
        print(f"D={D:.0f} dx={disk_scale:.0e} T={T:5d} M<l={frac_lt:.3f} "
              f"det={det.latency:.0f}/{det.cost:.0f} "
              f"rand={row['rand_lat'][0]:.0f}±{row['rand_lat'][1]:.0f}/"
              f"{row['rand_cost'][0]:.0f}±{row['rand_cost'][1]:.0f} "
              f"Δlat={row['paired_lat']['mean']:+.0f}±{row['paired_lat']['ci_half']:.0f} "
              f"Δcost={row['paired_cost']['mean']:+.0f}±{row['paired_cost']['ci_half']:.0f}")

out = dict(
    config=dict(prefixes=PREFIXES, seeds=seeds, K=K, configs=CONFIGS),
    rows=rows,
    note=("Prefixes of the 50k Google trace.  The MILP ALG/OPT vs T result is exp31; its "
          "objective is latency only subject to the budget and capacities."),
)
save(a, "exp39_horizon_prefixes", out)

print("\n### summary")
for r in rows:
    print(f"D={r['D']:.0f} dx={r['disk_scale']:.0e} T={r['prefix']:5d} "
          f"Δlat={r['paired_lat']['mean']:+.0f}±{r['paired_lat']['ci_half']:.0f} "
          f"Δcost={r['paired_cost']['mean']:+.0f}±{r['paired_cost']['ci_half']:.0f}")