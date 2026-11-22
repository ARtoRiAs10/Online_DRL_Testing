"""Phase-5 full 50,000-request Google trace run.

Runs Online-DRL, LL-RC, R3-only and randomized DRL on the full bundled `google.csv`
(50,000 requests, 4,619 services) for two download/latency regimes.  This is the
full-scale companion of exp16/exp22.
"""
from common import *
from sim.trace import load_trace_csv
from sim.algos import online_drl, ll_rc, rand_landlord, rand_drl

a = parse(__doc__, C=50, K=4)
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
path = os.path.join(DATA, "google.csv")
if not os.path.exists(path):
    raise SystemExit(f"missing {path}")

CFG = dict(id_col="JobID", time_col="Timestamp", cpu_col="CPU", ram_col="RAM",
           disk_col="Disk", time_scale_ms=1e-3, disk_unit_mb=1.0 / 1024.0,
           U=100.0)
CONFIGS = [(1000.0, 1.0), (1000.0, 1e7)]
seeds = 2
print(f"phase-5 full Google: rows=50000 seeds={seeds} configs={CONFIGS}")

rows = []
for D, disk_scale in CONFIGS:
    cat, tr = load_trace_csv(path, max_rows=50000, D=D, disk_scale=disk_scale, **CFG)
    C = min(a.C, max(1, cat.N // 2))
    frac_cross = float(np.mean(cat.M < cat.l))
    det = online_drl(cat, tr, C)
    ll = ll_rc(cat, tr, C)
    r3 = [rand_landlord(cat, tr, C, np.random.default_rng(s)) for s in range(seeds)]
    rf = [rand_drl(cat, tr, C, np.random.default_rng(s), K=a.K,
                   repair_mode="landlord", gate_downloads=True,
                   download_time_mode="ema") for s in range(seeds)]
    row = dict(
        D=D, disk_scale=disk_scale, n_services=int(cat.N), n_requests=int(len(tr)), C=int(C),
        repair_mode="landlord", download_time_mode="ema", charge_churn="none",
        frac_M_lt_l=frac_cross,
        det_lat=float(det.latency), det_cost=float(det.cost), det_hits=int(det.hits),
        ll_lat=float(ll.latency), ll_cost=float(ll.cost), ll_hits=int(ll.hits),
        r3_lat=ms([r.latency for r in r3]), r3_cost=ms([r.cost for r in r3]),
        r3_hits=ms([r.hits for r in r3]),
        rf_lat=ms([r.latency for r in rf]), rf_cost=ms([r.cost for r in rf]),
        rf_hits=ms([r.hits for r in rf]),
        rf_downloads=ms([r.downloads for r in rf]),
        rf_vs_det_lat=paired_ci([r.latency for r in rf], [det.latency] * seeds),
        rf_vs_det_cost=paired_ci([r.cost for r in rf], [det.cost] * seeds),
        rf_vs_ll_lat=paired_ci([r.latency for r in rf], [ll.latency] * seeds),
        rf_vs_ll_cost=paired_ci([r.cost for r in rf], [ll.cost] * seeds),
    )
    rows.append(row)
    print(f"D={D:.0f} disk_x={disk_scale:.0e} M<l={frac_cross:.3f} "
          f"det={row['det_lat']:.0f}/{row['det_cost']:.0f} "
          f"ll={row['ll_lat']:.0f}/{row['ll_cost']:.0f} "
          f"r3={row['r3_lat'][0]:.0f}/{row['r3_cost'][0]:.0f} "
          f"rf={row['rf_lat'][0]:.0f}/{row['rf_cost'][0]:.0f} "
          f"hits={row['rf_hits'][0]:.0f}")

out = dict(config=dict(rows=50000, seeds=seeds, configs=CONFIGS, K=a.K),
           rows=rows)
save(a, "exp27_full_google", out)

print("\n### full Google summary")
for r in rows:
    print(f"D={r['D']:.0f} disk_x={r['disk_scale']:.0e} "
          f"RF vs det lat {r['rf_vs_det_lat']['mean']:+.0f} "
          f"±{r['rf_vs_det_lat']['ci_half']:.0f} "
          f"cost {r['rf_vs_det_cost']['mean']:+.0f} "
          f"±{r['rf_vs_det_cost']['ci_half']:.0f}")