"""Phase-6 full 50,000-request Google trace run (multi-seed, CIs).

Runs Online-DRL, LL-RC, R3-only and randomized DRL (Landlord and stateless repairs) on
the full bundled ``google.csv`` (50,000 requests, 4,619 services) for two
download/latency regimes:

* ``disk_scale=1``    -- the raw Google disk column (all ``M_n < l_n``);
* ``disk_scale=1e7``  -- a documented rescaling that makes ``M_n`` and ``l_n`` cross
  (about two thirds of the services still have ``M_n < l_n``).

Every randomized number is reported with a sample standard deviation and a paired 95%
confidence interval against Online-DRL and LL-RC.  This is the multi-seed companion of
exp16/exp22 and the Phase-6 replacement for the old two-seed exp27.
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
seeds = min(a.seeds, 3) if not a.quick else a.seeds
K = 2   # the full 50k trace is run with K=2 for tractability; exp36 uses K=4 on the 10k slice
print(f"phase-6 full Google: rows=50000 seeds={seeds} K={K} configs={CONFIGS}")

rows = []
for D, disk_scale in CONFIGS:
    cat, tr = load_trace_csv(path, max_rows=50000, D=D, disk_scale=disk_scale, **CFG)
    C = min(a.C, max(1, cat.N // 2))
    frac_cross = float(np.mean(cat.M < cat.l))
    frac_eq = float(np.mean(np.isclose(cat.M, cat.l)))
    frac_gt = float(np.mean(cat.M > cat.l))
    det = online_drl(cat, tr, C)
    ll = ll_rc(cat, tr, C)
    r3 = [rand_landlord(cat, tr, C, np.random.default_rng(s)) for s in range(seeds)]
    rf_l = [rand_drl(cat, tr, C, np.random.default_rng(s), K=K,
                     repair_mode="landlord", gate_downloads=True,
                     download_time_mode="ema") for s in range(seeds)]
    rf_s = [rand_drl(cat, tr, C, np.random.default_rng(s), K=K,
                     repair_mode="stateless", gate_downloads=True,
                     download_time_mode="ema") for s in range(seeds)]
    row = dict(
        D=D, disk_scale=disk_scale, n_services=int(cat.N), n_requests=int(len(tr)), C=int(C),
        repair_mode="both", download_time_mode="ema", charge_churn="none",
        frac_M_lt_l=frac_cross, frac_M_eq_l=frac_eq, frac_M_gt_l=frac_gt,
        M_min=float(cat.M.min()), M_max=float(cat.M.max()),
        l_min=float(cat.l.min()), l_max=float(cat.l.max()),
        det_lat=float(det.latency), det_cost=float(det.cost), det_hits=int(det.hits),
        ll_lat=float(ll.latency), ll_cost=float(ll.cost), ll_hits=int(ll.hits),
        r3_lat=ms([r.latency for r in r3]), r3_cost=ms([r.cost for r in r3]),
        r3_hits=ms([r.hits for r in r3]),
        rf_landlord_lat=ms([r.latency for r in rf_l]),
        rf_landlord_cost=ms([r.cost for r in rf_l]),
        rf_landlord_hits=ms([r.hits for r in rf_l]),
        rf_landlord_downloads=ms([r.downloads for r in rf_l]),
        rf_stateless_lat=ms([r.latency for r in rf_s]),
        rf_stateless_cost=ms([r.cost for r in rf_s]),
        rf_stateless_hits=ms([r.hits for r in rf_s]),
        rf_stateless_downloads=ms([r.downloads for r in rf_s]),
        rf_landlord_vs_det_lat=paired_ci([r.latency for r in rf_l], [det.latency] * seeds),
        rf_landlord_vs_det_cost=paired_ci([r.cost for r in rf_l], [det.cost] * seeds),
        rf_landlord_vs_ll_lat=paired_ci([r.latency for r in rf_l], [ll.latency] * seeds),
        rf_landlord_vs_ll_cost=paired_ci([r.cost for r in rf_l], [ll.cost] * seeds),
        rf_stateless_vs_det_lat=paired_ci([r.latency for r in rf_s], [det.latency] * seeds),
        rf_stateless_vs_det_cost=paired_ci([r.cost for r in rf_s], [det.cost] * seeds),
        rf_stateless_vs_ll_lat=paired_ci([r.latency for r in rf_s], [ll.latency] * seeds),
        rf_stateless_vs_ll_cost=paired_ci([r.cost for r in rf_s], [ll.cost] * seeds),
    )
    rows.append(row)
    print(f"D={D:.0f} disk_x={disk_scale:.0e} M<l={frac_cross:.3f} "
          f"det={row['det_lat']:.0f}/{row['det_cost']:.0f} "
          f"ll={row['ll_lat']:.0f}/{row['ll_cost']:.0f} "
          f"r3={row['r3_lat'][0]:.0f}/{row['r3_cost'][0]:.0f} "
          f"rf_landlord={row['rf_landlord_lat'][0]:.0f}/{row['rf_landlord_cost'][0]:.0f} "
          f"rf_stateless={row['rf_stateless_lat'][0]:.0f}/{row['rf_stateless_cost'][0]:.0f}")

out = dict(config=dict(rows=50000, seeds=seeds, K=K, configs=CONFIGS),
           rows=rows,
           note=("disk_scale=1 is the raw Google disk column (all M<l).  "
                 "disk_scale=1e7 is a documented rescaling that makes M and l cross; "
                 "the fraction M<l is reported for every row."))
save(a, "exp27_full_google", out)

print("\n### full Google summary")
for r in rows:
    print(f"D={r['D']:.0f} disk_x={r['disk_scale']:.0e} M<l={r['frac_M_lt_l']:.3f}")
    for tag in ("rf_landlord", "rf_stateless"):
        lat = r[f"{tag}_vs_det_lat"]; cost = r[f"{tag}_vs_det_cost"]
        print(f"  {tag:14s} vs det lat {lat['mean']:+.0f} ±{lat['ci_half']:.0f} "
              f"cost {cost['mean']:+.0f} ±{cost['ci_half']:.0f}")