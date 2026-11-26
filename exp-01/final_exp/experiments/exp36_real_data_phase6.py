"""Phase-6 real-data experiment: two traces, M/l crossing, multi-seed CIs.

Traces (bundled Google cluster traces 2011, prepared with the paper's recipe):

* ``google.csv``               -- 50,000 requests / 4,619 services (main trace);
* ``google_dataset2_30.csv``   -- 1,000 requests / 30 services (paper-like dataset2);
* ``google_dataset2_paper.csv``-- 1,000 requests / 6 services (paper recipe).

SHARCNet is not reachable from this workspace; the two Google traces are the second
real trace.  For every trace we run two regimes:

* ``disk_scale=1``    -- the raw Google disk column; all ``M_n < l_n``;
* ``disk_scale=1e7``  -- a documented rescaling that makes ``M_n`` and ``l_n`` cross.

The fraction of services with ``M_n < l_n``, ``M_n = l_n`` and ``M_n > l_n`` is reported
for every row.  Every randomized number is a mean over seeds with a sample standard
deviation and a paired 95% confidence interval against Online-DRL and LL-RC.
"""
from common import *
from sim.trace import load_trace_csv
from sim.algos import online_drl, ll_rc, rand_landlord, rand_drl

a = parse(__doc__, C=50, K=4)
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CFG = dict(id_col="JobID", time_col="Timestamp", cpu_col="CPU", ram_col="RAM",
           disk_col="Disk", time_scale_ms=1e-3, disk_unit_mb=1.0 / 1024.0,
           U=100.0)
CONFIGS = [(1000.0, 1.0), (1000.0, 1e7)]
TRACES = [
    ("google.csv", 10000, 5),
    ("google_dataset2_30.csv", None, 10),
    ("google_dataset2_paper.csv", None, 10),
]
print(f"phase-6 real data: configs={CONFIGS}")

rows = []
for name, max_rows, seeds in TRACES:
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        print(f"missing {path}, skipping")
        continue
    for D, disk_scale in CONFIGS:
        cat, tr = load_trace_csv(path, max_rows=max_rows, D=D,
                                 disk_scale=disk_scale, **CFG)
        C = min(a.C, max(1, cat.N // 2))
        frac_lt = float(np.mean(cat.M < cat.l))
        frac_eq = float(np.mean(np.isclose(cat.M, cat.l)))
        frac_gt = float(np.mean(cat.M > cat.l))
        det = online_drl(cat, tr, C)
        ll = ll_rc(cat, tr, C)
        r3 = [rand_landlord(cat, tr, C, np.random.default_rng(s)) for s in range(seeds)]
        rf_l = [rand_drl(cat, tr, C, np.random.default_rng(s), K=a.K,
                         repair_mode="landlord", gate_downloads=True,
                         download_time_mode="ema") for s in range(seeds)]
        rf_s = [rand_drl(cat, tr, C, np.random.default_rng(s), K=a.K,
                         repair_mode="stateless", gate_downloads=True,
                         download_time_mode="ema") for s in range(seeds)]
        row = dict(
            trace=name, D=D, disk_scale=disk_scale, seeds=seeds,
            n_services=int(cat.N), n_requests=int(len(tr)), C=int(C),
            frac_M_lt_l=frac_lt, frac_M_eq_l=frac_eq, frac_M_gt_l=frac_gt,
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
        print(f"{name:26s} D={D:6.0f} disk_x={disk_scale:7.0e} "
              f"M<l={frac_lt:.3f} M=l={frac_eq:.3f} M>l={frac_gt:.3f} "
              f"det={det.latency:.0f}/{det.cost:.0f} "
              f"ll={ll.latency:.0f}/{ll.cost:.0f} "
              f"rf_l={row['rf_landlord_lat'][0]:.0f}/{row['rf_landlord_cost'][0]:.0f} "
              f"rf_s={row['rf_stateless_lat'][0]:.0f}/{row['rf_stateless_cost'][0]:.0f}")

out = dict(
    config=dict(configs=CONFIGS, traces=[t[0] for t in TRACES], C=a.C, K=a.K),
    rows=rows,
    note=("SHARCNet is not reachable; the two Google traces are the second real trace.  "
          "The 10,000-row slice of google.csv keeps the multi-seed run tractable; the full "
          "50,000-row trace is exp27.  disk_scale=1 is the raw disk column (all M<l); "
          "disk_scale=1e7 is a documented rescaling that makes M and l cross.  The fraction "
          "M<l is reported for every row."),
)
save(a, "exp36_real_data_phase6", out)

print("\n### summary")
for r in rows:
    print(f"\n{r['trace']} D={r['D']:.0f} disk_x={r['disk_scale']:.0e} "
          f"M<l={r['frac_M_lt_l']:.3f} M=l={r['frac_M_eq_l']:.3f} "
          f"M>l={r['frac_M_gt_l']:.3f}")
    print(f"  Online-DRL  lat={r['det_lat']:.0f} cost={r['det_cost']:.0f} hits={r['det_hits']}")
    print(f"  LL-RC       lat={r['ll_lat']:.0f} cost={r['ll_cost']:.0f} hits={r['ll_hits']}")
    for tag in ("rf_landlord", "rf_stateless"):
        lat = r[f"{tag}_vs_det_lat"]; cost = r[f"{tag}_vs_det_cost"]
        print(f"  {tag:12s} lat={r[f'{tag}_lat'][0]:.0f}±{r[f'{tag}_lat'][1]:.0f} "
              f"cost={r[f'{tag}_cost'][0]:.0f}±{r[f'{tag}_cost'][1]:.0f} "
              f"vs det lat {lat['mean']:+.0f}±{lat['ci_half']:.0f} "
              f"cost {cost['mean']:+.0f}±{cost['ci_half']:.0f}")