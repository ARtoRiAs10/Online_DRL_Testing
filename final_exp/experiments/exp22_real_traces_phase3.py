"""Phase-3 real-trace experiment: cross ``M_n`` and ``l_n``, and use two traces.

For each bundled Google trace and several downlink values ``D`` we report

* the fraction of services with ``M_n < l_n`` (download cheaper than forwarding);
* Online-DRL, LL-RC, R3-only and randomized DRL latency/cost;
* multiple seeds and confidence intervals for the randomized variants.

The bundled traces are ``google.csv`` (a 5 000-row slice by default) and
``google_dataset2_30.csv``.  A second real trace is therefore included; SHARCNet /
Azure / Alibaba are not reachable from this workspace and remain open items.
"""
from common import *
from sim.trace import load_trace_csv
from sim.algos import online_drl, ll_rc, rand_landlord, rand_drl

a = parse(__doc__, C=50, K=4)
extra = argparse.ArgumentParser(add_help=False)
extra.add_argument("--rows", type=int, default=5000)
opt, _unknown = extra.parse_known_args()
seeds = min(a.seeds, 5) if not a.quick else a.seeds
CONFIGS = ([(1000.0, 1.0), (1000.0, 1e6), (1000.0, 1e7),
             (1000.0, 1e8), (100.0, 1e7)]
           if not a.quick else [(1000.0, 1.0), (1000.0, 1e6), (1000.0, 1e7)])

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
TRACES = [
    ("google.csv", opt.rows,
     dict(id_col="JobID", time_col="Timestamp", cpu_col="CPU", ram_col="RAM",
          disk_col="Disk", time_scale_ms=1e-3, disk_unit_mb=1.0 / 1024.0,
          U=100.0)),
    ("google_dataset2_30.csv", None,
     dict(id_col="JobID", time_col="Timestamp", cpu_col="CPU", ram_col="RAM",
          disk_col="Disk", time_scale_ms=1e-3, disk_unit_mb=1.0 / 1024.0,
          U=100.0)),
]

rows = []
for name, max_rows, cfg in TRACES:
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        print(f"missing {path}, skipping")
        continue
    for D, disk_scale in CONFIGS:
        cat, tr = load_trace_csv(path, max_rows=max_rows, D=D,
                                 disk_scale=disk_scale, **cfg)
        frac_cross = float(np.mean(cat.M < cat.l))
        C = min(a.C, max(1, cat.N // 2))
        det = online_drl(cat, tr, C)
        ll = ll_rc(cat, tr, C)
        r3 = [rand_landlord(cat, tr, C, np.random.default_rng(s)) for s in range(seeds)]
        rf = [rand_drl(cat, tr, C, np.random.default_rng(s), K=a.K,
                       repair_mode="landlord", gate_downloads=True) for s in range(seeds)]
        row = dict(
            trace=name, D=D, disk_scale=disk_scale, n_services=int(cat.N), n_requests=int(len(tr)), C=int(C),
            repair_mode="landlord",
            frac_M_lt_l=frac_cross,
            M_min=float(cat.M.min()), M_max=float(cat.M.max()),
            l_min=float(cat.l.min()), l_max=float(cat.l.max()),
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
        print(f"{name:24s} D={D:6.0f} disk_x={disk_scale:5.0f} M<=l={frac_cross:.3f} "
              f"det={row['det_lat']:.0f}/{row['det_cost']:.0f} "
              f"rf={row['rf_lat'][0]:.0f}/{row['rf_cost'][0]:.0f} "
              f"hits={row['rf_hits'][0]:.0f}")

out = dict(config=dict(seeds=seeds, configs=CONFIGS, traces=[t[0] for t in TRACES]),
           rows=rows)
save(a, "exp22_real_traces_phase3", out)

print("\n### summary")
for r in rows:
    print(f"{r['trace']:24s} D={r['D']:6.0f} disk_x={r['disk_scale']:5.0f} "
          f"M<l={r['frac_M_lt_l']:.3f} "
          f"RF vs det lat {r['rf_vs_det_lat']['mean']:+.0f} "
          f"±{r['rf_vs_det_lat']['ci_half']:.0f} "
          f"cost {r['rf_vs_det_cost']['mean']:+.0f} "
          f"±{r['rf_vs_det_cost']['ci_half']:.0f}")