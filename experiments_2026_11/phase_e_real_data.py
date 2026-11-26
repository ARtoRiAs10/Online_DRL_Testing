"""Phase E: real-data expansion, natural M/l crossing, multi-seed CIs.

Outputs results/phase_e_real_data.json.
"""
from common import *
from sim.algos import online_drl, ll_rc, rand_landlord

REGIMES = [
    ("raw", dict(D=1000.0, disk_unit_mb=1.0/1024.0, disk_scale=1.0)),
    ("natural_crossing", dict(D=0.1, disk_unit_mb=1.0, disk_scale=1.0)),
]
rows = []
print("Phase E: real traces")


def run_trace(name, max_rows, seeds, K):
    path = os.path.join(FINAL, "data", name)
    if not os.path.exists(path):
        print("missing", path); return
    for rname, cfg in REGIMES:
        cat, tr = load_trace_csv(path, max_rows=max_rows, id_col="JobID",
                                 time_col="Timestamp", cpu_col="CPU", ram_col="RAM",
                                 disk_col="Disk", time_scale_ms=1e-3, U=100.0, **cfg)
        C = min(12, max(1, cat.N // 2))
        fl = float(np.mean(cat.M < cat.l))
        det = online_drl(cat, tr, C, theta=1.0)
        ll = ll_rc(cat, tr, C)
        r3 = [rand_landlord(cat, tr, C, np.random.default_rng(s)) for s in range(seeds)]
        rs = [rand_drl(cat, tr, C, np.random.default_rng(s), K=K,
                       repair_mode="stateless", gate_downloads=True,
                       download_time_mode="ema") for s in range(seeds)]
        rl = [rand_drl(cat, tr, C, np.random.default_rng(s), K=K,
                       repair_mode="landlord", gate_downloads=True,
                       download_time_mode="ema") for s in range(seeds)]
        rows.append(dict(trace=name, regime=rname, seeds=seeds, K=K,
                         n_services=int(cat.N), n_requests=int(len(tr)), C=int(C),
                         frac_M_lt_l=fl,
                         frac_M_eq_l=float(np.mean(np.isclose(cat.M, cat.l))),
                         frac_M_gt_l=float(np.mean(cat.M > cat.l)),
                         M_min=float(cat.M.min()), M_max=float(cat.M.max()),
                         l_min=float(cat.l.min()), l_max=float(cat.l.max()),
                         det_lat=float(det.latency), det_cost=float(det.cost),
                         det_hits=int(det.hits),
                         ll_lat=float(ll.latency), ll_cost=float(ll.cost),
                         r3_lat=ms([r.latency for r in r3]),
                         r3_cost=ms([r.cost for r in r3]),
                         stateless_lat=ms([r.latency for r in rs]),
                         stateless_cost=ms([r.cost for r in rs]),
                         stateless_hits=ms([r.hits for r in rs]),
                         stateless_vs_det_lat=paired_ci([r.latency for r in rs], [det.latency]*seeds),
                         stateless_vs_det_cost=paired_ci([r.cost for r in rs], [det.cost]*seeds),
                         landlord_lat=ms([r.latency for r in rl]),
                         landlord_cost=ms([r.cost for r in rl]),
                         landlord_hits=ms([r.hits for r in rl]),
                         landlord_vs_det_lat=paired_ci([r.latency for r in rl], [det.latency]*seeds),
                         landlord_vs_det_cost=paired_ci([r.cost for r in rl], [det.cost]*seeds)))
        print(f"{name} {rname}: M<l={fl:.3f} det={det.latency:.0f}/{det.cost:.0f} "
              f"stateless={rows[-1]['stateless_lat'][0]:.0f}/{rows[-1]['stateless_cost'][0]:.0f} "
              f"landlord={rows[-1]['landlord_lat'][0]:.0f}/{rows[-1]['landlord_cost'][0]:.0f}", flush=True)


run_trace("google_dataset2_30.csv", None, 5, 4)
run_trace("google_dataset2_paper.csv", None, 5, 4)
run_trace("google.csv", 10000, 3, 4)

out = dict(config=dict(regimes=REGIMES, seeds=dict(dataset2=5, google10k=3)),
           rows=rows,
           note=("Two Google cluster trace families are used: the 50k/10k main trace and "
                 "the dataset2 30-service / paper recipes.  The natural-crossing regime "
                 "uses D=0.1 and treats the disk column as MB, so M and l cross without "
                 "an artificial disk multiplier."))
save("phase_e_real_data", out)

print("\n### summary")
for r in rows:
    print(f"\n{r['trace']} {r['regime']} M<l={r['frac_M_lt_l']:.3f} "
          f"M=l={r['frac_M_eq_l']:.3f} M>l={r['frac_M_gt_l']:.3f}")
    print(f"  Online-DRL lat={r['det_lat']:.0f} cost={r['det_cost']:.0f} hits={r['det_hits']}")
    print(f"  LL-RC      lat={r['ll_lat']:.0f} cost={r['ll_cost']:.0f}")
    s=r['stateless_vs_det_lat']; c=r['stateless_vs_det_cost']
    print(f"  stateless  lat={r['stateless_lat'][0]:.0f}±{r['stateless_lat'][1]:.0f} "
          f"cost={r['stateless_cost'][0]:.0f}±{r['stateless_cost'][1]:.0f} "
          f"Δlat={s['mean']:+.0f}±{s['ci_half']:.0f} Δcost={c['mean']:+.0f}±{c['ci_half']:.0f}")
    s=r['landlord_vs_det_lat']; c=r['landlord_vs_det_cost']
    print(f"  landlord   lat={r['landlord_lat'][0]:.0f}±{r['landlord_lat'][1]:.0f} "
          f"cost={r['landlord_cost'][0]:.0f}±{r['landlord_cost'][1]:.0f} "
          f"Δlat={s['mean']:+.0f}±{s['ci_half']:.0f} Δcost={c['mean']:+.0f}±{c['ci_half']:.0f}")
