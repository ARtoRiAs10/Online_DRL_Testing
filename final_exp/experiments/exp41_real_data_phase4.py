"""Phase-4 (v2): real data, natural M/l crossing, multi-seed CIs.

Traces:
  * google.csv (50k / 4,619) -- main trace;
  * google_dataset2_30.csv (1k / 30) -- second real trace;
  * google_dataset2_paper.csv (1k / 6) -- paper recipe.

Regimes:
  * raw: D=1000, disk_unit_mb=1/1024 -- all M_n < l_n;
  * natural crossing: D=0.1, disk_unit_mb=1.0 -- the slow downlink makes M_n and l_n
    cross without an artificial disk multiplier (M<l ~ 0.65 on google, 0.067 on
    dataset2_30, 0.50 on dataset2_paper).

The 50k trace is run with 5 seeds (stateless repair, K=2) for tractability; the 10k and
dataset2 traces use 5-10 seeds with both repairs.
"""
from common import *
from sim.trace import load_trace_csv
from sim.algos import online_drl, ll_rc, rand_landlord, rand_drl

a = parse(__doc__, C=50, K=4)
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
BASE = dict(id_col="JobID", time_col="Timestamp", cpu_col="CPU", ram_col="RAM",
            disk_col="Disk", time_scale_ms=1e-3, U=100.0)
REGIMES = [
    ("raw", dict(D=1000.0, disk_unit_mb=1.0 / 1024.0, disk_scale=1.0)),
    ("natural_crossing", dict(D=0.1, disk_unit_mb=1.0, disk_scale=1.0)),
]
print(f"phase-4 v2 real data: regimes={REGIMES}")

rows = []


def run_trace(name, max_rows, seeds, both_repairs=True, K=4):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        print(f"missing {path}")
        return
    for rname, cfg in REGIMES:
        cat, tr = load_trace_csv(path, max_rows=max_rows, **BASE, **cfg)
        C = min(a.C, max(1, cat.N // 2))
        frac_lt = float(np.mean(cat.M < cat.l))
        frac_eq = float(np.mean(np.isclose(cat.M, cat.l)))
        frac_gt = float(np.mean(cat.M > cat.l))
        det = online_drl(cat, tr, C, theta=1.0)
        ll = ll_rc(cat, tr, C)
        r3 = [rand_landlord(cat, tr, C, np.random.default_rng(s)) for s in range(seeds)]
        rs = [rand_drl(cat, tr, C, np.random.default_rng(s), K=K,
                       repair_mode="stateless", gate_downloads=True,
                       download_time_mode="ema") for s in range(seeds)]
        rl = []
        if both_repairs:
            rl = [rand_drl(cat, tr, C, np.random.default_rng(s), K=K,
                           repair_mode="landlord", gate_downloads=True,
                           download_time_mode="ema") for s in range(seeds)]
        rows.append(dict(
            trace=name, regime=rname, seeds=seeds, K=K,
            n_services=int(cat.N), n_requests=int(len(tr)), C=int(C),
            frac_M_lt_l=frac_lt, frac_M_eq_l=frac_eq, frac_M_gt_l=frac_gt,
            M_min=float(cat.M.min()), M_max=float(cat.M.max()),
            l_min=float(cat.l.min()), l_max=float(cat.l.max()),
            det_lat=float(det.latency), det_cost=float(det.cost), det_hits=int(det.hits),
            ll_lat=float(ll.latency), ll_cost=float(ll.cost), ll_hits=int(ll.hits),
            r3_lat=ms([r.latency for r in r3]), r3_cost=ms([r.cost for r in r3]),
            r3_hits=ms([r.hits for r in r3]),
            stateless_lat=ms([r.latency for r in rs]),
            stateless_cost=ms([r.cost for r in rs]),
            stateless_hits=ms([r.hits for r in rs]),
            stateless_downloads=ms([r.downloads for r in rs]),
            stateless_vs_det_lat=paired_ci([r.latency for r in rs], [det.latency] * seeds),
            stateless_vs_det_cost=paired_ci([r.cost for r in rs], [det.cost] * seeds),
            stateless_vs_ll_lat=paired_ci([r.latency for r in rs], [ll.latency] * seeds),
            stateless_vs_ll_cost=paired_ci([r.cost for r in rs], [ll.cost] * seeds),
        ))
        if both_repairs:
            rows[-1].update(
                landlord_lat=ms([r.latency for r in rl]),
                landlord_cost=ms([r.cost for r in rl]),
                landlord_hits=ms([r.hits for r in rl]),
                landlord_downloads=ms([r.downloads for r in rl]),
                landlord_vs_det_lat=paired_ci([r.latency for r in rl], [det.latency] * seeds),
                landlord_vs_det_cost=paired_ci([r.cost for r in rl], [det.cost] * seeds),
                landlord_vs_ll_lat=paired_ci([r.latency for r in rl], [ll.latency] * seeds),
                landlord_vs_ll_cost=paired_ci([r.cost for r in rl], [ll.cost] * seeds),
            )
        print(f"{name} {rname}: M<l={frac_lt:.3f} det={det.latency:.0f}/{det.cost:.0f} "
              f"stateless={rows[-1]['stateless_lat'][0]:.0f}/{rows[-1]['stateless_cost'][0]:.0f}")


run_trace("google_dataset2_30.csv", None, 10, both_repairs=True, K=4)
run_trace("google_dataset2_paper.csv", None, 10, both_repairs=True, K=4)
run_trace("google.csv", 10000, 5, both_repairs=False, K=4)
run_trace("google.csv", 50000, 3, both_repairs=False, K=2)

out = dict(
    config=dict(regimes=REGIMES, K4=4, K50=2),
    rows=rows,
    note=("Natural crossing uses D=0.1 and treats the Google disk column as MB; the raw "
          "regime uses D=1000 and disk_unit_mb=1/1024.  The 50k trace uses 5 seeds and "
          "the stateless repair (K=2) for tractability."),
)
save(a, "exp41_real_data_phase4", out)

print("\n### summary")
for r in rows:
    print(f"\n{r['trace']} {r['regime']} M<l={r['frac_M_lt_l']:.3f} "
          f"M=l={r['frac_M_eq_l']:.3f} M>l={r['frac_M_gt_l']:.3f}")
    print(f"  Online-DRL  lat={r['det_lat']:.0f} cost={r['det_cost']:.0f} hits={r['det_hits']}")
    print(f"  LL-RC       lat={r['ll_lat']:.0f} cost={r['ll_cost']:.0f} hits={r['ll_hits']}")
    s = r['stateless_vs_det_lat']; c = r['stateless_vs_det_cost']
    print(f"  stateless   lat={r['stateless_lat'][0]:.0f}±{r['stateless_lat'][1]:.0f} "
          f"cost={r['stateless_cost'][0]:.0f}±{r['stateless_cost'][1]:.0f} "
          f"vs det lat {s['mean']:+.0f}±{s['ci_half']:.0f} cost {c['mean']:+.0f}±{c['ci_half']:.0f}")
    if 'landlord_lat' in r:
        s = r['landlord_vs_det_lat']; c = r['landlord_vs_det_cost']
        print(f"  landlord    lat={r['landlord_lat'][0]:.0f}±{r['landlord_lat'][1]:.0f} "
              f"cost={r['landlord_cost'][0]:.0f}±{r['landlord_cost'][1]:.0f} "
              f"vs det lat {s['mean']:+.0f}±{s['ci_half']:.0f} cost {c['mean']:+.0f}±{c['ci_half']:.0f}")