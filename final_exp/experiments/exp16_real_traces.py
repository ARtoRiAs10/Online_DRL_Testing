"""Experiment 16 -- the paper's real traces.

The original paper evaluates Online-DRL on the Google cluster traces 2011 (dataset2) and
on SHARCNet.  This workspace contains the Google traces prepared with
``data/prepare_google.py``:

  * ``data/google_dataset2_paper.csv`` -- the paper's dataset2 recipe (1000 requests,
    30 services), regenerated from clusterdata-2011-2;
  * ``data/google_dataset2.csv``        -- a small 1000-row slice;
  * ``data/google.csv``                 -- 50 000 requests / 4 619 services.

For every trace the script runs Online-DRL, LL-RC, R3-only and the updated
Landlord-Repair randomized DRL, and reports the paper's metrics (total latency, total
cost, hits, delayed hits, and the latency/cost difference with respect to LL-RC).
"""
from common import *
from sim.trace import load_trace_csv
from sim.algos import online_drl, ll_rc, rand_landlord, rand_drl
from sim.metrics import pct
from paper_claims import PAPER_EXP01_LATENCY_PCT, PAPER_EXP01_COST_PCT, PAPER_REAL_TRACES, in_range

a = parse(__doc__, C=50)
extra = argparse.ArgumentParser(add_help=False)
extra.add_argument("--csv", default=None, help="run only this preprocessed CSV")
extra.add_argument("--kind", default="google", choices=["google", "sharcnet", "generic"])
extra.add_argument("--rows", type=int, default=50000)
opt, _unknown = extra.parse_known_args()

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
candidates = []
if opt.csv:
    candidates = [opt.csv]
else:
    for name in ("google_dataset2_30.csv", "google_dataset2_paper.csv", "google_dataset2.csv",
                 "google.csv", "sharcnet.csv"):
        p = os.path.join(DATA, name)
        if os.path.exists(p):
            candidates.append(p)
if not candidates:
    msg = dict(status="NO REAL TRACE AVAILABLE", paper=PAPER_REAL_TRACES)
    save(a, "exp16_real_traces", msg)
    print(json.dumps(msg, indent=2))
    raise SystemExit(0)

KIND = {
    "google": dict(id_col="JobID", time_col="Timestamp", cpu_col="CPU", ram_col="RAM",
                   disk_col="Disk", time_scale_ms=1e-3, disk_unit_mb=1.0 / 1024.0,
                   U=100.0, D=1000.0),
    "sharcnet": dict(id_col="ExecutableID", time_col="SubmitTime", cpu_col="UsedCPUTime",
                     ram_col="UsedMemory", disk_col="UsedLocalDiskSpace", time_scale_ms=1000.0,
                     disk_unit_mb=1.0, U=200.0, D=2000.0),
    "generic": dict(id_col="service", time_col="time", cpu_col="cpu", ram_col="ram",
                    disk_col="disk", time_scale_ms=1.0, disk_unit_mb=1.0, U=100.0, D=1000.0),
}

all_rows = []
for path in candidates:
    if not os.path.exists(path):
        continue
    print("=" * 78, f"\n### real trace: {path}")
    cfg = KIND[opt.kind]
    cat, tr = load_trace_csv(path, max_rows=opt.rows, **cfg)
    C = min(a.C, max(1, cat.N // 2)) if cat.N > 1 else 1
    print(f"services={cat.N} requests={len(tr)} C={C} "
          f"l=[{cat.l.min():.1f},{cat.l.max():.1f}] ms M=[{cat.M.min():.1f},{cat.M.max():.1f}] ms")

    res = {
        "Online-DRL": online_drl(cat, tr, C),
        "LL-RC": ll_rc(cat, tr, C),
        "R3 only": rand_landlord(cat, tr, C, np.random.default_rng(0)),
        "Randomized DRL (updated landlord)": rand_drl(cat, tr, C, np.random.default_rng(0), K=4),
    }
    base = res["LL-RC"]
    rows = []
    for name, R in res.items():
        row = dict(algo=name, latency=float(R.latency), cost=float(R.cost), hits=R.hits,
                   delayed_hits=R.delayed_hits, forwards=R.forwards, downloads=R.downloads,
                   lat_vs_llrc_pct=pct(R.latency, base.latency),
                   cost_vs_llrc_pct=pct(R.cost, base.cost))
        rows.append(row)
        print(row)
    lat = next(r["lat_vs_llrc_pct"] for r in rows if r["algo"] == "Online-DRL")
    cost = next(r["cost_vs_llrc_pct"] for r in rows if r["algo"] == "Online-DRL")
    print(f"Online-DRL vs LL-RC: latency {lat:+.3f} % (paper {PAPER_EXP01_LATENCY_PCT}), "
          f"cost {cost:+.2f} % (paper {PAPER_EXP01_COST_PCT})")
    print(f"  latency in paper range? {in_range(lat, PAPER_EXP01_LATENCY_PCT)}; "
          f"cost in paper range? {in_range(cost, PAPER_EXP01_COST_PCT)}")
    all_rows.append(dict(trace=os.path.basename(path), kind=opt.kind, n_services=cat.N,
                         n_requests=len(tr), C=C, rows=rows,
                         online_drl_lat_pct=lat, online_drl_cost_pct=cost,
                         latency_in_paper_range=in_range(lat, PAPER_EXP01_LATENCY_PCT),
                         cost_in_paper_range=in_range(cost, PAPER_EXP01_COST_PCT)))

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    names = [r["algo"] for r in rows]
    colors = ["#1f3b73", "#8ea9d0", "#c58b2a", "#1b6b3a"]
    ax[0].bar(names, [r["latency"] for r in rows], color=colors)
    ax[0].set_title("total latency"); ax[0].tick_params(axis="x", labelsize=7, rotation=20)
    ax[1].bar(names, [r["cost"] for r in rows], color=colors)
    ax[1].set_title("total cost"); ax[1].tick_params(axis="x", labelsize=7, rotation=20)
    ax[2].bar(names, [r["hits"] for r in rows], color=colors)
    ax[2].set_title("hits"); ax[2].tick_params(axis="x", labelsize=7, rotation=20)
    for x in ax:
        x.grid(alpha=.3, axis="y")
    save(a, "exp16_real_traces_" + os.path.splitext(os.path.basename(path))[0],
         dict(trace=os.path.basename(path), kind=opt.kind, rows=rows,
              n_services=cat.N, n_requests=len(tr), C=C,
              paper=dict(latency_pct=PAPER_EXP01_LATENCY_PCT, cost_pct=PAPER_EXP01_COST_PCT)),
         fig)

save(a, "exp16_real_traces", dict(paper=PAPER_REAL_TRACES, traces=all_rows))
print("\n### summary")
for t in all_rows:
    print(f"{t['trace']:28s} Online-DRL vs LL-RC: latency {t['online_drl_lat_pct']:+.3f} % "
          f"cost {t['online_drl_cost_pct']:+.2f} %  "
          f"(paper ranges: {t['latency_in_paper_range']}/{t['cost_in_paper_range']})")