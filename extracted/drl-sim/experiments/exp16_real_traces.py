"""Experiment on the REAL traces used by the paper (Google cluster traces / SHARCNet).

The paper evaluates Online-DRL on
  * the Google cluster traces (2011), preprocessed into "dataset0".."dataset10" (we use dataset2),
  * the SHARCNet (Grid Workloads Archive) trace,
  * and a synthetic dataset built FROM the unique services of Google dataset2 with Zipf
    popularity and Poisson arrivals.

Those traces are NOT bundled with this repository (they are large and must be downloaded), so
every other experiment in this suite runs on synthetic traffic.  This script closes the loop:

  * if a preprocessed CSV is found (data/google.csv, data/sharcnet.csv or any file given on the
    command line), it runs Online-DRL, LL-RC, R3-only and Randomized DRL on it and reports the
    paper's metrics: total latency, total cost, hits, delayed hits, and the latency/cost
    difference of Online-DRL and of Randomized DRL with respect to LL-RC;
  * if no CSV is present it writes a JSON file that states the gap explicitly, so the experiment
    suite always documents which claims were tested on which data.

Usage
    python experiments/exp16_real_traces.py                      # auto-detect data/*.csv
    python experiments/exp16_real_traces.py --csv data/sharcnet.csv --kind sharcnet
"""
from common import *
from sim.trace import load_trace_csv
from sim.algos import online_drl, ll_rc, rand_landlord, rand_drl
from sim.metrics import pct

a = parse(__doc__, C=50)
ap_extra = a  # keep the namespace of common.parse

# ---------------------------------------------------------------- trace configuration
# The column names follow the preprocessing described in Sec. IV of the paper
# (data/README.md has the same recipe).
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
import argparse
extra = argparse.ArgumentParser(add_help=False)
extra.add_argument("--csv", default=None, help="preprocessed CSV (one row per request)")
extra.add_argument("--kind", default="generic", choices=list(KIND))
extra.add_argument("--rows", type=int, default=50000, help="max rows to load")
extra.add_argument("--sep", default=",")
opt, _unknown = extra.parse_known_args()

candidates = [opt.csv] if opt.csv else sorted(
    [os.path.join(a.out, "..", "data", f) for f in os.listdir(os.path.join(os.path.dirname(a.out), "..", "data"))
     if f.endswith(".csv")] if os.path.isdir(os.path.join(os.path.dirname(a.out), "..", "data")) else [])
candidates = [c for c in candidates if c and os.path.exists(c)]

if not candidates:
    msg = dict(status="NO REAL TRACE AVAILABLE",
               what_was_needed="Google cluster traces 2011 (dataset2) and/or SHARCNet, "
                               "preprocessed as described in Sec. IV of the paper",
               how_to_close_the_gap=[
                   "1. download the Google cluster traces 2011 (clusterdata-2011-2) and/or the "
                   "SHARCNet trace from the Grid Workloads Archive",
                   "2. preprocess them into a CSV with one row per request and the columns "
                   "JobID/service id, Timestamp, CPU, RAM, Disk (zeros -> column median; the "
                   "resources of a service are taken from its first occurrence; the disk size is "
                   "in KB for Google and in MB for SHARCNet)",
                   "3. put the file in drl-sim/data/ (e.g. data/google.csv) and run "
                   "python experiments/exp16_real_traces.py",
               ],
               claims_tested_only_on_synthetic_data=[
                   "G1 feasibility of the repair step", "G2 expected download budget",
                   "G3 surrogate regret", "G4 expected competitive ratio vs the MILP",
                   "G5 safety of the slot reductions",
                   "the removal of the linear-in-k blow-up by R1+R2+R3"])
    save(a, "exp16_real_traces", msg)
    print("### no real trace found in data/ -> wrote the gap report to", a.out)
    for c in msg["claims_tested_only_on_synthetic_data"]:
        print("   untested on the paper's data:", c)
    raise SystemExit(0)

path = candidates[0]
cfg = KIND[opt.kind]
print(f"### real trace: {path} (kind={opt.kind})")
cat, tr = load_trace_csv(path, max_rows=opt.rows, sep=opt.sep,
                          **{k: v for k, v in cfg.items() if k not in ("U", "D")},
                          U=cfg["U"], D=cfg["D"])
print(f"services={cat.N}  requests={len(tr)}  l=[{cat.l.min():.1f},{cat.l.max():.1f}] ms  "
      f"M=[{cat.M.min():.1f},{cat.M.max():.1f}] ms")

res = {}
res["Online-DRL"] = online_drl(cat, tr, a.C)
res["LL-RC"] = ll_rc(cat, tr, a.C)
res["R3 only"] = rand_landlord(cat, tr, a.C, np.random.default_rng(0))
res["Randomized DRL"] = rand_drl(cat, tr, a.C, np.random.default_rng(0), K=4)

rows = []
for name, R in res.items():
    rows.append(dict(algo=name, latency=R.latency, cost=R.cost, hits=R.hits,
                     delayed_hits=R.delayed_hits, forwards=R.forwards, downloads=R.downloads,
                     lat_vs_llrc_pct=pct(R.latency, res["LL-RC"].latency),
                     cost_vs_llrc_pct=pct(R.cost, res["LL-RC"].cost)))
    print(rows[-1])

fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
names = list(res)
ax[0].bar(names, [r["latency"] for r in rows], color=["#1f3b73", "#8ea9d0", "#c58b2a", "#1b6b3a"])
ax[0].set_title("total latency"); ax[0].tick_params(axis="x", labelsize=7, rotation=20)
ax[1].bar(names, [r["cost"] for r in rows], color=["#1f3b73", "#8ea9d0", "#c58b2a", "#1b6b3a"])
ax[1].set_title("total cost"); ax[1].tick_params(axis="x", labelsize=7, rotation=20)
ax[2].bar(names, [r["hits"] for r in rows], color=["#1f3b73", "#8ea9d0", "#c58b2a", "#1b6b3a"])
ax[2].set_title("hits"); ax[2].tick_params(axis="x", labelsize=7, rotation=20)
for x in ax: x.grid(alpha=.3, axis="y")
save(a, "exp16_real_traces", dict(trace=os.path.basename(path), kind=opt.kind, rows=rows,
                                  n_services=cat.N, n_requests=len(tr), C=a.C), fig)
