"""Export a single human-readable file with every experiment result and the dataset
used for each experiment.

Run from final_exp/:

    python experiments/export_results_and_datasets.py

It reads results/*.json + docs/REPORT.md and writes RESULTS_AND_DATASETS.md.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import statistics
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPORT = os.path.join(ROOT, "docs", "REPORT.md")
OUT = os.path.join(ROOT, "RESULTS_AND_DATASETS.md")


def csv_stats(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    svc = {r["JobID"] for r in rows}
    out = {"file": os.path.basename(path), "rows": n, "services": len(svc)}
    out["missing"] = sum(1 for r in rows if any(r.get(c, "") in ("", None) for c in ("CPU", "RAM", "Disk")))
    for col in ("CPU", "RAM", "Disk"):
        vals = []
        for r in rows:
            v = r.get(col, "")
            if v not in ("", None):
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
        out[col] = (min(vals), statistics.mean(vals), statistics.median(vals), max(vals))
    t = [float(r["Timestamp"]) for r in rows]
    out["t0"] = min(t)
    out["t1"] = max(t)
    out["span_s"] = (max(t) - min(t)) / 1e6
    return out


def fmt(x):
    if x == 0:
        return "0"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 1:
        return f"{x:.4g}"
    return f"{x:.3g}"


def dataset_section():
    files = ["google.csv", "google_dataset2_30.csv", "google_dataset2_paper.csv",
             "google_dataset2.csv"]
    stats = []
    for name in files:
        p = os.path.join(ROOT, "data", name)
        if os.path.exists(p):
            stats.append(csv_stats(p))
    lines = []
    lines.append("## 2. Datasets used by the experiments")
    lines.append("")
    lines.append("The simulator is in `final_exp/`.  Every experiment reads either a **synthetic "
                 "Zipf/Poisson world**, the **adversarial Large Sequence** from Theorem 1, or one "
                 "of the **real Google CSVs** in `final_exp/data/`.  Only exp16 uses the real "
                 "traces; all other experiments use synthetic or adversarial data, exactly as "
                 "noted in the table below.")
    lines.append("")
    lines.append("### 2.1 Real traces: Google cluster traces 2011")
    lines.append("")
    lines.append("Source: `clusterdata-2011-2` (`task_events`), prepared by "
                 "`final_exp/data/prepare_google.py` with the paper's recipe:")
    lines.append("")
    lines.append("* keep only `SUBMIT` events (event type 0);")
    lines.append("* one request per `(JobID, distinct Timestamp)`;")
    lines.append("* a service's CPU/RAM/disk is the max over the tasks of that job (first occurrence in the loader);")
    lines.append("* zeros/missing CPU/RAM/disk are replaced by the column median;")
    lines.append("* timestamps are converted from microseconds to milliseconds;")
    lines.append("* the loader treats the disk column as **KB** and converts it to MB (`disk_unit_mb=1/1024`).")
    lines.append("")
    lines.append("| file | requests | unique services | CPU min / mean / median / max | RAM min / mean / median / max | Disk min / mean / median / max (CSV units) | time span | used by |")
    lines.append("|---|---:|---:|---|---|---|---|---|")
    used = {
        "google.csv": "exp16 (main real-trace check, 50 000 requests)",
        "google_dataset2_30.csv": "exp16 (paper-like dataset2, 30 services)",
        "google_dataset2_paper.csv": "exp16 (1 000-row slice; actual file has 6 services)",
        "google_dataset2.csv": "exp16 (small 1 000-row slice; identical to google_dataset2_paper.csv)",
    }
    for s in stats:
        cpu = " / ".join(fmt(v) for v in s["CPU"])
        ram = " / ".join(fmt(v) for v in s["RAM"])
        disk = " / ".join(fmt(v) for v in s["Disk"])
        lines.append(f"| `{s['file']}` | {s['rows']:,} | {s['services']:,} | {cpu} | {ram} | {disk} | "
                     f"{s['span_s']/60:.1f} min | {used.get(s['file'], '')} |")
    lines.append("")
    lines.append("Notes on the real traces:")
    lines.append("")
    google_missing = next((s["missing"] for s in stats if s["file"] == "google.csv"), 0)
    lines.append(f"* `google.csv` is the largest bundled trace: 50 000 requests over 4 619 services "
                 f"(about 70 minutes of trace time).  It is the only file large enough for the "
                 f"50-slot capacity used in exp16.  It contains {google_missing} rows with blank "
                 f"CPU/RAM/disk fields; the loader replaces those by the column median, exactly "
                 f"as the paper's recipe does.")
    lines.append("* `google_dataset2_30.csv` is the paper-like `dataset2`: 1 000 requests, 30 unique "
                 "services.  The services are mostly grouped together, as the paper notes.")
    lines.append("* `google_dataset2_paper.csv` and `google_dataset2.csv` are both 1 000-row slices "
                 "with **6** unique services (despite the `_paper` name in the first file).  "
                 "The 30-service version is the closer match to the paper's `dataset2` description.")
    lines.append("* With the paper-style bandwidth parameters in `sim/catalog.py`, `M_n < l_n` for "
                 "every service in these traces.  Therefore the `theta=1` (Online-DRL) and "
                 "`theta=0` (LL-RC) triggers fire on the same requests, and Online-DRL and LL-RC "
                 "coincide in exp16.  The paper's exact latency/cost split needs its exact "
                 "bandwidth/`l_n` parameters, which are not recoverable from the damaged PDF.")
    lines.append("")
    lines.append("### 2.2 Synthetic Zipf/Poisson world")
    lines.append("")
    lines.append("Generated by `sim.trace.synthetic_world` (no CSV).  Default full-run parameters:")
    lines.append("")
    lines.append("| property | value |")
    lines.append("|---|---|")
    lines.append("| catalogue size `N` | 40 services (exp10/13 use 10-16; exp14 sweeps 20-320) |")
    lines.append("| trace length `T` | 8 000 requests (exp09/17 use the same default; exp13 family 2 uses 1 200) |")
    lines.append("| popularity | Zipf with exponent `rho = 1.2` (exp17 sweeps 1.0-2.0) |")
    lines.append("| arrivals | Poisson, rate 0.5 requests/ms (exp17 sweeps 0.25-2.0) |")
    lines.append("| raw demands | CPU `lognormal(0, 0.6)`, RAM `lognormal(1, 0.6)`, disk `lognormal(3.5, 0.8)` |")
    lines.append("| capacity | `max(8 * mean_demand, max_demand)` per resource by default; `cap_mult` varies in exp04/08/11/12 |")
    lines.append("| bandwidth | `U = 100` Mbps uplink, `D = 1000` Mbps downlink, `rtt = 20` ms, request `100` kbit |")
    lines.append("| seeds | 10 by default (3 in `--quick`) |")
    lines.append("")
    lines.append("Some experiments add an **imbalance** factor: one random resource of each "
                 "service is multiplied by 3-4, which makes slot reductions wasteful.  The "
                 "synthetic world is the only data source for the sensitivity sweeps (theta, "
                 "bandwidth, capacity, Zipf rho, request rate) because the real trace has only "
                 "one fixed resource vector per service.")
    lines.append("")
    lines.append("### 2.3 Adversarial Large Sequence (Theorem 1)")
    lines.append("")
    lines.append("Generated by `sim.trace.large_sequence` from the proof of Theorem 1 in the "
                 "paper.  It is not a random dataset; it is a constructed request sequence "
                 "(BS1, BS2, BS3, MS repeated `n_reps = 2k` times) whose offline optimum is "
                 "known exactly:")
    lines.append("")
    lines.append("* `k` = cache capacity, tested at `k = 3, 6, ..., 24`;")
    lines.append("* `M = 6`, `Ml = 12`, `l = 1`, burst fraction 0.5;")
    lines.append("* `OPT = (k + 2n) l` (paper eq. 17);")
    lines.append("* each `k/3` block is disk-heavy, RAM-heavy, or CPU-heavy, and the MS block "
                 "needs the large demand of every resource, forcing a full-cache eviction.")
    lines.append("")
    lines.append("Used by exp05 and by family 1 of exp13.  It is the only dataset that tests the "
                 "competitive-ratio scaling with `k` in the adversarial sense.")
    lines.append("")
    lines.append("### 2.4 Analytic / random inputs")
    lines.append("")
    lines.append("* **exp06 (ski rental)**: no dataset; analytic minimax ratio and a Monte-Carlo "
                 "check of the randomized threshold.")
    lines.append("* **exp07 (projection)**: 60 random instances (10 in `--quick`) with `N` drawn "
                 "uniformly from 10-80 and random demands `a ~ U(0.01, 0.6)`, compared against "
                 "the exact QP.")
    lines.append("* **exp14 (time complexity)**: synthetic worlds with `N = 20, 40, 80, 160, 320` "
                 "or `C = 4, 8, 16, 32, 64`; three repeats per point.")
    lines.append("")
    lines.append("### 2.5 SHARCNet (not available)")
    lines.append("")
    lines.append("The paper also evaluates on **SHARCNet** (Grid Workloads Archive).  That host "
                 "(`gwa.unc.edu`) is not reachable from this codespace, and no `sharcnet.csv` is "
                 "bundled.  The loader and column mapping are implemented in "
                 "`experiments/exp16_real_traces.py` (`kind=\"sharcnet\"`), but the dataset is "
                 "documented as an open item rather than silently substituted with synthetic data.")
    lines.append("")
    lines.append("### 2.6 Dataset-to-experiment map")
    lines.append("")
    lines.append("| # | experiment | dataset / trace | key parameters | result files |")
    lines.append("|---|---|---|---|---|")
    mapping = [
        ("01", "baseline vs LL-RC", "synthetic Zipf/Poisson", "N=40, T=8000, C=15, 10 seeds, rho=1.2, rate=0.5", "exp01_baseline_vs_llrc.json/.png"),
        ("02", "theta sweep", "synthetic Zipf/Poisson", "N=40, T=8000, C=15, seed=0; theta=0..16", "exp02_theta.json/.png"),
        ("03", "bandwidth U/D", "synthetic Zipf/Poisson", "N=40, T=8000, C=15, seed=0, req_kbit=2000; U=20..400, D=250..4000", "exp03_bandwidth.json/.png"),
        ("04", "capacity C and cap_mult", "synthetic Zipf/Poisson", "N=40, T=8000, seed=0; C=1..32, cap_mult=2..16", "exp04_capacity.json/.png"),
        ("05", "Large Sequence", "adversarial Large Sequence", "k=3..24, n=2k, 10 seeds per k", "exp05_large_sequence.json/.png"),
        ("06", "ski rental", "analytic + Monte-Carlo", "deterministic ratio 2; randomized e/(e-1)", "exp06_ski_rental.json/.png"),
        ("07", "projection", "random instances", "60 trials, N~U(10,80), a~U(0.01,0.6)", "exp07_projection.json/.png"),
        ("08", "G1/G2 feasibility and budget", "synthetic Zipf/Poisson", "N=40, T=8000, imbalance=3.0, cap_mult=4, 10 seeds", "exp08_feasibility_budget.json/.png"),
        ("09", "G3 surrogate regret", "synthetic Zipf/Poisson", "N=40, C=12, T=500..8000, 5 runs per T", "exp09_regret.json/.png"),
        ("10", "G4 expected ratio vs MILP", "synthetic Zipf/Poisson", "N=10, T=120, C=3, 5 traces, 10 seeds; exact MILP", "exp10_expected_ratio.json/.png"),
        ("11", "G5 slot reduction", "synthetic Zipf/Poisson", "N=40, T=8000, C=40, imbalance=4.0, cap_mult=5, 10 seeds", "exp11_slot_reduction.json/.png"),
        ("12", "multi-resource", "synthetic Zipf/Poisson", "N=40, T=8000, C=12, cap_mult=5, 10 seeds; imbalance 1.0 and 4.0", "exp12_multiresource_balanced.json/.png; exp12_multiresource_imbalanced.json/.png"),
        ("13", "gamma complexity", "Large Sequence + synthetic Zipf", "family 1 k=3..24; family 2 N=16, T=1200, 3 seeds, MILP", "exp13_gamma_complexity.json/.png; exp13_gamma_complexity_zipf.json/.png"),
        ("14", "time complexity", "synthetic Zipf/Poisson", "T=8000, seed=0, 3 repeats; N=20..320 and C=4..64", "exp14_time_complexity.json/.png"),
        ("15", "G2 budget + G3 regret", "synthetic Zipf/Poisson", "N=40, T=8000, imbalance=3.0, cap_mult=4, 10 seeds", "exp15_budget_regret.json/.png"),
        ("16", "real traces", "Google cluster traces 2011 CSVs", "google.csv (50k/4619), dataset2_30 (1k/30), dataset2_paper & dataset2 (1k/6); C=min(50, N//2)", "exp16_real_traces*.json/.png"),
        ("17", "Zipf rho / request rate", "synthetic Zipf/Poisson", "N=40, T=8000, C=15, 10 seeds; rho=1.0..2.0, rate=0.25..2.0", "exp17_paper_synthetic_params.json/.png"),
    ]
    for row in mapping:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def demote_headings(text):
    out = []
    for line in text.splitlines():
        if line.startswith("# "):
            out.append("## " + line[2:])
        elif line.startswith("## "):
            new = "### " + line[3:]
            m = re.match(r"### (\d+)([a-z]?\.) (.*)", new)
            if m:
                new = f"### 3.{m.group(1)}{m.group(2)} {m.group(3)}"
            elif new.startswith("### Summary:"):
                new = "### 3.18. Summary: confirmed / refuted / not testable"
            out.append(new)
        elif line.startswith("### "):
            out.append("#### " + line[4:])
        else:
            out.append(line)
    return "\n".join(out)


def appendix():
    lines = []
    lines.append("## 4. Appendix: raw per-seed and per-trace outputs")
    lines.append("")
    lines.append("### 4.1 exp01 per-seed outputs (synthetic Zipf/Poisson)")
    lines.append("")
    lines.append("| seed | Online-DRL lat vs LL-RC % | Online-DRL cost vs LL-RC % | Randomized lat vs LL-RC % | Randomized cost vs LL-RC % | hits DRL | hits LL-RC | hits rand | delayed hits DRL | delayed hits LL-RC | delayed hits rand | downloads rand |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    d = json.load(open(os.path.join(ROOT, "results", "exp01_baseline_vs_llrc.json")))
    for r in d["rows"]:
        lines.append("| {seed} | {a:.3f} | {b:.3f} | {c:.3f} | {d:.3f} | {e} | {f} | {g} | {h} | {i} | {j} | {k} |".format(
            seed=r["seed"], a=r["lat_drl_pct"], b=r["cost_drl_pct"], c=r["lat_rand_pct"],
            d=r["cost_rand_pct"], e=r["hits_drl"], f=r["hits_llrc"], g=r["hits_rand"],
            h=r["dh_drl"], i=r["dh_llrc"], j=r["dh_rand"], k=r["downloads_rand"]))
    lines.append("")
    lines.append("### 4.2 exp16 per-trace, per-algorithm outputs")
    lines.append("")
    lines.append("| trace | services | requests | C | algorithm | latency | cost | hits | delayed hits | forwards | downloads | latency vs LL-RC % | cost vs LL-RC % |")
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    d = json.load(open(os.path.join(ROOT, "results", "exp16_real_traces.json")))
    for t in d["traces"]:
        for r in t["rows"]:
            lines.append("| {trace} | {ns} | {nr} | {C} | {algo} | {lat:.3f} | {cost:.3f} | {hits} | {dh} | {fw} | {dl} | {lp:+.3f} | {cp:+.3f} |".format(
                trace=t["trace"], ns=t["n_services"], nr=t["n_requests"], C=t["C"],
                algo=r["algo"], lat=r["latency"], cost=r["cost"], hits=r["hits"],
                dh=r["delayed_hits"], fw=r["forwards"], dl=r["downloads"],
                lp=r["lat_vs_llrc_pct"], cp=r["cost_vs_llrc_pct"]))
    lines.append("")
    lines.append("### 4.3 Result artifacts")
    lines.append("")
    lines.append("| file | size | contents |")
    lines.append("|---|---:|---|")
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "*"))):
        name = os.path.basename(p)
        size = os.path.getsize(p)
        if name.endswith(".json"):
            kind = "full experiment output (JSON)"
        elif name.endswith(".png"):
            kind = "figure"
        else:
            kind = "other"
        lines.append(f"| `results/{name}` | {size/1024:.1f} KiB | {kind} |")
    lines.append("")
    lines.append("All JSON files are the machine-readable outputs.  The Markdown tables in "
                 "section 3 are copied from `docs/REPORT.md`, which is generated by "
                 "`experiments/summarize.py` from these JSON files.")
    lines.append("")
    lines.append("### 4.4 Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("cd final_exp")
    lines.append("pip install -r requirements.txt && pip install -e .")
    lines.append("make test")
    lines.append("python experiments/run_all.py --quick   # smoke run")
    lines.append("python experiments/run_all.py         # full run (N=40, T=8000, 10 seeds)")
    lines.append("python experiments/summarize.py       # rebuild docs/REPORT.md")
    lines.append("python experiments/export_results_and_datasets.py  # rebuild this file")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main():
    report = open(REPORT).read()
    # drop the report's H1 + its generated line, keep the rest
    parts = report.split("\n", 2)
    body = parts[2] if len(parts) == 3 else report
    body = demote_headings(body)

    header = f"""# Results and datasets — final_exp

**Generated:** {date.today().isoformat()}  
**Scope:** every experiment in `final_exp/`, the dataset/trace used for each one, and a
short description of every dataset.  The numerical results below are the outputs of
`experiments/run_all.py` (full run: N=40, T=8000, 10 seeds) and are stored in
`results/*.json`; `docs/REPORT.md` is the generator's report.

## 1. Quick answer: what was run on what

* **Most experiments (01-04, 08-12, 14-15, 17)** use a **synthetic Zipf/Poisson world**
  (`sim.trace.synthetic_world`): 40 services, 8 000 requests, Zipf `rho=1.2`, Poisson rate
  0.5 req/ms, 10 seeds.  This is the only way to run the parameter sweeps (theta, U/D, C,
  cap_mult, Zipf rho, request rate).
* **exp05 and exp13 family 1** use the **adversarial Large Sequence** from the proof of
  Theorem 1 (`sim.trace.large_sequence`), not a random trace.
* **exp10 and exp13 family 2** use small **synthetic Zipf** traces with the exact MILP
  optimum.
* **exp06** is analytic (ski-rental), **exp07** is random projection instances.
* **exp16 is the only experiment on real data**: the bundled **Google cluster traces 2011**
  CSVs (`data/google.csv`, `data/google_dataset2_30.csv`, `data/google_dataset2_paper.csv`,
  `data/google_dataset2.csv`).
* **SHARCNet** (the paper's second real dataset) is **not bundled and not reachable** from
  this codespace; it is documented as an open item, not replaced by synthetic data.

"""
    text = header + dataset_section() + "\n## 3. Full results output (all 17 experiments)\n\n" + body + "\n\n" + appendix()
    with open(OUT, "w") as f:
        f.write(text)
    print(f"wrote {OUT} ({len(text)/1024:.1f} KiB)")


if __name__ == "__main__":
    main()