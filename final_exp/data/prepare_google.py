"""Preprocess the Google cluster traces (clusterdata-2011-2) exactly as the paper does
(Sec. IV of "Edge Service Caching With Delayed Hits") and write a CSV that
`sim/trace.py::load_trace_csv` / `experiments/exp16_real_traces.py` can read.

The paper's recipe (see data/README.md of this repo):
  * every JobID occurrence with a distinct timestamp is a *service request*;
  * only the records with event type SUBMIT (0) are kept;
  * the resources of a service are taken from its first occurrence;
  * the zeros in CPU/RAM/Disk are replaced by the column median;
  * the disk size is in KB (the loader converts it with disk_unit_mb = 1/1024).

The raw trace is NOT stored in the repository (1.6 GB gzipped for task_events).  This script
downloads only the shards it needs and writes a small CSV (a few MB).

    python data/prepare_google.py --shards 10 --rows 50000 --out data/google.csv
    python data/prepare_google.py --shards 10 --rows 1000 --paper-dataset2 --out data/google_dataset2.csv

Then

    python experiments/exp16_real_traces.py --csv data/google.csv --kind google
"""
from __future__ import annotations
import argparse, io, os, sys, urllib.request
import numpy as np
import pandas as pd

BUCKET = "https://storage.googleapis.com/clusterdata-2011-2/task_events/part-%05d-of-00500.csv.gz"
COLS = ["time", "missing", "jobID", "task_index", "machine", "event_type", "user",
        "sched_class", "priority", "cpu", "ram", "disk", "diff_mach"]
USECOLS = ["time", "jobID", "task_index", "event_type", "cpu", "ram", "disk"]


def fetch(shard, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, f"task_events_part{shard:05d}.csv.gz")
    if not os.path.exists(p):
        print(f"  downloading shard {shard} ...", flush=True)
        with urllib.request.urlopen(BUCKET % shard, timeout=120) as r, open(p, "wb") as f:
            f.write(r.read())
    return p


def build(shards, cache_dir, rows, paper_dataset2=False, out="data/google.csv"):
    frames = []
    for s in shards:
        p = fetch(s, cache_dir)
        df = pd.read_csv(p, header=None, compression="gzip", names=COLS, usecols=USECOLS,
                         dtype={"jobID": "int64", "time": "int64", "task_index": "int32",
                                "event_type": "int8", "cpu": "float64", "ram": "float64",
                                "disk": "float64"})
        frames.append(df)
        print(f"  shard {s}: {len(df)} raw rows", flush=True)
    df = pd.concat(frames, ignore_index=True)
    del frames

    # (1) keep only the SUBMIT events (event type 0)
    df = df[df.event_type == 0]
    # (2) the resources of a service: the first occurrence of every (JobID, distinct Timestamp)
    df = df.sort_values(["time", "jobID", "task_index"], kind="stable")
    df = df.drop_duplicates(subset=["jobID", "time"], keep="first")
    # (3) the per-service footprint: sum of the task requests of that job
    per_job = df.groupby("jobID", sort=False)[["cpu", "ram", "disk"]].max()
    df["cpu"] = df["jobID"].map(per_job["cpu"])
    df["ram"] = df["jobID"].map(per_job["ram"])
    df["disk"] = df["jobID"].map(per_job["disk"])

    if paper_dataset2:
        # the paper's dataset2: 1000 rows, 30 unique services
        df = df[df.jobID.isin(df.jobID.value_counts().head(30).index)].head(1000)

    df = df.sort_values("time", kind="stable").head(rows)
    # (4) zeros -> column median (the paper), exactly as the loader does it again
    for c in ("cpu", "ram", "disk"):
        v = df[c].to_numpy(float).copy()
        med = np.median(v[v > 0]) if np.any(v > 0) else 0.0
        v[v <= 0] = med
        df[c] = v

    out_df = pd.DataFrame({
        "JobID": df["jobID"].to_numpy(),
        "Timestamp": df["time"].to_numpy(),
        "CPU": df["cpu"].to_numpy(),
        "RAM": df["ram"].to_numpy(),
        "Disk": df["disk"].to_numpy(),
    })
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"wrote {out}: {len(out_df)} requests, {out_df.JobID.nunique()} services, "
          f"{os.path.getsize(out)/1e6:.1f} MB")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=10, help="number of task_events shards to use")
    ap.add_argument("--rows", type=int, default=50000, help="number of requests to keep")
    ap.add_argument("--cache", default="/tmp/gcs_cache", help="where the raw shards are kept")
    ap.add_argument("--paper-dataset2", action="store_true", help="emulate the paper's dataset2")
    ap.add_argument("--out", default="data/google.csv")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(here))      # the drl-sim root: data/ is a subdirectory
    build(list(range(a.shards)), a.cache, a.rows, a.paper_dataset2, a.out)
