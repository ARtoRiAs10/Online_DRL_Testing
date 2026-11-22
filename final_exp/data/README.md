# Real traces (bundled Google CSVs)

The Google CSVs are **bundled** in this folder (prepared with the paper's recipe):

| file | requests | services | use |
|---|---|---|---|
| `google.csv` | 50 000 | 4 619 | main real-trace check (exp16) |
| `google_dataset2_30.csv` | 1 000 | 30 | paper-like `dataset2` (30 services) |
| `google_dataset2_paper.csv` | 1 000 | 6 | 1 000-row slice of the paper recipe |
| `google_dataset2.csv` | 1 000 | 6 | small slice |

The simulator also runs on synthetic data by default.

## Getting the paper's Google trace in this codespace (it works)

The Google cluster traces 2011 are **still downloadable** from the GCS bucket
`clusterdata-2011-2` (the old `google_cluster_data` bucket is gone).  The raw
`task_events` is 1.6 GB gzipped, but the paper's recipe only needs the `SUBMIT`
records, so a handful of shards is enough:

```bash
cd final_exp
python data/prepare_google.py --shards 10 --rows 50000 --out data/google.csv
python data/prepare_google.py --shards 10 --rows 1000 --paper-dataset2 --out data/google_dataset2_paper.csv
python experiments/exp16_real_traces.py --csv data/google.csv --kind google --rows 50000
```

`prepare_google.py` downloads the shards to `/tmp/gcs_cache` (10 shards = 20 MB), keeps the
records with event type `SUBMIT` (0), makes one row per (JobID, distinct Timestamp), takes the
resources of a service from its first occurrence, replaces the zeros/missing values by the
column median and writes a CSV of `JobID,Timestamp,CPU,RAM,Disk`.

Measured in this codespace (2 vCPU, 8 GB RAM): 10 shards -> 50 000 requests / 4 619 services ->
a 2.3 MB CSV; `exp16_real_traces.py` runs the four algorithms in **85 s** and **~1 GB** of RAM.
The MILP reference (`sim/opt.py`) also runs on the real trace, but only on short slices
(1 000 requests: 0.4 s), exactly the restriction the paper had on the Google data.

The SHARCNet host (`gwa.unc.edu`, Grid Workloads Archive) is **not reachable** from this
sandbox; that trace has to be fetched manually and preprocessed with the recipe below.

* Google cluster traces 2011 (clusterdata-2011-2): take `task_events`/`job_events` rows with event type SUBMIT.
* SHARCNet (Grid Workloads Archive): ExecutableID, SubmitTime, UsedCPUTime, UsedMemory.

Preprocess to a CSV with one row per request (columns: service id, timestamp, cpu, ram, disk), keep the first
10k-100k rows (Codespaces memory), then:

```python
from sim.trace import load_trace_csv
cat, tr = load_trace_csv("data/google.csv", id_col="JobID", time_col="Timestamp",
                         cpu_col="CPU", ram_col="RAM", disk_col="Disk",
                         time_scale_ms=1e-3,      # microseconds -> ms
                         disk_unit_mb=1/1024,     # KB -> MB
                         max_rows=50000, U=100, D=1000)
```
Zeros are replaced by the column median and a service's resources come from its first occurrence (as in the paper).
Then pass `cat, tr` to any function in `sim/algos.py`.

## The paper's own datasets (Sec. IV of the paper)

* **Google cluster traces 2011** (`clusterdata-2011-2`, Job and Task event files): every JobID
  occurrence with a distinct timestamp is a service request; only records with event type
  `SUBMIT` are kept.  The paper preprocesses them into 11 files `dataset0..dataset10` with the
  columns `JobID, Timestamp, CPU, RAM, Disk, Task Index, Status`, replaces the zeros in
  CPU/RAM/Disk by the **column median** and assumes the **disk size in KB**; it uses `dataset2`.
  Note: in `dataset2` only 30 unique services appear in 1 000 rows and they are mostly grouped
  together, which is why the paper does not run the OPT program on Google.
* **SHARCNet** (Grid Workloads Archive): columns `ExecutableID, SubmitTime, UsedCPUTime,
  UsedMemory, JobID, Status`; `UsedLocalDiskSpace` was -1 everywhere, so the `UsedMemory` values
  are copied into it; `SubmitTime` is shifted by the first value and equal timestamps are
  incremented; zeros -> column median; the resources of an executable are taken from its first
  occurrence; CPU/RAM/disk are in **MB** and are compared against the `MAXLIMIT` values.  The
  paper runs the OPT program here only, limited to 1 000 rows (420 unique services).
* **Synthetic dataset of the paper**: the *unique services of Google dataset2*, with a Zipf
  popularity (`rho`, default 1.6) and Poisson arrivals (`xi`, default 3e6).  `experiments/
  exp17_paper_synthetic_params.py` reproduces the *sweep* of Exps. 15-16 on a synthetic catalogue.

## Running the comparison on a real trace

`experiments/exp16_real_traces.py` runs Online-DRL, LL-RC, R3-only and Randomized DRL on any
preprocessed CSV in this directory and reports the paper's metrics (total latency, total cost,
hits, delayed hits, and the difference with respect to LL-RC):

```bash
python experiments/exp16_real_traces.py                        # auto-detects data/*.csv
python experiments/exp16_real_traces.py --csv data/google.csv --kind google
```

If no CSV is present the script writes `results/exp16_real_traces.json` with the status
`NO REAL TRACE AVAILABLE` and the list of the claims that are therefore still tested only on
synthetic data - the gap is documented instead of hidden.
