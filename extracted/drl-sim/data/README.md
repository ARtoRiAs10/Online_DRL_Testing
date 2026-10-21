# Real traces (optional)

Not bundled (large). The simulator runs on synthetic data by default.

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
