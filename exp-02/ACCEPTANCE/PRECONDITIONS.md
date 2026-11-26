# Preconditions (without these, nothing else counts)

These are frozen.  A failed precondition blocks every downstream claim; the
claim is reported as **not accepted**, not as a negative result about R3/MG.

## 1. Baseline reproduced

Online-DRL and LL-RC must match the paper's published figures within about
**2%** (absolute percentage points for the relative metrics).

* **Google:** latency **0.016% to 7.41% above LL-RC**, and cost
  **77.57% to 92.54% lower**.
* **SHARCNet:** latency **9.34% to 26.62% above OPT**.

If this fails, any improvement may just reflect a different baseline.

**Frozen tolerance:** a reproduced value passes if it lies inside the published
interval expanded by 2 percentage points on each side.  For a published range
`[a, b]`, the acceptance interval is `[a - 2, b + 2]`.

## 2. Reduction holds

With randomization off, results are identical across seeds, and `lam = 0`
equals LL-RC.

* deterministic Online-DRL: identical `(latency, cost, hits, delayed_hits,
  downloads)` for every seed;
* `drl_r3(..., lam=0)` equals `ll_rc(...)` exactly;
* `theta = 0` equals `lam = 0`.

## 3. Invariants hold

Zero resource, cardinality or budget violations across every run.

* after every eviction/admission: `sum_r a_{n,r} <= 1` for CPU, RAM and disk;
* `|cache| <= C`;
* `cost <= Bmax` whenever a budget is set;
* marking invariant: a marked item is never evicted while an unmarked item
  exists.

A single violation in any run fails the precondition.