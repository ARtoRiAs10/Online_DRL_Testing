# Phase 6 report — real data

**Status: done, with one documented limitation (SHARCNet is not reachable from this
workspace).**  The two bundled Google traces are used as the primary and second real
trace.  Every randomized number is now multi-seed with a paired 95% CI.

---

## 1. Data

| file | requests | services | role |
|---|---:|---:|---|
| `google.csv` | 50,000 | 4,619 | main real trace (full run, exp27) |
| `google.csv` (first 10,000) | 10,000 | 4,619 | multi-seed slice (exp36, exp37) |
| `google_dataset2_30.csv` | 1,000 | 30 | second real trace, paper-like `dataset2` (exp36) |
| `google_dataset2_paper.csv` | 1,000 | 6 | paper recipe (exp36) |
| SHARCNet | — | — | **not reachable**; documented as an open item |

The old 2,000-row `google.csv` slice (0 hits for every method) is not used; the full
50,000-row `google.csv` replaces it.  The duplicate `google_dataset2.csv` (identical to
`google_dataset2_paper.csv`) is dropped from the experiment candidates.

### 1.1 `M_n` vs `l_n` crossing

With the raw Google disk column (`disk_scale=1`) every service has `M_n < l_n`:
`M_n ≈ 11 ms`, `l_n = 21 ms`.  The paper's model needs a regime where the download can
be more expensive than forwarding, so we use a **documented rescaling** of the disk
column, `disk_scale=1e7` (equivalently, a much slower downlink or larger service
images).  The fraction of services with `M_n < l_n` is reported for every row:

| trace | `disk_scale=1` `M<l` | `disk_scale=1e7` `M<l` | `M=l` | `M>l` |
|---|---:|---:|---:|---:|
| `google.csv` 50k | 1.000 | 0.664 | 0.000 | 0.336 |
| `google.csv` 10k | 1.000 | 0.654 | 0.000 | 0.346 |
| `google_dataset2_30.csv` | 1.000 | 0.067 | 0.000 | 0.933 |
| `google_dataset2_paper.csv` | 1.000 | 0.500 | 0.000 | 0.500 |

The crossing is therefore not natural in the raw trace; it is induced by the documented
`disk_scale` and the fraction `M<l` is reported so the reader can judge the regime.

---

## 2. Experiments

| script | trace | seeds | K | repair rules | purpose |
|---|---|---:|---:|---|---|
| `exp36_real_data_phase6.py` | 10k google, dataset2_30, dataset2_paper | 5 / 10 / 10 | 4 | Landlord, stateless | multi-seed real-trace comparison with CIs |
| `exp27_full_google.py` | 50k google | 3 | 2 | Landlord, stateless | full-scale run with CIs |
| `exp37_real_k_sensitivity.py` | 10k google | 3 | 2, 4, 8 | stateless | check that the result is not a `K` artifact |

All randomized runs use `gate_downloads=True` and `download_time_mode="ema"`.

---

## 3. Results

### 3.1 exp36 — 10k google slice and the two dataset2 traces

`Δ` is the paired difference vs deterministic Online-DRL; `±` is the 95% CI half-width.

| trace | regime | `M<l` | Online-DRL lat/cost | Landlord Δlat | Landlord Δcost | Stateless Δlat | Stateless Δcost |
|---|---|---:|---:|---:|---:|---:|---:|
| google 10k | raw | 1.000 | 138,921 / 47,366 | +4,055 ±1,267 | −18,284 ±547 | +309 ±152 | −20,044 ±305 |
| google 10k | crossing | 0.654 | 183,876 / 41,940 | +3,981 ±1,180 | −674 ±1,656 | +443 ±179 | −3,476 ±889 |
| dataset2_30 | raw | 1.000 | 9,834 / 561 | +1,767 ±264 | +153 ±89 | +1,327 ±108 | −163 ±28 |
| dataset2_30 | crossing | 0.067 | 18,739 / 1,320 | +1,050 ±266 | +540 ±157 | +256 ±84 | −30 ±77 |
| dataset2_paper | raw | 1.000 | 1,016 / 396 | +841 ±775 | −117 ±207 | +228 ±205 | −303 ±61 |
| dataset2_paper | crossing | 0.500 | 1,088 / 532 | +880 ±789 | +87 ±462 | +194 ±233 | −299 ±179 |

### 3.2 exp27 — full 50k google trace

| regime | `M<l` | Online-DRL lat/cost | Landlord Δlat | Landlord Δcost | Stateless Δlat | Stateless Δcost |
|---|---:|---:|---:|---:|---:|---:|
| raw | 1.000 | 499,143 / 60,434 | +34,797 ±3,593 | −33,781 ±1,995 | **−2,448 ±676** | **−42,658 ±206** |
| crossing | 0.664 | 861,093 / 62,152 | +40,789 ±1,897 | −15,934 ±5,157 | **−2,215 ±782** | **−33,867 ±1,021** |

On the full 50k trace the **stateless repair improves both latency and cost** relative to
Online-DRL, with tight paired CIs.  The Landlord repair reduces cost but has a large
latency penalty.

### 3.3 exp37 — K sensitivity (10k slice, stateless)

| regime | K | Δlat vs det | Δcost vs det |
|---|---:|---:|---:|
| raw | 2 | +229 ±89 | −19,939 ±216 |
| raw | 4 | +316 ±185 | −19,895 ±264 |
| raw | 8 | +260 ±322 | −20,112 ±277 |
| crossing | 2 | +352 ±71 | −3,727 ±1,108 |
| crossing | 4 | +425 ±189 | −3,162 ±884 |
| crossing | 8 | +439 ±337 | −4,417 ±508 |

The 10k-slice result is **not** a `K` artifact: for every `K ∈ {2,4,8}` the stateless
repair trades a small latency increase for a large cost reduction.  The full-50k latency
improvement in exp27 is therefore a **trace-length / warm-up effect**, not a `K` effect.

---

## 4. What can be said

1. **Cost:** the randomized stateless repair reduces download cost on every real-trace
   configuration, from −30 (dataset2_30 crossing) to −42,658 (full 50k raw).  The
   reduction is largest when downloads are expensive (`disk_scale=1e7`) or when the trace
   is long.
2. **Latency:** the sign is trace-dependent.  On the 10k slice and the two dataset2
   traces the stateless repair has a small latency penalty (+194 to +1,327); on the full
   50k trace it has a small latency improvement (−2,215 to −2,448).  The paper should
   report both and not claim a universal latency win.
3. **Landlord vs stateless:** the Landlord repair always has a larger latency penalty on
   the real traces; the stateless repair is the better default.  The Landlord repair is
   the conservative feasibility variant.
4. **M/l crossing:** the raw Google disk column gives `M_n < l_n` for every service; the
   crossing regime requires the documented `disk_scale=1e7`.  The fraction `M<l` is
   reported for every row.
5. **Second trace:** `google_dataset2_30.csv` is the second real trace.  SHARCNet remains
   an open item.

---

## 5. Files

- `final_exp/experiments/exp36_real_data_phase6.py`
- `final_exp/experiments/exp27_full_google.py` (updated: 3 seeds, K=2, both repairs, CIs)
- `final_exp/experiments/exp37_real_k_sensitivity.py`
- `final_exp/results/exp36_real_data_phase6.json`
- `final_exp/results/exp27_full_google.json`
- `final_exp/results/exp37_real_k_sensitivity.json`