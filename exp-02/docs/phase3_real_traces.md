# Phase 3 -- real traces

Seeds: 30.  Soft-MG tau=0.15.  SHARCNet: **BLOCKED** (sharcnet.csv not bundled; gwa.unc.edu unreachable).

| dataset | regime | C | M>l | MG lat % | MG cost % | SoftMG lat % | SoftMG cost % | R3+SoftMG lat % | R3+SoftMG cost % |
|---|---|---|---|---|---|---|---|---|---|
| google_dataset2_30 | paper_recipe | 10 | 0.000 | -0.13 | -1.18 | +0.19 | +1.76 | +0.12 | +1.11 |
| google_dataset2_30 | natural_crossing | 10 | 0.933 | +0.10 | +3.23 | -0.05 | +0.00 | -0.08 | +14.52 |
| google_dataset2_30 | calibrated | 10 | 1.000 | +0.11 | +0.00 | +0.00 | +0.00 | +0.07 | +1.34 |
| google_50k | paper_recipe | 50 | 0.000 | -0.05 | +0.28 | -0.00 | +0.00 | -0.00 | +0.00 |
| google_50k | natural_crossing | 50 | 0.336 | +0.23 | +2.80 | +0.00 | -0.00 | +0.08 | +41.85 |
| google_50k | calibrated | 50 | 1.000 | +2.35 | +5.83 | +0.00 | -0.00 | -0.55 | +6.94 |

MG no harm (<=2%) hard: **False**; Soft-MG no harm (<=2%): **True**
