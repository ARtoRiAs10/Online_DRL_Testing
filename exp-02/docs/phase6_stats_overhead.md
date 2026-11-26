# Phase 6 -- statistics and overhead

Seeds: 30

## Synthetic Zipf (N=20, T=2000, C=10)

| config | mean | 95% CI | worst | p95 | p | p Holm | win rate |
|---|---|---|---|---|---|---|---|
| R3 | 19217 | [18840, 19599] | 20689 | 20543 | 4.41e-05 | 4.41e-05 | 0.73 |
| MG | 15909 | [15909, 15909] | 15909 | 15909 | 4.32e-08 | 8.64e-08 | 1.00 |
| R3+MG | 17574 | [17157, 18003] | 19909 | 19726 | 1.86e-09 | 5.59e-09 | 1.00 |

## Cyclic scan (k=16)

| config | mean | 95% CI | worst | p95 | p | p Holm |
|---|---|---|---|---|---|---|
| R3 | 170 | [170, 170] | 170 | 170 | 1 | 1 |
| MG | 48 | [47, 49] | 60 | 54 | 1.66e-06 | 4.97e-06 |
| R3+MG | 48 | [47, 50] | 58 | 55 | 1.67e-06 | 4.97e-06 |

## Overhead (us/request, median of 15)

| config | median | ratio to det | within 2x |
|---|---|---|---|
| Online-DRL | 13.630 | 1.00 | True |
| R3 | 9.768 | 0.72 | True |
| MG | 9.493 | 0.70 | True |
| R3+MG | 9.228 | 0.68 | True |
