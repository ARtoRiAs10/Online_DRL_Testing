# Phase 4 -- matched cost

Levels are fractions of the mean deterministic theta=1 cost.

## synthetic

| level | frac | covered | gap | 95% CI | p | p Holm | win |
|---|---|---|---|---|---|---|---|
| 74569.0 | 0.50 | 30/30 | +1162.7 | [+273.6, +1981.2] | 0.0277 | 0.0555 | False |
| 111853.5 | 0.75 | 30/30 | +75.7 | [-498.3, +644.1] | 0.761 | 0.761 | False |
| 149137.9 | 1.00 | 30/30 | +1456.8 | [+988.8, +1953.3] | 3.24e-06 | 9.72e-06 | False |
| 186422.4 | 1.25 | 30/30 | +1992.3 | [+1513.1, +2490.1] | 6.15e-08 | 2.46e-07 | False |

wins 0/4; pass False

## google_dataset2_30

| level | frac | covered | gap | 95% CI | p | p Holm | win |
|---|---|---|---|---|---|---|---|
| 671.2 | 0.50 | 0/30 | +nan | [+nan, +nan] | 1 | 1 | False |
| 1006.8 | 0.75 | 0/30 | +nan | [+nan, +nan] | 1 | 1 | False |
| 1342.4 | 1.00 | 30/30 | +27.8 | [+22.2, +34.2] | 3.73e-09 | 1.49e-08 | False |
| 1678.0 | 1.25 | 30/30 | -10.4 | [-18.0, -2.4] | 0.0175 | 0.0525 | True |

wins 0/4; pass False

## google_50k

| level | frac | covered | gap | 95% CI | p | p Holm | win |
|---|---|---|---|---|---|---|---|
| 31286.3 | 0.50 | 30/30 | +972.1 | [+819.0, +1120.3] | 1.86e-09 | 7.45e-09 | False |
| 46929.5 | 0.75 | 30/30 | +714.9 | [+606.2, +835.2] | 1.86e-09 | 7.45e-09 | False |
| 62572.6 | 1.00 | 30/30 | +699.3 | [+552.3, +852.4] | 1.86e-09 | 7.45e-09 | False |
| 78215.8 | 1.25 | 30/30 | +125.5 | [+38.3, +218.0] | 0.0248 | 0.0248 | False |

wins 0/4; pass False

Overall pass (R3 at matched cost): **False**; datasets passing: []
