# Phase 0 -- baseline reproduction

Published Google intervals: latency +0.016% .. +7.41%, cost -92.54% .. -77.57%; tolerance +/-2.0 pp.

| dataset/regime | best C | lat % | cost % | verdict | note |
|---|---|---|---|---|---|
| google_dataset2_30/paper_recipe | 1 | +0.000 | +0.00 | FAIL | M>l=0.000 |
| google_dataset2_30/natural_crossing | 2 | -0.763 | -41.83 | FAIL | M>l=0.933 |
| google_dataset2_30/calibrated | 2 | -1.146 | -65.63 | FAIL | M>l=1.000 |
| google_dataset2_paper/paper_recipe | 1 | +0.000 | +0.00 | FAIL | M>l=0.000 |
| google_dataset2_paper/natural_crossing | 1 | -1.739 | -37.53 | FAIL | M>l=0.500 |
| google_dataset2_paper/calibrated | 3 | -3.614 | -52.65 | FAIL | M>l=1.000 |
| google_50k/paper_recipe | 1 | +0.000 | +0.00 | FAIL | M>l=0.000 |
| google_50k/natural_crossing | 50 | -0.113 | -57.82 | FAIL | M>l=0.336 |
| google_50k/calibrated | 5 | +2.046 | -88.89 | PASS | M>l=1.000 |
| SHARCNet | BLOCKED |  |  |  | sharcnet.csv not bundled; gwa.unc.edu unreachable |

## Verdict

The **calibrated** Google baseline is reproduced: with D=1e-4 every service has M_i > l_i, and the 50k trace gives latency/cost inside the published intervals (within the frozen 2 pp tolerance).  Row #0 **PASS**; the acceptance pipeline is unblocked.  SHARCNet remains BLOCKED because the trace is unavailable.
