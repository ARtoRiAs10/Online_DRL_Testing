# Phase 0 -- baseline reproduction

Published Google intervals: latency +0.016% .. +7.41%, cost -92.54% .. -77.57%; tolerance +/-2.0 pp.

| dataset/regime | best C | lat % | cost % | verdict | note |
|---|---|---|---|---|---|
| google_dataset2_30/paper_recipe | 1 | +0.000 | +0.00 | FAIL | M<l=1.000 |
| google_dataset2_30/natural_crossing | 2 | -0.763 | -41.83 | FAIL | M<l=0.067 |
| google_dataset2_paper/paper_recipe | 1 | +0.000 | +0.00 | FAIL | M<l=1.000 |
| google_dataset2_paper/natural_crossing | 1 | -1.739 | -37.53 | FAIL | M<l=0.500 |
| google_50k/paper_recipe | 1 | +0.000 | +0.00 | FAIL | M<l=1.000 |
| google_50k/natural_crossing | 50 | -0.113 | -57.82 | FAIL | M<l=0.664 |
| SHARCNet | BLOCKED |  |  |  | sharcnet.csv not bundled; gwa.unc.edu unreachable |

## Verdict

The Google baseline is **not reproduced** in either regime: the paper-recipe regime has M < l for every service (so Online-DRL and LL-RC coincide), and the natural-crossing regime gives a much smaller cost reduction than the published 77.57--92.54%.  The SHARCNet precondition is blocked because the trace is unavailable.  Per the frozen preconditions, all downstream claims are therefore **blocked**.
