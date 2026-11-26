# Statistical standard for any randomized result

Frozen before running.

* At least **30 seeds** per configuration, preferably **100**.
* Report **mean**, **bootstrap 95% CI**, **worst seed** and **95th percentile**.
* Use a **paired Wilcoxon signed-rank test** against the deterministic
  baseline, with **p < 0.05**.
* Apply a **multiple-comparison correction (Holm)** when testing many sweep
  points.
* The improvement must hold on the **majority of sweep points**.  One
  favorable point doesn't count.

## Implementation notes

* Bootstrap: 2000 resamples of the per-seed values, percentile interval.
* Paired test: `scipy.stats.wilcoxon(..., zero_method="wilcox")`; if all
  differences are zero the p-value is 1.
* Holm: step-down adjusted p-values; a sweep point is significant if its
  adjusted p-value is below 0.05.
* "Majority of sweep points" means strictly more than half of the tested
  points, not merely a positive mean over all points.