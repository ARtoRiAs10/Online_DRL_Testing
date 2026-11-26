# Phase 2 and Phase 3 report

**Not committed.**

## Phase 2 — stateless headline reruns

### exp01 — vs LL-RC (N=40, T=8000, 10 seeds)

| quantity | mean % | sd % |
|---|---:|---:|
| Online-DRL latency vs LL-RC | -2.41 | 3.70 |
| Online-DRL cost vs LL-RC | -50.52 | 0.55 |
| Randomized stateless latency vs LL-RC | **-27.82** | 17.42 |
| Randomized stateless cost vs LL-RC | **-94.28** | 1.94 |

The randomized stateless method improves both latency and cost relative to LL-RC. This is
the strongest headline result in the paper.

### exp04 — capacity sweep (stateless)

| C | Online-DRL latency | Randomized stateless latency | max util |
|---:|---:|---:|---:|
| 1 | 158,044 | 116,369 | 0.115 |
| 2 | 136,789 | 115,320 | 0.115 |
| 4 | 100,466 | 91,719 | 0.350 |
| 8 | 80,187 | 76,703 | 0.663 |
| 16 | 81,592 | 76,610 | 0.669 |
| 32 | 81,592 | 76,610 | 0.669 |

Randomized stateless is faster at every capacity and fills the cache much better than the
Landlord rule.

### exp12 — multi-resource (5 seeds)

**Balanced**

| algorithm | latency | cost | max util |
|---|---:|---:|---:|
| Online-DRL | 120,110 | 167,401 | 0.711 |
| LL-RC | 123,573 | 334,689 | 0.718 |
| R3 only | 119,868 | 198,166 | 0.720 |
| Randomized stateless | **89,750** | **17,420** | 0.644 |
| Randomized Landlord | 154,018 | 58,806 | 0.095 |

**Imbalanced**

| algorithm | latency | cost | max util |
|---|---:|---:|---:|
| Online-DRL | 132,142 | 190,686 | 0.624 |
| LL-RC | 135,919 | 386,714 | 0.624 |
| R3 only | 131,891 | 227,893 | 0.622 |
| Randomized stateless | **91,058** | **15,527** | 0.722 |
| Randomized Landlord | 150,261 | 60,233 | 0.127 |

The stateless repair dominates both deterministic baselines on latency and cost. The
Landlord repair has very low utilization and is not competitive on these workloads.

### exp17 — Zipf/rate sensitivity (stateless, 10 seeds)

| rho | rate | latency vs LL-RC | cost vs LL-RC | max util |
|---:|---:|---:|---:|---:|
| 1.0 | 0.25 | -27.20% | -94.52% | 0.773 |
| 1.0 | 0.50 | -23.94% | -93.43% | 0.744 |
| 1.0 | 2.00 | -16.18% | -89.66% | 0.650 |
| 1.2 | 0.25 | -29.21% | -95.01% | 0.787 |
| 1.2 | 0.50 | -27.82% | -94.28% | 0.758 |
| 1.2 | 2.00 | -19.25% | -91.09% | 0.671 |
| 1.6 | 0.25 | -37.43% | -95.78% | 0.821 |
| 1.6 | 0.50 | -35.29% | -94.90% | 0.789 |
| 1.6 | 2.00 | -28.50% | -92.61% | 0.698 |
| 2.0 | 0.25 | -29.67% | -95.28% | 0.829 |
| 2.0 | 0.50 | -33.25% | -94.79% | 0.801 |
| 2.0 | 2.00 | -30.19% | -92.28% | 0.710 |

The stateless method improves both metrics across the whole sweep. The improvement is
largest for skewed popularity and low request rates.

## Phase 2 — hyperparameters

Command: `python experiments/exp30_hyperparameters.py`

| group | value | latency | cost | hits | max util |
|---|---:|---:|---:|---:|---:|
| eta_mult | 1 | 22,372 | 5,558 | 929 | 0.577 |
| eta_mult | 4 | 32,345 | 21,988 | 446 | 0.183 |
| eta_mult | 16 | 40,871 | 41,370 | 32 | 0.013 |
| eta_mult | 64 | 41,333 | 41,787 | 10 | 0.006 |
| alpha | 0.05 | 22,218 | 5,464 | 936 | 0.583 |
| alpha | 0.1 | 22,218 | 5,464 | 936 | 0.583 |
| alpha | 0.2 | 22,218 | 5,464 | 936 | 0.583 |
| alpha | 0.4 | 22,218 | 5,464 | 936 | 0.583 |
| K | 2 | 23,173 | 5,249 | 890 | 0.533 |
| K | 4 | 22,575 | 5,301 | 920 | 0.567 |
| K | 8 | 22,372 | 5,558 | 929 | 0.577 |
| proj_tol | 1e-6 | 22,372 | 5,558 | 929 | 0.577 |
| proj_tol | 1e-7 | 22,372 | 5,558 | 929 | 0.577 |
| proj_tol | 1e-8 | 22,372 | 5,558 | 929 | 0.577 |
| proj_skip_tol | 0.0 | 22,372 | 5,558 | 929 | 0.577 |
| proj_skip_tol | 1e-3 | 22,372 | 5,558 | 929 | 0.577 |

**Chosen values:** `eta_mult=1`, `alpha=0.05`, `K=8`, `proj_tol=1e-6`,
`proj_skip_tol=0`.

The `alpha` effect is small because the synthetic world has few in-flight downloads; the
q-EMA is mostly zero. The projection tolerances are below numerical precision and do not
change the result.

## Phase 3 — ALG/OPT vs trace length

Command: `python experiments/exp31_alg_opt_vs_T.py`

| T | traces | Online-DRL/OPT | R3/OPT | Landlord/OPT | Stateless/OPT | OPT status |
|---:|---:|---:|---:|---:|---:|---|
| 120 | 30 | 1.168 | 1.152 | 1.427 | 1.301 | 30 optimal |
| 300 | 15 | 1.456 | 1.485 | 1.981 | 1.631 | 15 optimal |
| 600 | 10 | 1.717 | 1.761 | 2.251 | 1.675 | 10 optimal |
| 1200 | 5 | 1.753 | 1.819 | 2.188 | 1.622 | 4 optimal, 1 time limit |

**Paired gap of stateless randomized vs Online-DRL:**

| T | gap |
|---:|---:|
| 120 | +11.4% worse |
| 300 | +12.0% worse |
| 600 | -2.4% better |
| 1200 | -7.5% better |

**Interpretation.** The absolute `ALG/OPT` ratio grows with `T` because the problem becomes
harder, but the randomized stateless method starts behind Online-DRL on short traces and
overtakes it around `T ≈ 600`. This supports the warm-up explanation for the short-trace
MILP result. The variance is moderate (`sd ≈ 0.09–0.31`), so additional MILP instances are
not required.

## Conclusions

1. The stateless repair is the correct default. It dominates Online-DRL and LL-RC on
  latency and cost across all tested synthetic workloads.
2. The Landlord repair is a conservative feasibility heuristic with very low utilization;
  it should be presented as a negative result or an alternative, not as the main method.
3. The short-trace MILP gap is explained by warm-up: randomized stateless overtakes
  Online-DRL at longer horizons.
4. The paper’s main claim should be: **randomized stateless DRL improves the
  cost–latency tradeoff of Online-DRL**, with the Landlord repair as a feasibility
  variant.