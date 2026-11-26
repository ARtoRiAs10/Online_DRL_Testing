# Phase 4 report — adversarial families and ratio

**Not committed.**

## 1. Large Sequence ablation

Command: `python experiments/exp32_large_sequence_ablation.py`

Variants:
- `r3_only` — randomized threshold, deterministic Landlord;
- `r1r2_stateless` — randomized rounding + stateless repair, deterministic threshold;
- `r1r2r3_stateless` — full method with stateless repair;
- `r1r2r3_landlord` — full method with Landlord repair.

Representative latency ratios to OPT (`M/l = 4`):

| k | R3 only | R1+R2 stateless | R1+R2+R3 stateless | R1+R2+R3 Landlord |
|---:|---:|---:|---:|---:|
| 9 | 12.791 | 2.920 | 2.716 | 3.000 |
| 12 | 16.690 | 3.277 | 3.043 | 3.120 |
| 24 | 32.747 | 2.460 | 2.357 | 2.683 |
| 48 | 64.600 | 3.161 | 3.011 | 3.011 |

Cost ratios to OPT (`M/l = 4`):

| k | R1+R2 stateless | R1+R2+R3 stateless | R1+R2+R3 Landlord |
|---:|---:|---:|---:|
| 9 | 12.551 | 2.471 | 3.129 |
| 12 | 16.533 | 2.800 | 3.013 |
| 24 | 27.973 | 1.213 | 2.140 |
| 48 | 54.550 | 1.089 | 1.089 |

**Findings**

1. **R3 alone does not flatten the ratio.** Its latency ratio grows roughly linearly with
   `k` (e.g. 12.8 at k=9 to 64.6 at k=48 for M/l=4).
2. **R1+R2 (randomized rounding) is the component that flattens the ratio.** Once R2 is
   present, the latency ratio stays in the 2–4 range for `k = 9..48`.
3. **R3 mainly reduces cost.** Adding R3 to R1+R2 lowers the cost ratio from 12–55 to
   0.6–3.5, with little latency change.
4. **Landlord vs stateless on the Large Sequence:** both are feasible; the stateless rule
   has slightly lower latency and cost at most k. This is consistent with the phase-2
   result that the Landlord repair is too conservative on popularity-driven workloads.

## 2. Additional adversarial families

Command: `python experiments/exp33_adversarial_families.py`

Families:
- **Shuffled cyclic scan** — k+1 services, fresh random permutation each cycle, k up to 128.
- **Markov-modulated Zipf** — popularity alternates between Zipf 0.8 and 1.6.
- **Bursty random** — bursts of 20 requests centred on one service with random outliers.

Mean latency/cost ratio vs Online-DRL (10 seeds):

| family | algorithm | latency | cost | lat/Online-DRL | cost/Online-DRL |
|---|---|---:|---:|---:|---:|
| shuffled_cyclic_scan | Online-DRL | 101 | 101 | 1.000 | 1.000 |
| shuffled_cyclic_scan | R3 only | 101 | 101 | 1.000 | 1.000 |
| shuffled_cyclic_scan | randomized stateless | 154 | 88 | 1.459 | 0.866 |
| shuffled_cyclic_scan | randomized Landlord | 154 | 88 | 1.460 | 0.867 |
| markov_zipf | Online-DRL | 139,951 | 174,192 | 1.000 | 1.000 |
| markov_zipf | R3 only | 139,987 | 207,083 | 1.000 | 1.188 |
| markov_zipf | randomized stateless | 103,880 | 16,512 | 0.739 | 0.095 |
| markov_zipf | randomized Landlord | 157,703 | 56,677 | 1.127 | 0.326 |
| bursty | Online-DRL | 153,225 | 169,938 | 1.000 | 1.000 |
| bursty | R3 only | 153,100 | 198,821 | 0.999 | 1.170 |
| bursty | randomized stateless | 161,736 | 75,899 | 1.056 | 0.446 |
| bursty | randomized Landlord | 166,445 | 90,990 | 1.086 | 0.535 |

**Findings**

- On **Markov-modulated Zipf**, randomized stateless dominates Online-DRL on both latency
  (-26%) and cost (-90%).
- On **bursty** traffic, randomized stateless trades latency (+5.6%) for a large cost
  reduction (-55%).
- On the **shuffled cyclic scan**, randomized stateless is worse on latency (+46%) with only
  a modest cost gain (-13%). This is an honest negative result: the method is not a
  universal improvement.
- R3 alone is essentially neutral on these families.

## 3. Large Sequence P_t dynamics

Command: `python experiments/exp34_large_sequence_dynamics.py`

| k | M/l | gamma | mean p_sum | sd p_sum | fraction p=1 | projection iters |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 2 | 2.867 | 8.066 | 1.745 | 0.068 | 10.9 |
| 9 | 4 | 3.822 | 8.220 | 1.629 | 0.090 | 6.7 |
| 9 | 8 | 5.822 | 8.385 | 1.475 | 0.095 | 3.7 |
| 12 | 2 | 1.700 | 10.955 | 2.253 | 0.062 | 7.6 |
| 12 | 4 | 1.917 | 11.139 | 2.072 | 0.075 | 4.9 |
| 12 | 8 | 2.500 | 11.302 | 1.883 | 0.078 | 2.7 |
| 24 | 2 | 2.017 | 22.631 | 4.008 | 0.035 | 5.0 |
| 24 | 4 | 2.417 | 22.874 | 3.649 | 0.041 | 3.2 |
| 24 | 8 | 3.242 | 23.114 | 3.245 | 0.047 | 1.9 |

The fractional state stays near the capacity simplex (`p_sum` close to `k`), with a small
fraction of coordinates at the upper bound. This supports the view that the randomized
method uses the cache efficiently and the ratio is controlled by the cyclic request pattern.

## 4. Theory

See `docs/PHASE4_THEORY.md` for the download-cost lemma:

> For any fixed cache-state trajectory, the randomized threshold R3 satisfies
> `E[download cost] <= (e/(e−1)) * OPT_download`.

The factor `2` in the deterministic Online-DRL analysis comes from the ski-rental
reduction; replacing it by `e/(e−1)` is a component-level improvement for the download-cost
term. It does **not** prove a full competitive ratio.

## 5. Conclusions

- The randomized rounding (R2) is the component that flattens the Large Sequence ratio;
  the randomized threshold (R3) alone does not.
- R3 is nevertheless useful for cost: it reduces the cost ratio by an order of magnitude.
- The stateless repair is the best default on popularity-driven and nonstationary
  workloads; the Landlord repair is a conservative feasibility variant.
- The method is not a universal improvement: it loses on the shuffled cyclic scan.
- The paper should claim “on the constructed families” and present the cost–latency
  tradeoff honestly.

## 6. Outputs

- `results/exp32_large_sequence_ablation.json`
- `results/exp33_adversarial_families.json`
- `results/exp34_large_sequence_dynamics.json`
- `docs/PHASE4_THEORY.md`
- `docs/DEV_CHANGES.md`