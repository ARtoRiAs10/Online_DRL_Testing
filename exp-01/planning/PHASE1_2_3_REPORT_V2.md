# Phase 1–3 report v2 — frontier, horizon, adversarial

**Status: done.**  All new files are in `planning/` and are **not committed**.

---

## Phase 1 — deterministic cost–latency frontier (`exp38`)

### Synthetic world (N=40, T=8000, 10 seeds)

| θ | Online-DRL latency | Online-DRL cost |
|---:|---:|---:|
| 0 | 144,537 | 366,644 |
| 0.25 | 140,543 | 254,503 |
| 0.5 | 141,548 | 223,834 |
| 1 | 141,598 | 184,001 |
| 2 | 141,210 | 135,412 |
| 4 | 142,294 | 89,424 |
| 8 | 144,962 | 51,024 |
| 16 | 143,712 | 24,691 |

Randomized stateless (same seeds):

| variant | latency | cost |
|---|---:|---:|
| default (η×1, α=0.1, K=4) | **93,237 ± 20,785** | **12,011 ± 6,363** |
| η×4 | 119,024 ± 27,767 | 66,347 ± 16,621 |
| K=8 | 99,988 ± 25,936 | 15,528 ± 10,051 |

**Result:** the randomized default is **below the entire Online-DRL frontier**: lower
latency than θ=0 (144,537) and lower cost than θ=16 (24,691).  It is not a
single-point trade-off; it dominates the deterministic curve in this world.

### Google 10k slice (3 seeds, K=4)

| θ | latency | cost |
|---:|---:|---:|
| 0 | 161,818 | 82,510 |
| 1 | 161,833 | 44,956 |
| 4 | 161,982 | 3,044 |
| 16 | 163,253 | 2,830 |

Randomized stateless: **161,669 ± 22,536**, **34,002 ± 5,737**.

**Result:** slightly lower latency than every θ; lower cost than θ=0/1, but higher cost
than θ=4/16.  On/below the frontier in the moderate-cost range; at very low cost the
deterministic method is cheaper with essentially the same latency.

### Google 50k (3 seeds, K=2)

| θ | latency | cost |
|---:|---:|---:|
| 0 | 689,898 | 107,107 |
| 1 | 690,095 | 64,688 |
| 4 | 690,530 | 13,259 |
| 16 | 697,240 | 12,687 |

Randomized stateless: **680,384 ± 181,273**, **24,831 ± 5,985**.

**Result:** lower latency than every θ; lower cost than θ=0/1, higher than θ=4/16.
On/below the frontier in the moderate-cost range.

### Budget-matched table (synthetic, 10 seeds)

| `B_max` | Online-DRL latency | Randomized latency | paired Δlat | paired Δcost |
|---:|---:|---:|---:|---:|
| 20% | 154,637 | 97,257 | **−57,380 ± 16,679** | −28,572 ± 3,597 |
| 50% | 148,803 | 94,712 | **−54,091 ± 11,515** | −82,591 ± 5,154 |
| 100% | 141,598 | 93,941 | **−47,658 ± 11,721** | −174,005 ± 7,873 |

**Phase-1 decision:** randomized stateless is on or below the deterministic frontier over
a meaningful range (synthetic and moderate-cost real traces).  The claim can be
"improves the cost–latency tradeoff", with the caveat that at very low download budgets
the deterministic method is cheaper at similar latency.

---

## Phase 2 — horizon and consistency (`exp39`, `exp31`)

### Trace-prefix sweep (50k Google, 3 seeds, K=2)

| prefix | raw Δlat | raw Δcost | crossing Δlat | crossing Δcost |
|---:|---:|---:|---:|---:|
| 10k | +221 ± 169 | −18,498 ± 79 | +353 ± 143 | −1,676 ± 1,741 |
| 20k | +50 ± 129 | −27,551 ± 158 | −4 ± 140 | −14,371 ± 887 |
| 30k | +159 ± 109 | −32,732 ± 187 | −54 ± 164 | −20,902 ± 813 |
| 50k | **−2,448 ± 676** | **−42,658 ± 206** | **−2,215 ± 782** | **−33,867 ± 1,021** |

**Result:** the latency penalty shrinks with the horizon and becomes a gain at 50k;
the cost reduction grows monotonically.  This supports the warm-up hypothesis.

### MILP ALG/OPT vs T (`exp31`)

| T | Online-DRL | R3 only | Landlord | Stateless |
|---:|---:|---:|---:|---:|
| 120 | 1.168 | 1.152 | 1.427 | 1.301 |
| 300 | 1.456 | 1.485 | 1.981 | 1.631 |
| 600 | 1.717 | 1.761 | 2.251 | 1.675 |
| 1200 | 1.753 | 1.819 | 2.188 | 1.622 |

The stateless gap to Online-DRL closes: **+11% at T=120, +12% at T=300, −2% at T=600,
−7% at T=1200**.  The absolute ratio grows with T for all algorithms, but the randomized
stateless gap closes.  This supports the warm-up explanation.

**MILP objective:** latency only, subject to the download budget and per-step
CPU/RAM/disk capacities.  Download cost is charged in the constraint, not in the
objective.

---

## Phase 3 — adversarial families and ablation (`exp40`)

### A. Shuffled cyclic scan — where is the +46% latency loss?

| variant | latency | ratio vs Online-DRL |
|---|---:|---:|
| Online-DRL | 90 | 1.000 |
| R3 only | 90 | 1.000 |
| R1+R2 stateless (det threshold) | 128 | 1.490 |
| R1+R2 stateless + gate | 128 | 1.490 |
| R1+R2+R3 stateless + gate | 128 | 1.490 |
| R1+R2+R3 stateless, no gate | 128 | 1.490 |
| R1+R2+R3 Landlord + gate | 128 | 1.483 |

**Result:** the loss is caused by **R1+R2 (randomized rounding + repair)**, not by R3 or
the download gate.  R3 alone is neutral on this family.  The fractional OGD state does
not align with the cyclic scan, and the rounding/repair causes extra misses.

### B. Large Sequence ablation (median latency ratio to closed-form OPT)

| k | M/l | R3 only | R1+R2 stateless | R1+R2+R3 stateless | R1+R2+R3 Landlord |
|---:|---:|---:|---:|---:|---:|
| 9 | 2 | 7.867 | 2.444 | 2.200 | 2.533 |
| 9 | 4 | 12.844 | 3.000 | 2.778 | 3.222 |
| 9 | 8 | 22.444 | 3.133 | 2.844 | 2.844 |
| 12 | 2 | 10.267 | 1.900 | 1.817 | 2.117 |
| 12 | 4 | 16.633 | 2.283 | 2.117 | 2.300 |
| 12 | 8 | 29.067 | 2.883 | 2.817 | 2.817 |
| 24 | 2 | 19.675 | 2.067 | 1.967 | 2.108 |
| 24 | 4 | 32.825 | 2.483 | 2.400 | 2.517 |
| 24 | 8 | 58.433 | 3.117 | 3.025 | 3.258 |
| 48 | 2 | 38.317 | 2.592 | 2.475 | 2.533 |
| 48 | 4 | 64.692 | 3.121 | 2.962 | 2.962 |
| 48 | 8 | 116.033 | 4.033 | 3.900 | 3.900 |

**Result:** R1+R2 flattens the ratio; adding R3 slightly improves it at most k; Landlord
is similar or slightly worse than stateless.  R3 alone remains linear in k.  Medians and
IQRs are reported; the mean can be higher due to outliers (e.g. k=9 M/l=8 median 2.84,
mean 3.55).

### C. Additional families (median latency, ratio vs Online-DRL)

| family | Online-DRL | R3 only | R1+R2 stateless | R1+R2+R3 stateless | Landlord |
|---|---:|---:|---:|---:|---:|
| hotspot | 136,468 | 133,638 (0.990) | **115,830 (0.862)** | 120,137 (0.890) | 157,961 (1.150) |
| markov_zipf | 141,701 | 140,453 (1.007) | **95,217 (0.709)** | 96,116 (0.716) | 162,782 (1.147) |
| periodic_burst | 138,933 | 136,276 (0.982) | **85,964 (0.633)** | 87,155 (0.642) | 162,524 (1.151) |
| random_walk_zipf | 87,933 | 88,540 (1.001) | **69,932 (0.739)** | 69,726 (0.748) | 134,859 (1.356) |
| bursty | 153,050 | 151,364 (0.992) | 159,118 (1.034) | 161,108 (1.043) | 166,748 (1.086) |

**Result:** stateless is better on hotspot, Markov-Zipf, periodic-burst and random-walk
Zipf; slightly worse on bursty; loses on the shuffled cyclic scan.  Landlord is worse on
all five.  The random-walk Zipf has high variance (median 0.739, IQR 0.626–0.762, mean
0.823 ± 0.453); the median is the more robust summary.

### D. OPT check

| k | M/l | closed `(k+2n)l` | MILP | MILP/closed |
|---:|---:|---:|---:|---:|
| 3 | 2 | 15 | 14 | 0.933 |
| 3 | 4 | 15 | 14 | 0.933 |
| 6 | 2 | 30 | 29 | 0.967 |
| 6 | 4 | 30 | 29 | 0.967 |

**Result:** the closed form `(k+2n)l` is an **upper bound**, not the exact optimum; the
MILP is 1 unit lower for every instance checked.  The ratios in the Large Sequence
ablation are therefore slightly overestimated (by ~3–7%).  The paper should either use
the MILP optimum where tractable or state `OPT ≤ (k+2n)l`.

---

## Bottom line for Phases 1–3

1. **Phase 1:** randomized stateless is on/below the deterministic frontier over a
   meaningful range; the budget-matched table shows a large latency reduction at equal
   budget.  The claim "improves the cost–latency tradeoff" is supported.
2. **Phase 2:** the latency sign change is a warm-up/horizon effect; the ALG/OPT gap to
   Online-DRL closes with T.
3. **Phase 3:** the shuffled-cyclic loss is a rounding/repair effect, not R3 or the gate.
   R1+R2 is the component that flattens the Large Sequence ratio; +R3 gives a small
   additional improvement.  Landlord is consistently worse than stateless.  The
   closed-form OPT is an upper bound, not exact.
4. **Caveats:** at very low download budgets the deterministic method is cheaper on the
   Google traces; the random-walk Zipf has high variance; the OPT check is only for
   small k (3 and 6).