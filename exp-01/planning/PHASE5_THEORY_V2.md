# Phase 5 (v2) — theory: churn, repair fetches, ζ, restricted ratio

**Status: done.**  This note complements `docs/PHASE5_THEORY.md` (the Proposition-2
re-derivation) and records the Phase-5 v2 results from `exp42_theory_phase5_v2.py`.

---

## 1. Proposition 2 (recap)

For the pre-update OGD sequence `p_{t-1}` with `p_t = Π(p_{t-1} + η c~_t e_{n*_t})`:

```
(i)  Σ_t ⟨∇F_t(p_{t-1}), p_{t-1} − P*⟩ ≤ sqrt(N) ℓ_max sqrt(T).
(ii) E[Σ_t L_t(p_{t-1})] − E[min_P Σ_t L_t(P)]
       ≤ 2 sqrt(N) ℓ_max sqrt(T) + ζ T.
```

The proof is in `docs/PHASE5_THEORY.md`.  The factor 2 is the OGD term plus the
martingale term; `ζ` is the predictor bias.

---

## 2. Rounding-churn lemma — rigorous statement and approximation

### 2.1 The clean (independent) case

Let `x_{k,n,t} = 1[U_{k,n} < p_{n,t}]` with persistent uniforms `U_{k,n}` independent
of the past.  Then

```
Pr[x_{k,n,t-1}=0, x_{k,n,t}=1] = (p_{n,t} − p_{n,t-1})_+.
```

Since `p_t, p_{t-1} ∈ D_vec` and the projection is nonexpansive,
`||p_t − p_{t-1}||_2 ≤ η c~_t ≤ η ℓ_max`, so `||p_t − p_{t-1}||_1 ≤ sqrt(N) η ℓ_max`.
With `η = sqrt(N)/(ℓ_max sqrt(T))`,

```
Σ_t Σ_n M_n Pr[x_{k,n,t-1}=0, x_{k,n,t}=1]
    ≤ M_max Σ_t ||p_t − p_{t-1}||_1
    ≤ M_max N sqrt(T).
```

This is the draft's Lemma 3.  It is exact **if the uniforms are independent of the
fractional state**.

### 2.2 Why Algorithm 2 violates the independence assumption

In Algorithm 2, `p_t` is updated with a gradient that depends on `q_t` and `dm_t`;
`q_t` is the in-flight indicator, which depends on whether the *sample* path contains
the service.  Hence `p_t` and the rounding event `x_{k,n,t}` are dependent.  The clean
lemma is therefore an **approximation** for the algorithm itself.

A rigorous version can be written as

```
E[Σ_t Σ_n M_n 1{x_{k,n,t-1}=0, x_{k,n,t}=1}]
  ≤ M_max Σ_t ||p_t − p_{t-1}||_1
    + M_max Σ_t P( q_t changes the update at t ),
```

where the second term is the dependence correction.  The correction is not bounded by
the L1 movement alone; it requires a bound on how often the in-flight indicator changes
the gradient.  We do not have such a bound, so the paper should state Lemma 3 as an
**approximation** and report the measured churn and repair-induced fetches.

### 2.3 Measured terms (exp42, N=20, stateless repair, K=4)

| T | mean `δ_l1` | rounding churn `Σ_t δ_churn` | repair-induced fetches (gate off) | refetch / churn |
|---:|---:|---:|---:|---:|
| 8,000 | 0.0486 | 485 | 1 | 0.002 |
| 16,000 | 0.0352 | 817 | 5 | 0.006 |
| 32,000 | 0.0247 | 1,563 | 11 | 0.007 |

The repair-induced fetches are a small fraction of the rounding churn (0.2–0.7%) and
grow roughly linearly with `T`, like the rounding churn.  The ratio is stable, which
supports treating the repair fetches as a lower-order correction in the churn bound.

---

## 3. Does the EMA drive ζ → 0?

`ζ` is the mean absolute predictor error `|c_t − c~_t|`.  For the realized gradient
`c~_t = c_t`, so `ζ = 0` trivially.  For the expected gradient (EMA), the measured
values are:

| T | ζ_mae (expectation) | ζ_signed |
|---:|---:|---:|
| 8,000 | 0.0335 | −0.0335 |
| 16,000 | 0.0400 | −0.0400 |
| 32,000 | 0.0523 | −0.0523 |

**Result: ζ does not decrease with the horizon; it grows slowly.**  This is consistent
with the first-order EMA tracking error `drift/α + sqrt(α)σ`, which is a constant bias
plus noise, not a vanishing term.  The `ζ T` term in Proposition 2(ii) therefore remains
linear unless the predictor is improved (e.g. a better model of `q_t` and `dm_t`).  The
paper should not claim that the EMA makes the regret `o(T)`; it should say the regret is
`O(sqrt(T) + ζ T)` with a measured `ζ`.

---

## 4. Repair-induced fetches

The repair can evict a service that the next draw would have kept; the next request then
triggers a new download.  This is not covered by the rounding-churn lemma.  The measured
counts (exp35, N=40, T=8000, gate off) are:

| repair rule | repair-induced refetches | rounding churn | ratio |
|---|---:|---:|---:|
| stateless | 45.3 | 154.3 | 0.29 |
| Landlord | 42.0 | 651.7 | 0.06 |

On the longer N=20 horizons (exp42), the ratio is smaller (0.002–0.007) because the
stateless repair is less aggressive.  The paper should report the gate as the mechanism
that prevents repair-induced fetches (`gate_downloads=True` makes them zero by
construction), and report the gate-off counts as the cost of the literal Algorithm 1.

---

## 5. Restricted competitive ratio

Phase 1 succeeded: the randomized stateless repair is on/below the deterministic
frontier over a meaningful range.  This justifies an attempt at a restricted ratio.

**What we can state (empirical, not a proof):**

> On the constructed Large Sequence with `k ≤ 48` and `M/ℓ ∈ {2,4,8}`, the randomized
> stateless repair has latency ratio to the closed-form `OPT = (k+2n)ℓ` with median
> between 1.8 and 4.1, and a 95% interval that stays below 5.8 (exp40).  R1+R2 is the
> component that flattens the ratio; adding R3 gives a small additional improvement.
> The Landlord repair is similar or slightly worse.

**What we cannot state:** a theorem.  The ratio depends on the cyclic request pattern and
the fractional state; a proof would require a contraction/boundedness argument for the
cyclic map `P_t`, which remains open.  The paper should present the restricted ratio as
**empirical evidence on a constructed family**, not as a competitive-ratio theorem.

**Attempted proof route:** on the Large Sequence, `P_t` is driven by a periodic request
pattern.  The OGD update is linear between projections and the projection is
nonexpansive, so one can try to show the cyclic map has a bounded invariant set.  We did
not complete this; the empirical ratio is the evidence.

---

## 6. Summary of Phase-5 v2

1. **Churn lemma:** the clean lemma is exact under independence; Algorithm 2 violates the
   independence assumption, so the lemma is an approximation.  The measured
   repair-induced fetches are 0.2–0.7% of the rounding churn on the N=20 horizons and
   6–29% on the N=40 horizons.
2. **ζ:** the EMA does **not** drive `ζ → 0`; `ζ_mae` grows from 0.0335 to 0.0523 as
   `T` goes from 8k to 32k.  The `ζ T` term remains linear.
3. **Restricted ratio:** Phase 1 succeeded, but a formal restricted ratio remains open;
   the empirical ratio on the Large Sequence is ≤ 5.8 (median 1.8–4.1).