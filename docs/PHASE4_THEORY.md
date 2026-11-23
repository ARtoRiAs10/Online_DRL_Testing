# Phase 4 — theory: the e/(e−1) download-cost lemma

**Not committed.**

## 1. Where the factor 2 enters

The deterministic Online-DRL analysis uses a ski-rental reduction for the download
decision. For a miss that would benefit from a download, the deterministic rule waits
until the accumulated latency reaches `M_n` and then downloads. In the worst case the
algorithm pays `2 M_n`: it waits `M_n` and then pays `M_n` for the download, while the
offline optimum pays `M_n`. This is the source of the factor `2` in the download-cost term
of the competitive-ratio proof.

The randomized threshold R3 replaces the deterministic threshold by
`θ_n ~ f(θ) = e^{θ/M_n} / (M_n (e−1))` on `[0, M_n]`. The classic ski-rental analysis
shows that the randomized rule pays at most `e/(e−1) M_n` in expectation, which is strictly
better than `2 M_n`.

## 2. Lemma (download-cost term)

**Lemma.** Fix any request sequence and any cache state trajectory. Consider only the
download-cost subproblem: whenever a service is not cached and a download is triggered, the
algorithm pays `M_n`; the offline optimum pays `M_n` if it downloads. Then R3 satisfies

```
E[ download cost of R3 ] <= (e / (e - 1)) * OPT_download.
```

**Proof sketch.**

1. Condition on the cache state and the miss epoch. The decision is a rent-or-buy problem:
   wait `t` and then download, or download immediately.
2. The deterministic rule has worst-case ratio `2` at `t = M_n`.
3. The randomized threshold with density
   `f(θ) = e^{θ/M_n} / (M_n (e−1))` has expected cost
   `E[cost(t)] = ∫_0^{min(t,M_n)} (θ + M_n) f(θ) dθ + t (1 - F(min(t,M_n)))`.
4. A direct calculation gives `E[cost(t)] / min(t, M_n) <= e/(e−1)` for all `t`, with
   equality at `t = M_n`.
5. Summing over miss epochs preserves the bound because the cache-state trajectory is
   fixed and the threshold draws are independent of the future. ∎

**Scope.** This lemma bounds only the download-cost component. It does **not** bound the
latency component, and it does **not** prove a competitive ratio for the full algorithm.
It is a component-level replacement for the factor 2 in the Online-DRL proof.

## 3. How to use it in the paper

- State the lemma after the randomized threshold definition.
- In the competitive-ratio proof, replace the deterministic download-cost term `2 M_n` by
  `(e/(e−1)) M_n`.
- Keep the latency term separate; the lemma says nothing about it.
- Be explicit: the full ratio remains empirical on the Large Sequence and other families.

## 4. Large Sequence ratio — what is proved and what is measured

The Large Sequence has a fixed cyclic structure. On that family:

- `OPT = (k + 2 n_reps) l`;
- deterministic Online-DRL has `gamma = 2.00 k + 0.333` (`R^2 = 1.000`);
- R3 alone has `gamma ≈ 1.88 k` (still linear);
- the full randomized method has `gamma ≈ 3–5` for `k = 3..48`, with a linear-fit slope
  near `0.01–0.07` depending on `M/l`.

This is **evidence on constructed families**, not a general competitive-ratio proof. The
paper should say “on the constructed families” rather than “removes the linear-in-k ratio”.

## 5. Cyclic dynamics

On the Large Sequence, `P_t` is driven by a periodic request pattern. The OGD update is
linear between projections, and the projection is nonexpansive. One can analyse the
resulting cyclic map to bound `E[ALG]/OPT` for that family. The empirical dynamics check
(`exp34_large_sequence_dynamics.py`) records:

- `p_sum` mean/std/min/max;
- fraction of `p_n` at 0 and 1;
- projection iterations and dual variables;
- realized `gamma`.

This is a starting point for a family-specific proof, not a proof itself.

## 6. Open theory items

- [ ] Prove the cyclic map contraction / boundedness for the Large Sequence.
- [ ] Extend the `e/(e−1)` lemma to the multi-resource download-cost term with repair.
- [ ] Bound repair-induced fetches separately from rounding churn.
- [ ] Make the EMA tracking-error bound exact for the implementation.