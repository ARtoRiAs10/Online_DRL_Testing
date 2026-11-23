# Phase 5 report — theory clean-up

**Status: done.  Phase 5 deliverables are in `docs/PHASE5_THEORY.md`,
`final_exp/experiments/exp35_phase5_theory_check.py`, and `results/exp35_phase5_theory_check.json`.**

This phase did four things:

1. re-derived Proposition 2 (surrogate regret) and fixed an indexing error;
2. decided the fate of the old download-charged bound (G3b): removed as a theorem;
3. added per-step measurements of the predictor bias `ζ` and the rounding distortion `δ`;
4. separated repair-induced fetches from rounding churn and measured both.

---

## 1. Proposition 2 (re-derived)

The draft's Proposition 2(i) wrote the inner product with `P_t` but its proof used
`p_{t-1}`.  The correct statement is for the **pre-update** sequence:

```
Σ_t ⟨ ∇F_t(p_{t-1}), p_{t-1} − P* ⟩ ≤ sqrt(N) ℓ_max sqrt(T),
```

with `p_0 = 0` and `η = sqrt(N)/(ℓ_max sqrt(T))`.  The proof is the standard projected-OGD
telescoping argument; it needs an exact Euclidean projection.

Proposition 2(ii) decomposes `c_t = c~_t + ζ_t + ε_t` and gives

```
E[ Σ_t L_t(p_{t-1}) ] − E[ min_P Σ_t L_t(P) ]
    ≤ 2 sqrt(N) ℓ_max sqrt(T) + ζ T.
```

The factor 2 is the OGD term plus the martingale term.  If `c~_t = c_t` (the
`grad_mode="realized"` implementation), `ζ = ε = 0` and the bound reduces to the one-term
`sqrt(N) ℓ_max sqrt(T)`.

The full derivation, the conditions on the request sequence, and the projection caveat are
in `docs/PHASE5_THEORY.md`.

**Code fix.**  `sim/regret.py::ogd_run` used to return the post-update state when
`peek=True`, which made the measured regret inconsistent with the bound.  It now returns
the pre-update (played) state in both modes.  `FracPolicy` now records `P_pre_hist` and
`P_post_hist` separately; `exp35` uses the pre-update history for the theory object.

---

## 2. G3b — removed as a theorem

The old draft's download-charged bound contained a `M_max N sqrt(T)` term intended to cover
repair-induced fetches.  No derivation supports it:

- Lemma 3 bounds only the *rounding* churn
  `Pr[x_{k,n,t-1}=0, x_{k,n,t}=1] = (p_{n,t} − p_{n,t-1})_+`;
- a repair eviction can remove a service that the next draw would have kept, and the next
  request then triggers a new download that Lemma 3 does not count;
- the `M_max N sqrt(T)` term is a placeholder.

**Decision:** keep `sim/regret_bound.py` as a diagnostic, remove G3b from the claims, and
state in the paper that the download-charged regret is open.

**Measured gap (exp35, N=40, T=8000, 10 seeds, `gate_downloads=False`).**

| repair rule | repair evictions | repair-evicted requests | repair-induced refetches | refetch cost | rounding churn (Σ δ_churn) |
|---|---:|---:|---:|---:|---:|
| stateless | 18,740 | 283.7 | 45.3 | 6,352 | 154.3 |
| Landlord | 21,811 | 556.7 | 42.0 | 5,992 | 651.7 |

Repair-induced refetches are ~29% of the rounding churn for the stateless repair and ~6%
for Landlord.  With the default download gate on, they are zero by construction: a download
can only start if the serving sample contains the service, and a repair eviction removes
it.  The paper should report the gate as the mechanism that prevents repair-induced
fetches, and the gate-off numbers as the cost of the literal Algorithm 1.

---

## 3. `ζ` and `δ` — measured in every run

### 3.1 Predictor bias

`sim/rand_drl.py` now logs, for every step:

- `c_realized = (1−q) l_n + q min(l_n, dm)`;
- `c_pred = (1−q_ema,t−1) l_n + q_ema,t−1 min(l_n, dl_ema,t−1)`;
- `zeta = c_realized − grad`;
- `zeta_pred = c_realized − c_pred`.

`diagnostics()` reports `mean_zeta`, `mean_abs_zeta`; `regret_bound_terms()` reports
`zeta_signed`, `zeta_mae`, `zeta_pred_mae`.

**Measured (N=40, T=8000, 10 seeds):**

| grad mode | repair rule | ζ_signed | ζ_mae | ζ_pred_mae |
|---|---|---:|---:|---:|
| realized | stateless | 0.000 | 0.000 | 0.000 |
| realized | Landlord | 0.000 | 0.000 | 0.000 |
| expectation | stateless | −0.049 | 0.049 | 0.049 |
| expectation | Landlord | −0.288 | 0.289 | 0.289 |

The negative sign means the EMA predictor overestimates the realized cost on average.  The
bias is small relative to `ℓ_max ≈ 21 ms`, but the `ζ T` term is `O(T)`.

### 3.2 Rounding distortion

`sim/rand_drl.py` now logs:

- `delta_l1 = ||p_t − p_{t-1}||_1`;
- `delta_round = |p_{n*_t,used} − x_{k*,n*_t,t}|`;
- `delta_churn = 1[x_{k*,n*_t}(t-1)=0, x_{k*,n*_t}(t)=1]`;
- `delta_churn_all = Σ_n 1[x_{k*,n}(t-1)=0, x_{k*,n}(t)=1]`.

`diagnostics()` reports `mean_delta_l1`, `mean_delta_round`, `delta_churn_sum`,
`delta_churn_all_sum`; `regret_bound_terms()` reports `delta_l1_mean`,
`delta_round_mean`, `delta_churn_rate`, `delta_churn_sum`, `delta_churn_all_sum`.

**Measured (N=40, T=8000, 10 seeds):**

| grad mode | repair rule | mean δ_l1 | mean δ_round | δ_churn rate | Σ δ_churn_all |
|---|---|---:|---:|---:|---:|
| realized | stateless | 0.0769 | 0.106 | 0.0155 | 154.3 |
| realized | Landlord | 0.0767 | 0.199 | 0.1067 | 651.7 |
| expectation | stateless | 0.0772 | 0.105 | 0.0151 | 154.3 |
| expectation | Landlord | 0.0772 | 0.196 | 0.1071 | 651.7 |

The stateless repair has much lower rounding distortion and churn than the Landlord
repair, consistent with the Phase-2 utilization result.

---

## 4. exp35 — numerical check

`python experiments/exp35_phase5_theory_check.py` (full: N=40, T=8000, 10 seeds):

| quantity | realized/stateless | realized/Landlord | expectation/stateless | expectation/Landlord |
|---|---:|---:|---:|---:|
| surrogate regret | 3,883 | 3,859 | 3,891 | 3,891 |
| Prop. 2(i) bound | 11,879 | 11,879 | 11,879 | 11,879 |
| realized regret | 3,883 | 3,859 | 3,869 | 3,826 |
| Prop. 2(ii) bound | 23,759 | 23,759 | 24,151 | 26,070 |
| surrogate / bound | 0.327 | 0.325 | 0.327 | 0.327 |
| realized / bound | 0.163 | 0.163 | 0.160 | 0.147 |
| surrogate hold rate | 1.00 | 1.00 | 1.00 | 1.00 |
| realized hold rate | 1.00 | 1.00 | 1.00 | 1.00 |

The deterministic Proposition 2(i) check (5 random exogenous sequences) also holds on
every trial (regret ≈ 0.35–0.38 of the bound).

**Non-peeking check (`peek=False`, realized gradient).**  The proposition is stated for
the pre-update sequence; the proper online variant plays exactly that sequence.  The
non-peeking runs give surrogate/realized regret 3,680 (stateless) and 3,676 (Landlord)
against the same 11,879 bound (ratio ≈ 0.31), with a 1.00 hold rate.  The default
`peek=True` variant is an optimistic heuristic; its regret on the pre-update sequence is
3,883 / 3,859, also below the bound, but the peeking update itself is not covered by the
proposition without the extra `Σ_t ⟨g_t, p_t − p_{t-1}⟩` term.

**Interpretation.**

- The surrogate regret is about one third of the Proposition-2(i) bound in every
  configuration.  The bound is valid but loose.
- The realized regret is about one sixth of the general Proposition-2(ii) bound.  For
  `grad_mode="realized"` it equals the surrogate regret, as expected.
- The `ζ_mae T` term is small compared with the OGD term; the bound is dominated by
  `2 sqrt(N) ℓ_max sqrt(T)` in this world.
- The literal realized-latency regret (exp09) is a different quantity: 3–10x the bound and
  growing with `T`.  Proposition 2 is a surrogate statement, not a competitive-ratio
  theorem.

---

## 5. Files changed

- `final_exp/sim/regret.py` — `ogd_run` returns pre-update states; docstring corrected.
- `final_exp/sim/rand_drl.py` — per-step `c_realized`, `c_pred`, `zeta`, `zeta_pred`,
  `delta_l1`, `delta_round`, `delta_churn`, `delta_churn_all`; `P_pre_hist`/`P_post_hist`;
  repair-evicted-request and repair-refetch counters; diagnostics.
- `final_exp/sim/regret_bound.py` — Proposition-2 fields (`surrogate_regret_pre`,
  `realized_regret_pre`, `prop2_bound_1`, `prop2_bound_2`, `zeta_signed`, `zeta_mae`,
  `zeta_pred_mae`, `delta_*`), while keeping the G3b diagnostic fields for backwards
  compatibility.
- `final_exp/experiments/exp35_phase5_theory_check.py` — new experiment.
- `final_exp/tests/test_phase5.py` — new tests (pre-update OGD, phase-5 log fields,
  `ζ` modes, Prop-2 terms, repair-refetch counters, churn consistency).
- `docs/PHASE5_THEORY.md` — full re-derivation, G3b decision, `ζ`/`δ` definitions.
- `docs/PHASE5_REPORT.md` — this report.

---

## 6. Remaining Phase 5 items

- [x] Re-derive Proposition 2 (pre-update indexing fixed).
- [x] Decide G3b: removed as a theorem; diagnostic kept.
- [x] Measure `ζ` and `δ` in every run.
- [x] Bound/measure repair-induced fetches separately from rounding churn.
- [ ] (optional) Prove the peeking-variant regret with the extra
  `Σ_t ⟨g_t, p_t − p_{t-1}⟩` term.
- [ ] (optional) Extend the `e/(e−1)` lemma to the multi-resource download-cost term
  with repair.
- [ ] (optional) Make the EMA tracking-error bound exact for the implementation.