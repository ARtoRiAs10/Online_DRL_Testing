"""Phase-5 theory check: Proposition 2, zeta/delta, and repair-induced fetches.

This experiment is the numerical counterpart of ``docs/PHASE5_THEORY.md``.

It reports, for both gradient modes (realized / expectation) and both repair rules
(stateless / Landlord):

* the surrogate regret of the **pre-update** OGD sequence and the Proposition-2 bound
  ``sqrt(N) l_max sqrt(T)`` (part i);
* the realized regret against the best fixed fractional cache and the
  ``2 sqrt(N) l_max sqrt(T) + zeta_hat T`` bound (part ii);
* the predictor bias ``zeta`` (signed and mean absolute) and the one-step EMA
  prediction error;
* the rounding distortion ``delta`` (L1 movement, per-request rounding error, 0->1
  churn) and the realized repair-induced fetches.

The G3b download-charged bound is *not* claimed here; it is reported only as a
diagnostic and is explicitly not a theorem (see the phase-5 theory notes).
"""
from common import *
from sim.trace import synthetic_world
from sim.algos import rand_drl
from sim.projection import normalize
from sim.regret import ogd_run, surrogate_regret
from sim.regret_bound import regret_bound_terms

a = parse(__doc__, C=12, K=4)
N = 20 if a.quick else 40
T = 2000 if a.quick else 8000
seeds = 3 if a.quick else a.seeds
print(f"phase-5 theory check: N={N} T={T} C={a.C} K={a.K} seeds={seeds}")

# ---------------------------------------------------------------- deterministic (i)
cat0, tr0 = synthetic_world(N=N, T=10, seed=0)
lmax0 = float(cat0.l.max())
An0 = normalize(np.vstack([np.ones(N), cat0.a.T]), np.r_[float(a.C), 1, 1, 1])
rng = np.random.default_rng(123)
pop = 1.0 / np.arange(1, N + 1) ** 1.2
pop /= pop.sum()
det_rows = []
for trial in range(5):
    n_seq = rng.choice(N, T, p=pop)
    g_seq = rng.uniform(0.5, 1.5, T) * cat0.l[n_seq]
    eta = np.sqrt(N) / (lmax0 * np.sqrt(T))
    P = ogd_run(n_seq, g_seq, An0, eta)
    reg = surrogate_regret(n_seq, g_seq, P, An0)
    bound = float(np.sqrt(N) * lmax0 * np.sqrt(T))
    det_rows.append(dict(trial=trial, regret=float(reg), bound=bound,
                         ratio=float(reg / bound),
                         holds=bool(reg <= bound + 1e-9)))
    print(f"deterministic trial {trial}: regret={reg:.1f} bound={bound:.1f} "
          f"ratio={reg / bound:+.3f} holds={reg <= bound + 1e-9}")

# ---------------------------------------------------------------- stochastic (ii)
rows = []
for grad_mode in ("realized", "expectation"):
    for repair_mode in ("stateless", "landlord"):
        for seed in range(seeds):
            cat, tr = synthetic_world(N=N, T=T, seed=seed, imbalance=2.0, cap_mult=4)
            R = rand_drl(cat, tr, a.C, np.random.default_rng(seed), K=a.K,
                         log_steps=True, record=True, repair_mode=repair_mode,
                         grad_mode=grad_mode, download_time_mode="ema")
            terms = regret_bound_terms(R.extra["step_log"], cat, a.C)
            rows.append(dict(
                grad_mode=grad_mode, repair_mode=repair_mode, seed=seed,
                latency=float(R.latency), cost=float(R.cost),
                surrogate_regret=terms["surrogate_regret_pre"],
                realized_regret=terms["realized_regret_pre"],
                bound_1=terms["prop2_bound_1"],
                bound_2=terms["prop2_bound_2"],
                surrogate_holds=bool(terms["surrogate_regret_pre"]
                                     <= terms["prop2_bound_1"] + 1e-9),
                realized_holds=bool(terms["realized_regret_pre"]
                                    <= terms["prop2_bound_2"] + 1e-9),
                surrogate_holds_tight=bool(terms["surrogate_regret_pre"]
                                           <= terms["prop2_bound_1"] + 1e-9),
                realized_holds_tight=bool(
                    terms["realized_regret_pre"]
                    <= (terms["prop2_bound_1"] if grad_mode == "realized"
                        else terms["prop2_bound_2"]) + 1e-9),
                zeta_signed=terms["zeta_signed"],
                zeta_mae=terms["zeta_mae"],
                zeta_pred_mae=terms["zeta_pred_mae"],
                delta_l1=terms["delta_l1_mean"],
                delta_round=terms["delta_round_mean"],
                delta_churn_rate=terms["delta_churn_rate"],
                repair_evictions=int(R.extra.get("repair_evictions", 0)),
                repair_evicted_requests=int(R.extra.get("repair_evicted_requests", 0)),
                repair_refetches=int(R.extra.get("repair_refetches", 0)),
                repair_refetch_cost=float(R.extra.get("repair_refetch_cost", 0.0)),
                churn_events=int(R.extra.get("churn_events", 0)),
                comparator=terms["realized_comparator"],
                surrogate_loss=terms["surrogate_loss"],
                realized_loss=terms["realized_loss"],
            ))
        print(f"finished grad_mode={grad_mode} repair_mode={repair_mode}")

# ---------------------------------------------------------------- repair-induced fetch probe
# With gate_downloads=True the serving sample must contain the service before a download
# can start, so a repair eviction cannot cause a fetch.  The probe below turns the gate
# off and measures the fetches that repair-induced misses actually cause.
probe_rows = []
for repair_mode in ("stateless", "landlord"):
    for seed in range(min(3, seeds)):
        cat, tr = synthetic_world(N=N, T=T, seed=seed, imbalance=2.0, cap_mult=4)
        R = rand_drl(cat, tr, a.C, np.random.default_rng(seed), K=a.K,
                     log_steps=True, repair_mode=repair_mode,
                     grad_mode="realized", gate_downloads=False)
        probe_rows.append(dict(
            repair_mode=repair_mode, seed=seed,
            latency=float(R.latency), cost=float(R.cost),
            repair_evictions=int(R.extra.get("repair_evictions", 0)),
            repair_evicted_requests=int(R.extra.get("repair_evicted_requests", 0)),
            repair_refetches=int(R.extra.get("repair_refetches", 0)),
            repair_refetch_cost=float(R.extra.get("repair_refetch_cost", 0.0)),
            delta_churn_sum=int(R.extra.get("delta_churn_sum", 0)),
            churn_events=int(R.extra.get("churn_events", 0)),
            downloads=int(R.downloads),
        ))
        print(f"probe {repair_mode} seed {seed}: "
              f"evicted_requests={probe_rows[-1]['repair_evicted_requests']} "
              f"refetches={probe_rows[-1]['repair_refetches']} "
              f"refetch_cost={probe_rows[-1]['repair_refetch_cost']:.0f}")

probe_summary = {}
for repair_mode in ("stateless", "landlord"):
    fr = [r for r in probe_rows if r["repair_mode"] == repair_mode]
    probe_summary[repair_mode] = {
        m: ms([r[m] for r in fr])
        for m in ("repair_evictions", "repair_evicted_requests",
                  "repair_refetches", "repair_refetch_cost",
                  "delta_churn_sum", "churn_events", "downloads")
    }

# ---------------------------------------------------------------- non-peeking check
# The proposition is stated for the pre-update sequence.  The default implementation
# uses peek=True (update before serving).  This probe runs the proper online variant
# (peek=False) and checks that the pre-update regret is the one the theory bounds.
nonpeek_rows = []
for repair_mode in ("stateless", "landlord"):
    for seed in range(min(3, seeds)):
        cat, tr = synthetic_world(N=N, T=T, seed=seed, imbalance=2.0, cap_mult=4)
        R = rand_drl(cat, tr, a.C, np.random.default_rng(seed), K=a.K,
                     log_steps=True, record=True, repair_mode=repair_mode,
                     grad_mode="realized", peek=False, download_time_mode="ema")
        terms = regret_bound_terms(R.extra["step_log"], cat, a.C)
        nonpeek_rows.append(dict(
            repair_mode=repair_mode, seed=seed,
            latency=float(R.latency), cost=float(R.cost),
            surrogate_regret=terms["surrogate_regret_pre"],
            realized_regret=terms["realized_regret_pre"],
            bound_1=terms["prop2_bound_1"],
            bound_2=terms["prop2_bound_2"],
            surrogate_holds=bool(terms["surrogate_regret_pre"]
                                 <= terms["prop2_bound_1"] + 1e-9),
            realized_holds=bool(terms["realized_regret_pre"]
                                <= terms["prop2_bound_2"] + 1e-9),
        ))
        print(f"non-peek {repair_mode} seed {seed}: "
              f"surrogate={nonpeek_rows[-1]['surrogate_regret']:.0f} "
              f"bound1={nonpeek_rows[-1]['bound_1']:.0f} "
              f"realized={nonpeek_rows[-1]['realized_regret']:.0f} "
              f"bound2={nonpeek_rows[-1]['bound_2']:.0f}")

nonpeek_summary = {}
for repair_mode in ("stateless", "landlord"):
    fr = [r for r in nonpeek_rows if r["repair_mode"] == repair_mode]
    nonpeek_summary[repair_mode] = {
        m: ms([r[m] for r in fr])
        for m in ("surrogate_regret", "realized_regret", "bound_1", "bound_2")
    }
    nonpeek_summary[repair_mode]["surrogate_hold_rate"] = float(
        np.mean([r["surrogate_holds"] for r in fr]))
    nonpeek_summary[repair_mode]["realized_hold_rate"] = float(
        np.mean([r["realized_holds"] for r in fr]))

# ---------------------------------------------------------------- summary
summary = {}
for grad_mode in ("realized", "expectation"):
    for repair_mode in ("stateless", "landlord"):
        fr = [r for r in rows if r["grad_mode"] == grad_mode
              and r["repair_mode"] == repair_mode]
        key = f"{grad_mode}/{repair_mode}"
        metrics = ("latency", "cost", "surrogate_regret", "realized_regret",
                   "bound_1", "bound_2", "zeta_signed", "zeta_mae",
                   "zeta_pred_mae", "delta_l1", "delta_round",
                   "delta_churn_rate", "repair_evictions",
                   "repair_evicted_requests", "repair_refetches",
                   "repair_refetch_cost", "churn_events")
        summary[key] = {m: ms([r[m] for r in fr]) for m in metrics}
        summary[key]["surrogate_hold_rate"] = float(
            np.mean([r["surrogate_holds"] for r in fr]))
        summary[key]["realized_hold_rate"] = float(
            np.mean([r["realized_holds"] for r in fr]))
        summary[key]["surrogate_hold_rate_tight"] = float(
            np.mean([r["surrogate_holds_tight"] for r in fr]))
        summary[key]["realized_hold_rate_tight"] = float(
            np.mean([r["realized_holds_tight"] for r in fr]))
        summary[key]["surrogate_ratio"] = ms(
            [r["surrogate_regret"] / r["bound_1"] for r in fr])
        summary[key]["realized_ratio"] = ms(
            [r["realized_regret"] / r["bound_2"] for r in fr])
        summary[key]["surrogate_ratio_tight"] = ms(
            [r["surrogate_regret"] / r["bound_1"] for r in fr])
        summary[key]["realized_ratio_tight"] = ms(
            [r["realized_regret"] / (r["bound_1"] if grad_mode == "realized"
                                     else r["bound_2"]) for r in fr])

out = dict(
    config=dict(N=N, T=T, C=a.C, K=a.K, seeds=seeds,
                grad_modes=["realized", "expectation"],
                repair_modes=["stateless", "landlord"]),
    deterministic=det_rows,
    rows=rows,
    probe_rows=probe_rows,
    probe_summary=probe_summary,
    nonpeek_rows=nonpeek_rows,
    nonpeek_summary=nonpeek_summary,
    summary=summary,
    note=("Proposition 2 is a statement about the pre-update OGD sequence.  "
          "The realized regret is a different (literal) quantity; the G3b "
          "download-charged bound is a diagnostic, not a theorem."),
)
save(a, "exp35_phase5_theory_check", out)

print("\n### summary (mean ± sd)")
for key, s in summary.items():
    print(f"\n{key}")
    for m in ("surrogate_regret", "bound_1", "realized_regret", "bound_2",
              "zeta_signed", "zeta_mae", "zeta_pred_mae", "delta_l1",
              "delta_round", "delta_churn_rate", "repair_evictions",
              "repair_evicted_requests", "repair_refetches",
              "repair_refetch_cost", "churn_events"):
        print(f"  {m:24s} {s[m][0]:12.4g} ± {s[m][1]:.4g}")
    print(f"  surrogate hold rate    {s['surrogate_hold_rate']:.2f}")
    print(f"  realized hold rate     {s['realized_hold_rate']:.2f}")
    print(f"  surrogate / bound      {s['surrogate_ratio'][0]:+.3f}")
    print(f"  realized / bound       {s['realized_ratio'][0]:+.3f}")
    print(f"  realized / tight bound {s['realized_ratio_tight'][0]:+.3f}")

print("\n### non-peeking check (peek=False, realized gradient)")
for key, s in nonpeek_summary.items():
    print(f"\n{key}")
    for m in ("surrogate_regret", "bound_1", "realized_regret", "bound_2"):
        print(f"  {m:24s} {s[m][0]:12.4g} ± {s[m][1]:.4g}")
    print(f"  surrogate hold rate    {s['surrogate_hold_rate']:.2f}")
    print(f"  realized hold rate     {s['realized_hold_rate']:.2f}")

print("\n### repair-induced fetch probe (gate_downloads=False)")
for key, s in probe_summary.items():
    print(f"\n{key}")
    for m in ("repair_evictions", "repair_evicted_requests",
              "repair_refetches", "repair_refetch_cost",
              "delta_churn_sum", "churn_events", "downloads"):
        print(f"  {m:24s} {s[m][0]:12.4g} ± {s[m][1]:.4g}")

print("\n### deterministic Proposition 2(i) check")
for r in det_rows:
    print(f"  trial {r['trial']}: regret={r['regret']:.1f} "
          f"bound={r['bound']:.1f} ratio={r['ratio']:+.3f} holds={r['holds']}")