"""Phase-2 numerical check of the download-charged regret bound (G3b).

Runs a short logged randomized-DRL trace, extracts every RHS term from the per-step
log, solves the comparator LP, and compares the bound with the realized
``latency + downloads``.  This is a consistency check, not a proof: the theorem's LHS
is an expectation and the surrogate ``F~_t`` is not the realized latency.
"""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, rand_drl
from sim.regret_bound import regret_bound_terms, summarize_step_log

a = parse(__doc__, C=12, K=4)
N = 16 if a.quick else 20
T = 500 if a.quick else 1000
K = a.K
C = a.C
cat, tr = synthetic_world(N=N, T=T, seed=0, imbalance=2.0, cap_mult=4)
det = online_drl(cat, tr, C)
print(f"phase-2 bound check: N={N} T={T} K={K} C={C} seeds={a.seeds}")
print(f"Online-DRL: latency={det.latency:.0f} cost={det.cost:.0f}")

rows = []
for seed in range(a.seeds):
    R = rand_drl(cat, tr, C, np.random.default_rng(seed), K=K, log_steps=True,
                 repair_mode="landlord", gate_downloads=True)
    terms = regret_bound_terms(R.extra["step_log"], cat, C)
    summary = summarize_step_log(R.extra["step_log"])
    row = {
        "seed": seed,
        "latency": float(R.latency),
        "downloads": float(R.cost),
        "realized_latency_plus_downloads": float(R.latency + R.cost),
        "sum_F_tilde_Pt": terms["sum_F_tilde_Pt"],
        "comparator_objective": terms["comparator_objective"],
        "base_term": terms["base_term"],
        "zeta": terms["zeta"],
        "ema_term": terms["ema_term"],
        "sum_delta_abs_px": terms["sum_delta_abs_px"],
        "churn_term": terms["churn_term"],
        "budget_term": terms["budget_term"],
        "regret_bound_rhs": terms["regret_bound_rhs"],
        "bound_minus_realized": float(terms["regret_bound_rhs"] -
                                      (R.latency + R.cost)),
        "mean_repair_evicted_fraction": summary.get("mean_repair_evicted_fraction"),
        "mean_p_sum": summary.get("mean_p_sum"),
    }
    rows.append(row)
    print(f"  seed {seed}: realized={row['realized_latency_plus_downloads']:.0f} "
          f"rhs={row['regret_bound_rhs']:.0f} "
          f"gap={row['bound_minus_realized']:+.0f} "
          f"zeta={row['zeta']:.3f} "
          f"sum|delta|={row['sum_delta_abs_px']:.3f}")

out = {
    "config": dict(N=N, T=T, K=K, C=C, seeds=a.seeds),
    "online_drl": dict(latency=float(det.latency), cost=float(det.cost)),
    "rows": rows,
    "mean": {k: float(np.mean([r[k] for r in rows]))
             for k in rows[0] if k != "seed"},
}
save(a, "phase2_bound_check", out)