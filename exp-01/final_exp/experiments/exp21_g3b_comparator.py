"""Phase-3 G3b comparator check.

For every seed we run the randomized DRL with per-step logging, solve the G3b
comparator LP, and compare the realized ``latency + downloads`` with the comparator
objective and the full regret bound.  This is the empirical counterpart of the phase-2
theorem.
"""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, rand_drl
from sim.regret_bound import regret_bound_terms, summarize_step_log

a = parse(__doc__, C=12, K=4, D=10000.0)
N = 16 if a.quick else 20
T = 500 if a.quick else 1000
seeds = min(a.seeds, 10) if not a.quick else a.seeds
print(f"phase-3 G3b comparator: N={N} T={T} K={a.K} C={a.C} D={a.D} seeds={seeds}")

rows = []
for seed in range(seeds):
    cat, tr = synthetic_world(N=N, T=T, seed=seed, imbalance=2.0, cap_mult=4, D=a.D)
    det = online_drl(cat, tr, a.C)
    R = rand_drl(cat, tr, a.C, np.random.default_rng(seed), K=a.K,
                 log_steps=True, repair_mode="landlord", gate_downloads=True)
    terms = regret_bound_terms(R.extra["step_log"], cat, a.C)
    summary = summarize_step_log(R.extra["step_log"])
    realized = float(R.latency + R.cost)
    row = dict(
        seed=seed, repair_mode="landlord", download_time_mode="ema",
        latency=float(R.latency), downloads=float(R.cost), realized=realized,
        det_latency=float(det.latency), det_cost=float(det.cost),
        sum_F_tilde_Pt=terms["sum_F_tilde_Pt"],
        comparator_objective=terms["comparator_objective"],
        comparator_gap=float(realized - terms["comparator_objective"]),
        bound=terms["regret_bound_rhs"],
        bound_gap=float(terms["regret_bound_rhs"] - realized),
        zeta=terms["zeta"], sum_delta=terms["sum_delta_abs_px"],
        base_term=terms["base_term"], ema_term=terms["ema_term"],
        churn_term=terms["churn_term"], budget_term=terms["budget_term"],
        mean_p_sum=summary.get("mean_p_sum"),
        mean_repair_evicted_fraction=summary.get("mean_repair_evicted_fraction"),
    )
    rows.append(row)
    print(f"seed {seed}: realized={realized:.0f} "
          f"comparator={row['comparator_objective']:.0f} "
          f"gap={row['comparator_gap']:+.0f} "
          f"bound={row['bound']:.0f} zeta={row['zeta']:.3f} "
          f"sum|d|={row['sum_delta']:.1f}")

metrics = ("realized", "comparator_objective", "comparator_gap", "bound",
           "bound_gap", "zeta", "sum_delta")
summary = {m: ms([r[m] for r in rows]) for m in metrics}
summary["paired"] = {
    "realized_vs_comparator": paired_ci(
        [r["realized"] for r in rows], [r["comparator_objective"] for r in rows]),
    "realized_vs_det": paired_ci(
        [r["realized"] for r in rows],
        [r["det_latency"] + r["det_cost"] for r in rows]),
}
out = dict(config=dict(N=N, T=T, K=a.K, C=a.C, seeds=seeds), rows=rows, summary=summary)
save(a, "exp21_g3b_comparator", out)

print("\n### summary")
for m in metrics:
    print(f"{m:24s} {summary[m][0]:.1f} ± {summary[m][1]:.1f}")
print("realized - comparator (paired):",
      summary["paired"]["realized_vs_comparator"])
print("realized - Online-DRL (paired):",
      summary["paired"]["realized_vs_det"])