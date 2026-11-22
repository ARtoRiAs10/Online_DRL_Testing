"""Phase-5 long-horizon G3b bound check.

Runs the randomized DRL with per-step logging on a long trace, computes every RHS term
of the download-charged regret bound, and reports the gap to the realized
``latency + downloads``.  This is the long-horizon companion of exp21.
"""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, rand_drl
from sim.regret_bound import regret_bound_terms

a = parse(__doc__, C=12, K=4, D=10000.0)
N = 16 if a.quick else 20
T = 2000 if a.quick else 10000
seeds = min(a.seeds, 5) if not a.quick else a.seeds
print(f"phase-5 long horizon: N={N} T={T} K={a.K} C={a.C} seeds={seeds}")

rows = []
for seed in range(seeds):
    cat, tr = synthetic_world(N=N, T=T, seed=seed, imbalance=2.0, cap_mult=4, D=a.D)
    det = online_drl(cat, tr, a.C)
    R = rand_drl(cat, tr, a.C, np.random.default_rng(seed), K=a.K,
                 log_steps=True, repair_mode="landlord",
                 gate_downloads=True, download_time_mode="ema")
    terms = regret_bound_terms(R.extra["step_log"], cat, a.C)
    realized = float(R.latency + R.cost)
    rows.append(dict(
        seed=seed, repair_mode="landlord", download_time_mode="ema",
        latency=float(R.latency), downloads=float(R.cost),
        realized=realized, det_latency=float(det.latency), det_cost=float(det.cost),
        comparator_objective=float(terms["comparator_objective"]),
        comparator_gap=float(realized - terms["comparator_objective"]),
        bound=float(terms["regret_bound_rhs"]),
        bound_gap=float(terms["regret_bound_rhs"] - realized),
        zeta_q_hat=float(terms["zeta_q_hat"]),
        zeta_dl_hat=float(terms["zeta_dl_hat"]),
        zeta_hat=float(terms["zeta_hat"]),
        sum_delta=float(terms["sum_delta_abs_px"]),
        churn_credit=float(R.extra.get("churn_credit", 0.0)),
    ))
    print(f"seed {seed}: realized={realized:.0f} "
          f"comparator={rows[-1]['comparator_objective']:.0f} "
          f"bound={rows[-1]['bound']:.0f} "
          f"zeta_q={rows[-1]['zeta_q_hat']:.3f} "
          f"zeta_dl={rows[-1]['zeta_dl_hat']:.3f}")

metrics = ("realized", "comparator_objective", "comparator_gap", "bound",
           "bound_gap", "zeta_q_hat", "zeta_dl_hat", "zeta_hat",
           "sum_delta", "churn_credit")
summary = {m: ms([r[m] for r in rows]) for m in metrics}
out = dict(config=dict(N=N, T=T, K=a.K, C=a.C, D=a.D, seeds=seeds),
           rows=rows, summary=summary)
save(a, "exp26_long_horizon_bound", out)

print("\n### long-horizon summary")
for m in metrics:
    print(f"{m:24s} {summary[m][0]:.1f} ± {summary[m][1]:.1f}")