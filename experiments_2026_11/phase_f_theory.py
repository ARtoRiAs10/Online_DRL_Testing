"""Phase F: churn/rounding distortion, repair-induced fetches, R3 overshoot.

Outputs results/phase_f_theory.json.
"""
from common import *
from sim.regret_bound import regret_bound_terms
from sim.engine import RandThreshold

rows = []
print("Phase F: theory diagnostics")
for T in (8000, 16000, 32000):
    for grad_mode in ("realized", "expectation"):
        cat, tr = synthetic_world(N=20, T=T, seed=0, imbalance=2.0, cap_mult=4)
        R = rand_drl(cat, tr, 8, np.random.default_rng(0), K=4,
                     log_steps=True, record=True, repair_mode="stateless",
                     grad_mode=grad_mode, gate_downloads=True,
                     download_time_mode="ema")
        terms = regret_bound_terms(R.extra["step_log"], cat, 8)
        Rg = rand_drl(cat, tr, 8, np.random.default_rng(0), K=4,
                      log_steps=True, repair_mode="stateless",
                      grad_mode=grad_mode, gate_downloads=False,
                      download_time_mode="ema")
        rows.append(dict(T=T, grad_mode=grad_mode,
                         zeta_signed=terms["zeta_signed"], zeta_mae=terms["zeta_mae"],
                         zeta_pred_mae=terms["zeta_pred_mae"],
                         delta_l1=terms["delta_l1_mean"],
                         delta_round=terms["delta_round_mean"],
                         delta_churn_rate=terms["delta_churn_rate"],
                         delta_churn_sum=terms["delta_churn_sum"],
                         repair_evictions=int(R.extra.get("repair_evictions", 0)),
                         repair_evicted_requests=int(R.extra.get("repair_evicted_requests", 0)),
                         repair_refetches=int(Rg.extra.get("repair_refetches", 0)),
                         repair_refetch_cost=float(Rg.extra.get("repair_refetch_cost", 0.0)),
                         downloads=int(Rg.downloads),
                         latency=float(R.latency), cost=float(R.cost),
                         surrogate_regret=terms["surrogate_regret_pre"],
                         realized_regret=terms["realized_regret_pre"],
                         bound1=terms["prop2_bound_1"], bound2=terms["prop2_bound_2"]))
        print(f"T={T} {grad_mode}: zeta_mae={terms['zeta_mae']:.4f} "
              f"delta_round={terms['delta_round_mean']:.4f} "
              f"refetch={rows[-1]['repair_refetches']}", flush=True)

# F3: R3 threshold support check
rng = np.random.default_rng(0)
M = np.array([1.0, 5.0, 20.0])
thr = RandThreshold(rng)
draws = np.array([[thr.draw(i) * M[i] for _ in range(10000)] for i in range(len(M))])
r3_check = dict(max_ratio=float((draws / M[:, None]).max()),
                min_ratio=float((draws / M[:, None]).min()),
                all_within_support=bool(np.all(draws >= 0) and np.all(draws <= M[:, None] + 1e-12)))

summary = {}
for grad_mode in ("realized", "expectation"):
    fr = [r for r in rows if r["grad_mode"] == grad_mode]
    summary[grad_mode] = {
        "T": [r["T"] for r in fr],
        "zeta_mae": [r["zeta_mae"] for r in fr],
        "delta_l1": [r["delta_l1"] for r in fr],
        "delta_round": [r["delta_round"] for r in fr],
        "refetch": [r["repair_refetches"] for r in fr],
        "churn": [r["delta_churn_sum"] for r in fr],
        "refetch_per_churn": [r["repair_refetches"]/max(r["delta_churn_sum"], 1) for r in fr],
    }
out = dict(config=dict(N=20, Ts=[8000,16000,32000], C=8, K=4), rows=rows,
           summary=summary, r3_check=r3_check,
           note=("F1/F2 measure the churn and rounding-distortion terms; the full "
                 "dependent-churn theorem is not proved. F3 checks the R3 support "
                 "numerically; the analytic bound is theta_n <= M_n."))
save("phase_f_theory", out)
print("\n### summary")
for mode, s in summary.items():
    print(mode)
    for i, T in enumerate(s["T"]):
        print(f"  T={T} zeta={s['zeta_mae'][i]:.4f} delta_round={s['delta_round'][i]:.4f} "
              f"refetch/churn={s['refetch_per_churn'][i]:.4f}")
print("R3 support:", r3_check)
