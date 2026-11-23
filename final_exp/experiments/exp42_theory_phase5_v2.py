"""Phase-5 (v2): churn approximation, repair-induced fetches, EMA zeta.

For T in {8000, 16000, 32000} on a synthetic world (N=20):

* measure the predictor bias zeta (signed and MAE) for the realized and expected
  gradients, to see whether the EMA drives zeta toward 0 as the horizon grows;
* measure the rounding churn (L1 movement, 0->1 transitions) and the repair-induced
  fetches (gate off), to see whether the ratio stabilises;
* check the churn-lemma approximation error (the independence assumption).

The churn lemma is stated as an approximation in `docs/PHASE5_THEORY_V2.md`; this
experiment measures the terms that the rigorous version would need.
"""
from common import *
from sim.trace import synthetic_world
from sim.algos import rand_drl
from sim.regret_bound import regret_bound_terms

a = parse(__doc__, C=8, K=4)
N = 20
Ts = [8000, 16000, 32000] if not a.quick else [2000, 4000]
print(f"phase-5 v2: N={N} Ts={Ts}")

rows = []
for T in Ts:
    for grad_mode in ("realized", "expectation"):
        cat, tr = synthetic_world(N=N, T=T, seed=0, imbalance=2.0, cap_mult=4)
        R = rand_drl(cat, tr, a.C, np.random.default_rng(0), K=a.K,
                     log_steps=True, record=True, repair_mode="stateless",
                     grad_mode=grad_mode, gate_downloads=True,
                     download_time_mode="ema")
        terms = regret_bound_terms(R.extra["step_log"], cat, a.C)
        # gate-off run for repair-induced fetches
        Rg = rand_drl(cat, tr, a.C, np.random.default_rng(0), K=a.K,
                      log_steps=True, repair_mode="stateless",
                      grad_mode=grad_mode, gate_downloads=False,
                      download_time_mode="ema")
        rows.append(dict(
            T=T, grad_mode=grad_mode,
            zeta_signed=terms["zeta_signed"], zeta_mae=terms["zeta_mae"],
            zeta_pred_mae=terms["zeta_pred_mae"],
            delta_l1=terms["delta_l1_mean"], delta_round=terms["delta_round_mean"],
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
            bound1=terms["prop2_bound_1"], bound2=terms["prop2_bound_2"],
        ))
        print(f"T={T} {grad_mode}: zeta_mae={terms['zeta_mae']:.4f} "
              f"delta_l1={terms['delta_l1_mean']:.4f} "
              f"refetch={rows[-1]['repair_refetches']} "
              f"churn={terms['delta_churn_sum']}")

# summary: does zeta decrease with T?
summary = {}
for grad_mode in ("realized", "expectation"):
    fr = [r for r in rows if r["grad_mode"] == grad_mode]
    summary[grad_mode] = {
        "T": [r["T"] for r in fr],
        "zeta_mae": [r["zeta_mae"] for r in fr],
        "zeta_signed": [r["zeta_signed"] for r in fr],
        "delta_l1": [r["delta_l1"] for r in fr],
        "refetch": [r["repair_refetches"] for r in fr],
        "churn": [r["delta_churn_sum"] for r in fr],
        "refetch_per_churn": [r["repair_refetches"] / max(r["delta_churn_sum"], 1)
                              for r in fr],
    }

out = dict(config=dict(N=N, Ts=Ts, C=a.C, K=a.K), rows=rows, summary=summary,
           note=("The churn lemma is an approximation; this measures the terms.  "
                 "zeta_mae is the mean absolute predictor error; refetch is the number "
                 "of downloads caused by repair evictions (gate off)."))
save(a, "exp42_theory_phase5_v2", out)

print("\n### summary")
for grad_mode, s in summary.items():
    print(f"\n{grad_mode}")
    for i, T in enumerate(s["T"]):
        print(f"  T={T:6d} zeta_mae={s['zeta_mae'][i]:.4f} "
              f"zeta_signed={s['zeta_signed'][i]:+.4f} "
              f"delta_l1={s['delta_l1'][i]:.4f} "
              f"refetch/churn={s['refetch_per_churn'][i]:.3f} "
              f"({s['refetch'][i]}/{s['churn'][i]})")