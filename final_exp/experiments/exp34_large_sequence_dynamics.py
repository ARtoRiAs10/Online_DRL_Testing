"""Phase-4 (optional): empirical dynamics of P_t on the Large Sequence.

Records the fractional state trajectory to see whether the randomized method settles
into a stable cycle or keeps moving.  This is evidence for the family-specific ratio,
not a proof.
"""
from common import *
from sim.trace import large_sequence
from sim.rand_drl import FracPolicy
from sim.engine import Engine, RandThreshold

a = parse(__doc__, C=12, K=4)
ks = [9, 12, 24] if not a.quick else [9]
ratios = [2.0, 4.0, 8.0] if not a.quick else [2.0]
print(f"phase-4 Large Sequence dynamics: k={ks} M/l={ratios}")

rows = []
for k in ks:
    for ratio in ratios:
        cat, tr, opt = large_sequence(k, 2 * k, M=ratio, Ml=2.0 * ratio, l=1.0)
        pol = FracPolicy(cat, k, np.random.default_rng(0), K=a.K,
                         repair_mode="stateless", gate_downloads=True,
                         record=True, log_steps=False)
        R = Engine(cat, pol, RandThreshold(np.random.default_rng(1))).run(tr)
        P = np.asarray(pol.P_hist, float)
        p_sum = P.sum(1)
        row = dict(
            k=k, M_over_l=ratio, opt=float(opt),
            latency=float(R.latency), cost=float(R.cost),
            gamma=float(R.latency / opt),
            p_sum_mean=float(p_sum.mean()), p_sum_std=float(p_sum.std()),
            p_sum_min=float(p_sum.min()), p_sum_max=float(p_sum.max()),
            p_max_mean=float(P.max(1).mean()), p_min_mean=float(P.min(1).mean()),
            frac_at_zero=float((P <= 1e-9).mean()),
            frac_at_one=float((P >= 1 - 1e-9).mean()),
            mean_proj_iters=float(R.extra.get("mean_proj_iters", 0.0)),
            mean_proj_calls=float(R.extra.get("proj_calls", 0.0)),
            mean_rho=float(np.mean(np.abs(R.extra.get("rho_final", np.zeros(1))))),
            hits=int(R.hits), downloads=int(R.downloads),
        )
        rows.append(row)
        print(row)

out = dict(config=dict(ks=ks, ratios=ratios, K=a.K), rows=rows)
save(a, "exp34_large_sequence_dynamics", out)

print("\n### dynamics summary")
for r in rows:
    print(f"k={r['k']:2d} M/l={r['M_over_l']:3.0f} gamma={r['gamma']:.3f} "
          f"p_sum={r['p_sum_mean']:.3f}±{r['p_sum_std']:.3f} "
          f"p_max={r['p_max_mean']:.3f} at0={r['frac_at_zero']:.3f} "
          f"at1={r['frac_at_one']:.3f}")