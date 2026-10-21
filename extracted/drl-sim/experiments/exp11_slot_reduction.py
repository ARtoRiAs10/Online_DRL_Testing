"""G5: vector formulation vs slot reductions -- capacity loss Lambda, utilization, stranded capacity,
true-resource violations and latency, on a workload with imbalanced (single-resource-heavy) services."""
from common import *
from sim.trace import synthetic_world
from sim.algos import rand_drl
from sim.metrics import resource_metrics
from sim import slots

a = parse(__doc__, C=40)
cat, tr = synthetic_world(N=a.N, T=a.T, seed=0, imbalance=4.0, cap_mult=5)
A = cat.a
vec = [rand_drl(cat, tr, a.C, np.random.default_rng(s), K=4) for s in range(a.seeds)]
rho = np.mean([r.extra["rho_final"][1:] for r in vec], 0); rho = rho if rho.sum() > 0 else np.ones(3)
variants = {
    "vector": None,
    "dominant (max-norm)": slots.phi_dominant(A),
    "linear alpha=1 (safe)": slots.phi_linear(A, np.ones(3)),
    "l2-norm alpha=1": slots.phi_lp(A, 2),
    "linear entropy (simplex)": slots.phi_linear(A, slots.entropy_weights(A)),
    "linear AHP (simplex)": slots.phi_linear(A, slots.ahp_weights()),
    "shadow price (mean rho)": slots.phi_shadow(A, rho)}
out = {}
for name, phi in variants.items():
    R = vec if phi is None else [rand_drl(cat, tr, a.C, np.random.default_rng(s), K=4, A_rows=phi[None, :]) for s in range(a.seeds)]
    m = [resource_metrics(r.util, r.latency) for r in R]
    out[name] = dict(Lambda=None if phi is None else slots.capacity_loss(A, phi), latency=ms([r.latency for r in R]),
                     Umax=ms([x["Umax_mean"] for x in m]), stranded=ms([x["stranded_mean"] for x in m]),
                     imbalance=ms([x["imbalance_mean"] for x in m]), violation_rate=ms([x["violation_rate"] for x in m]),
                     max_util=float(max(x["max_util"] for x in m)))
    print(f"{name:26s} Lambda={out[name]['Lambda']}  lat={out[name]['latency'][0]:.0f}  Umax={out[name]['Umax'][0]:.3f}  stranded={out[name]['stranded'][0]:.3f}  viol={out[name]['violation_rate'][0]:.3f}")
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4)); ks = list(out)
for i, (k, t) in enumerate((("latency", "latency"), ("stranded", "stranded capacity W"), ("violation_rate", "true-resource violation rate"))):
    ax[i].bar(range(len(ks)), [out[n][k][0] for n in ks]); ax[i].set_xticks(range(len(ks))); ax[i].set_xticklabels(ks, rotation=40, ha="right", fontsize=7); ax[i].set_title(t)
save(a, "exp11_slot_reduction", out, fig)
