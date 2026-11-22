"""Multi-resource (CPU, RAM, disk) comparison -- Sec. 1-4 of randomized_drl_multiresource.pdf.

For every algorithm we report
  * latency, cost, #downloads                                        (Sec. 4.2 / 4.3)
  * utilization U_{r,t} per resource, U^max_t, imbalance I_t, stranded capacity W_t  (eqs. 30-33)
  * violation rate of the *true* multi-resource constraints            (goal G1)
  * latency per unit of bottleneck resource                           (eq. 34)
  * Lambda_phi: theoretical (eq. 18) and empirical estimate of the slot-reduction loss
and we compare the *vector* formulation (D_vec, eq. 8) with the *slot* formulations
(dominant resource / l_p / linear scalarization / shadow prices) of Sec. 2.
"""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, ll_rc, rand_landlord, rand_drl
from sim.metrics import resource_metrics, pct
from sim import slots

a = parse(__doc__, C=12)
ALGOS = {
    "Online-DRL (det)": lambda cat, tr, C, s: online_drl(cat, tr, C),
    "LL-RC (theta=0)": lambda cat, tr, C, s: ll_rc(cat, tr, C),
    "R3 only (rand thr + Landlord)": lambda cat, tr, C, s: rand_landlord(cat, tr, C, np.random.default_rng(s)),
    "Randomized DRL (R1+R2+R3)": lambda cat, tr, C, s: rand_drl(cat, tr, C, np.random.default_rng(s), K=4),
}

for imbalance, tag in ((1.0, "balanced"), (4.0, "imbalanced")):
    print("=" * 78, "\nworkload:", tag, flush=True)
    out = {}
    for seed in range(a.seeds):
        cat, tr = synthetic_world(N=a.N, T=a.T, seed=seed, imbalance=imbalance, cap_mult=5)
        for name, fn in ALGOS.items():
            R = fn(cat, tr, a.C, seed)
            m = resource_metrics(R.util, R.latency)
            out.setdefault(name, []).append(dict(
                latency=R.latency, cost=R.cost, downloads=R.downloads,
                hits=R.hits, delayed_hits=R.delayed_hits,
                U=m["U_mean"], Umax=m["Umax_mean"], imbalance=m["imbalance_mean"],
                stranded=m["stranded_mean"], viol=m["violation_rate"],
                lat_per_res=m["latency_per_unit_resource"],
                Lambda_hat=slots.lambda_estimate(R.util)))
    for name, rows in out.items():
        agg = {k: ms([r[k] for r in rows]) for k in ("latency", "cost", "downloads", "Umax", "imbalance",
                                                    "stranded", "viol", "lat_per_res", "Lambda_hat")}
        print(f"{name:34s} lat={agg['latency'][0]:10.1f} cost={agg['cost'][0]:9.1f} dl={agg['downloads'][0]:6.1f} "
              f"Umax={agg['Umax'][0]:.3f} imb={agg['imbalance'][0]:.3f} strand={agg['stranded'][0]:.3f} "
              f"viol={agg['viol'][0]:.3f} lat/res={agg['lat_per_res'][0]:.1f} Lam_hat={agg['Lambda_hat'][0]:.2f}")
        agg["U_per_resource"] = ms([np.mean(r["U"]) for r in rows])
        out[name] = agg

    # ---- slot reductions vs the exact vector formulation (goal G5) ----
    print("-" * 78, "slot reductions (Randomized DRL), theoretical Lambda vs empirical Lambda_hat")
    cat, tr = synthetic_world(N=a.N, T=a.T, seed=0, imbalance=imbalance, cap_mult=5)
    A = cat.a
    variants = {"vector (exact)": None,
                "dominant slots": slots.phi_dominant(A),
                "linear alpha=1": slots.phi_linear(A, np.ones(3)),
                "l2-norm alpha=1": slots.phi_lp(A, 2),
                "linear entropy (simplex, unsafe)": slots.phi_linear(A, slots.entropy_weights(A)),
                "shadow prices (rho=1)": slots.phi_shadow(A, np.ones(3))}
    slot_res = {}
    for name, phi in variants.items():
        R = [rand_drl(cat, tr, a.C, np.random.default_rng(s), K=4,
                      **({} if phi is None else dict(A_rows=phi[None, :]))) for s in range(a.seeds)]
        m = [resource_metrics(r.util, r.latency) for r in R]
        slot_res[name] = dict(Lambda=None if phi is None else slots.capacity_loss(A, phi),
                              latency=ms([r.latency for r in R]), cost=ms([r.cost for r in R]),
                              Umax=ms([x["Umax_mean"] for x in m]),
                              stranded=ms([x["stranded_mean"] for x in m]),
                              imbalance=ms([x["imbalance_mean"] for x in m]),
                              violation=ms([x["violation_rate"] for x in m]),
                              Lambda_hat=ms([slots.lambda_estimate(r.util) for r in R]))
        r = slot_res[name]
        print(f"  {name:34s} Lambda_th={str(r['Lambda']):>6s} Lambda_hat={r['Lambda_hat'][0]:.2f} "
              f"lat={r['latency'][0]:9.1f} Umax={r['Umax'][0]:.3f} strand={r['stranded'][0]:.3f} "
              f"viol(true resources)={r['violation'][0]:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    names = list(out)
    ax[0].bar(range(len(names)), [out[n]["latency"][0] for n in names]); ax[0].set_title("latency")
    ax[1].bar(range(len(names)), [out[n]["cost"][0] for n in names]); ax[1].set_title("download cost")
    ax[2].bar(range(len(names)), [out[n]["Umax"][0] for n in names]); ax[2].set_title("bottleneck utilization")
    for x in ax: x.set_xticks(range(len(names))); x.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    save(a, f"exp12_multiresource_{tag}", dict(algos=out, slots=slot_res), fig)
