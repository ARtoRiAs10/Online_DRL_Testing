"""Randomized DRL v2 (the revised method) versus Online-DRL and Randomized DRL v1.

v2 = v1 + the revisions of `docs/Proposed_revisions_randomized_DRL.pdf`:
  N1  the download decision becomes a learned probability Q_n (the benefit of a residency is
      measured and compared with its cost M_n),
  N2  the gradient uses the expected in-flight probability (an EMA) instead of the realized one,
  N7  the projection is computed on the active set, the rounding refreshes and repairs only the
      served path, and the repair is vectorised.

Measured on
  * the paper's adversarial Large Sequence (competitive ratio vs k),
  * synthetic Zipf traffic against the exact MILP optimum,
  * the multi-resource workload (latency, cost, utilization, violations),
  * the corrected goals G3a / G3b / G3c',
  * the per-request computing time.
"""
from common import *
from sim.trace import synthetic_world, large_sequence
from sim.algos import online_drl, ll_rc, rand_drl
from sim.engine import Engine, DetThreshold, Policy
from sim.projection import normalize
from sim.regret import best_fixed_value, surrogate_regret
from sim.opt import solve_opt

a = parse(__doc__, C=12)

V2 = dict(download_prob_mode="learned", grad_mode="expectation", proj_mode="active",
          rounding_mode="served", proj_tol=1e-4)


def v1(cat, tr, C, seed): return rand_drl(cat, tr, C, np.random.default_rng(seed), K=4)
def v2(cat, tr, C, seed): return rand_drl(cat, tr, C, np.random.default_rng(seed), K=4, **V2)
def det(cat, tr, C, seed): return online_drl(cat, tr, C)


# ------------------------------------------------------------------ 1 adversarial family
print("### Family 1: the paper's adversarial Large Sequence (OPT = (k+2n)l)")
rows1 = []
for k in ([3, 6, 9, 12] if a.quick else [3, 6, 9, 12, 15, 18, 21, 24]):
    n = 2 * k
    cat, tr, opt = large_sequence(k, n)
    d = det(cat, tr, k, 0)
    r1 = [v1(cat, tr, k, s) for s in range(a.seeds)]
    r2 = [v2(cat, tr, k, s) for s in range(a.seeds)]
    row = dict(k=k, opt=opt, det=d.latency / opt,
               v1=ms([r.latency / opt for r in r1]), v2=ms([r.latency / opt for r in r2]),
               v1_cost=ms([r.cost for r in r1]), v2_cost=ms([r.cost for r in r2]),
               v1_us=ms([1e6 * r.time_s / len(tr) for r in r1]),
               v2_us=ms([1e6 * r.time_s / len(tr) for r in r2]),
               det_us=1e6 * d.time_s / len(tr))
    rows1.append(row); print(row)

# ------------------------------------------------------------------ 2 Zipf vs the MILP
print("\n### Family 2: synthetic Zipf traffic, offline optimum from the MILP")
T2 = 400 if a.quick else 1200
N2 = 12 if a.quick else 16
rows2 = []
for k in ([2, 4, 6, 8] if a.quick else [2, 4, 6, 8, 10, 12]):
    acc = []
    for seed in range(min(a.seeds, 3)):
        cat, tr = synthetic_world(N=N2, T=T2, seed=seed, imbalance=2.0, cap_mult=4)
        o = solve_opt(cat, tr, k, time_limit=60)
        if o["latency"] is None or o["status"] != 0: continue
        d = det(cat, tr, k, seed)
        r1 = v1(cat, tr, k, seed); r2 = v2(cat, tr, k, seed)
        acc.append(dict(opt=o["latency"], det=d.latency / o["latency"], v1=r1.latency / o["latency"],
                        v2=r2.latency / o["latency"], det_cost=d.cost, v1_cost=r1.cost,
                        v2_cost=r2.cost))
    if acc:
        row = dict(k=k, det=ms([x["det"] for x in acc]), v1=ms([x["v1"] for x in acc]),
                   v2=ms([x["v2"] for x in acc]), det_cost=ms([x["det_cost"] for x in acc]),
                   v1_cost=ms([x["v1_cost"] for x in acc]), v2_cost=ms([x["v2_cost"] for x in acc]))
        rows2.append(row); print(row)

# ------------------------------------------------------------------ 3 multi-resource
print("\n### Family 3: the multi-resource workload (imbalance = 3)")
cat, tr = synthetic_world(N=a.N, T=a.T, seed=0, imbalance=3.0, cap_mult=4)
algos = {"Online-DRL (det)": det, "Randomized DRL v1": v1, "Randomized DRL v2": v2}
rows3 = {}
for name, fn in algos.items():
    rs = [fn(cat, tr, a.C, s) for s in range(a.seeds)]
    rows3[name] = dict(latency=ms([r.latency for r in rs]), cost=ms([r.cost for r in rs]),
                       downloads=ms([r.downloads for r in rs]), hits=ms([r.hits for r in rs]),
                       delayed=ms([r.delayed_hits for r in rs]),
                       viol=ms([float((r.util.max(1) > 1 + 1e-9).mean()) for r in rs]),
                       us=ms([1e6 * r.time_s / len(tr) for r in rs]),
                       wasted=ms([r.extra.get("wasted_downloads", 0) for r in rs]),
                       meanQ=ms([r.extra.get("mean_dprob", 1.0) for r in rs]),
                       unprof=ms([r.extra.get("unprofitable", 0) for r in rs]))
    print(name, rows3[name])

# ------------------------------------------------------------------ 4 corrected goals
print("\n### The corrected goals G3a / G3b / G3c' for v2")
cat2, tr2 = synthetic_world(N=a.N, T=8000 if not a.quick else 2000, seed=0)
An = normalize(np.vstack([np.ones(cat2.N), cat2.a.T]), np.r_[float(a.C), 1, 1, 1])
N, lmax = cat2.N, float(cat2.l.max())
rng = np.random.default_rng(1)
pop = 1 / np.arange(1, N + 1) ** 1.2; pop /= pop.sum()
rows4 = []
for T in ([500, 1000, 2000] if a.quick else [500, 1000, 2000, 4000, 8000]):
    tr = tr2.head(T)
    n_seq = tr.n.astype(int)
    g = rng.uniform(.5, 1.5, T) * cat2.l[n_seq]
    rr = rand_drl(cat2, tr, a.C, np.random.default_rng(0), K=4, record=True, **V2)
    ex = rr.extra
    reg_a = surrogate_regret(ex["n_hist"], ex["g_hist"], ex["P_hist"], An)          # G3a
    literal = rr.latency - (ex["g_hist"].sum() - best_fixed_value(ex["n_hist"], ex["g_hist"], An))
    detr = online_drl(cat2, tr, a.C)
    bound = float(np.sqrt(N) * lmax * np.sqrt(T))
    rows4.append(dict(T=T, bound_prop1=bound, G3a_surrogate=float(reg_a),
                       G3b_literal=float(literal), G3b_bound=bound + float(detr.cost),
                       det_cost=float(detr.cost), G3a_ok=bool(reg_a <= bound),
                       G3b_ok=bool(literal <= bound + detr.cost)))
    print(rows4[-1])

# ------------------------------------------------------------------ 5 tuning of eta_Q
print("\n### Family 5: sensitivity of the learned download probability (eta_Q)")
rows5 = []
for eq in (0.05, 0.1, 0.25, 0.5):
    rs = [rand_drl(cat, tr, a.C, np.random.default_rng(s), K=4, eta_q=eq, **V2) for s in range(min(3, a.seeds))]
    row = dict(eta_q=eq, latency=ms([r.latency for r in rs]), cost=ms([r.cost for r in rs]),
               downloads=ms([r.downloads for r in rs]), hits=ms([r.hits for r in rs]),
               wasted=ms([r.extra.get("wasted_downloads", 0) for r in rs]),
               meanQ=ms([r.extra.get("mean_dprob", 1.0) for r in rs]))
    rows5.append(row); print(row)

# ------------------------------------------------------------------ figures
fig, ax = plt.subplots(1, 4, figsize=(16.5, 3.6))
ks = [r["k"] for r in rows1]
ax[0].plot(ks, [r["det"] for r in rows1], "o-", color="#1f3b73", label="Online-DRL")
ax[0].plot(ks, [r["v1"][0] for r in rows1], "s--", color="#8ea9d0", label="Randomized v1")
ax[0].plot(ks, [r["v2"][0] for r in rows1], "d-", color="#1b6b3a", label="Randomized v2")
ax[0].set_xlabel("k"); ax[0].set_ylabel("ALG / OPT"); ax[0].set_title("Adversarial family: ratio vs k", fontsize=10)
ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
ax[1].bar([r["k"] for r in rows1], [r["v1_cost"][0] for r in rows1], 0.4, color="#8ea9d0", label="v1")
ax[1].bar([r["k"] + .4 for r in rows1], [r["v2_cost"][0] for r in rows1], 0.4, color="#1b6b3a", label="v2")
ax[1].set_xlabel("k"); ax[1].set_ylabel("download cost"); ax[1].set_title("Download cost", fontsize=10)
ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, axis="y")
ax[2].bar([r["k"] for r in rows2], [r["det"][0] for r in rows2], 0.4, color="#1f3b73", label="Online-DRL")
ax[2].bar([r["k"] + .4 for r in rows2], [r["v1"][0] for r in rows2], 0.4, color="#8ea9d0", label="v1")
ax[2].bar([r["k"] + .8 for r in rows2], [r["v2"][0] for r in rows2], 0.4, color="#1b6b3a", label="v2")
ax[2].set_xlabel("k"); ax[2].set_ylabel("ALG / OPT (MILP)"); ax[2].set_title("Zipf traffic vs the optimum", fontsize=10)
ax[2].legend(fontsize=7); ax[2].grid(alpha=.3, axis="y")
names = list(rows3)
ax[3].bar(range(len(names)), [rows3[n]["us"][0] for n in names],
          color=["#1f3b73", "#8ea9d0", "#1b6b3a"])
ax[3].set_xticks(range(len(names))); ax[3].set_xticklabels(["Online-DRL", "Random. v1", "Random. v2"],
                                                          rotation=20, fontsize=8)
ax[3].set_ylabel("us / request"); ax[3].set_title("Computing time per request", fontsize=10)
ax[3].grid(alpha=.3, axis="y")
for x in ax: x.set_facecolor("white")
save(a, "exp19_randomized_drl_v2", dict(adversarial=rows1, zipf=rows2, multiresource=rows3,
                                         goals=rows4, eta_q_sensitivity=rows5, v2_options=V2), fig)
