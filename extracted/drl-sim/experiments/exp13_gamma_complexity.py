"""Does Randomized DRL reach / improve the paper's complexity statement?

Theorem 1 of the paper (Deka & Sukapuram, IEEE TCC 2025):
        gamma = ALG(I)/OPT(I)  in  O(k (1 + 3 beta_up M_x))   and   in  Omega(k (1 + 3 beta_down M_min))
where k = cache capacity, M_x / M_min = largest / smallest download time and beta_up (beta_down)
is the fraction of M_x (M_min) requests that are forwarded to the cloud instead of buffered.

This script
  (1) measures beta_up, beta_down, M_x, M_min from the simulation itself,
  (2) evaluates the O-bound and the Omega-bound numerically,
  (3) measures the empirical competitive ratio gamma of Online-DRL (deterministic),
      LL-RC, R3-only and the full Randomized DRL,
  (4) fits gamma(k) against a linear model k, a logarithmic model log k and a power law
      k^b, to see whether the linear-in-k behaviour is removed by randomization.

Family 1: the paper's adversarial Large Sequence (BS1|BS2|BS3|MS)^n, OPT = (k+2n)l.
Family 2: Zipf/Poisson synthetic traffic with the offline optimum from the MILP (small sizes).
"""
from common import *
from sim.trace import large_sequence, synthetic_world
from sim.algos import online_drl, ll_rc, rand_landlord, rand_drl
from sim.opt import solve_opt
from sim.analysis import measure_betas, ratio_bounds, compare_models

a = parse(__doc__)
print("### Family 1: Large Sequence (adversarial) -- OPT = (k+2n)l  [paper eq. 17]")
ks = [3, 6, 9, 12] if a.quick else [3, 6, 9, 12, 15, 18, 21, 24]
rows = []
for k in ks:
    n = 2 * k
    cat, tr, opt_paper = large_sequence(k, n)
    det = online_drl(cat, tr, k); ll = ll_rc(cat, tr, k)
    rl = [rand_landlord(cat, tr, k, np.random.default_rng(s)) for s in range(a.seeds)]
    rf = [rand_drl(cat, tr, k, np.random.default_rng(s), K=4) for s in range(a.seeds)]
    bdet = measure_betas(det, cat, tr); brf = measure_betas(rf[0], cat, tr)
    Mx, Mmin = float(cat.M.max()), float(cat.M.min())
    row = dict(k=k, n=n, opt=opt_paper, Mx=Mx, Mmin=Mmin,
               beta_up=brf["beta_up"], beta_down=brf["beta_down"],
               det=det.latency / opt_paper, llrc=ll.latency / opt_paper,
               det_cost=det.cost, llrc_cost=ll.cost)
    row["rand_landlord"], row["rand_landlord_sd"] = ms([r.latency / opt_paper for r in rl])
    row["rand_drl"], row["rand_drl_sd"] = ms([r.latency / opt_paper for r in rf])
    row["rand_drl_cost"] = ms([r.cost for r in rf])
    row.update({"O_bound": ratio_bounds(k, brf["beta_up"], brf["beta_down"], Mx, Mmin)["O_bound"],
                "Omega_bound": ratio_bounds(k, brf["beta_up"], brf["beta_down"], Mx, Mmin)["Omega_bound"]})
    rows.append(row)
    print({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in row.items()})

fits = {m: compare_models([r["k"] for r in rows], [r[m] for r in rows])
        for m in ("det", "llrc", "rand_landlord", "rand_drl")}
print("\ngrowth models for gamma(k) on the Large Sequence:")
for m, f in fits.items():
    print(f"  {m:14s} linear: slope={f['linear']['slope']:.3f} R2={f['linear']['R2']:.3f} | "
          f"log: slope={f['log']['slope']:.3f} R2={f['log']['R2']:.3f} | "
          f"power: k^{f['power']['b']:.3f} R2={f['power']['R2']:.3f}")
print("bound check: measured det gamma <= O-bound?",
      [(r["k"], round(r["det"], 2), round(r["O_bound"], 2)) for r in rows])
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
for m, lab in (("det", "Online-DRL"), ("llrc", "LL-RC"), ("rand_landlord", "R3 only"), ("rand_drl", "Randomized DRL")):
    ax[0].plot(ks, [r[m] for r in rows], "o-", label=lab)
ax[0].plot(ks, [r["O_bound"] for r in rows], "k--", label="O(k(1+3 beta_up M_x))")
ax[0].set_xlabel("k (cache capacity)"); ax[0].set_ylabel("ALG / OPT"); ax[0].legend(fontsize=7)
ax[0].set_title("Large Sequence: competitive ratio vs k")
for m, lab in (("det", "Online-DRL"), ("rand_drl", "Randomized DRL")):
    ax[1].plot(ks, [r[m] for r in rows], "o-", label=lab)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].plot(ks, [r["det"] for r in rows], "o--", label="k reference")
ax[1].set_xlabel("k (log)"); ax[1].set_ylabel("gamma (log)"); ax[1].legend(fontsize=7)
ax[1].set_title("log-log view of gamma(k)")
save(a, "exp13_gamma_complexity", dict(family1=rows, fits=fits), fig)

# ---------------------------------------------------------------------------------
print("\n### Family 2: Zipf synthetic traffic, offline OPT from the MILP")
T2 = 400 if a.quick else 1200
N2 = 12 if a.quick else 16
rows2 = []
for k in ([2, 4, 6, 8] if a.quick else [2, 4, 6, 8, 10, 12]):
    per_seed = []
    for seed in range(min(a.seeds, 3)):
        cat, tr = synthetic_world(N=N2, T=T2, seed=seed, imbalance=2.0, cap_mult=4)
        o = solve_opt(cat, tr, k, time_limit=60)
        if o["latency"] is None or not np.isfinite(o["latency"]) or o["status"] != 0:
            continue
        det = online_drl(cat, tr, k); ll = ll_rc(cat, tr, k)
        rf = rand_drl(cat, tr, k, np.random.default_rng(seed), K=4)
        per_seed.append(dict(opt=o["latency"], det=det.latency / o["latency"],
                              llrc=ll.latency / o["latency"], rand_drl=rf.latency / o["latency"],
                              det_cost=det.cost, rand_cost=rf.cost, opt_status=o["status"]))
    if per_seed:
        row = dict(k=k, det=ms([p["det"] for p in per_seed]), llrc=ms([p["llrc"] for p in per_seed]),
                   rand_drl=ms([p["rand_drl"] for p in per_seed]),
                   det_cost=ms([p["det_cost"] for p in per_seed]), rand_cost=ms([p["rand_cost"] for p in per_seed]),
                   opt_lat=ms([p["opt"] for p in per_seed]))
        rows2.append(row); print(row)
fits2 = {m: compare_models([r["k"] for r in rows2], [r[m][0] for r in rows2])
         for m in ("det", "llrc", "rand_drl")}
print("growth models for gamma(k) on Zipf traffic:")
for m, f in fits2.items():
    print(f"  {m:10s} linear R2={f['linear']['R2']:.3f} | log R2={f['log']['R2']:.3f} | "
          f"power k^{f['power']['b']:.3f} R2={f['power']['R2']:.3f}")
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for m, lab in (("det", "Online-DRL"), ("llrc", "LL-RC"), ("rand_drl", "Randomized DRL")):
    ax.plot([r["k"] for r in rows2], [r[m][0] for r in rows2], "o-", label=lab)
ax.set_xlabel("k"); ax.set_ylabel("ALG / OPT (MILP)"); ax.legend(); ax.set_title("Zipf traffic: ratio vs k")
save(a, "exp13_gamma_complexity_zipf", dict(rows=rows2, fits=fits2), fig)
