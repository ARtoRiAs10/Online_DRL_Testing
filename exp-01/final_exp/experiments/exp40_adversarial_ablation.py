"""Phase-3 (v2): adversarial families, ablation and medians/quantiles.

A. Shuffled cyclic scan ablation: which component causes the +46% latency loss?
   Variants: Online-DRL, R3 only, R1+R2 (det threshold), R1+R2+R3 (gate on/off),
   R1+R2+R3 Landlord.

B. Large Sequence ablation: R1+R2 vs R1+R2+R3, with medians and quartiles.

C. New adversarial families not built against Online-DRL:
   hotspot, random-walk Zipf, periodic burst, plus Markov-modulated Zipf and bursty.

D. OPT check: solve the MILP on small Large Sequence instances and compare with the
   closed form ``OPT = (k + 2 n_reps) l``.
"""
from common import *
from sim.trace import (large_sequence, shuffled_cyclic_scan, markov_zipf_world,
                       bursty_world, hotspot_world, random_walk_zipf_world,
                       periodic_burst_world)
from sim.algos import online_drl, ll_rc, rand_landlord, rand_drl
from sim.opt import solve_opt

a = parse(__doc__, C=12, K=4)
print("phase-3 v2: adversarial ablation")


def qstats(x):
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                median=float(np.median(x)),
                q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)),
                min=float(x.min()), max=float(x.max()))


rows = []

# ---------------------------------------------------------------- A. shuffled cyclic scan
for k in ([16, 32, 64, 128] if not a.quick else [16, 32]):
    for s in range(10 if not a.quick else 3):
        cat, tr = shuffled_cyclic_scan(k, 10, seed=s)
        det = online_drl(cat, tr, k)
        variants = {
            "online_drl": det,
            "r3_only": rand_landlord(cat, tr, k, np.random.default_rng(s)),
            "r1r2_stateless_det": rand_drl(cat, tr, k, np.random.default_rng(s), K=a.K,
                                           thr="det", repair_mode="stateless",
                                           gate_downloads=False),
            "r1r2_stateless_det_gate": rand_drl(cat, tr, k, np.random.default_rng(s), K=a.K,
                                                thr="det", repair_mode="stateless",
                                                gate_downloads=True),
            "r1r2r3_stateless_gate": rand_drl(cat, tr, k, np.random.default_rng(s), K=a.K,
                                              thr="random", repair_mode="stateless",
                                              gate_downloads=True),
            "r1r2r3_stateless_nogate": rand_drl(cat, tr, k, np.random.default_rng(s), K=a.K,
                                                thr="random", repair_mode="stateless",
                                                gate_downloads=False),
            "r1r2r3_landlord_gate": rand_drl(cat, tr, k, np.random.default_rng(s), K=a.K,
                                             thr="random", repair_mode="landlord",
                                             gate_downloads=True),
        }
        for name, R in variants.items():
            rows.append(dict(family="shuffled_cyclic_scan", k=k, seed=s, variant=name,
                             latency=float(R.latency), cost=float(R.cost),
                             hits=int(R.hits), downloads=int(R.downloads),
                             lat_ratio_vs_det=float(R.latency / det.latency),
                             cost_ratio_vs_det=float(R.cost / det.cost) if det.cost else float("nan")))
        print(f"shuffled k={k} seed={s} done")

# ---------------------------------------------------------------- B. Large Sequence ablation
for k in ([9, 12, 24, 48] if not a.quick else [9, 12]):
    seeds = (3 if k >= 48 else 5) if not a.quick else 3
    for ratio in ([2.0, 4.0, 8.0] if not a.quick else [2.0, 4.0]):
        cat, tr, opt = large_sequence(k, 2 * k, M=ratio, Ml=2.0 * ratio, l=1.0)
        det = online_drl(cat, tr, k)
        for s in range(seeds):
            for name, fn in (
                ("r3_only", lambda: rand_landlord(cat, tr, k, np.random.default_rng(s))),
                ("r1r2_stateless", lambda: rand_drl(cat, tr, k, np.random.default_rng(s),
                                                    K=a.K, thr="det",
                                                    repair_mode="stateless",
                                                    gate_downloads=False)),
                ("r1r2r3_stateless", lambda: rand_drl(cat, tr, k, np.random.default_rng(s),
                                                      K=a.K, thr="random",
                                                      repair_mode="stateless",
                                                      gate_downloads=True)),
                ("r1r2r3_landlord", lambda: rand_drl(cat, tr, k, np.random.default_rng(s),
                                                     K=a.K, thr="random",
                                                     repair_mode="landlord",
                                                     gate_downloads=True)),
            ):
                R = fn()
                rows.append(dict(family="large_sequence", k=k, M_over_l=ratio, seed=s,
                                 variant=name, opt=float(opt),
                                 latency=float(R.latency), cost=float(R.cost),
                                 hits=int(R.hits),
                                 lat_ratio_opt=float(R.latency / opt),
                                 cost_ratio_opt=float(R.cost / opt),
                                 det_ratio_opt=float(det.latency / opt)))
        print(f"large k={k} M/l={ratio:g} done")

# ---------------------------------------------------------------- C. new families
families = {
    "hotspot": lambda s: hotspot_world(N=40, T=8000, seed=s, rho=1.2, switch=0.01,
                                       hot_frac=0.5, rate=0.5, cap_mult=4),
    "random_walk_zipf": lambda s: random_walk_zipf_world(N=40, T=8000, seed=s, rho=1.2,
                                                         step=0.05, rate=0.5, cap_mult=4),
    "periodic_burst": lambda s: periodic_burst_world(N=40, T=8000, seed=s, rho=1.2,
                                                     period=500, rate=0.5, cap_mult=4),
    "markov_zipf": lambda s: markov_zipf_world(N=40, T=8000, seed=s, rho1=0.8, rho2=1.6,
                                               switch=0.01, rate=0.5, cap_mult=4),
    "bursty": lambda s: bursty_world(N=40, T=8000, seed=s, burst_len=20, rate=0.5,
                                     cap_mult=4),
}
for fam, gen in families.items():
    for s in range(5 if not a.quick else 2):
        cat, tr = gen(s)
        C = a.C
        det = online_drl(cat, tr, C)
        variants = {
            "online_drl": det,
            "r3_only": rand_landlord(cat, tr, C, np.random.default_rng(s)),
            "r1r2_stateless": rand_drl(cat, tr, C, np.random.default_rng(s), K=a.K,
                                       thr="det", repair_mode="stateless",
                                       gate_downloads=False),
            "r1r2r3_stateless": rand_drl(cat, tr, C, np.random.default_rng(s), K=a.K,
                                         thr="random", repair_mode="stateless",
                                         gate_downloads=True),
            "r1r2r3_landlord": rand_drl(cat, tr, C, np.random.default_rng(s), K=a.K,
                                        thr="random", repair_mode="landlord",
                                        gate_downloads=True),
        }
        for name, R in variants.items():
            rows.append(dict(family=fam, seed=s, variant=name,
                             latency=float(R.latency), cost=float(R.cost),
                             hits=int(R.hits), downloads=int(R.downloads),
                             lat_ratio_vs_det=float(R.latency / det.latency),
                             cost_ratio_vs_det=float(R.cost / det.cost) if det.cost else float("nan")))
        print(f"family {fam} seed {s} done")

# ---------------------------------------------------------------- D. OPT check
opt_rows = []
for k in ([3, 6] if not a.quick else [3]):
    for ratio in ([2.0, 4.0] if not a.quick else [2.0]):
        cat, tr, opt_closed = large_sequence(k, 2 * k, M=ratio, Ml=2.0 * ratio, l=1.0)
        o = solve_opt(cat, tr, k, bmax=None, time_limit=20.0)
        opt_rows.append(dict(k=k, M_over_l=ratio, opt_closed=float(opt_closed),
                             opt_milp=float(o["latency"]), status=int(o["status"]),
                             gap=float(o["gap"]) if np.isfinite(o["gap"]) else None,
                             ratio=float(o["latency"] / opt_closed)
                             if np.isfinite(o["latency"]) else None))
        print(f"OPT check k={k} M/l={ratio:g}: closed={opt_closed:.1f} "
              f"milp={o['latency']:.1f} status={o['status']}")

# ---------------------------------------------------------------- summaries
summary = {}
for fam in sorted(set(r["family"] for r in rows)):
    if fam == "large_sequence":
        continue
    summary[fam] = {}
    variants = sorted(set(r["variant"] for r in rows if r["family"] == fam))
    for name in variants:
        fr = [r for r in rows if r["family"] == fam and r["variant"] == name]
        if not fr:
            continue
        summary[fam][name] = {
            "latency": qstats([r["latency"] for r in fr]),
            "cost": qstats([r["cost"] for r in fr]),
            "lat_ratio_vs_det": qstats([r["lat_ratio_vs_det"] for r in fr]),
            "cost_ratio_vs_det": qstats([r["cost_ratio_vs_det"] for r in fr]),
        }

ls_summary = {}
for k in sorted(set(r["k"] for r in rows if r["family"] == "large_sequence")):
    for ratio in sorted(set(r["M_over_l"] for r in rows
                            if r["family"] == "large_sequence" and r["k"] == k)):
        for name in ("r3_only", "r1r2_stateless", "r1r2r3_stateless", "r1r2r3_landlord"):
            fr = [r for r in rows if r["family"] == "large_sequence" and r["k"] == k
                  and r["M_over_l"] == ratio and r["variant"] == name]
            if fr:
                ls_summary[f"k={k},M/l={ratio:g},{name}"] = {
                    "lat_ratio_opt": qstats([r["lat_ratio_opt"] for r in fr]),
                    "cost_ratio_opt": qstats([r["cost_ratio_opt"] for r in fr]),
                    "det_ratio_opt": float(fr[0]["det_ratio_opt"]),
                }

out = dict(
    config=dict(K=a.K, C=a.C, seeds=dict(shuffled=10, large=5, families=5)),
    rows=rows,
    summary=summary,
    large_sequence=ls_summary,
    opt_check=opt_rows,
    note=("Medians and quartiles are reported because the spread exceeds the mean in "
          "some cells.  The shuffled-cyclic ablation separates R1+R2 from R1+R2+R3 and "
          "the gate on/off."),
)
save(a, "exp40_adversarial_ablation", out)

print("\n### summary (median latency ratio vs Online-DRL)")
for fam, s in summary.items():
    print(f"\n{fam}")
    for name in ("online_drl", "r3_only", "r1r2_stateless", "r1r2r3_stateless",
                 "r1r2r3_landlord", "r1r2_stateless_det", "r1r2_stateless_det_gate",
                 "r1r2r3_stateless_gate", "r1r2r3_stateless_nogate",
                 "r1r2r3_landlord_gate"):
        if name in s:
            q = s[name]["lat_ratio_vs_det"]
            print(f"  {name:26s} lat={s[name]['latency']['median']:9.0f} "
                  f"ratio med={q['median']:.3f} IQR=[{q['q25']:.3f},{q['q75']:.3f}] "
                  f"mean={q['mean']:.3f}±{q['sd']:.3f}")
print("\n### Large Sequence ablation (median latency ratio to OPT)")
for key, s in ls_summary.items():
    q = s["lat_ratio_opt"]
    print(f"  {key:34s} med={q['median']:.3f} IQR=[{q['q25']:.3f},{q['q75']:.3f}] "
          f"mean={q['mean']:.3f}±{q['sd']:.3f} det={s['det_ratio_opt']:.3f}")
print("\n### OPT check")
for r in opt_rows:
    print(f"  k={r['k']} M/l={r['M_over_l']:g} closed={r['opt_closed']:.1f} "
          f"milp={r['opt_milp']:.1f} status={r['status']} ratio={r['ratio']}")