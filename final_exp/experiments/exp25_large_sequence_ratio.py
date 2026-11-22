"""Phase-5 family-specific ratio on the Large Sequence.

Runs the deterministic and randomized algorithms on the Large Sequence for several
capacities and download/latency ratios, with 20 seeds for the randomized variants.
Reports ``ALG / OPT`` with 95% confidence intervals and a linear fit in ``k``.
"""
from common import *
from sim.trace import large_sequence
from sim.algos import online_drl, ll_rc, rand_landlord, rand_drl

a = parse(__doc__, K=4)
ks = [9, 12, 24, 48] if not a.quick else [9, 12]
ratios = [2.0, 4.0, 8.0] if not a.quick else [2.0, 4.0]
seeds = max(a.seeds, 20) if not a.quick else a.seeds
print(f"phase-5 large-sequence ratio: k={ks} M/l={ratios} seeds={seeds}")

rows = []
for k in ks:
    for ratio in ratios:
        cat, tr, opt = large_sequence(k, 2 * k, M=ratio, Ml=2.0 * ratio, l=1.0)
        det = online_drl(cat, tr, k)
        ll = ll_rc(cat, tr, k)
        for s in range(seeds):
            r3 = rand_landlord(cat, tr, k, np.random.default_rng(s))
            rf = rand_drl(cat, tr, k, np.random.default_rng(s), K=a.K,
                          repair_mode="landlord", gate_downloads=True)
            rows.append(dict(
                k=k, M_over_l=ratio, seed=s, opt=float(opt),
                det_ratio=float(det.latency / opt),
                ll_ratio=float(ll.latency / opt),
                r3_ratio=float(r3.latency / opt),
                rf_ratio=float(rf.latency / opt),
                det_cost=float(det.cost), ll_cost=float(ll.cost),
                r3_cost=float(r3.cost), rf_cost=float(rf.cost),
                det_lat=float(det.latency), r3_lat=float(r3.latency),
                rf_lat=float(rf.latency),
            ))

summary = {}
for k in ks:
    for ratio in ratios:
        fr = [r for r in rows if r["k"] == k and r["M_over_l"] == ratio]
        key = f"k={k},M/l={ratio:g}"
        summary[key] = {
            "det_ratio": float(fr[0]["det_ratio"]),
            "ll_ratio": float(fr[0]["ll_ratio"]),
            "r3_ratio": ms([r["r3_ratio"] for r in fr]),
            "rf_ratio": ms([r["rf_ratio"] for r in fr]),
            "rf_ci": paired_ci([r["rf_ratio"] for r in fr],
                               [r["det_ratio"] for r in fr]),
            "det_cost": float(fr[0]["det_cost"]),
            "rf_cost": ms([r["rf_cost"] for r in fr]),
        }

# Linear fit of the randomized ratio in k, per M/l ratio.
fits = {}
for ratio in ratios:
    fr = [r for r in rows if r["M_over_l"] == ratio]
    x = np.array([r["k"] for r in fr], float)
    y = np.array([r["rf_ratio"] for r in fr], float)
    A = np.c_[x, np.ones_like(x)]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss = ((y - pred) ** 2).sum()
    r2 = float(1.0 - ss / ((y - y.mean()) ** 2).sum())
    fits[f"M/l={ratio:g}"] = dict(slope=float(coef[0]), intercept=float(coef[1]), R2=r2)

out = dict(config=dict(ks=ks, ratios=ratios, seeds=seeds, K=a.K),
           rows=rows, summary=summary, fits=fits)
save(a, "exp25_large_sequence_ratio", out)

print("\n### large-sequence ratio")
for key, s in summary.items():
    print(f"{key:18s} det={s['det_ratio']:8.3f} r3={s['r3_ratio'][0]:8.3f} "
          f"rf={s['rf_ratio'][0]:8.3f} ±{s['rf_ratio'][1]:.3f} "
          f"cost det={s['det_cost']:.0f} rf={s['rf_cost'][0]:.0f}")
print("\nfits:", fits)