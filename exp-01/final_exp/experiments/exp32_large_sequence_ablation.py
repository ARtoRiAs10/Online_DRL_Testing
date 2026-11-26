"""Phase-4: Large Sequence ablation — which component flattens the ratio?

Compares:
  * R3 only;
  * R1+R2 with Landlord repair;
  * R1+R2 with stateless repair;
  * R1+R2+R3 with Landlord repair;
  * R1+R2+R3 with stateless repair.

20 seeds for k <= 48, 10 seeds for k = 96.  Ratios are to the exact Large Sequence
optimum ``OPT = (k + 2 n_reps) l``.
"""
from common import *
from sim.trace import large_sequence
from sim.algos import online_drl, ll_rc, rand_landlord, rand_drl

a = parse(__doc__, C=12, K=4)
ks = [9, 12, 24, 48] if not a.quick else [9, 12]
ratios = [2.0, 4.0, 8.0] if not a.quick else [2.0, 4.0]
print(f"phase-4 Large Sequence ablation: k={ks} M/l={ratios}")

VARIANTS = {
    "r3_only": lambda cat, tr, k, rng: rand_landlord(cat, tr, k, rng),
    "r1r2_stateless": lambda cat, tr, k, rng: rand_drl(
        cat, tr, k, rng, K=a.K, thr="det", repair_mode="stateless", gate_downloads=False),
    "r1r2r3_stateless": lambda cat, tr, k, rng: rand_drl(
        cat, tr, k, rng, K=a.K, thr="random", repair_mode="stateless", gate_downloads=True),
    "r1r2r3_landlord": lambda cat, tr, k, rng: rand_drl(
        cat, tr, k, rng, K=a.K, thr="random", repair_mode="landlord", gate_downloads=True),
}

rows = []
for k in ks:
    seeds = (3 if k >= 48 else 5) if not a.quick else 3
    for ratio in ratios:
        cat, tr, opt = large_sequence(k, 2 * k, M=ratio, Ml=2.0 * ratio, l=1.0)
        det = online_drl(cat, tr, k)
        ll = ll_rc(cat, tr, k)
        for s in range(seeds):
            for name, fn in VARIANTS.items():
                R = fn(cat, tr, k, np.random.default_rng(s))
                rows.append(dict(
                    k=k, M_over_l=ratio, seed=s, variant=name,
                    opt=float(opt),
                    latency_ratio=float(R.latency / opt),
                    cost_ratio=float(R.cost / opt),
                    latency=float(R.latency), cost=float(R.cost),
                    hits=int(R.hits),
                    det_ratio=float(det.latency / opt),
                    ll_ratio=float(ll.latency / opt),
                ))

summary = {}
for k in ks:
    for ratio in ratios:
        for name in VARIANTS:
            fr = [r for r in rows if r["k"] == k and r["M_over_l"] == ratio
                  and r["variant"] == name]
            if not fr:
                continue
            key = f"k={k},M/l={ratio:g},{name}"
            summary[key] = {
                "latency_ratio": ms([r["latency_ratio"] for r in fr]),
                "cost_ratio": ms([r["cost_ratio"] for r in fr]),
                "latency_ratio_ci": paired_ci(
                    [r["latency_ratio"] for r in fr],
                    [r["det_ratio"] for r in fr]),
            }

out = dict(config=dict(ks=ks, ratios=ratios, K=a.K, variants=list(VARIANTS)),
           rows=rows, summary=summary)
save(a, "exp32_large_sequence_ablation", out)

print("\n### ablation summary (latency ratio to OPT)")
print(f"{'k':>3s} {'M/l':>4s} {'variant':20s} {'lat/OPT':>10s} {'sd':>8s} {'cost/OPT':>10s}")
for k in ks:
    for ratio in ratios:
        base = next((r for r in rows if r["k"] == k and r["M_over_l"] == ratio), None)
        if base:
            print(f"{k:3d} {ratio:4.0f} {'Online-DRL':20s} "
                  f"{base['det_ratio']:10.3f} {'':8s} {'':10s}")
        for name in VARIANTS:
            key = f"k={k},M/l={ratio:g},{name}"
            s = summary.get(key)
            if s:
                print(f"{k:3d} {ratio:4.0f} {name:20s} "
                      f"{s['latency_ratio'][0]:10.3f} {s['latency_ratio'][1]:8.3f} "
                      f"{s['cost_ratio'][0]:10.3f}")