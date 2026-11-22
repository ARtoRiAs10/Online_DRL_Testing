"""Phase-3 adversarial families.

1. Classic cyclic scan over ``k+1`` services with capacity ``k`` (``k`` up to 64).
2. The paper's Large Sequence with several download/latency ratios ``M/l``.
3. A trace built by running deterministic Online-DRL and always requesting the service
   it just evicted.  The trace is fixed in advance and therefore oblivious to the
   randomized algorithm's coin flips.
4. Larger Large-Sequence capacities (``k=24``).

All randomized variants use the same seed per algorithm and trace, so the comparisons
are paired.
"""
from common import *
from sim.trace import (synthetic_world, cyclic_scan, large_sequence,
                       deterministic_eviction_trace)
from sim.algos import online_drl, ll_rc, rand_landlord, rand_drl

a = parse(__doc__, C=12, K=4)
seeds = min(a.seeds, 5) if not a.quick else a.seeds
print(f"phase-3 adversarial: seeds={seeds}")

rows = []


def add_family(family, **kw):
    row = dict(family=family, **kw)
    rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})


# ---------------------------------------------------------------- cyclic scan
ks = [8, 16, 32, 64] if not a.quick else [8, 16]
n_reps = 10 if not a.quick else 5
for k in ks:
    cat, tr = cyclic_scan(k, n_reps)
    det = online_drl(cat, tr, k)
    ll = ll_rc(cat, tr, k)
    for s in range(seeds):
        r3 = rand_landlord(cat, tr, k, np.random.default_rng(s))
        rf = rand_drl(cat, tr, k, np.random.default_rng(s), K=a.K,
                      repair_mode="landlord", gate_downloads=True)
        add_family("cyclic_scan", k=k, seed=s, n_requests=len(tr),
                   det_lat=float(det.latency), ll_lat=float(ll.latency),
                   r3_lat=float(r3.latency), rf_lat=float(rf.latency),
                   det_cost=float(det.cost), ll_cost=float(ll.cost),
                   r3_cost=float(r3.cost), rf_cost=float(rf.cost),
                   det_ratio=float(det.latency / det.latency),
                   r3_ratio=float(r3.latency / det.latency),
                   rf_ratio=float(rf.latency / det.latency))

# ---------------------------------------------------------------- Large Sequence M/l
ks2 = [9, 12, 24] if not a.quick else [9]
ratios = [2.0, 4.0, 8.0, 16.0] if not a.quick else [2.0, 4.0]
for k in ks2:
    for ratio in ratios:
        cat, tr, opt = large_sequence(k, 2 * k, M=ratio, Ml=2.0 * ratio, l=1.0)
        det = online_drl(cat, tr, k)
        ll = ll_rc(cat, tr, k)
        for s in range(seeds):
            r3 = rand_landlord(cat, tr, k, np.random.default_rng(s))
            rf = rand_drl(cat, tr, k, np.random.default_rng(s), K=a.K,
                          repair_mode="landlord", gate_downloads=True)
            add_family("large_sequence", k=k, M_over_l=ratio, seed=s,
                       n_requests=len(tr), opt=float(opt),
                       det_ratio=float(det.latency / opt),
                       ll_ratio=float(ll.latency / opt),
                       r3_ratio=float(r3.latency / opt),
                       rf_ratio=float(rf.latency / opt),
                       det_cost=float(det.cost), ll_cost=float(ll.cost),
                       r3_cost=float(r3.cost), rf_cost=float(rf.cost))

# ---------------------------------------------------------------- deterministic-eviction trace
for s in range(seeds):
    cat, init = synthetic_world(N=20, T=500, seed=s, imbalance=2.0, cap_mult=4)
    tr2, base, ev = deterministic_eviction_trace(cat, init, C=8)
    det = online_drl(cat, tr2, 8)
    ll = ll_rc(cat, tr2, 8)
    r3 = rand_landlord(cat, tr2, 8, np.random.default_rng(s))
    rf = rand_drl(cat, tr2, 8, np.random.default_rng(s), K=a.K,
                  repair_mode="landlord", gate_downloads=True)
    add_family("deterministic_eviction", seed=s, n_requests=len(tr2),
               n_evictions=len(ev), base_requests=len(base),
               det_lat=float(det.latency), ll_lat=float(ll.latency),
               r3_lat=float(r3.latency), rf_lat=float(rf.latency),
               det_cost=float(det.cost), ll_cost=float(ll.cost),
               r3_cost=float(r3.cost), rf_cost=float(rf.cost),
               det_ratio=float(det.latency / det.latency),
               r3_ratio=float(r3.latency / det.latency),
               rf_ratio=float(rf.latency / det.latency))

# ---------------------------------------------------------------- summary
families = sorted(set(r["family"] for r in rows))
summary = {}
for fam in families:
    fr = [r for r in rows if r["family"] == fam]
    summary[fam] = {}
    for metric in ("r3_ratio", "rf_ratio"):
        summary[fam][metric] = ms([r[metric] for r in fr])
    for metric in ("det_lat", "r3_lat", "rf_lat", "det_cost", "r3_cost", "rf_cost"):
        vals = [r[metric] for r in fr if metric in r]
        if vals:
            summary[fam][metric] = ms(vals)

out = dict(config=dict(seeds=seeds, K=a.K), rows=rows, summary=summary)
save(a, "exp19_adversarial", out)

print("\n### summary")
for fam in families:
    s = summary[fam]
    det_lat = s.get("det_lat", [float("nan")])[0]
    r3_lat = s.get("r3_lat", [float("nan")])[0]
    rf_lat = s.get("rf_lat", [float("nan")])[0]
    print(f"{fam:24s} R3/det={s['r3_ratio'][0]:.3f}  "
          f"RF/det={s['rf_ratio'][0]:.3f}  "
          f"lat det={det_lat:.0f} r3={r3_lat:.0f} rf={rf_lat:.0f}")