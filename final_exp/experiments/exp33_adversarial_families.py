"""Phase-4: additional adversarial families not built against Online-DRL.

1. Shuffled cyclic scan: k+1 services, random permutation each cycle.
2. Markov-modulated Zipf: popularity alternates between two Zipf laws.
3. Bursty random: each burst is centred on one service with random outliers.

For each family we compare Online-DRL, LL-RC, R3 only, randomized stateless and
randomized Landlord.  Ratios are to Online-DRL on the same trace.
"""
from common import *
from sim.trace import (synthetic_world, shuffled_cyclic_scan, markov_zipf_world,
                       bursty_world)
from sim.algos import online_drl, ll_rc, rand_landlord, rand_drl

a = parse(__doc__, C=12, K=4)
seeds = min(a.seeds, 10) if not a.quick else a.seeds
print(f"phase-4 adversarial families: seeds={seeds}")

rows = []


def run_family(family, cat, tr, C, seed):
    det = online_drl(cat, tr, C)
    ll = ll_rc(cat, tr, C)
    r3 = rand_landlord(cat, tr, C, np.random.default_rng(seed))
    rs = rand_drl(cat, tr, C, np.random.default_rng(seed), K=a.K,
                  repair_mode="stateless", gate_downloads=True)
    rl = rand_drl(cat, tr, C, np.random.default_rng(seed), K=a.K,
                  repair_mode="landlord", gate_downloads=True)
    for name, R in (("online_drl", det), ("ll_rc", ll), ("r3_only", r3),
                    ("rand_stateless", rs), ("rand_landlord", rl)):
        rows.append(dict(
            family=family, seed=seed, variant=name,
            latency=float(R.latency), cost=float(R.cost),
            hits=int(R.hits), downloads=int(R.downloads),
            latency_ratio_vs_det=float(R.latency / det.latency),
            cost_ratio_vs_det=float(R.cost / det.cost) if det.cost else float("nan"),
            latency_ratio_vs_llrc=float(R.latency / ll.latency),
            cost_ratio_vs_llrc=float(R.cost / ll.cost) if ll.cost else float("nan"),
            det_latency=float(det.latency), det_cost=float(det.cost),
            ll_latency=float(ll.latency), ll_cost=float(ll.cost),
        ))


# ---------------------------------------------------------------- shuffled cyclic scan
for k in ([16, 32, 64, 128] if not a.quick else [16, 32]):
    cat, tr = shuffled_cyclic_scan(k, 10, seed=0)
    for s in range(seeds):
        cat_s, tr_s = shuffled_cyclic_scan(k, 10, seed=s)
        run_family("shuffled_cyclic_scan", cat_s, tr_s, k, s)

# ---------------------------------------------------------------- Markov Zipf
for s in range(seeds):
    cat, tr = markov_zipf_world(N=40, T=8000, seed=s, rho1=0.8, rho2=1.6,
                                switch=0.01, rate=0.5, cap_mult=4)
    run_family("markov_zipf", cat, tr, a.C, s)

# ---------------------------------------------------------------- bursty
for s in range(seeds):
    cat, tr = bursty_world(N=40, T=8000, seed=s, burst_len=20, rate=0.5,
                           cap_mult=4)
    run_family("bursty", cat, tr, a.C, s)

summary = {}
for fam in sorted(set(r["family"] for r in rows)):
    summary[fam] = {}
    for name in ("online_drl", "ll_rc", "r3_only", "rand_stateless", "rand_landlord"):
        fr = [r for r in rows if r["family"] == fam and r["variant"] == name]
        if not fr:
            continue
        summary[fam][name] = {
            "latency": ms([r["latency"] for r in fr]),
            "cost": ms([r["cost"] for r in fr]),
            "hits": ms([r["hits"] for r in fr]),
            "latency_ratio_vs_det": ms([r["latency_ratio_vs_det"] for r in fr]),
            "cost_ratio_vs_det": ms([r["cost_ratio_vs_det"] for r in fr]),
            "latency_ratio_vs_llrc": ms([r["latency_ratio_vs_llrc"] for r in fr]),
            "cost_ratio_vs_llrc": ms([r["cost_ratio_vs_llrc"] for r in fr]),
        }

out = dict(config=dict(seeds=seeds, K=a.K, C=a.C), rows=rows, summary=summary)
save(a, "exp33_adversarial_families", out)

print("\n### summary (mean latency ratio vs Online-DRL)")
for fam, s in summary.items():
    print(f"\n{fam}")
    for name in ("online_drl", "ll_rc", "r3_only", "rand_stateless", "rand_landlord"):
        if name in s:
            print(f"  {name:16s} lat={s[name]['latency'][0]:9.0f} "
                  f"cost={s[name]['cost'][0]:9.0f} "
                  f"lat/det={s[name]['latency_ratio_vs_det'][0]:.3f} "
                  f"cost/det={s[name]['cost_ratio_vs_det'][0]:.3f}")