"""Phase A: accounting audit, peek mode, gate/trigger order, K, Landlord ties.

Outputs results/phase_a_audit.json.
"""
from common import *
from sim.engine import Engine, RandThreshold

SEEDS = [0, 1, 2]
rows = []
krows = []
tie_rows = []
identical_rows = []


def run_one(cat, tr, C, seed, repair, peek, K=4, dl_mode="ema"):
    rng = np.random.default_rng(seed)
    pol = AuditPolicy(cat, C, rng, K=K, repair_mode=repair, gate_downloads=True,
                      peek=peek, download_time_mode=dl_mode)
    res = Engine(cat, pol, RandThreshold(rng)).run(tr)
    a = pol.audit
    return res, pol, a


print("Phase A: synthetic audit")
for seed in SEEDS:
    cat, tr = synthetic_world(N=40, T=8000, seed=seed, imbalance=2.0, cap_mult=4)
    C = 12
    for repair in ("stateless", "landlord"):
        for peek in (True, False):
            res, pol, a = run_one(cat, tr, C, seed, repair, peek)
            rows.append(dict(family="synthetic", seed=seed, repair=repair, peek=peek,
                             K=4, latency=float(res.latency), cost=float(res.cost),
                             hits=int(res.hits), downloads=int(res.downloads),
                             violations=int(a["violations"]),
                             hits_checked=int(a["hits_checked"]),
                             strict_extra_cost=float(a["violation_cost"]),
                             strict_cost=float(res.cost + a["violation_cost"]),
                             serving_entries=int(a["serving_entries"]),
                             entry_cost_serving=float(a["entry_cost_serving"]),
                             charge_all_upper=float(res.cost + a["entry_cost_serving"]),
                             all_entries=int(a["all_entries"]),
                             entry_cost_all=float(a["entry_cost_all"])))
            bs = pol.landlord_stats.get("bankrupt_sizes", [])
            tie_rows.append(dict(family="synthetic", seed=seed, repair=repair, peek=peek,
                                 evictions=len(bs), mean_size=float(np.mean(bs)) if bs else 0.0,
                                 max_size=int(np.max(bs)) if bs else 0,
                                 frac_size_one=float(np.mean(np.array(bs) == 1)) if bs else 0.0))
            print(f"syn seed={seed} {repair} peek={peek}: cost={res.cost:.0f} "
                  f"viol={a['violations']} entries={a['serving_entries']} "
                  f"entrycost={a['entry_cost_serving']:.0f}")

print("Phase A: google10k audit")
for seed in (0, 1):
    cat, tr = load_google("google.csv", max_rows=10000, D=1000.0, disk_scale=1.0)
    C = min(12, max(1, cat.N // 2))
    for repair in ("stateless", "landlord"):
        res, pol, a = run_one(cat, tr, C, seed, repair, True, K=4)
        rows.append(dict(family="google10k", seed=seed, repair=repair, peek=True,
                         K=4, latency=float(res.latency), cost=float(res.cost),
                         hits=int(res.hits), downloads=int(res.downloads),
                         violations=int(a["violations"]),
                         hits_checked=int(a["hits_checked"]),
                         strict_extra_cost=float(a["violation_cost"]),
                         strict_cost=float(res.cost + a["violation_cost"]),
                         serving_entries=int(a["serving_entries"]),
                         entry_cost_serving=float(a["entry_cost_serving"]),
                         charge_all_upper=float(res.cost + a["entry_cost_serving"]),
                         all_entries=int(a["all_entries"]),
                         entry_cost_all=float(a["entry_cost_all"])))
        bs = pol.landlord_stats.get("bankrupt_sizes", [])
        tie_rows.append(dict(family="google10k", seed=seed, repair=repair, peek=True,
                             evictions=len(bs), mean_size=float(np.mean(bs)) if bs else 0.0,
                             max_size=int(np.max(bs)) if bs else 0,
                             frac_size_one=float(np.mean(np.array(bs) == 1)) if bs else 0.0))
        print(f"g10k seed={seed} {repair}: cost={res.cost:.0f} viol={a['violations']} "
              f"entries={a['serving_entries']}")

print("Phase A: K effect (synthetic, same seed)")
cat, tr = synthetic_world(N=40, T=8000, seed=0, imbalance=2.0, cap_mult=4)
for K in (1, 2, 4, 8):
    rng = np.random.default_rng(0)
    pol = AuditPolicy(cat, 12, rng, K=K, repair_mode="stateless", gate_downloads=True,
                      peek=True, download_time_mode="ema")
    res = Engine(cat, pol, RandThreshold(rng)).run(tr)
    krows.append(dict(K=K, latency=float(res.latency), cost=float(res.cost),
                      hits=int(res.hits), downloads=int(res.downloads),
                      violations=int(pol.audit["violations"]),
                      serving_entries=int(pol.audit["serving_entries"])))
    print(f"K={K}: lat={res.latency:.0f} cost={res.cost:.0f} hits={res.hits}")

print("Phase A: dl_ema vs base (download_time_mode ema vs half)")
for seed in (0, 1):
    cat, tr = synthetic_world(N=40, T=8000, seed=seed, imbalance=2.0, cap_mult=4)
    for mode in ("ema", "half"):
        rng = np.random.default_rng(seed)
        pol = AuditPolicy(cat, 12, rng, K=4, repair_mode="stateless", gate_downloads=True,
                          peek=True, download_time_mode=mode)
        res = Engine(cat, pol, RandThreshold(rng)).run(tr)
        identical_rows.append(dict(seed=seed, mode=mode, latency=float(res.latency),
                                   cost=float(res.cost), hits=int(res.hits),
                                   downloads=int(res.downloads)))
        print(f"seed={seed} mode={mode}: lat={res.latency:.0f} cost={res.cost:.0f}")

summary = {}
for fam in ("synthetic", "google10k"):
    for repair in ("stateless", "landlord"):
        fr = [r for r in rows if r["family"] == fam and r["repair"] == repair and r["peek"]]
        if fr:
            summary[f"{fam}/{repair}"] = {
                "cost": ms([r["cost"] for r in fr]),
                "strict_cost": ms([r["strict_cost"] for r in fr]),
                "charge_all_upper": ms([r["charge_all_upper"] for r in fr]),
                "violations": ms([r["violations"] for r in fr]),
                "serving_entries": ms([r["serving_entries"] for r in fr]),
                "entry_cost_serving": ms([r["entry_cost_serving"] for r in fr]),
            }

out = dict(config=dict(seeds=SEEDS, synthetic=dict(N=40, T=8000, C=12),
                       google10k=dict(rows=10000, C=12)),
           rows=rows, k_effect=krows, landlord_ties=tie_rows,
           dl_mode=identical_rows, summary=summary,
           note=("current = engine download charges; strict = current + M_n for every "
                 "hit without a completed charged download; charge_all_upper = current + "
                 "M_n for every serving-path 0->1 transition (an upper bound, not the "
                 "paper's accounting)."))
save("phase_a_audit", out)

print("\n### summary")
for k, s in summary.items():
    print(f"{k:24s} cost={s['cost'][0]:.0f} strict={s['strict_cost'][0]:.0f} "
          f"charge_all_upper={s['charge_all_upper'][0]:.0f} "
          f"viol={s['violations'][0]:.2f} entries={s['serving_entries'][0]:.1f}")
print("\n### K effect")
for r in krows:
    print(r)
print("\n### landlord bankrupt set sizes")
for r in tie_rows:
    print(r)
