"""Phase-1 diagnostics: step size, repair, top-up, and per-step logging.

Three hypotheses from the week-1 plan, run on one fixed synthetic world:

  1. step size too small: eta x {1, 4, 16, 64};
  2. repair evicts too much: none / stateless / landlord (gate off to isolate repair);
  3. cache not filled: optional top-up after repair.

Every run uses ``log_steps=True`` so the diagnostic also exercises the new per-step
log (p sums, resource loads, hit ranks, repair-eviction fraction, c_t, c~_t,
p_{n*,t-1} - xbar_{n*,t-1}) and writes a small summary.
"""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, rand_drl
from sim.rand_drl import FracPolicy
from sim.regret_bound import summarize_step_log

a = parse(__doc__, C=12, K=4)
# Keep the diagnostic small enough to run in a couple of minutes.
N = 16 if a.quick else 20
T = 500 if a.quick else 1000
K = a.K
C = a.C
print(f"phase-1 diagnostic: N={N} T={T} K={K} C={C} seeds={a.seeds}")

cat, tr = synthetic_world(N=N, T=T, seed=0, imbalance=2.0, cap_mult=4)
det = online_drl(cat, tr, C)
print(f"Online-DRL reference: latency={det.latency:.0f} cost={det.cost:.0f} hits={det.hits}")
pol0 = FracPolicy(cat, C, np.random.default_rng(0), K=K, eta_mode="horizon")
pol0.set_horizon(T)
base_eta = float(pol0.eta)
print(f"base eta = {base_eta:.6g} (l_max={cat.l.max():.3f}, M_max={cat.M.max():.3f})")


def run_one(label, **kw):
    runs = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        R = rand_drl(cat, tr, C, rng, K=K, log_steps=True, **kw)
        s = summarize_step_log(R.extra.get("step_log", []))
        util = R.util.mean(0)
        runs.append(dict(
            seed=seed,
            latency=float(R.latency),
            cost=float(R.cost),
            hits=int(R.hits),
            hit_ratio=float(R.hits / R.n_req),
            delayed=int(R.delayed_hits),
            util_cpu=float(util[0]),
            util_ram=float(util[1]),
            util_disk=float(util[2]),
            util_max=float(util.max()),
            mean_p_sum=s.get("mean_p_sum"),
            mean_p_sum_after=s.get("mean_p_sum_after"),
            mean_repair_evicted_fraction=s.get("mean_repair_evicted_fraction"),
            mean_grad=s.get("mean_grad"),
            mean_abs_p_minus_xbar=s.get("mean_abs_p_minus_xbar"),
            hit_rank_mean=s.get("hit_rank_mean"),
            outcomes=s.get("outcomes"),
        ))
    mean = lambda key: float(np.mean([r[key] for r in runs if r[key] is not None]))
    summary = dict(
        label=label,
        latency=mean("latency"),
        cost=mean("cost"),
        hits=mean("hits"),
        hit_ratio=mean("hit_ratio"),
        util_cpu=mean("util_cpu"),
        util_ram=mean("util_ram"),
        util_disk=mean("util_disk"),
        util_max=mean("util_max"),
        mean_p_sum=mean("mean_p_sum"),
        mean_p_sum_after=mean("mean_p_sum_after"),
        mean_repair_evicted_fraction=mean("mean_repair_evicted_fraction"),
        mean_grad=mean("mean_grad"),
        mean_abs_p_minus_xbar=mean("mean_abs_p_minus_xbar"),
        hit_rank_mean=mean("hit_rank_mean"),
        runs=runs,
    )
    print(f"  {label:32s} lat={summary['latency']:9.0f} cost={summary['cost']:8.0f} "
          f"hits={summary['hits']:6.0f} util={summary['util_max']:.3f} "
          f"p_sum={summary['mean_p_sum']:.3f} repair_frac={summary['mean_repair_evicted_fraction']:.3f}")
    return summary


out = {"config": dict(N=N, T=T, K=K, C=C, seeds=a.seeds, base_eta=base_eta,
                      l_max=float(cat.l.max()), M_max=float(cat.M.max())),
       "online_drl": dict(latency=float(det.latency), cost=float(det.cost), hits=int(det.hits)),
       "eta_sweep": [], "repair_ablation": [], "topup": [], "lazy_topup": []}

print("\n[1] eta sweep (landlord repair, gate on)")
for mult in (1, 4, 16, 64):
    out["eta_sweep"].append(run_one(
        f"eta x{mult:g}", eta=base_eta * mult, repair_mode="landlord", gate_downloads=True))

print("\n[2] repair ablation (gate off, to isolate repair)")
for mode in ("none", "stateless", "landlord", "landlord_pop", "landlord_hybrid"):
    out["repair_ablation"].append(run_one(
        f"repair={mode}", repair_mode=mode, gate_downloads=False))

print("\n[3] lazy repair / top-up (landlord repair, gate on)")
for lazy in (False, True):
    for topup in (False, True):
        out["lazy_topup"].append(run_one(
            f"landlord lazy={lazy} topup={topup}",
            repair_mode="landlord", gate_downloads=True, topup=topup, lazy=lazy))

save(a, "diagnose_phase1", out)

# Print a compact comparison table.
print("\n### summary")
for group in ("eta_sweep", "repair_ablation", "lazy_topup"):
    print(f"\n{group}")
    for r in out[group]:
        print(f"  {r['label']:32s} latency={r['latency']:9.0f} cost={r['cost']:8.0f} "
              f"hits={r['hits']:6.0f} util={r['util_max']:.3f} "
              f"p_sum={r['mean_p_sum']:.3f} repair_frac={r['mean_repair_evicted_fraction']:.3f}")