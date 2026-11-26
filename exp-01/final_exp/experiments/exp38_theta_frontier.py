"""Phase-1 (v2): the deterministic cost-latency frontier.

Online-DRL already trades cost for latency through theta.  A cost reduction is an
improvement only if the randomized method lies on or below that curve.

This experiment sweeps

* theta in {0, 0.25, 0.5, 1, 2, 4, 8, 16} for Online-DRL;
* randomized stateless DRL with three knob settings (eta multiplier, alpha, K);

on

* the synthetic Zipf/Poisson world (N=40, T=8000, 10 seeds);
* the 10k Google slice (3 seeds, both M/l regimes);
* the full 50k Google trace (3 seeds, K=2, both regimes; reduced theta grid).

It also reports a budget-matched table at 20/50/100% of the theta=1 Online-DRL cost.
"""
from common import *
from sim.trace import synthetic_world, load_trace_csv
from sim.algos import online_drl, rand_drl

a = parse(__doc__, C=12, K=4)
THETAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
THETAS_SMALL = [0.0, 1.0, 4.0, 16.0]
VARIANTS = {
    "default": dict(eta_mult=1.0, alpha=0.1, K=4),
    "eta4": dict(eta_mult=4.0, alpha=0.1, K=4),
    "K8": dict(eta_mult=1.0, alpha=0.05, K=8),
}
seeds_syn = 1 if a.quick else 10
print(f"phase-1 v2 frontier: synthetic seeds={seeds_syn}, thetas={THETAS}")

rows = []


def run_synthetic():
    for s in range(seeds_syn):
        cat, tr = synthetic_world(N=40, T=8000, seed=s, imbalance=2.0, cap_mult=4)
        C = a.C
        lmax = float(cat.l.max()); T = len(tr)
        eta0 = np.sqrt(cat.N) / (lmax * np.sqrt(T))
        for theta in THETAS:
            det = online_drl(cat, tr, C, theta=theta)
            rows.append(dict(family="synthetic", seed=s, algo="online_drl", theta=theta,
                             variant="det", latency=float(det.latency), cost=float(det.cost),
                             hits=int(det.hits), downloads=int(det.downloads)))
            print(f"syn seed {s} theta={theta:g}: det lat={det.latency:.0f} cost={det.cost:.0f}")
        for name, kw in VARIANTS.items():
            rr = rand_drl(cat, tr, C, np.random.default_rng(s), K=kw["K"],
                          repair_mode="stateless", gate_downloads=True,
                          download_time_mode="ema",
                          eta=kw["eta_mult"] * eta0, q_ema_alpha=kw["alpha"])
            rows.append(dict(family="synthetic", seed=s, algo="rand_stateless",
                             theta=float("nan"), variant=name,
                             latency=float(rr.latency), cost=float(rr.cost),
                             hits=int(rr.hits), downloads=int(rr.downloads)))
            print(f"syn seed {s} rand {name}: lat={rr.latency:.0f} cost={rr.cost:.0f}")


def run_google10k():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "google.csv")
    CFG = dict(id_col="JobID", time_col="Timestamp", cpu_col="CPU", ram_col="RAM",
               disk_col="Disk", time_scale_ms=1e-3, disk_unit_mb=1.0 / 1024.0, U=100.0)
    seeds = 1 if a.quick else 3
    configs = [(1000.0, 1.0)] if a.quick else [(1000.0, 1.0), (1000.0, 1e7)]
    for D, disk_scale in configs:
        cat, tr = load_trace_csv(path, max_rows=10000, D=D, disk_scale=disk_scale, **CFG)
        C = min(a.C, max(1, cat.N // 2))
        frac_lt = float(np.mean(cat.M < cat.l))
        for s in range(seeds):
            for theta in THETAS_SMALL:
                det = online_drl(cat, tr, C, theta=theta)
                rows.append(dict(family="google10k", seed=s, algo="online_drl",
                                 theta=theta, variant="det", D=D, disk_scale=disk_scale,
                                 frac_M_lt_l=frac_lt,
                                 latency=float(det.latency), cost=float(det.cost),
                                 hits=int(det.hits), downloads=int(det.downloads)))
            rr = rand_drl(cat, tr, C, np.random.default_rng(s), K=a.K,
                          repair_mode="stateless", gate_downloads=True,
                          download_time_mode="ema")
            rows.append(dict(family="google10k", seed=s, algo="rand_stateless",
                             theta=float("nan"), variant="default", D=D,
                             disk_scale=disk_scale, frac_M_lt_l=frac_lt,
                             latency=float(rr.latency), cost=float(rr.cost),
                             hits=int(rr.hits), downloads=int(rr.downloads)))
        print(f"google10k D={D:.0f} dx={disk_scale:.0e} M<l={frac_lt:.3f}: done")


def run_google50k():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "google.csv")
    CFG = dict(id_col="JobID", time_col="Timestamp", cpu_col="CPU", ram_col="RAM",
               disk_col="Disk", time_scale_ms=1e-3, disk_unit_mb=1.0 / 1024.0, U=100.0)
    seeds = 3
    K = 2
    configs = [(1000.0, 1.0)] if a.quick else [(1000.0, 1.0), (1000.0, 1e7)]
    for D, disk_scale in configs:
        cat, tr = load_trace_csv(path, max_rows=50000, D=D, disk_scale=disk_scale, **CFG)
        C = min(a.C, max(1, cat.N // 2))
        frac_lt = float(np.mean(cat.M < cat.l))
        for s in range(seeds):
            for theta in THETAS_SMALL:
                det = online_drl(cat, tr, C, theta=theta)
                rows.append(dict(family="google50k", seed=s, algo="online_drl",
                                 theta=theta, variant="det", D=D, disk_scale=disk_scale,
                                 frac_M_lt_l=frac_lt,
                                 latency=float(det.latency), cost=float(det.cost),
                                 hits=int(det.hits), downloads=int(det.downloads)))
            rr = rand_drl(cat, tr, C, np.random.default_rng(s), K=K,
                          repair_mode="stateless", gate_downloads=True,
                          download_time_mode="ema")
            rows.append(dict(family="google50k", seed=s, algo="rand_stateless",
                             theta=float("nan"), variant="default", D=D,
                             disk_scale=disk_scale, frac_M_lt_l=frac_lt,
                             latency=float(rr.latency), cost=float(rr.cost),
                             hits=int(rr.hits), downloads=int(rr.downloads)))
        print(f"google50k D={D:.0f} dx={disk_scale:.0e} M<l={frac_lt:.3f}: done")


def run_budget_matched():
    for s in range(seeds_syn):
        cat, tr = synthetic_world(N=40, T=8000, seed=s, imbalance=2.0, cap_mult=4)
        C = a.C
        det_full = online_drl(cat, tr, C, theta=1.0)
        for frac in (0.2, 0.5, 1.0):
            B = frac * det_full.cost
            det_b = online_drl(cat, tr, C, theta=1.0, bmax=B)
            rr = rand_drl(cat, tr, C, np.random.default_rng(s), K=a.K,
                          repair_mode="stateless", gate_downloads=True,
                          download_time_mode="ema", bmax=B)
            rows.append(dict(family="budget", seed=s, frac=frac, Bmax=float(B),
                             det_lat=float(det_b.latency), det_cost=float(det_b.cost),
                             rand_lat=float(rr.latency), rand_cost=float(rr.cost)))


print("running synthetic ...")
run_synthetic()
print("running google10k ...")
run_google10k()
print("running google50k ...")
if not a.quick:
    run_google50k()
print("running budget-matched ...")
run_budget_matched()

# ---------------------------------------------------------------- summaries
summary = {}
for family in ("synthetic", "google10k", "google50k"):
    for variant in sorted(set(r["variant"] for r in rows
                              if r["family"] == family and r["algo"] == "rand_stateless")):
        fr = [r for r in rows if r["family"] == family and r["algo"] == "rand_stateless"
              and r["variant"] == variant]
        summary[f"{family}/rand/{variant}"] = {
            "latency": ms([r["latency"] for r in fr]),
            "cost": ms([r["cost"] for r in fr]),
            "hits": ms([r["hits"] for r in fr]),
        }
    for theta in THETAS:
        fr = [r for r in rows if r["family"] == family and r["algo"] == "online_drl"
              and r["theta"] == theta]
        if fr:
            summary[f"{family}/det/theta={theta:g}"] = {
                "latency": ms([r["latency"] for r in fr]),
                "cost": ms([r["cost"] for r in fr]),
                "hits": ms([r["hits"] for r in fr]),
            }

budget_summary = {}
for frac in (0.2, 0.5, 1.0):
    fr = [r for r in rows if r["family"] == "budget" and r["frac"] == frac]
    if fr:
        budget_summary[str(frac)] = {
            "Bmax": ms([r["Bmax"] for r in fr]),
            "det_lat": ms([r["det_lat"] for r in fr]),
            "rand_lat": ms([r["rand_lat"] for r in fr]),
            "det_cost": ms([r["det_cost"] for r in fr]),
            "rand_cost": ms([r["rand_cost"] for r in fr]),
            "paired_lat": paired_ci([r["rand_lat"] for r in fr],
                                    [r["det_lat"] for r in fr]),
            "paired_cost": paired_ci([r["rand_cost"] for r in fr],
                                     [r["det_cost"] for r in fr]),
        }

# ---------------------------------------------------------------- plot
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for family, color in (("synthetic", "#1f3b73"), ("google10k", "#1b6b3a"),
                      ("google50k", "#c58b2a")):
    fr = [r for r in rows if r["family"] == family and r["algo"] == "online_drl"]
    if fr:
        # average over seeds for each theta
        thetas = sorted(set(r["theta"] for r in fr))
        lat = [np.mean([r["latency"] for r in fr if r["theta"] == t]) for t in thetas]
        cost = [np.mean([r["cost"] for r in fr if r["theta"] == t]) for t in thetas]
        ax[0].plot(cost, lat, "o-", color=color, label=f"{family} Online-DRL")
    fr = [r for r in rows if r["family"] == family and r["algo"] == "rand_stateless"
          and r["variant"] == "default"]
    if fr:
        ax[0].scatter(np.mean([r["cost"] for r in fr]),
                      np.mean([r["latency"] for r in fr]),
                      marker="*", s=140, color=color, zorder=5)
ax[0].set_xlabel("download cost"); ax[0].set_ylabel("latency")
ax[0].set_title("Cost-latency frontier (stars = randomized stateless)")
ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

for family, color in (("synthetic", "#1f3b73"), ("google10k", "#1b6b3a"),
                      ("google50k", "#c58b2a")):
    for variant in VARIANTS:
        fr = [r for r in rows if r["family"] == family and r["algo"] == "rand_stateless"
              and r["variant"] == variant]
        if fr:
            ax[1].scatter(np.mean([r["cost"] for r in fr]),
                          np.mean([r["latency"] for r in fr]),
                          marker="o", s=45, color=color, alpha=.7)
ax[1].set_xlabel("download cost"); ax[1].set_ylabel("latency")
ax[1].set_title("Randomized knob variants")
ax[1].grid(alpha=.3)

out = dict(
    config=dict(thetas=THETAS, variants=VARIANTS, seeds_synthetic=seeds_syn),
    rows=rows,
    summary=summary,
    budget=budget_summary,
    note=("Online-DRL theta sweep vs randomized stateless.  The randomized method is on "
          "the same axes; the budget-matched table uses the theta=1 Online-DRL cost."),
)
save(a, "exp38_theta_frontier", out, fig)

print("\n### summary")
for k, s in summary.items():
    print(f"{k:34s} lat={s['latency'][0]:10.0f} ±{s['latency'][1]:8.0f} "
          f"cost={s['cost'][0]:10.0f} ±{s['cost'][1]:8.0f}")
print("\n### budget-matched")
for frac, s in budget_summary.items():
    print(f"B={frac}: det lat={s['det_lat'][0]:.0f} rand lat={s['rand_lat'][0]:.0f} "
          f"paired {s['paired_lat']['mean']:+.0f} ±{s['paired_lat']['ci_half']:.0f} "
          f"cost paired {s['paired_cost']['mean']:+.0f} ±{s['paired_cost']['ci_half']:.0f}")