"""Phase B: cost-latency frontier, paired interpolation, equal-cost comparison.

Outputs results/phase_b_frontier.json and .png.
"""
from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THETAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
THETAS_SMALL = [0.0, 1.0, 4.0, 16.0]
VARIANTS = {
    "default": dict(eta_mult=1.0, alpha=0.1, K=4),
    "eta4": dict(eta_mult=4.0, alpha=0.1, K=4),
    "K8": dict(eta_mult=1.0, alpha=0.05, K=8),
}
rows = []
print("Phase B: synthetic frontier (10 seeds)")
for s in range(10):
    cat, tr = synthetic_world(N=40, T=8000, seed=s, imbalance=2.0, cap_mult=4)
    C = 12
    eta0 = np.sqrt(cat.N) / (float(cat.l.max()) * np.sqrt(len(tr)))
    for th in THETAS:
        d = online_drl(cat, tr, C, theta=th)
        rows.append(dict(family="synthetic", regime="raw", seed=s, algo="online_drl",
                         theta=th, variant="det", latency=float(d.latency),
                         cost=float(d.cost), hits=int(d.hits), downloads=int(d.downloads),
                         capacity=C, frac_M_lt_l=float(np.mean(cat.M < cat.l))))
    for name, kw in VARIANTS.items():
        r = rand_drl(cat, tr, C, np.random.default_rng(s), K=kw["K"],
                     repair_mode="stateless", gate_downloads=True,
                     download_time_mode="ema", eta=kw["eta_mult"] * eta0,
                     q_ema_alpha=kw["alpha"])
        rows.append(dict(family="synthetic", regime="raw", seed=s, algo="rand_stateless",
                         theta=float("nan"), variant=name, latency=float(r.latency),
                         cost=float(r.cost), hits=int(r.hits), downloads=int(r.downloads),
                         capacity=C, frac_M_lt_l=float(np.mean(cat.M < cat.l))))
    print(f"syn seed {s} done")

print("Phase B: google10k frontier (3 seeds, both regimes)")
for regime, kw in (("raw", dict(D=1000.0, disk_scale=1.0)),
                   ("crossing", dict(D=1000.0, disk_scale=1e7))):
    cat, tr = load_google("google.csv", max_rows=10000, **kw)
    C = min(12, max(1, cat.N // 2))
    fl = float(np.mean(cat.M < cat.l))
    for s in range(3):
        for th in THETAS_SMALL:
            d = online_drl(cat, tr, C, theta=th)
            rows.append(dict(family="google10k", regime=regime, seed=s, algo="online_drl",
                             theta=th, variant="det", latency=float(d.latency),
                             cost=float(d.cost), hits=int(d.hits), downloads=int(d.downloads),
                             capacity=C, frac_M_lt_l=fl))
        r = rand_drl(cat, tr, C, np.random.default_rng(s), K=4,
                     repair_mode="stateless", gate_downloads=True,
                     download_time_mode="ema")
        rows.append(dict(family="google10k", regime=regime, seed=s, algo="rand_stateless",
                         theta=float("nan"), variant="default", latency=float(r.latency),
                         cost=float(r.cost), hits=int(r.hits), downloads=int(r.downloads),
                         capacity=C, frac_M_lt_l=fl))
    print(f"google10k {regime} done")

print("Phase B: google50k frontier (1 seed, raw regime)")
for regime, kw in (("raw", dict(D=1000.0, disk_scale=1.0)),):
    cat, tr = load_google("google.csv", max_rows=50000, **kw)
    C = min(12, max(1, cat.N // 2))
    fl = float(np.mean(cat.M < cat.l))
    for s in range(1):
        for th in THETAS_SMALL:
            d = online_drl(cat, tr, C, theta=th)
            rows.append(dict(family="google50k", regime=regime, seed=s, algo="online_drl",
                             theta=th, variant="det", latency=float(d.latency),
                             cost=float(d.cost), hits=int(d.hits), downloads=int(d.downloads),
                             capacity=C, frac_M_lt_l=fl))
        r = rand_drl(cat, tr, C, np.random.default_rng(s), K=2,
                     repair_mode="stateless", gate_downloads=True,
                     download_time_mode="ema")
        rows.append(dict(family="google50k", regime=regime, seed=s, algo="rand_stateless",
                         theta=float("nan"), variant="default", latency=float(r.latency),
                         cost=float(r.cost), hits=int(r.hits), downloads=int(r.downloads),
                         capacity=C, frac_M_lt_l=fl))
    print(f"google50k {regime} done")

# B3: paired interpolation at the randomized realized cost
interp_rows = []
for fam in ("synthetic", "google10k", "google50k"):
    regimes = sorted(set(r["regime"] for r in rows if r["family"] == fam))
    for regime in regimes:
        for s in sorted(set(r["seed"] for r in rows if r["family"] == fam and r["regime"] == regime)):
            det = [r for r in rows if r["family"] == fam and r["regime"] == regime
                   and r["seed"] == s and r["algo"] == "online_drl"]
            rr = [r for r in rows if r["family"] == fam and r["regime"] == regime
                  and r["seed"] == s and r["algo"] == "rand_stateless"]
            for r in rr:
                lat = interp_log_cost(det, r["cost"])
                interp_rows.append(dict(family=fam, regime=regime, seed=s,
                                        variant=r["variant"], rand_lat=r["latency"],
                                        rand_cost=r["cost"], det_interp_lat=lat,
                                        diff=(r["latency"] - lat) if np.isfinite(lat) else None))

summary = {}
for fam in ("synthetic", "google10k", "google50k"):
    for regime in sorted(set(r["regime"] for r in rows if r["family"] == fam)):
        for variant in sorted(set(r["variant"] for r in rows if r["family"] == fam
                                  and r["regime"] == regime and r["algo"] == "rand_stateless")):
            fr = [r for r in rows if r["family"] == fam and r["regime"] == regime
                  and r["algo"] == "rand_stateless" and r["variant"] == variant]
            if fr:
                summary[f"{fam}/{regime}/rand/{variant}"] = {
                    "latency": ms([r["latency"] for r in fr]),
                    "cost": ms([r["cost"] for r in fr]),
                    "hits": ms([r["hits"] for r in fr])}
        for th in sorted(set(r["theta"] for r in rows if r["family"] == fam
                             and r["regime"] == regime and r["algo"] == "online_drl")):
            fr = [r for r in rows if r["family"] == fam and r["regime"] == regime
                  and r["algo"] == "online_drl" and r["theta"] == th]
            if fr:
                summary[f"{fam}/{regime}/det/theta={th:g}"] = {
                    "latency": ms([r["latency"] for r in fr]),
                    "cost": ms([r["cost"] for r in fr]),
                    "hits": ms([r["hits"] for r in fr])}

interp_summary = {}
for fam in ("synthetic", "google10k", "google50k"):
    for regime in sorted(set(r["regime"] for r in rows if r["family"] == fam)):
        fr = [r for r in interp_rows if r["family"] == fam and r["regime"] == regime
              and r["diff"] is not None]
        if fr:
            d = np.array([r["diff"] for r in fr], float)
            interp_summary[f"{fam}/{regime}"] = dict(
                n=len(d), mean=float(d.mean()), ci_half=float(1.96*d.std(ddof=1)/np.sqrt(len(d)))
                if len(d) > 1 else 0.0, win_rate=float(np.mean(d < 0)))

b5 = {}
for fam in ("synthetic", "google10k", "google50k"):
    for regime in sorted(set(r["regime"] for r in rows if r["family"] == fam)):
        fr = [r for r in rows if r["family"] == fam and r["regime"] == regime
              and r["algo"] == "online_drl"]
        if not fr:
            continue
        lat = [r["latency"] for r in fr]; cost = [r["cost"] for r in fr]
        b5[f"{fam}/{regime}"] = dict(
            latency_min=float(np.min(lat)), latency_max=float(np.max(lat)),
            latency_range_pct=float(100*(np.max(lat)-np.min(lat))/np.mean(lat)),
            cost_min=float(np.min(cost)), cost_max=float(np.max(cost)),
            cost_ratio=float(np.max(cost)/max(np.min(cost), 1e-12)),
            hits_mean=float(np.mean([r["hits"] for r in fr])),
            frac_M_lt_l=float(fr[0]["frac_M_lt_l"]),
            capacity=int(fr[0]["capacity"]))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for fam, color in (("synthetic", "#1f3b73"), ("google10k", "#1b6b3a"),
                   ("google50k", "#c58b2a")):
    for regime, ls in (("raw", "-"), ("crossing", "--")):
        fr = [r for r in rows if r["family"] == fam and r["regime"] == regime
              and r["algo"] == "online_drl"]
        if fr:
            ths = sorted(set(r["theta"] for r in fr))
            lat = [np.mean([r["latency"] for r in fr if r["theta"] == t]) for t in ths]
            cost = [np.mean([r["cost"] for r in fr if r["theta"] == t]) for t in ths]
            ax[0].plot(cost, lat, ls, color=color, label=f"{fam}/{regime}")
        fr = [r for r in rows if r["family"] == fam and r["regime"] == regime
              and r["algo"] == "rand_stateless"]
        if fr:
            ax[0].scatter(np.mean([r["cost"] for r in fr]),
                          np.mean([r["latency"] for r in fr]), marker="*",
                          s=120, color=color, zorder=5)
ax[0].set_xlabel("download cost"); ax[0].set_ylabel("latency")
ax[0].set_title("Cost-latency frontier (stars = randomized)")
ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
for fam, color in (("synthetic", "#1f3b73"), ("google10k", "#1b6b3a")):
    for variant in VARIANTS:
        fr = [r for r in rows if r["family"] == fam and r["regime"] == "raw"
              and r["algo"] == "rand_stateless" and r["variant"] == variant]
        if fr:
            ax[1].scatter(np.mean([r["cost"] for r in fr]),
                          np.mean([r["latency"] for r in fr]), s=45, color=color, alpha=.7)
ax[1].set_xlabel("download cost"); ax[1].set_ylabel("latency")
ax[1].set_title("Randomized knob variants"); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "phase_b_frontier.png"), dpi=130)

out = dict(config=dict(thetas=THETAS, thetas_small=THETAS_SMALL, variants=VARIANTS,
                       seeds=dict(synthetic=10, google10k=3, google50k=1)),
           rows=rows, interp_rows=interp_rows, interp_summary=interp_summary,
           summary=summary, b5=b5,
           note=("B3 interpolates deterministic latency linearly in log cost at the "
                 "randomized realized cost. The 50k trace uses 1 seed for tractability; "
                 "the equal-cost comparison is in phase_b_equal.json."))
save("phase_b_frontier", out)
print("\n### frontier summary")
for k, s in summary.items():
    if "/rand/" in k:
        print(f"{k:34s} lat={s['latency'][0]:10.0f} cost={s['cost'][0]:10.0f}")
print("\n### interpolated paired latency (rand - det)")
for k, s in interp_summary.items():
    print(f"{k:24s} n={s['n']:3d} diff={s['mean']:+.0f} ±{s['ci_half']:.0f} win={s['win_rate']:.2f}")
print("\n### B5")
for k, s in b5.items():
    print(k, s)
