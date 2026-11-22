"""Phase-4 EMA tracking check: drift/alpha + sqrt(alpha)*sigma.

Simulates a random walk with drift and noisy observations, compares the empirical MSE
of an EMA with the theoretical tracking-error expression used in the phase-4 notes.
"""
from common import *
from sim.regret_bound import ema_tracking_error, optimal_ema_alpha

a = parse(__doc__)
T = 2000 if a.quick else 10000
runs = 50 if a.quick else 200
sigmas = [0.5, 1.0, 2.0]
drifts = [0.0, 0.01, 0.1, 1.0]
alphas = [0.01, 0.03, 0.1, 0.3, 1.0]
print(f"phase-4 EMA tracking: T={T} runs={runs}")

rows = []
for sigma in sigmas:
    for drift in drifts:
        for alpha in alphas:
            se = 0.0
            for r in range(runs):
                rng = np.random.default_rng(1000 + r)
                x = 0.0
                ema = 0.0
                for _ in range(T):
                    x += drift
                    y = x + rng.normal(0.0, sigma)
                    ema = (1.0 - alpha) * ema + alpha * y
                    se += (ema - x) ** 2
            rmse = float(np.sqrt(se / (runs * T)))
            theory = ema_tracking_error(alpha, sigma, drift)
            rows.append(dict(sigma=sigma, drift=drift, alpha=alpha,
                             rmse=rmse, theory=float(theory)))

summary = {}
for sigma in sigmas:
    for drift in drifts:
        fr = [r for r in rows if r["sigma"] == sigma and r["drift"] == drift]
        best = min(fr, key=lambda r: r["rmse"])
        alpha_star = optimal_ema_alpha(sigma, drift)
        summary[f"sigma={sigma},drift={drift}"] = dict(
            best_alpha=best["alpha"], best_rmse=best["rmse"],
            theory_alpha=float(alpha_star),
            theory_error=float(ema_tracking_error(alpha_star, sigma, drift)),
        )

out = dict(config=dict(T=T, runs=runs, sigmas=sigmas, drifts=drifts, alphas=alphas),
           rows=rows, summary=summary)
save(a, "exp24_ema_tracking", out)

print("\n### summary (best empirical alpha vs theory)")
for key, s in summary.items():
    print(f"{key:28s} best_alpha={s['best_alpha']:.2f} "
          f"rmse={s['best_rmse']:.3f} | "
          f"theory_alpha={s['theory_alpha']:.3f} "
          f"theory_err={s['theory_error']:.3f}")