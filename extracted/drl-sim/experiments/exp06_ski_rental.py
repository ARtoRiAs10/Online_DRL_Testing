"""R3: randomized download threshold improves the rent-or-buy ratio from 2 to e/(e-1)."""
from common import *
from scipy.integrate import quad
from sim.engine import RandThreshold

a = parse(__doc__)
M, e = 1.0, np.e
f = lambda th: np.exp(th / M) / (M * (e - 1)); F = lambda th: (np.exp(th / M) - 1) / (e - 1)
ts = np.linspace(0.02, 3, 150)
def exp_cost(t):
    if t >= M: return quad(lambda th: (th + M) * f(th), 0, M)[0]
    return quad(lambda th: (th + M) * f(th), 0, t)[0] + t * (1 - F(t))
opt = np.minimum(ts, M)
r_rand = np.array([exp_cost(t) for t in ts]) / opt
r_det = np.where(ts < M, ts, 2 * M) / opt
rng = np.random.default_rng(0); N = 200_000 if not a.quick else 50_000
thr = np.array([RandThreshold(rng).draw(0) for _ in range(N)]) * M
mc = np.array([np.where(t < thr, t, thr + M).mean() / min(t, M) for t in ts[::15]])
print(f"deterministic worst ratio = {r_det.max():.4f} (theory 2)")
print(f"randomized worst ratio    = {r_rand.max():.4f}, min = {r_rand.min():.4f} (theory e/(e-1) = {e/(e-1):.4f}, constant for t<=M)")
print(f"Monte-Carlo (sampler used by the simulator) max deviation from analytic: {np.abs(mc - r_rand[::15]).max():.4f}")
fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.plot(ts / M, r_det, label="deterministic"); ax.plot(ts / M, r_rand, label="randomized"); ax.plot(ts[::15] / M, mc, "x", label="Monte-Carlo")
ax.axhline(e / (e - 1), ls=":", c="k"); ax.set_xlabel("stop time / M"); ax.set_ylabel("ALG / OPT"); ax.legend()
save(a, "exp06_ski_rental", dict(det_max=float(r_det.max()), rand_max=float(r_rand.max()), rand_min=float(r_rand.min()), theory=e / (e - 1)), fig)
