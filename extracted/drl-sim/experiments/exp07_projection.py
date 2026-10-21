"""Projection correctness: dual ascent (eqs. 22-25) vs an exact QP (cvxpy)."""
from common import *
from sim.projection import project_dual, project_exact, normalize

a = parse(__doc__)
rng = np.random.default_rng(0); trials = 10 if a.quick else 60
rows = []
for _ in range(trials):
    N = int(rng.integers(10, 80))
    aa = rng.uniform(.01, .6, (N, 3)); C = float(rng.integers(2, max(3, N // 3)))
    An = normalize(np.vstack([np.ones(N), aa.T]), np.r_[C, 1, 1, 1]); z = rng.uniform(-.3, 1.6, N)
    p, rho, it = project_dual(z, An, tol=1e-9, max_iter=5000); q = project_exact(z, An)
    kkt = abs(rho @ (An @ p - 1))
    rows.append(dict(N=N, iters=it, l2_err=float(np.linalg.norm(p - q)), obj_gap=float(.5 * ((p - z) ** 2).sum() - .5 * ((q - z) ** 2).sum()),
                     max_viol=float((An @ p - 1).max()), compl_slack=float(kkt)))
for k in ("iters", "l2_err", "obj_gap", "max_viol", "compl_slack"):
    v = np.array([r[k] for r in rows]); print(f"{k:12s} mean={v.mean():.3e} max={v.max():.3e}")
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].hist(np.log10([r["l2_err"] + 1e-16 for r in rows])); ax[0].set_title("log10 ||p_dual - p_QP||")
ax[1].hist([r["iters"] for r in rows]); ax[1].set_title("dual-ascent iterations")
save(a, "exp07_projection", dict(rows=rows), fig)
