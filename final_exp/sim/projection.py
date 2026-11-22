"""Euclidean projection onto D_vec = {P in [0,1]^N : A P <= b}  (eqs. 21-25 of the proposal)."""
from __future__ import annotations
import numpy as np


def normalize(A, b):
    """Scale rows so that every constraint reads  An p <= 1."""
    return A / np.asarray(b, float)[:, None]


def project_dual(z, An, rho0=None, L=None, tol=1e-7, max_iter=1000):
    """KKT / dual-ascent projection.  p(rho) = clip(z - An^T rho, 0, 1)   (eq. 22)

    Accelerated projected dual ascent with step 1/L, L = ||An||_2^2 (the dual gradient
    An p(rho) - 1 is L-Lipschitz), gradient restart, warm start rho0.  A final scaling
    (eq. 25) makes the result exactly feasible.  Returns (p, rho, iterations).
    """
    p = np.clip(z, 0.0, 1.0)
    if np.all(An @ p <= 1.0 + tol):
        return p, np.zeros(An.shape[0]), 0
    if L is None:
        L = np.linalg.norm(An, 2) ** 2
    rho = np.zeros(An.shape[0]) if rho0 is None else rho0.copy()
    y, t = rho.copy(), 1.0
    it = 0
    for it in range(1, max_iter + 1):
        g = An @ np.clip(z - An.T @ y, 0.0, 1.0) - 1.0
        rho_new = np.maximum(y + g / L, 0.0)
        if g @ (rho_new - rho) < 0:           # momentum restart
            t, y = 1.0, rho_new.copy()
        else:
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            y = rho_new + (t - 1) / t_new * (rho_new - rho)
            t = t_new
        rho = rho_new
        gc = An @ np.clip(z - An.T @ rho, 0.0, 1.0) - 1.0
        if gc.max() <= tol and abs(rho @ gc) <= tol:
            break
    p = np.clip(z - An.T @ rho, 0.0, 1.0)
    p = p / max(1.0, float((An @ p).max()))     # eq. (25)
    return p, rho, it


def project_exact(z, An):
    """Reference solution of the QP (21) with cvxpy (used for testing only)."""
    import cvxpy as cp
    p = cp.Variable(len(z))
    cp.Problem(cp.Minimize(0.5 * cp.sum_squares(p - z)), [p >= 0, p <= 1, An @ p <= 1]).solve()
    return np.clip(p.value, 0, 1)


def project_dual_active(z, An, active, rho0=None, L=None, tol=1e-7, max_iter=1000):
    """Projection onto D_vec restricted to the *active set* (revision N7.1).

    Services outside `active` have z_j <= 0, so their projection is 0 and they contribute
    nothing to A p; restricting the projection to the support of p is therefore exact, and the
    cost drops from O(N) to O(|active|) per dual-ascent iteration.  `active` must contain every
    index with z_j > 0.
    """
    idx = np.asarray(active, int)
    if idx.size == 0:
        return np.zeros(An.shape[1]), (np.zeros(An.shape[0]) if rho0 is None else rho0), 0
    A_sub = An[:, idx]
    z_sub = z[idx]
    if L is None:
        # Frobenius bound of the spectral norm: O(4|idx|) instead of an SVD per request
        L = float((A_sub ** 2).sum())
    p_sub, rho, it = project_dual(z_sub, A_sub, rho0, L, tol=tol, max_iter=max_iter)
    p = np.zeros(An.shape[1])
    p[idx] = p_sub
    return p, rho, it
