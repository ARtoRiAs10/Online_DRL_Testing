"""Surrogate-loss regret (G3 / Proposition 2)."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from .projection import project_dual


def best_fixed_value(n_seq, g_seq, An):
    """max_P sum_t p_{n_t} g_t over D_vec  (so min_P sum F_t(P) = sum g - this)."""
    N = An.shape[1]
    G = np.bincount(n_seq, weights=g_seq, minlength=N)
    r = linprog(-G, A_ub=An, b_ub=np.ones(An.shape[0]), bounds=(0, 1), method="highs")
    return -r.fun


def surrogate_regret(n_seq, g_seq, P_seq, An):
    """sum_t F_t(P_t) - min_P sum_t F_t(P),   F_t(P) = (1 - p_{n_t}) g_t."""
    n_seq = np.asarray(n_seq); g_seq = np.asarray(g_seq)
    loss = float(((1 - P_seq[np.arange(len(n_seq)), n_seq]) * g_seq).sum())
    return loss - (g_seq.sum() - best_fixed_value(n_seq, g_seq, An))


def ogd_run(n_seq, g_seq, An, eta, peek=True, eta_sched=None):
    """Projected OGD on a *given* (exogenous) gradient sequence; returns the played states.

    The returned ``P[i]`` is always the state **before** the i-th update (the point that
    is played at round i).  This is the convention of Proposition 2: the loss at round i
    is evaluated at ``P[i]``, and the gradient ``g_seq[i]`` is then used to move to
    ``P[i+1]``.  The ``peek`` flag is kept for backwards compatibility; it only controls
    whether the caller evaluates the gradient before or after the update, not which state
    is returned.  Earlier versions returned the post-update state when ``peek=True``,
    which made the measured regret inconsistent with the OGD bound.

    eta_sched=None  : a constant step size eta (Prop. 2: eta = sqrt(N)/(l_max sqrt(T))).
    eta_sched="1/sqrt(t)" : the horizon-free schedule eta_t = eta / sqrt(t).
    """
    N = An.shape[1]; L = np.linalg.norm(An, 2) ** 2
    p = np.zeros(N); rho = None; P = np.zeros((len(n_seq), N), np.float32)
    for i, (n, g) in enumerate(zip(n_seq, g_seq)):
        P[i] = p                                   # pre-update / played state
        e = eta if eta_sched is None else eta / np.sqrt(i + 1)
        z = p.copy(); z[n] += e * g
        p, rho, _ = project_dual(z, An, rho, L)
    return P
