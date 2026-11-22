"""Surrogate-loss regret (G3 / Proposition 1)."""
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
    """Plain projected OGD on a *given* (exogenous) gradient sequence; returns the P_t played.

    eta_sched=None  : a constant step size eta (Prop. 1: eta = sqrt(N)/(l_max sqrt(T))).
    eta_sched="1/sqrt(t)" : the horizon-free schedule eta_t = eta / sqrt(t) (revision R-B),
                            for which the standard OGD bound becomes 2 sqrt(N) l_max sqrt(T).
    """
    N = An.shape[1]; L = np.linalg.norm(An, 2) ** 2
    p = np.zeros(N); rho = None; P = np.zeros((len(n_seq), N), np.float32)
    for i, (n, g) in enumerate(zip(n_seq, g_seq)):
        e = eta if eta_sched is None else eta / np.sqrt(i + 1)
        if peek:
            z = p.copy(); z[n] += e * g; p, rho, _ = project_dual(z, An, rho, L)
            P[i] = p
        else:
            P[i] = p
            z = p.copy(); z[n] += e * g; p, rho, _ = project_dual(z, An, rho, L)
    return P
