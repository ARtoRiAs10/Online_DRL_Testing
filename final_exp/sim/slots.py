"""Slot reductions phi(a) (Sec. 2) returning w_n / C_slot, i.e. the weight in a single constraint
sum_n phi_n p_n <= 1, plus weight-selection helpers and the capacity loss Lambda_phi."""
from __future__ import annotations
import numpy as np


def phi_dominant(a): return a.max(1)
def phi_linear(a, alpha): return a @ np.asarray(alpha, float)
def phi_lp(a, p, alpha=None):
    alpha = np.ones(a.shape[1]) if alpha is None else np.asarray(alpha, float)
    return ((alpha * a ** p).sum(1)) ** (1.0 / p)
def phi_shadow(a, rho):
    """Surrogate relaxation sum_r rho_r (sum_n a_nr p_n - 1) <= 0, normalized by sum(rho)."""
    rho = np.asarray(rho, float); return a @ rho / rho.sum()


def capacity_loss(a, phi):
    """Lambda_phi = max_n phi(a_n) / max_r a_{n,r}   (eq. 18)."""
    return float((phi / a.max(1)).max())


def ahp_weights(P=None):
    P = np.array([[1, 2, 3], [1 / 2, 1, 2], [1 / 3, 1 / 2, 1]]) if P is None else np.asarray(P, float)
    w, v = np.linalg.eig(P); x = np.abs(np.real(v[:, np.argmax(np.real(w))])); return x / x.sum()


def entropy_weights(a):
    q = a / a.sum(0, keepdims=True)
    H = -(q * np.log(q + 1e-12)).sum(0) / np.log(a.shape[0])
    w = 1 - H; return w / w.sum()


def utilization_weights(util):
    w = util.mean(0) + 1e-12; return w / w.sum()


def slot_rows(a, mode="dominant", alpha=None, p=2):
    """Row(s) of the *slot* (single-constraint) formulation of D_vec.

    mode = 'dominant'  : w_n = max_r a_{n,r}                       (eq. 15, always safe)
    mode = 'linear'    : w_n = sum_r alpha_r a_{n,r}                (eq. 16)
    mode = 'lp'        : w_n = (sum_r alpha_r a_{n,r}^p)^{1/p}      (eq. 17)
    mode = 'shadow'    : w_n = sum_r rho_r a_{n,r}                  (Sec. 2.5)
    Returns a (1, N) array, to be passed as A_rows to FracPolicy.
    """
    a = np.asarray(a, float)
    if mode == "dominant":
        w = phi_dominant(a)
    elif mode == "linear":
        alpha = np.ones(a.shape[1]) if alpha is None else np.asarray(alpha, float)
        w = phi_linear(a, alpha)
    elif mode == "lp":
        w = phi_lp(a, p, alpha)
    elif mode == "shadow":
        rho = np.ones(a.shape[1]) if alpha is None else np.asarray(alpha, float)
        w = a @ rho / rho.sum()
    else:
        raise ValueError(mode)
    return np.asarray(w, float)[None, :]


def lambda_estimate(util):
    """Empirical estimate of the capacity loss Lambda_phi from a realized utilization trace.

    The dominant-resource (slot) view charges a service the *largest* of its three demands,
    so the slot occupancy is U^max_t while the balanced usage is the mean over resources.
    Lambda_hat = E[U^max] / E[mean_r U_r]  >= 1  is the factor the slot reduction wastes.
    """
    util = np.asarray(util, float)
    if util.size == 0: return float("nan")
    Umax = util.max(1)
    return float(Umax.mean() / max(util.mean(1).mean(), 1e-12))
