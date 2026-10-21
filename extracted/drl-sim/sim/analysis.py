"""Helpers to connect the simulator with the *mathematical* statements of the documents.

  * measure the beta_up / beta_down fractions that appear in Theorem 1 of the paper
    (the fraction of M_x / M_min requests that are forwarded to the cloud while a
    download is in flight, instead of being buffered as a delayed hit);
  * evaluate the theoretical competitive-ratio bounds  gamma in O(k(1+3 beta_up M_x))
    and  gamma in Omega(k(1+3 beta_down M_min));
  * fit the *empirical* growth of gamma with k (linear vs logarithmic) and fit a power law
    to the measured running time (time complexity of the algorithms themselves).
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- betas
def measure_betas(res, cat, trace):
    """beta_up / beta_down estimated from a simulation run.

    For every service that had a download in flight we count the requests that were
    forwarded to the cloud (remaining download time > forwarding latency l_n) instead of
    being buffered (delayed hit).  The paper's beta for a service is the fraction of its
    M_i download slots that are forwarded, i.e. beta_i = fwd_i / M_i.

    beta_up   := max_i beta_i          (the largest fraction, used with M_x in the O-bound)
    beta_down := mean_i beta_i         (the typical fraction, used with M_min in the Omega-bound)
    """
    T = len(trace)
    fwd = np.asarray(res.fwd_srv, float); buf = np.asarray(res.buf_srv, float)
    served = fwd + buf
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = np.where(served > 0, fwd / np.maximum(cat.M, 1e-9), 0.0)
    eps[served == 0] = 0.0
    # beta_up   := max_i beta_i          (the largest fraction, used with M_x in the O-bound)
    # beta_down := mean_i beta_i         (the typical fraction, used with M_min in the Omega-bound)
    beta_up = float(eps.max()) if eps.size else 0.0
    beta_down = float(eps[served > 0].mean()) if (served > 0).any() else 0.0
    return dict(beta_up=beta_up, beta_down=beta_down,
                beta_min_service=float(eps[np.argmin(cat.M)]) if eps.size else 0.0,
                engine_fwd_inflight=int(res.fwd_inflight), engine_inflight=int(res.inflight_req))


def ratio_bounds(k, beta_up, beta_down, Mx, Mmin):
    """The paper's Theorem 1 evaluated numerically (k = cache capacity)."""
    return dict(O_bound=k * (1 + 3 * beta_up * Mx),
                Omega_bound=k * (1 + 3 * beta_down * Mmin))


# --------------------------------------------------------------------------- fits
def power_fit(x, y):
    """Least-squares fit y = a * x^b (log-log) -> empirical exponent b."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = (x > 0) & (y > 0)
    if m.sum() < 2: return dict(a=float("nan"), b=float("nan"))
    A = np.c_[np.log(x[m]), np.ones(m.sum())]
    c, *_ = np.linalg.lstsq(A, np.log(y[m]), rcond=None)
    pred = np.exp(A @ c)
    ss = ((np.log(y[m]) - np.log(pred)) ** 2).sum()
    tot = ((np.log(y[m]) - np.log(y[m]).mean()) ** 2).sum()
    return dict(a=float(np.exp(c[1])), b=float(c[0]), R2=float(1 - ss / tot) if tot > 0 else float("nan"))


def lin_fit(x, y, logx=False):
    """Least squares fit of y on x (or on log x); returns slope, intercept and R^2."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    f = np.log(x) if logx else x
    A = np.c_[f, np.ones_like(f)]
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss = ((y - A @ c) ** 2).sum(); tot = ((y - y.mean()) ** 2).sum()
    return dict(slope=float(c[0]), intercept=float(c[1]),
                R2=float(1 - ss / tot) if tot > 0 else float("nan"))


def compare_models(x, y):
    """Which growth model explains gamma(k) better: linear in k, logarithmic in k, or power law?"""
    return dict(linear=lin_fit(x, y), log=lin_fit(x, y, logx=True), power=power_fit(x, y))
