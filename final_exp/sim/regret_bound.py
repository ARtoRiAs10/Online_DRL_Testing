"""Download-charged regret bound (G3b) utilities.

The theorem proved in ``notes/phase2_theorem.md`` is

    E[latency + downloads]
        <= min_P [ sum_t F~_t(P) + sum_n M_n P_n ]
           + 2 sqrt(N) l_max sqrt(T)
           + zeta T
           + l_max sum_t delta_t
           + M_max N sqrt(T).

This module turns a per-step log produced by ``FracPolicy(log_steps=True)`` into the
numerical terms of that bound and solves the comparator LP with SciPy.

Sign convention
---------------
The step log stores ``grad`` = g_t, the (positive) cost coefficient used by the OGD
gradient for the requested service, and ``benefit`` = l_n - g_t.  The plan writes the
comparator objective as ``sum_n p_n (M_n - W_n)`` with ``W_n = sum c~_t``; here
``c~_t = benefit_t = l_n - g_t`` is the latency avoided by caching.  Both conventions
are exposed below so the paper can use either one.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def catalog_constraints(cat, C):
    """Return the normalized constraint matrix ``An`` with ``An p <= 1``.

    Row 0 is the cardinality row ``sum_n p_n <= C``; rows 1..3 are the normalized
    CPU/RAM/disk rows.  This matches ``FracPolicy.An``.
    """
    N = cat.N
    A = np.vstack([np.ones(N), cat.a.T])
    b = np.r_[float(C), np.ones(cat.a.shape[1])]
    return A / b[:, None]


def g3b_comparator_lp(cat, C, W, method="highs"):
    """Solve the G3b comparator LP.

    Minimize ``sum_n p_n (M_n - W_n)`` subject to ``p in D_vec`` and ``0 <= p <= 1``.
    ``W`` is the accumulated benefit ``sum_t c~_t`` per service (positive is good).
    Returns ``(p_star, objective)``.
    """
    An = catalog_constraints(cat, C)
    c = np.asarray(cat.M, float) - np.asarray(W, float)
    res = linprog(c, A_ub=An, b_ub=np.ones(An.shape[0]),
                  bounds=(0.0, 1.0), method=method)
    if not res.success:
        raise RuntimeError(f"G3b comparator LP failed: {res.message}")
    return np.asarray(res.x, float), float(res.fun)


def _as_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def regret_bound_terms(step_log, cat, C, zeta=None, T=None):
    """Extract the RHS terms of the G3b bound from a per-step log.

    Parameters
    ----------
    step_log : list[dict]
        Records written by ``FracPolicy`` with ``log_steps=True``.  Each record must
        contain ``n``, ``grad`` and ``p_n_star_prev``; ``q_ema_before``, ``q`` and
        ``p_minus_xbar`` are used when present.
    cat : Catalog
    C : int
        Cache capacity used for the comparator LP.
    zeta : float | None
        EMA-bias constant.  If None, estimated as ``mean |q_ema_before - q|``.
    T : int | None
        Horizon.  Defaults to ``len(step_log)``.
    """
    step_log = list(step_log)
    N = cat.N
    T = len(step_log) if T is None else int(T)
    G = np.zeros(N, float)
    B = np.zeros(N, float)
    sum_F_tilde_Pt = 0.0
    sum_abs_px = 0.0
    q_bias = []
    dl_bias = []
    for r in step_log:
        n = int(r["n"])
        g = _as_float(r.get("grad"))
        G[n] += g
        # The benefit of caching is l_n - g.  Older logs without a benefit field
        # fall back to l_n - grad (l_n is usually logged as well).
        benefit = _as_float(r.get("benefit"),
                            _as_float(r.get("l_n"), 0.0) - g)
        B[n] += benefit
        sum_F_tilde_Pt += g * _as_float(r.get("p_n_star_prev"))
        sum_abs_px += abs(_as_float(r.get("p_minus_xbar")))
        if "q_ema_before" in r and "q" in r:
            q_bias.append(abs(_as_float(r["q_ema_before"]) - _as_float(r["q"])))
        if _as_float(r.get("q")) > 0 and "dl_ema_before" in r:
            dl_bias.append(abs(_as_float(r["dl_ema_before"]) - _as_float(r.get("dm"))))

    # W is the accumulated benefit c~ = l_n - g.
    W = B
    P, comp_obj = g3b_comparator_lp(cat, C, W)
    zeta_q_hat = float(np.mean(q_bias)) if q_bias else 0.0
    zeta_dl_hat = float(np.mean(dl_bias)) if dl_bias else 0.0
    zeta_hat = zeta_q_hat + zeta_dl_hat
    if zeta is None:
        zeta = zeta_hat

    lmax = float(cat.l.max())
    Mmax = float(cat.M.max())
    sum_Mp_comp = float(np.dot(cat.M, P))
    sum_F_tilde_comp = float(np.dot(G, P))
    base = 2.0 * np.sqrt(N) * lmax * np.sqrt(T)
    churn = lmax * sum_abs_px
    ema_term = float(zeta) * T
    budget_term = Mmax * N * np.sqrt(T)
    rhs = float(comp_obj + base + ema_term + churn + budget_term)

    return {
        "N": int(N),
        "T": int(T),
        "C": float(C),
        "l_max": lmax,
        "M_max": Mmax,
        "sum_F_tilde_Pt": float(sum_F_tilde_Pt),
        "sum_F_tilde_comparator": sum_F_tilde_comp,
        "sum_Mp_comparator": sum_Mp_comp,
        "comparator_objective": float(comp_obj),
        "comparator_P": P.tolist(),
        "G": G.tolist(),
        "W": W.tolist(),
        "B": B.tolist(),
        "zeta": float(zeta),
        "zeta_hat": float(zeta_hat),
        "zeta_q_hat": float(zeta_q_hat),
        "zeta_dl_hat": float(zeta_dl_hat),
        "sum_delta_abs_px": float(sum_abs_px),
        "base_term": float(base),
        "ema_term": float(ema_term),
        "churn_term": float(churn),
        "budget_term": float(budget_term),
        "regret_bound_rhs": rhs,
        "bound_without_zeta_delta": float(comp_obj + base + budget_term),
    }


def ema_tracking_error(alpha, sigma, drift):
    """Tracking error of an EMA under drift and noise.

    ``error(alpha) ~= drift / alpha + sqrt(alpha) * sigma``.
    """
    alpha = float(alpha)
    if alpha <= 0:
        return float("inf")
    return float(drift) / alpha + np.sqrt(alpha) * float(sigma)


def optimal_ema_alpha(sigma, drift):
    """Alpha minimizing ``drift/alpha + sqrt(alpha)*sigma``.

    For ``sigma > 0`` and ``drift > 0`` the minimizer is
    ``alpha* = (2*drift/sigma)^(2/3)``.
    """
    sigma = float(sigma)
    drift = float(drift)
    if sigma <= 0 or drift <= 0:
        return 1.0
    return float((2.0 * drift / sigma) ** (2.0 / 3.0))


def summarize_step_log(step_log):
    """Small numerical summary of a per-step log (for the phase-1 diagnostic)."""
    if not step_log:
        return {}
    p_sum = np.array([_as_float(r.get("p_sum"), np.nan) for r in step_log], float)
    p_after = np.array([_as_float(r.get("p_sum_after"), np.nan) for r in step_log], float)
    grad = np.array([_as_float(r.get("grad"), np.nan) for r in step_log], float)
    qe = np.array([_as_float(r.get("q_ema_before"), np.nan) for r in step_log], float)
    ct = np.array([_as_float(r.get("c_t"), np.nan) for r in step_log], float)
    px = np.array([_as_float(r.get("p_minus_xbar"), np.nan) for r in step_log], float)
    frac = np.array([_as_float(r.get("repair_evicted_fraction"), np.nan)
                     for r in step_log], float)
    outcomes = [r.get("outcome") for r in step_log]
    hits = sum(1 for o in outcomes if o == "hit")
    delayed = sum(1 for o in outcomes if o == "delayed")
    fwd = sum(1 for o in outcomes if o in ("forward", "forward_inflight"))
    ranks = [r.get("hit_rank") for r in step_log if r.get("outcome") == "hit"
             and r.get("hit_rank") is not None]

    def _mean(x):
        x = x[np.isfinite(x)]
        return float(x.mean()) if x.size else None

    return {
        "n_steps": len(step_log),
        "mean_p_sum": _mean(p_sum),
        "mean_p_sum_after": _mean(p_after),
        "mean_grad": _mean(grad),
        "mean_q_ema_before": _mean(qe),
        "mean_c_t": _mean(ct),
        "mean_abs_p_minus_xbar": _mean(np.abs(px)),
        "mean_repair_evicted_fraction": _mean(frac),
        "hit_rank_mean": float(np.mean(ranks)) if ranks else None,
        "hit_rank_min": int(np.min(ranks)) if ranks else None,
        "hit_rank_max": int(np.max(ranks)) if ranks else None,
        "outcomes": {"hit": hits, "delayed": delayed, "forward": fwd},
    }