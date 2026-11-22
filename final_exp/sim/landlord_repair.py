"""Landlord-based feasibility repair (Algorithm 3 of
``Randomized_DRL_Pseudocode__updated_landlord_algo.pdf``).

The randomized rounding of R2 draws binary sample paths with marginals ``p`` but the
realization can violate the multi-resource polytope ``A x <= b``.  This module implements
the *stateful* repair of the updated algorithm:

  * every sample path ``k`` owns a persistent credit vector ``cr[k, n]``;
  * a service that is rounded into the path and has no credit receives a credit equal to
    its download cost ``M_n``; a service that is absent has its credit cleared (Step 0);
  * while the path is infeasible, every cached service pays rent proportional to its
    bottleneck size ``w_n = max_r a_{n,r}``; the rent step is the largest one that keeps
    every credit non-negative, so at least one service goes bankrupt;
  * only bankrupt (zero-credit) services may be evicted.  Among them the one whose
    eviction leaves the smallest residual overflow ``D_f`` is evicted (ties: smaller
    ``p_f / w_f``), and ``D_f`` is recomputed after every eviction.

Unlike the stateless rule ``evict in increasing order of p_n / max_r a_{n,r}`` this
repair is *download-cost aware*: an expensive service survives longer because its credit
is larger.  The function returns the repaired binary state and updates ``credit`` in
place.

The algorithm terminates after at most ``N`` outer passes (each pass evicts at least one
service and the empty cache is feasible), so the repaired state is feasible almost
surely (goal G1).
"""
from __future__ import annotations

import numpy as np


def _as_arrays(A, b):
    A = np.asarray(A, float)
    b = np.asarray(b, float).reshape(-1)
    if A.ndim != 2 or A.shape[0] != b.size:
        raise ValueError("A must be (rows, N) and b must have one entry per row")
    return A, b


def sync_credit(x, M, credit, tol=1e-9):
    """Step 0 of Algorithm 3: synchronize the persistent credit vector.

    ``x`` is modified in place only if the caller wants it (it is not modified here);
    ``credit`` is updated in place.  A cached service without credit receives ``M_n``;
    an absent service has its credit cleared.
    """
    x = np.asarray(x, bool)
    M = np.asarray(M, float)
    credit = np.asarray(credit, float)
    cached = x
    absent = ~x
    credit[cached & (credit <= tol)] = M[cached & (credit <= tol)]
    credit[absent] = 0.0
    return credit


def is_feasible(x, A, b, tol=1e-9):
    """``A x <= b + tol`` for every constraint row (including the cardinality row)."""
    return bool(np.all(np.asarray(A, float) @ np.asarray(x, float) <= np.asarray(b, float) + tol))


def landlord_repair(x, p, A, b, M, credit, tol=1e-9, max_passes=None,
                    stats=None, evict_mode="deficit", pop_weight=0.0):
    """Repair a binary cache realization with the stateful Landlord rule.

    Parameters
    ----------
    x : (N,) bool/uint8 array
        Sampled binary state.  Modified in place and returned.
    p : (N,) float array
        Current marginals (only used for the ``p_f / w_f`` tie-break).
    A : (4, N) array
        Constraint matrix.  Row 0 is the cardinality row (all ones), rows 1..3 are the
        raw CPU / RAM / disk demands.  The caller passes the *unnormalized* matrix and
        ``b = [C, 1, 1, 1]``; the function uses ``w_n = max_r a_{n,r}`` and the
        deficit-aligned rule on the raw scale.
    b : (4,) array
        Right-hand side ``[C, 1, 1, 1]``.
    M : (N,) array
        Download times, i.e. the initial credit of a service.
    credit : (N,) array
        Persistent credit of this sample path.  Updated in place.
    tol : float
        Numerical tolerance for the zero-credit test and the feasibility check.
    max_passes : int | None
        Safety bound on the number of outer rent/evict passes.  The algorithm needs at
        most ``N``; the default is ``N + 1``.
    stats : dict | None
        Optional dictionary updated with ``rent_rounds``, ``evictions`` and ``passes``.
    evict_mode : str
        ``"deficit"`` (default): Algorithm 3's smallest-residual-overflow rule.
        ``"popularity"``: among bankrupt services, evict the least popular ``p_f`` first.
        ``"hybrid"``: minimize ``D_f - pop_weight * p_f`` (protect popular services).
    pop_weight : float
        Weight of the popularity term in the ``"hybrid"`` mode.

    Returns
    -------
    x : (N,) bool array
        The repaired, feasible state (the same object that was passed in).
    """
    A, b = _as_arrays(A, b)
    if not isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=bool)
    elif x.dtype != bool:
        # Preserve the caller's object so the repair is visible in place.
        x[:] = x.astype(bool)
    p = np.asarray(p, float)
    M = np.asarray(M, float)
    credit = np.asarray(credit, float)
    N = x.size
    if A.shape[1] != N or p.size != N or M.size != N or credit.size != N:
        raise ValueError("inconsistent dimensions in landlord_repair")

    sync_credit(x, M, credit, tol=tol)
    if is_feasible(x, A, b, tol=tol):
        return x

    if stats is not None:
        stats["rent_rounds"] = stats.get("rent_rounds", 0)
        stats["evictions"] = stats.get("evictions", 0)
        stats["passes"] = stats.get("passes", 0)

    # Bottleneck size used for rent: max over the three *resource* rows, not the
    # cardinality row.  Rows 1..3 are the normalized CPU/RAM/disk demands.
    w = A[1:].max(axis=0) if A.shape[0] > 1 else np.ones(N)
    w = np.maximum(w, 1e-12)
    max_passes = N + 1 if max_passes is None else int(max_passes)

    for _ in range(max_passes):
        if is_feasible(x, A, b, tol=tol):
            break
        active = np.flatnonzero(x)
        if active.size == 0:                       # the empty cache is always feasible
            break
        if stats is not None:
            stats["passes"] += 1

        # ---- rent collection -------------------------------------------------
        ratio = credit[active] / w[active]
        d_rent = float(ratio.min())
        if d_rent < 0 and d_rent > -tol:
            d_rent = 0.0
        bankrupt = active[ratio - d_rent <= tol]
        credit[active] = np.maximum(0.0, credit[active] - d_rent * w[active])
        credit[bankrupt] = 0.0
        if stats is not None:
            stats["rent_rounds"] += 1

        # ---- deficit-aligned eviction among bankrupt services ----------------
        E = bankrupt
        while E.size and not is_feasible(x, A, b, tol=tol):
            def_r = np.maximum(0.0, A @ x.astype(float) - b)     # overflow of every row
            # D_f = sum_{r=1..3} max(0, def_r - a_{f,r}) + max(0, def_0 - 1)
            # The cardinality row is handled separately: evicting one service always
            # reduces the count by one, hence the `def_0 - 1` term.
            D = np.maximum(0.0, def_r[1:, None] - A[1:, E]).sum(axis=0)
            D = D + max(0.0, float(def_r[0]) - 1.0)
            if evict_mode == "popularity":
                # Protect popular services: evict the least popular bankrupt service first.
                f_star = int(E[np.argmin(p[E])])
            elif evict_mode == "hybrid":
                # Deficit reduction minus a popularity reward.
                score = D - float(pop_weight) * p[E]
                f_star = int(E[np.argmin(score)])
            else:
                prio = p[E] / w[E]
                # np.lexsort uses the LAST key as the primary key: D primary, prio secondary.
                f_star = int(E[np.lexsort((prio, D))[0]])
            x[f_star] = False
            credit[f_star] = 0.0
            E = E[E != f_star]
            if stats is not None:
                stats["evictions"] += 1

    return x


def landlord_repair_batch(S, P, A, b, M, credits, tol=1e-9, stats=None):
    """Repair every row of a ``(K, N)`` sample matrix.

    ``credits`` is a ``(K, N)`` array updated in place.  Returns the repaired matrix.
    """
    S = np.asarray(S)
    P = np.asarray(P)
    credits = np.asarray(credits)
    K, N = S.shape
    out = np.array(S, dtype=bool, copy=True)
    for k in range(K):
        landlord_repair(out[k], P[k] if P.ndim == 2 else P, A, b, M, credits[k],
                        tol=tol, stats=stats)
    return out