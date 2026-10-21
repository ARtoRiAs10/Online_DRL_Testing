"""Offline optimum (OPT) as a MILP solved with HiGHS (scipy.optimize.milp) -- no Gurobi needed.

Variables (only for services present in the trace):
  x[n,t] in {0,1}  service n is cached when request t arrives
  u[s]   in {0,1}  a download of n_s is initiated at request s (charged M once)
A download started at request s completes at the first request index c(s) with T_c >= T_s + M.
  * cache dynamics:  x[n,t] <= x[n,t-1] + sum_{s: c(s)=t, n_s=n} u[s]      (evictions are free)
  * in flight at request t (same service): s < t < c(s).  x and in-flight are exclusive.
  * latency of request t: l if neither cached nor in flight; min(l, T_s+M-T_t) if download s in flight.
  * per-step capacity (3 resources + count), budget  sum u M <= Bmax, one download in flight per service.
This is the formulation of Sec. 1.2 of the proposal (the original paper's (1-x) in (4)-(6) is read as x).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix


def solve_opt(cat, trace, C, bmax=None, time_limit=60.0, mip_gap=1e-4):
    T = len(trace); t = trace.t; sv = trace.n
    present = np.unique(sv); idx = {int(n): i for i, n in enumerate(present)}
    P = len(present); nx = P * T; nv = nx + T
    X = lambda n, tt: idx[int(n)] * T + tt
    comp = np.array([np.searchsorted(t, t[s] + cat.M[sv[s]], side="left") for s in range(T)])
    by = {int(n): np.flatnonzero(sv == n) for n in present}
    inflight = [[] for _ in range(T)]
    for n, reqs in by.items():
        for s in reqs:
            for tt in reqs:
                if s < tt < comp[s]: inflight[tt].append(s)
    c = np.zeros(nv); const = 0.0
    for tt in range(T):
        n = sv[tt]; l = cat.l[n]
        const += l; c[X(n, tt)] -= l
        for s in inflight[tt]:
            c[nx + s] += -l + min(l, t[s] + cat.M[n] - t[tt])
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    def add(entries, l_, h_):
        nonlocal r
        for j, v in entries: rows.append(r); cols.append(j); vals.append(v)
        lo.append(l_); hi.append(h_); r += 1
    done_at = {}
    for s in range(T):
        if comp[s] < T: done_at.setdefault((int(sv[s]), int(comp[s])), []).append(s)
    for n in present:
        for tt in range(T):                       # cache dynamics
            e = [(X(n, tt), 1.0)]
            if tt > 0: e.append((X(n, tt - 1), -1.0))
            e += [(nx + s, -1.0) for s in done_at.get((int(n), tt), [])]
            add(e, -np.inf, 0.0)
    for tt in range(T):                           # x and in-flight exclusive; u only if not cached
        add([(X(sv[tt], tt), 1.0)] + [(nx + s, 1.0) for s in inflight[tt]], -np.inf, 1.0)
        add([(nx + tt, 1.0), (X(sv[tt], tt), 1.0)], -np.inf, 1.0)
    for n, reqs in by.items():                    # one download in flight per service
        for s in reqs:
            act = [(nx + s2, 1.0) for s2 in reqs if s2 <= s < comp[s2]]
            if len(act) > 1: add(act, -np.inf, 1.0)
    for tt in range(T):                           # capacity
        for rr in range(3):
            add([(X(n, tt), cat.a[n, rr]) for n in present], -np.inf, 1.0)
        add([(X(n, tt), 1.0) for n in present], -np.inf, float(C))
    if bmax is not None:
        add([(nx + s, cat.M[sv[s]]) for s in range(T)], -np.inf, float(bmax))
    A = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(nv),
               bounds=Bounds(np.zeros(nv), np.ones(nv)),
               options=dict(time_limit=time_limit, mip_rel_gap=mip_gap, disp=False))
    if res.x is None:
        return dict(status=res.status, latency=np.nan, cost=np.nan, downloads=0, gap=np.nan, message=res.message)
    u = np.round(res.x[nx:])
    return dict(status=res.status, latency=const + res.fun, cost=float((u * cat.M[sv]).sum()),
                downloads=int(u.sum()), gap=getattr(res, "mip_gap", np.nan), message=res.message)
