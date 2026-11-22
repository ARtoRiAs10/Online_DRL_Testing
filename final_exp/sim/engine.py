"""Event-driven edge node shared by every algorithm.

The engine owns everything the paper treats as 'environment': request classification (hit /
delayed hit / forward), download bookkeeping, the miss epoch (T_i, L_i of Algorithm 1) and the
accounting of latency, cost and resource utilization.  A *Policy* only decides what is cached;
a *threshold sampler* decides when a download is triggered (deterministic = Online-DRL,
random = R3).
"""
from __future__ import annotations
import heapq, math, time
from dataclasses import dataclass, field
import numpy as np


class Policy:
    """Cache-management hooks.  Subclasses: LandlordPolicy, FracPolicy."""
    name = "base"

    def attach(self, eng): self.eng = eng
    def set_horizon(self, T): pass
    def before_request(self, t, n, q, dm): pass
    def after_request(self, t, n, q, dm): pass
    def cached(self, n) -> bool: raise NotImplementedError
    def cached_list(self): raise NotImplementedError
    def on_hit(self, n, saved=0.0):
        """A hit: the request was served from the cache, so `saved` (the latency that was
        avoided) is l_n.  Used by the learned download probability of the revised policy."""
    def on_delayed(self, n, saved=0.0):
        """A delayed hit: the request was buffered; `saved` = l_n - remaining download time."""
    def on_complete(self, n, t) -> bool: raise NotImplementedError
    def allow_download(self, n, t) -> bool: return True
    def download_prob(self, n, t, budget_left=None) -> float:
        """Probability of *starting* a download once the (random) threshold is met.

        1.0 = deterministic trigger (Online-DRL).  Randomized DRL lowers it to respect the
        download budget in expectation (goal G2 of the multi-resource proposal)."""
        return 1.0
    def diagnostics(self) -> dict: return {}


class DetThreshold:
    """Online-DRL: download when T_i >= theta*M_i or L_i >= theta*M_i (theta=1 in the paper)."""
    def __init__(self, theta=1.0): self.theta = theta
    def draw(self, n): return self.theta


class RandThreshold:
    """R3: threshold theta_n ~ f(theta) = e^{theta/M_n} / (M_n (e-1)) on [0, M_n].

    Inverse CDF:  F(theta) = (e^{theta/M_n} - 1)/(e-1)  =>  theta = M_n * ln(1 + (e-1) U).
    `draw` returns the *fraction* of M_n (the engine multiplies by M_n).
    """
    def __init__(self, rng, theta_max=1.0): self.rng, self.theta_max = rng, theta_max
    def draw(self, n): return self.theta_max * math.log1p((math.e - 1.0) * self.rng.random())


@dataclass
class Result:
    latency: float
    cost: float
    n_req: int
    hits: int
    delayed_hits: int
    forwards: int
    downloads: int
    lat_per_req: np.ndarray
    util: np.ndarray
    extra: dict = field(default_factory=dict)
    time_s: float = 0.0
    fwd_inflight: int = 0     # requests forwarded *while a download was in flight*
    inflight_req: int = 0     # requests served while a download was in flight (delayed hit or forward)
    fwd_srv: np.ndarray = field(default_factory=lambda: np.zeros(0))
    buf_srv: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def summary(self):
        return dict(latency=self.latency, cost=self.cost, hits=self.hits, delayed_hits=self.delayed_hits,
                    forwards=self.forwards, downloads=self.downloads, hit_ratio=self.hits / self.n_req,
                    us_per_req=1e6 * self.time_s / max(1, self.n_req))


class Engine:
    def __init__(self, cat, policy, thr, bmax=None, rng=None):
        self.cat, self.policy, self.thr, self.bmax = cat, policy, thr, bmax
        self.rng = np.random.default_rng(0) if rng is None else rng

    def reset_miss(self, n):
        self.miss_ts[n] = np.nan
        self.fcount[n] = 0

    def run(self, trace) -> Result:
        cat, pol = self.cat, self.policy
        N, T = cat.N, len(trace)
        self.pending, heap = {}, []
        self.miss_ts = np.full(N, np.nan)
        self.fcount = np.zeros(N, int)
        self.fwd_srv = np.zeros(N, int); self.buf_srv = np.zeros(N, int)
        self.thr_abs = np.zeros(N)
        pol.attach(self); pol.set_horizon(T)
        cost = 0.0
        hits = dh = fw = dl = 0
        fwd_inflight = inflight_req = 0
        lat = np.zeros(T); util = np.zeros((T, 3))
        wall = time.perf_counter()
        for i in range(T):
            t, n = float(trace.t[i]), int(trace.n[i])
            while heap and heap[0][0] <= t:                      # downloads finished before t
                tc, m = heapq.heappop(heap)
                self.pending.pop(m, None)
                pol.on_complete(m, tc)
                self.reset_miss(m)
            if n in self.pending:
                q, dm = 1.0, self.pending[n] - t
            else:
                q, dm = 0.0, 0.0
            pol.before_request(t, n, q, dm)
            cl = pol.cached_list()
            if len(cl):
                util[i] = cat.a[cl].sum(0)
            l = cat.l[n]
            if pol.cached(n):
                hits += 1; pol.on_hit(n, saved=l)
            elif n in self.pending:
                rem = self.pending[n] - t
                lat[i] = min(l, rem)
                inflight_req += 1
                if rem <= l:
                    dh += 1; self.buf_srv[n] += 1; pol.on_delayed(n, saved=l - rem)
                else: fw += 1; fwd_inflight += 1; self.fwd_srv[n] += 1
            else:
                if np.isnan(self.miss_ts[n]):
                    self.miss_ts[n] = t; self.fcount[n] = 0
                    self.thr_abs[n] = self.thr.draw(n) * cat.M[n]
                self.fcount[n] += 1
                Ti, Li = t - self.miss_ts[n], l * self.fcount[n]
                bleft = None if self.bmax is None else self.bmax - cost
                p_dl = float(np.clip(pol.download_prob(n, t, bleft), 0.0, 1.0))
                if (Ti >= self.thr_abs[n] or Li >= self.thr_abs[n]) and p_dl > 0 and pol.allow_download(n, t) \
                        and (self.bmax is None or cost + cat.M[n] <= self.bmax) \
                        and (p_dl >= 1.0 or self.rng.random() < p_dl):
                    self.pending[n] = t + cat.M[n]
                    heapq.heappush(heap, (t + cat.M[n], n))
                    cost += cat.M[n]; dl += 1
                lat[i] = l; fw += 1
            pol.after_request(t, n, q, dm)
        wall = time.perf_counter() - wall
        return Result(lat.sum(), cost, T, hits, dh, fw, dl, lat, util, pol.diagnostics(),
                      time_s=wall, fwd_inflight=fwd_inflight, inflight_req=inflight_req,
                      fwd_srv=self.fwd_srv.copy(), buf_srv=self.buf_srv.copy())
