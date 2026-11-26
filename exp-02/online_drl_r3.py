"""R3 + Marking-Gated Landlord on Online-DRL.

Self-contained simulator for the experiments in
``Experimentation Plan R3 and Marking-Gated Landlord on Online-DRL.md``.

The module implements

* the deterministic Online-DRL of Deka & Sukapuram (Algorithms 1 and 2),
  with the residual-deficit Landlord tie-break;
* LL-RC, the ``theta = 0`` / ``lam = 0`` extreme;
* R3, the randomized ski-rental download threshold
  ``theta_n = lam * M_n * ln(1 + u (e - 1))``, redrawn after a download or
  an eviction;
* MG, the marking-gated Landlord: one bit per cached service, rent is paid
  only by unmarked services, hits and admissions set the mark, and when all
  cached services are marked and the cache is still infeasible the marks are
  cleared (a new phase).  Zero-credit victims are drawn uniformly.

All algorithms share one event-driven engine, so the only differences are the
threshold sampler and the eviction policy.  With randomization off (``lam=0``
or ``theta`` fixed) and MG off the engine reduces to deterministic Online-DRL.

The module deliberately has no dependency on the older ``final_exp`` package.
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

E = math.e
TOL = 1e-9


# ---------------------------------------------------------------------------
# Catalog and trace
# ---------------------------------------------------------------------------
@dataclass
class Catalog:
    """Service catalogue.

    ``l``  forwarding latency (ms)
    ``M``  download time (ms)
    ``a``  normalized demands (N, 3): CPU, RAM, disk, each in (0, 1]
    """

    l: np.ndarray
    M: np.ndarray
    a: np.ndarray
    raw: Optional[np.ndarray] = None
    caps: Optional[np.ndarray] = None

    @property
    def N(self) -> int:
        return len(self.l)


@dataclass
class Trace:
    t: np.ndarray   # strictly increasing arrival times
    n: np.ndarray   # service index per request

    def __len__(self) -> int:
        return len(self.t)

    def head(self, T: int) -> "Trace":
        return Trace(self.t[:T], self.n[:T])


def from_physical(raw, caps=None, cap_mult=8.0, U=100.0, D=1000.0,
                  rtt=20.0, req_kbit=100.0) -> Catalog:
    """Build a catalogue from raw ``(N, 3)`` demands ``[cpu, ram, disk_MB]``.

    ``l_n = rtt + req/U`` and ``M_n = rtt/2 + req/U + S_n/D`` (eq. (26) of the
    paper).  Capacities default to ``max(cap_mult * mean, max)`` per resource.
    """
    raw = np.asarray(raw, float)
    N = raw.shape[0]
    if caps is None:
        caps = np.maximum(cap_mult * raw.mean(0), raw.max(0))
    caps = np.asarray(caps, float)
    a = raw / caps
    req = np.broadcast_to(np.asarray(req_kbit, float), (N,))
    l = rtt + req / U
    M = rtt / 2 + req / U + raw[:, 2] * 8000.0 / D
    return Catalog(l=l, M=M, a=a, raw=raw, caps=caps)


# ---------------------------------------------------------------------------
# Threshold samplers
# ---------------------------------------------------------------------------
class DetThreshold:
    """Deterministic Online-DRL threshold: ``theta * M_n``."""

    is_random = False

    def __init__(self, theta: float = 1.0):
        self.theta = float(theta)

    def draw(self, n: int) -> float:
        return self.theta


class RandThreshold:
    """R3 threshold as a fraction of ``M_n``.

    ``theta_n / M_n = lam * ln(1 + u (e - 1))``, ``u ~ U[0, 1)``.
    ``lam = 0`` gives the LL-RC threshold 0.
    """

    is_random = True

    def __init__(self, rng: np.random.Generator, lam: float = 1.0):
        self.rng = rng
        self.lam = float(lam)

    def draw(self, n: int) -> float:
        if self.lam == 0.0:
            return 0.0
        return self.lam * math.log1p((E - 1.0) * self.rng.random())

    def observe(self, n: int, t: float, hit: bool):
        pass


class BayesThreshold:
    """Posterior-rate (Bayes-R3) threshold.

    A Gamma(alpha, beta) posterior is kept for the request rate of every
    service.  The threshold fraction is

        theta_n / M_n = lam / (1 + c * E[lambda_n] * M_n)

    so a service with a high estimated request rate downloads earlier and a
    cold service waits longer.  ``lam`` is the global scale used for the
    matched-cost sweep; ``lam = 0`` is LL-RC.
    """

    is_random = True

    def __init__(self, cat: Catalog, rng: np.random.Generator, lam: float = 1.0,
                 c: float = 1.0, prior_shape: float = 1.0,
                 prior_rate: float = 1.0):
        self.cat = cat
        self.rng = rng
        self.lam = float(lam)
        self.c = float(c)
        self.alpha = np.full(cat.N, float(prior_shape))
        self.beta = np.full(cat.N, float(prior_rate))
        self.last_t = np.full(cat.N, np.nan)

    def observe(self, n: int, t: float, hit: bool):
        if np.isnan(self.last_t[n]):
            self.last_t[n] = t
            return
        dt = max(0.0, t - self.last_t[n])
        self.last_t[n] = t
        # Every request carries one unit of rate evidence; the elapsed time
        # is the exposure.
        self.alpha[n] += 1.0
        self.beta[n] += dt

    def draw(self, n: int) -> float:
        if self.lam == 0.0:
            return 0.0
        rate = self.alpha[n] / max(self.beta[n], 1e-12)
        return self.lam / (1.0 + self.c * rate * self.cat.M[n])


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
class Policy:
    name = "base"

    def attach(self, eng: "Engine"):
        self.eng = eng

    def set_horizon(self, T: int):
        pass

    def before_request(self, t: float, n: int, q: float, dm: float):
        pass

    def after_request(self, t: float, n: int, q: float, dm: float):
        pass

    def cached(self, n: int) -> bool:
        raise NotImplementedError

    def cached_list(self):
        raise NotImplementedError

    def on_hit(self, n: int, saved: float = 0.0):
        pass

    def on_delayed(self, n: int, saved: float = 0.0):
        pass

    def on_complete(self, n: int, t: float) -> bool:
        raise NotImplementedError

    def allow_download(self, n: int, t: float) -> bool:
        return True

    def diagnostics(self) -> dict:
        return {}


class LandlordPolicy(Policy):
    """Algorithm 2 of the paper (resource-aware Landlord).

    ``marked=True`` switches to the marking-gated variant (MG).
    """

    name = "landlord"

    def __init__(self, cat: Catalog, C: int, size_mode: str = "sum",
                 tie_break: str = "deficit", victim_mode: Optional[str] = None,
                 marked: bool = False, rng: Optional[np.random.Generator] = None,
                 tol: float = TOL, soft_tau: float = 0.0):
        self.cat = cat
        self.C = int(C)
        self.size_mode = size_mode
        self.tie_break = tie_break
        # MG draws uniformly among zero-credit victims; the deterministic
        # Landlord uses the residual-deficit tie-break.
        self.victim_mode = victim_mode or ("uniform" if marked else "deficit")
        self.marked_enabled = bool(marked)
        # Soft-MG: marked services pay (1 - soft_tau) of the rent and may be
        # evicted.  soft_tau = 0 is the hard marking-gated rule.
        self.soft_tau = float(soft_tau)
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.tol = float(tol)
        self.size = cat.a.sum(1) if size_mode == "sum" else cat.a.max(1)
        self.credit: dict[int, float] = {}
        self.marked: set[int] = set()
        self.last_use: dict[int, int] = {}
        self.n_evict = 0
        self.evictions: list[tuple[int, int]] = []
        self.phase_clears = 0
        self.marking_violations = 0
        self.step = 0
        self.max_marked = 0

    # -- hooks ------------------------------------------------------------
    def before_request(self, t, n, q, dm):
        self.step += 1
        if n in self.credit:
            self.last_use[n] = self.step

    def cached(self, n) -> bool:
        return n in self.credit

    def cached_list(self):
        return list(self.credit.keys())

    def on_hit(self, n, saved=0.0):
        self.credit[n] = self.cat.M[n]
        self.last_use[n] = self.step
        if self.marked_enabled:
            self.marked.add(n)
            self.max_marked = max(self.max_marked, len(self.marked))

    def on_complete(self, g, t):
        self._admit(g)
        return True

    def diagnostics(self):
        return dict(evictions=self.n_evict, phase_clears=self.phase_clears,
                    marking_violations=self.marking_violations,
                    max_marked=self.max_marked,
                    soft_tau=self.soft_tau,
                    final_cached=len(self.credit),
                    final_marked=len(self.marked))

    # -- feasibility ------------------------------------------------------
    def _used(self) -> np.ndarray:
        if not self.credit:
            return np.zeros(3)
        return self.cat.a[list(self.credit)].sum(0)

    def _feasible_with(self, g: int) -> bool:
        if len(self.credit) + (0 if g in self.credit else 1) > self.C:
            return False
        used = self._used()
        if g not in self.credit:
            used = used + self.cat.a[g]
        return bool(np.all(used <= 1.0 + self.tol))

    def _decrease(self, eligible):
        if not eligible:
            return []
        d = min(self.credit[f] / self.size[f] for f in eligible)
        for f in eligible:
            self.credit[f] = max(0.0, self.credit[f] - d * self.size[f])
        return [f for f in eligible if self.credit[f] <= self.tol]

    def _need(self, g: int) -> np.ndarray:
        used = self._used()
        if g not in self.credit:
            used = used + self.cat.a[g]
        return np.maximum(used - 1.0, 0.0)

    def _residual_deficit(self, f: int, need: np.ndarray) -> float:
        return float(np.maximum(self.cat.a[f] - need, 0.0).sum())

    def _order(self, zero, need):
        if self.tie_break == "lru":
            return sorted(zero, key=lambda f: self.last_use.get(f, -1))
        return sorted(zero, key=lambda f: self._residual_deficit(f, need))

    def _choose_victim(self, zero, need=None):
        if self.victim_mode == "uniform":
            return int(zero[self.rng.integers(len(zero))])
        if self.victim_mode == "weighted":
            w = np.array([self.size[f] / (self.credit[f] + 1e-12) for f in zero], float)
            w = np.maximum(w, 1e-12)
            w /= w.sum()
            return int(zero[self.rng.choice(len(zero), p=w)])
        # deterministic residual-deficit choice
        if need is None:
            need = self._need(-1)
        return min(zero, key=lambda f: self._residual_deficit(f, need))

    def _evict(self, f: int):
        if self.marked_enabled and self.soft_tau == 0.0 and f in self.marked and \
                any(g not in self.marked for g in self.credit):
            self.marking_violations += 1
        self.credit.pop(f, None)
        self.marked.discard(f)
        self.last_use.pop(f, None)
        self.n_evict += 1
        self.evictions.append((self.step, int(f)))
        self.eng.on_evict(f)
        self.eng.check_invariants(f"evict {f}")

    def _admit_soft(self, g: int):
        """Soft-MG: rent everyone, but marked items pay (1 - tau) rent."""
        while not self._feasible_with(g):
            if not self.credit:
                break
            d = min(self.credit[f] /
                    (self.size[f] * (1.0 - self.soft_tau if f in self.marked else 1.0))
                    for f in self.credit)
            for f in self.credit:
                mult = (1.0 - self.soft_tau) if f in self.marked else 1.0
                self.credit[f] = max(0.0, self.credit[f] - d * self.size[f] * mult)
            zero = [f for f in self.credit if self.credit[f] <= self.tol]
            if not zero:
                f = min(self.credit, key=lambda x: self.credit[x] / self.size[x])
                self.credit[f] = 0.0
                zero = [f]
            unmarked_zero = [f for f in zero if f not in self.marked]
            pool = unmarked_zero if unmarked_zero else zero
            victim = self._choose_victim(pool, self._need(g))
            self._evict(victim)

    def _admit(self, g: int):
        if g in self.credit:
            self.credit[g] = self.cat.M[g]
            self.last_use[g] = self.step
            if self.marked_enabled:
                self.marked.add(g)
            return

        if self.marked_enabled and self.soft_tau > 0.0:
            self._admit_soft(g)
            self.credit[g] = self.cat.M[g]
            self.last_use[g] = self.step
            self.marked.add(g)
            self.max_marked = max(self.max_marked, len(self.marked))
            self.eng.check_invariants(f"admit {g}")
            return

        if not self.marked_enabled:
            while not self._feasible_with(g):
                eligible = list(self.credit)
                if not eligible:
                    break
                need = self._need(g)
                zero = self._decrease(eligible)
                if not zero:
                    # Numerical safety: force the smallest ratio to zero.
                    f = min(eligible, key=lambda x: self.credit[x] / self.size[x])
                    self.credit[f] = 0.0
                    zero = [f]
                for f in self._order(zero, need):
                    if self._feasible_with(g):
                        break
                    self._evict(f)
        else:
            # Marking-gated Landlord.
            while not self._feasible_with(g):
                unmarked = [f for f in self.credit if f not in self.marked]
                if not unmarked:
                    # Every cached service is marked and the cache is still
                    # infeasible: start a new phase.
                    self.marked.clear()
                    self.phase_clears += 1
                    continue
                zero = self._decrease(unmarked)
                if not zero:
                    f = min(unmarked, key=lambda x: self.credit[x] / self.size[x])
                    self.credit[f] = 0.0
                    zero = [f]
                victim = self._choose_victim(zero, self._need(g))
                self._evict(victim)

        self.credit[g] = self.cat.M[g]
        self.last_use[g] = self.step
        if self.marked_enabled:
            self.marked.add(g)
            self.max_marked = max(self.max_marked, len(self.marked))
        self.eng.check_invariants(f"admit {g}")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
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

    def summary(self):
        return dict(latency=self.latency, cost=self.cost, hits=self.hits,
                    delayed_hits=self.delayed_hits, forwards=self.forwards,
                    downloads=self.downloads,
                    hit_ratio=self.hits / max(1, self.n_req),
                    us_per_req=1e6 * self.time_s / max(1, self.n_req),
                    **self.extra)


class Engine:
    def __init__(self, cat: Catalog, policy: Policy, thr, bmax=None,
                 rng: Optional[np.random.Generator] = None,
                 redraw_mode: str = "download_or_evict",
                 check_invariants: bool = True):
        self.cat = cat
        self.policy = policy
        self.thr = thr
        self.bmax = bmax
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.redraw_mode = redraw_mode
        self.check = check_invariants
        self.violations: dict[str, int] = {}
        self.n_redraws = 0

    # -- threshold handling ----------------------------------------------
    def _draw_threshold(self, n: int):
        self.thr_abs[n] = self.thr.draw(n) * self.cat.M[n]

    def redraw(self, n: int, reason: str):
        if not getattr(self.thr, "is_random", False):
            return
        do = (self.redraw_mode == "always"
              or (self.redraw_mode == "download_or_evict"
                  and reason in ("download", "evict"))
              or (self.redraw_mode == "miss_epoch" and reason == "miss"))
        if do:
            self._draw_threshold(n)
            self.n_redraws += 1

    def reset_miss(self, n: int):
        self.miss_ts[n] = np.nan
        self.fcount[n] = 0

    def on_evict(self, n: int):
        self.reset_miss(n)
        self.redraw(n, "evict")

    # -- invariants -------------------------------------------------------
    def check_invariants(self, where: str = ""):
        if not self.check:
            return
        pol = self.policy
        cached = pol.cached_list()
        if len(cached) > pol.C:
            self.violations["cardinality"] = self.violations.get("cardinality", 0) + 1
        if cached:
            used = self.cat.a[cached].sum(0)
            if np.any(used > 1.0 + 1e-8):
                self.violations["resource"] = self.violations.get("resource", 0) + 1
        if self.bmax is not None and self.cost > self.bmax + 1e-8:
            self.violations["budget"] = self.violations.get("budget", 0) + 1
        if getattr(pol, "marked_enabled", False):
            marked = pol.marked
            if marked and any(f not in marked for f in cached):
                # A marked item may coexist with unmarked items; that is fine.
                pass
            if getattr(pol, "marking_violations", 0):
                self.violations["marking"] = pol.marking_violations

    # -- main loop --------------------------------------------------------
    def run(self, trace: Trace) -> Result:
        cat, pol = self.cat, self.policy
        N, T = cat.N, len(trace)
        self.pending: dict[int, float] = {}
        heap: list[tuple[float, int]] = []
        self.miss_ts = np.full(N, np.nan)
        self.fcount = np.zeros(N, int)
        self.thr_abs = np.zeros(N)
        pol.attach(self)
        pol.set_horizon(T)

        cost = 0.0
        self.cost = 0.0
        hits = dh = fw = dl = 0
        lat = np.zeros(T)
        util = np.zeros((T, 3))
        wall = time.perf_counter()

        for i in range(T):
            t, n = float(trace.t[i]), int(trace.n[i])

            # downloads that finished before t
            while heap and heap[0][0] <= t:
                tc, m = heapq.heappop(heap)
                self.pending.pop(m, None)
                pol.on_complete(m, tc)
                self.reset_miss(m)
                self.redraw(m, "download")

            q = 1.0 if n in self.pending else 0.0
            dm = (self.pending[n] - t) if n in self.pending else 0.0
            pol.before_request(t, n, q, dm)

            if self.redraw_mode == "request":
                self.redraw(n, "request")

            cl = pol.cached_list()
            if cl:
                util[i] = cat.a[cl].sum(0)

            l = cat.l[n]
            if pol.cached(n):
                hits += 1
                pol.on_hit(n, saved=l)
            elif n in self.pending:
                rem = self.pending[n] - t
                lat[i] = min(l, rem)
                if rem <= l:
                    dh += 1
                    pol.on_delayed(n, saved=l - rem)
                else:
                    fw += 1
            else:
                if np.isnan(self.miss_ts[n]):
                    self.miss_ts[n] = t
                    self.fcount[n] = 0
                    self._draw_threshold(n)
                    self.redraw(n, "miss")
                self.fcount[n] += 1
                Ti = t - self.miss_ts[n]
                Li = l * self.fcount[n]
                bleft = None if self.bmax is None else self.bmax - cost
                if (Ti >= self.thr_abs[n] or Li >= self.thr_abs[n]) \
                        and pol.allow_download(n, t) \
                        and (self.bmax is None or cost + cat.M[n] <= self.bmax + 1e-9):
                    self.pending[n] = t + cat.M[n]
                    heapq.heappush(heap, (t + cat.M[n], n))
                    cost += cat.M[n]
                    self.cost = cost
                    dl += 1
                lat[i] = l
                fw += 1

            if hasattr(self.thr, "observe"):
                self.thr.observe(n, t, pol.cached(n))
            pol.after_request(t, n, q, dm)

        wall = time.perf_counter() - wall
        extra = pol.diagnostics()
        extra.update(violations=dict(self.violations), redraws=self.n_redraws,
                     check_invariants=self.check)
        return Result(float(lat.sum()), float(cost), T, hits, dh, fw, dl,
                      lat, util, extra, time_s=wall)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------
def online_drl(cat, trace, C, theta=1.0, bmax=None, **kw) -> Result:
    """Deterministic Online-DRL.  ``theta=0`` is LL-RC."""
    pol = LandlordPolicy(cat, C, **{k: v for k, v in kw.items()
                                    if k in ("size_mode", "tie_break",
                                             "victim_mode", "tol")})
    return Engine(cat, pol, DetThreshold(theta), bmax,
                  redraw_mode=kw.get("redraw_mode", "download_or_evict"),
                  check_invariants=kw.get("check_invariants", True)).run(trace)


def ll_rc(cat, trace, C, bmax=None, **kw) -> Result:
    return online_drl(cat, trace, C, theta=0.0, bmax=bmax, **kw)


def drl_r3(cat, trace, C, lam=1.0, seed=0, bmax=None, **kw) -> Result:
    """R3 only: randomized threshold + deterministic Landlord."""
    rng = np.random.default_rng(seed)
    pol = LandlordPolicy(cat, C, **{k: v for k, v in kw.items()
                                    if k in ("size_mode", "tie_break",
                                             "victim_mode", "tol")})
    return Engine(cat, pol, RandThreshold(rng, lam), bmax, rng=rng,
                  redraw_mode=kw.get("redraw_mode", "download_or_evict"),
                  check_invariants=kw.get("check_invariants", True)).run(trace)


def drl_mg(cat, trace, C, seed=0, bmax=None, **kw) -> Result:
    """Deterministic threshold + marking-gated Landlord."""
    rng = np.random.default_rng(seed)
    pol = LandlordPolicy(cat, C, marked=True, rng=rng,
                         **{k: v for k, v in kw.items()
                            if k in ("size_mode", "tie_break",
                                     "victim_mode", "tol", "soft_tau")})
    return Engine(cat, pol, DetThreshold(kw.get("theta", 1.0)), bmax, rng=rng,
                  redraw_mode=kw.get("redraw_mode", "download_or_evict"),
                  check_invariants=kw.get("check_invariants", True)).run(trace)


def drl_r3_mg(cat, trace, C, lam=1.0, seed=0, bmax=None, **kw) -> Result:
    """R3 + marking-gated Landlord."""
    rng = np.random.default_rng(seed)
    pol = LandlordPolicy(cat, C, marked=True, rng=rng,
                         **{k: v for k, v in kw.items()
                            if k in ("size_mode", "tie_break",
                                     "victim_mode", "tol", "soft_tau")})
    return Engine(cat, pol, RandThreshold(rng, lam), bmax, rng=rng,
                  redraw_mode=kw.get("redraw_mode", "download_or_evict"),
                  check_invariants=kw.get("check_invariants", True)).run(trace)


def drl_bayes_r3(cat, trace, C, lam=1.0, seed=0, bmax=None, **kw) -> Result:
    """Bayes-R3: posterior-rate threshold + deterministic Landlord."""
    rng = np.random.default_rng(seed)
    pol = LandlordPolicy(cat, C, **{k: v for k, v in kw.items()
                                    if k in ("size_mode", "tie_break",
                                             "victim_mode", "tol")})
    thr = BayesThreshold(cat, rng, lam=lam, c=kw.get("bayes_c", 1.0),
                         prior_shape=kw.get("prior_shape", 1.0),
                         prior_rate=kw.get("prior_rate", 1.0))
    return Engine(cat, pol, thr, bmax, rng=rng,
                  redraw_mode=kw.get("redraw_mode", "download_or_evict"),
                  check_invariants=kw.get("check_invariants", True)).run(trace)


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------
def synthetic_world(N=40, T=8000, zipf=1.2, rate=0.5, seed=0,
                    imbalance=1.0, **phys):
    """Zipf popularity + Poisson arrivals + lognormal resource demands."""
    rng = np.random.default_rng(seed)
    raw = np.c_[rng.lognormal(0, .6, N), rng.lognormal(1, .6, N),
                rng.lognormal(3.5, .8, N)]
    if imbalance != 1.0:
        dom = rng.integers(0, 3, N)
        raw[np.arange(N), dom] *= imbalance
    cat = from_physical(raw, **phys)
    pop = 1.0 / np.arange(1, N + 1) ** zipf
    pop /= pop.sum()
    perm = rng.permutation(N)
    svc = perm[rng.choice(N, size=T, p=pop)]
    t = np.cumsum(rng.exponential(1.0 / rate, T))
    return cat, Trace(t, svc.astype(int))


def zipf_world(N=40, T=8000, rho=1.2, seed=0, rate=0.5, **phys):
    return synthetic_world(N=N, T=T, zipf=rho, rate=rate, seed=seed, **phys)


def load_trace_csv(path, id_col, time_col, cpu_col, ram_col, disk_col,
                   time_scale_ms=1.0, disk_unit_mb=1.0, max_rows=None,
                   sep=",", disk_scale=1.0, **phys):
    """Load a preprocessed Google/SHARCNet CSV.

    Zeros/missing CPU/RAM/disk are replaced by the column median; a service's
    resources come from its first occurrence; timestamps are made strictly
    increasing.
    """
    import pandas as pd
    df = pd.read_csv(path, sep=sep, nrows=max_rows)
    for c in (cpu_col, ram_col, disk_col):
        col = pd.to_numeric(df[c], errors="coerce")
        med = col.loc[col > 0].median()
        if not np.isfinite(med):
            med = 0.0
        df[c] = col.where(col > 0, med)
    df = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    ids, inv = np.unique(df[id_col].to_numpy(), return_inverse=True)
    first = df.groupby(id_col).first().loc[ids]
    raw = np.c_[first[cpu_col], first[ram_col],
                first[disk_col] * disk_unit_mb * float(disk_scale)]
    t = df[time_col].to_numpy(float) * time_scale_ms
    t = t + np.arange(len(t)) * 1e-6
    return from_physical(raw, **phys), Trace(t - t[0], inv.astype(int))


def large_sequence(k, n_reps, M=6.0, Ml=12.0, l=1.0, burst_frac=0.5):
    """The adversarial Large Sequence of Theorem 1.

    Returns ``(catalog, trace, opt_latency)`` with ``OPT = (k + 2 n) l``.
    """
    assert k % 3 == 0 and k >= 3
    m = k // 3
    Rl, Cl, R, Cs = Ml, Ml, M, M
    caps = np.array([m * (Cl + 2 * Cs), m * (Rl + 2 * R),
                     m * (Ml + 2 * M)], float)
    raw, dl = [], []
    for _ in range(m):
        raw.append([Cs, R, Ml]); dl.append(Ml)   # BS1 disk-heavy
    for _ in range(m):
        raw.append([Cs, Rl, M]); dl.append(M)    # BS2 RAM-heavy
    for _ in range(m):
        raw.append([Cl, R, M]); dl.append(M)     # BS3 CPU-heavy
    raw.append([m * Cl, m * Rl, m * Ml]); dl.append(m * Ml)  # MS
    raw = np.array(raw, float)
    cat = Catalog(l=np.full(len(raw), float(l)), M=np.array(dl, float),
                  a=raw / caps, raw=raw, caps=caps)
    ev, t = [], 0.0

    def basic(s):
        nonlocal t
        Mi = cat.M[s]
        ev.append((t, s)); t += Mi + 1
        for _ in range(max(1, int(round(burst_frac * Mi)))):
            ev.append((t, s)); t += 1
        t += Mi

    for _ in range(n_reps):
        for s in range(k):
            basic(s)
        Mm = cat.M[k]
        ev.append((t, k)); t += Mm + 1
        ev.append((t, k)); t += 1 + Mm
    tt = np.array([e[0] for e in ev])
    ss = np.array([e[1] for e in ev], int)
    return cat, Trace(tt, ss), (k + 2 * n_reps) * l


def cyclic_scan(k, n_reps, l=1.0, M=1.0):
    """Classic cyclic scan over ``k+1`` services with cache capacity ``k``."""
    n = k + 1
    cat = Catalog(l=np.full(n, float(l)), M=np.full(n, float(M)),
                  a=np.ones((n, 3), float) / float(k))
    t = np.arange(n_reps * n, dtype=float)
    svc = np.tile(np.arange(n), n_reps).astype(int)
    return cat, Trace(t, svc)


def cyclic_burst_scan(k, n_reps, burst=4, l=1.0, M=None):
    """Cyclic scan in which every service is requested ``burst`` times in a row.

    With ``M = burst`` the download triggered by the first request of a burst
    finishes exactly when the burst ends.  This is the cyclic C+1 adversary
    used for the eviction-lockstep check.
    """
    n = k + 1
    if M is None:
        M = float(burst)
    cat = Catalog(l=np.full(n, float(l)), M=np.full(n, float(M)),
                  a=np.ones((n, 3), float) / float(k))
    ts, svc = [], []
    t = 0.0
    for _ in range(n_reps):
        for s in range(n):
            for _ in range(burst):
                ts.append(t); svc.append(s); t += 1.0
    return cat, Trace(np.asarray(ts, float), np.asarray(svc, int))


def shuffled_cyclic_scan(k, n_reps, seed=0, l=1.0, M=1.0):
    n = k + 1
    cat = Catalog(l=np.full(n, float(l)), M=np.full(n, float(M)),
                  a=np.ones((n, 3), float) / float(k))
    rng = np.random.default_rng(seed)
    svc = np.concatenate([rng.permutation(n) for _ in range(n_reps)]).astype(int)
    t = np.arange(len(svc), dtype=float)
    return cat, Trace(t, svc)


def deterministic_eviction_trace(cat, base_trace, C):
    """Request what deterministic Online-DRL has just evicted.

    The trace is generated with the deterministic policy only, so it is fixed
    in advance and oblivious to the randomized algorithm's coin flips.
    """
    from collections import defaultdict
    pol = LandlordPolicy(cat, C)
    Engine(cat, pol, DetThreshold(1.0), rng=np.random.default_rng(0)).run(base_trace)
    by_step = defaultdict(list)
    for step, f in pol.evictions:
        by_step[step].append(f)
    svc = []
    for i, n in enumerate(base_trace.n):
        svc.append(int(n))
        for f in by_step.get(i + 1, []):
            svc.append(int(f))
    dt = float(base_trace.t[1] - base_trace.t[0]) if len(base_trace.t) > 1 else 1.0
    t = np.arange(len(svc), dtype=float) * dt
    return Trace(t, np.asarray(svc, int)), base_trace, pol.evictions


# ---------------------------------------------------------------------------
# Small statistical helpers (kept here so the module is self-contained)
# ---------------------------------------------------------------------------
def bootstrap_ci(x, n_boot=2000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_wilcoxon(a, b):
    """Paired Wilcoxon signed-rank test; returns (statistic, pvalue)."""
    from scipy.stats import wilcoxon
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0.0):
        return 0.0, 1.0
    try:
        return tuple(wilcoxon(a, b, zero_method="wilcox"))
    except ValueError:
        return 0.0, 1.0


def holm(pvals, alpha=0.05):
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj