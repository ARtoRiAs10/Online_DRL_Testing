"""Randomized DRL cache policy.

R1 (fractional OGD state) + R2 (randomized rounding with repair) + R3 (randomized download
threshold), plus the revisions proposed in `docs/Proposed_revisions_randomized_DRL.pdf`:

  N1  `download_prob_mode="learned"`: the download decision becomes a *decision variable* Q_n.
      The algorithm measures the benefit of a residency (the latency avoided while the service
      was cached) and updates
          Q_n  <-  clip( Q_n + eta_Q (benefit_n - M_n) / M_n , 0, 1 )
      when the residency ends, and the download of n is then started with probability Q_n
      (multiplied by the budget factor).  Services whose download never pays off are therefore
      not downloaded again.
  N2  `grad_mode="expectation"`: the gradient uses the *expected* in-flight probability
      q_bar_n (an exponential moving average of the realized q) and the expected remaining
      download time M_n/2, instead of the realized q and dm.  The realized gradient becomes
      noise around it, which is what the O( sigma sqrt(T) ) term of the bound accounts for.
  N7  `proj_mode="active"`: the projection is computed on the support of p only (exact, since
      the other services have p = 0 and contribute nothing to A p);
      `rounding_mode="served"`: only the serving path is refreshed and repaired at every
      request, the other K-1 paths are refreshed round-robin every `refresh_every` steps.

Design decisions that the proposal leaves open (assumptions A1-A4, see README):
 A1  q_{n,t} = 1 if the edge node itself has a download of n in flight, else 0.
 A2  Rounding uses K persistent uniform vectors U_k: s_{k,n,t} = 1[U_{k,n} < p_{n,t}].  This gives
     E[s]=p and E|s_t - s_{t-1}| = |p_t - p_{t-1}|_1 (the coupling asked for in Sec. 3.4).
 A3  The serving path k* is fixed per run (`kstar_policy="run"`; "miss"/"step" are the two
     alternative readings of the proposal and are much worse - see exp18).
 A4  Downloads are triggered by the randomized threshold and, with `gate_downloads=True`, only if
     the sample currently wants the service (otherwise it would be discarded).
"""
from __future__ import annotations
import numpy as np
from .engine import Policy
from .projection import normalize, project_dual, project_exact, project_dual_active
from .landlord_repair import landlord_repair, sync_credit, is_feasible


class FracPolicy(Policy):
    name = "frac"

    def __init__(self, cat, C, rng, K=8, eta=None, A_rows=None, repair=True, proj="dual",
                 peek=True, resample_every=None, gate_downloads=True, record=False,
                 repair_order="p_over_max", budget_prob=True, eta_mode="horizon",
                 kstar_policy="run", download_prob_mode="budget", eta_q=0.1,
                 grad_mode="realized", q_ema_alpha=0.1, proj_mode="full",
                 rounding_mode="all", refresh_every=8, proj_tol=1e-7, proj_skip_tol=0.0,
                 repair_mode=None, log_steps=False, topup=False, store_p=False,
                 lazy=False, repair_pop_weight=0.1):
        self.cat, self.C, self.rng, self.K = cat, C, rng, K
        N = cat.N
        rows = cat.a.T if A_rows is None else np.atleast_2d(A_rows)
        A = np.vstack([np.ones(N), rows])
        b = np.r_[float(C), np.ones(rows.shape[0])]
        self.b = b
        self.A_raw = A
        self.b_raw = b
        self.An = normalize(A, b)                      # An p <= 1
        self.L = np.linalg.norm(self.An, 2) ** 2
        self.amax = self.An[1:].max(0) + 1e-12         # max_r a_{n,r}
        # repair_mode: "landlord" = Algorithm 3 of the updated pseudocode (stateful credits,
        # download-cost aware, deficit-aligned eviction); "stateless" = the v1 rule
        # (evict in increasing p_n / max_r a_{n,r}); None/"none" = no repair.
        if repair_mode is None:
            repair_mode = "landlord" if repair else "none"
        elif repair_mode in (True, False):
            repair_mode = "landlord" if repair_mode else "none"
        self.repair_mode = repair_mode
        self.repair = repair_mode not in (None, "none", False)
        self.log_steps, self.topup, self.store_p = log_steps, topup, store_p
        self.lazy = lazy
        self.repair_pop_weight = float(repair_pop_weight)
        self.popularity_rank = None
        self.eta_user, self.proj = eta, proj
        self.peek, self.resample_every, self.gate, self.record = peek, resample_every, gate_downloads, record
        self.repair_order, self.budget_prob = repair_order, budget_prob
        # eta_mode: "horizon"  = eta = sqrt(N)/(l_max sqrt(T))  (Prop. 1, needs the horizon T)
        #            "adaptive" = eta_t = eta / sqrt(t)         (horizon-free revision N4)
        # kstar_policy: "run"  = one serving path for the whole run (sticky, default)
        #                "miss" = redraw the serving path on every miss
        #                "step" = redraw at every request (the literal reading of the proposal)
        # download_prob_mode: "budget"  = only the budget clips the trigger (v1 behaviour)
        #                      "learned" = the trigger is also clipped by the learned Q_n (N1)
        # grad_mode: "realized" (v1) | "expectation" (N2)
        # proj_mode: "full" (v1) | "active" (N7.1) ; rounding_mode: "all" (v1) | "served" (N7.2)
        self.eta_mode, self.kstar_policy = eta_mode, kstar_policy
        self.download_prob_mode, self.eta_q = download_prob_mode, eta_q
        self.grad_mode, self.q_ema_alpha = grad_mode, q_ema_alpha
        self.proj_mode, self.rounding_mode, self.refresh_every = proj_mode, rounding_mode, refresh_every
        # proj_tol: tolerance of the dual ascent; proj_skip_tol: skip the projection when the
        # update is below that L1 size (the state is already feasible, so this is exact enough
        # and it is where most of the per-request cost of v1 was spent).
        self.proj_tol, self.proj_skip_tol = proj_tol, proj_skip_tol
        self.U = rng.random((K, N))
        self.kstar = int(rng.integers(K))
        self.T = None

    # ----- horizon-dependent step size (Prop. 1): eta = sqrt(N) / (l_max sqrt(T)) -----
    def set_horizon(self, T):
        N = self.cat.N
        self.T = T
        self.lmax = float(self.cat.l.max())
        if self.eta_user:
            self.eta = self.eta_user
        elif self.eta_mode == "adaptive":
            self.eta = 1.0 / self.lmax               # eta_t = self.eta / sqrt(t)
        else:
            self.eta = np.sqrt(N) / (self.lmax * np.sqrt(T))
        self.p = np.zeros(N); self.rho = np.zeros(self.An.shape[0])
        self.mask = np.zeros(N, bool)
        self.S = np.zeros((self.K, N), bool)
        # Persistent per-path Landlord credits (Algorithm 3, Step 0).  A credit is the
        # remaining "rent" a cached service has before it may be evicted.
        self.credit = np.zeros((self.K, N), float)
        self.landlord_stats = dict(rent_rounds=0, evictions=0, passes=0)
        self.steps = 0
        # --- revision state ---
        self.dprob = np.ones(N)                      # Q_n: the learned download probability (N1)
        self.benefit = np.zeros(N)                   # latency avoided during the current residency
        self.downloaded = np.zeros(N, bool)          # the service is in a residency
        self.q_ema = np.zeros(N)                     # EMA of the in-flight indicator (N2)
        self.hit_counts_by_rank = np.zeros(N, dtype=np.int64)
        self.step_log = []
        self._current_rec = None
        self.last_repair_stats = {}
        self._g = None
        self.st = dict(raw_violations=0, path_steps=0, repairs=0, repair_evictions=0,
                       proj_calls=0, proj_iters=0, wasted_downloads=0, dropped=0,
                       kstar_redraws=0, residencies=0, unprofitable=0, dprob_updates=0,
                       active_proj=0, active_size=0.0, refreshes=0, q_ema_sum=0.0,
                       proj_skipped=0, landlord_rent_rounds=0, landlord_evictions=0,
                       landlord_passes=0, topup_added=0)
        self.P_hist, self.g_hist, self.n_hist = [], [], []

    # ----- R1: projected OGD step (eqs. 19-22) -----
    def _update_q_ema(self, n, q):
        """EMA update for Q.  Call this *after* the OGD gradient uses the old Q."""
        a = self.q_ema_alpha
        self.q_ema[n] = (1 - a) * self.q_ema[n] + a * q
        self.st["q_ema_sum"] += float(self.q_ema[n])

    def _ogd(self, n, q, dm, t=1):
        l = self.cat.l[n]
        if self.grad_mode == "expectation":          # N2: expectation over the algorithm's noise
            qe = float(self.q_ema[n])
            dme = self.cat.M[n] / 2.0
            g = (1 - qe) * l + qe * min(l, dme)
        else:
            g = (1 - q) * l + q * min(l, dm)
        eta = (self.eta / np.sqrt(max(t, 1))) if (self.eta_mode == "adaptive" and not self.eta_user) \
            else self.eta
        z = self.p.copy(); z[n] += eta * g
        if self.proj_skip_tol and np.abs(z - self.p).sum() <= self.proj_skip_tol:
            # small update: skip the expensive dual ascent and apply only the cheap feasible
            # scaling of eq. (25) (the state stays feasible; the projection is approximate).
            self.st["proj_skipped"] += 1
            p = np.clip(z, 0.0, 1.0)
            m = float((self.An @ p).max())
            if m > 1.0: p = p / m
            self.p = p
            self._g = g
            return
        if self.proj == "scale":
            # fast feasible projection: clip to the box and apply the scaling of eq. (25).
            # It is feasible but NOT the exact Euclidean projection, so the regret bound of
            # Prop. 1 acquires an additive O(T eps) projection-error term.
            p = np.clip(z, 0.0, 1.0)
            m = float((self.An @ p).max())
            if m > 1.0: p = p / m
            self.p = p
            self.st["proj_iters"] += 1
        elif self.proj == "cvx":
            self.p = project_exact(z, self.An)
        elif self.proj_mode == "active":             # N7.1: project on the support of p
            idx = np.flatnonzero(z > 1e-12)
            # L = None -> the Frobenius bound of the sub-problem (O(4|idx|), no SVD per request)
            self.p, self.rho, it = project_dual_active(z, self.An, idx, self.rho, None,
                                                        tol=self.proj_tol)
            self.st["active_proj"] += 1; self.st["active_size"] += len(idx)
        else:
            self.p, self.rho, it = project_dual(z, self.An, self.rho, self.L, tol=self.proj_tol)
            self.st["proj_iters"] += it
        self.st["proj_calls"] += 1
        self._g = g

    # ----- R2: rounding + multi-resource repair (Sec. 3.4) -----
    def _repair_row(self, k):
        """Stateless v1 repair: evict in increasing order of p_n / max_r a_{n,r}."""
        s = self.S[k]
        idx = np.flatnonzero(s)
        if self.repair_order == "alignment":
            # Sec. 2.5: order candidates by the alignment <a_n, u> with the residual-capacity
            # vector u, i.e. evict the services that fit the leftover capacity best (Tetris-style).
            load = self.An[:, idx].sum(1)
            u = np.maximum(1.0 - load, 0.0)
            align = self.An[:, idx].T @ u
            order = idx[np.argsort(-align)]
        else:
            # Sec. 3.4: increasing order of p_{n,t} / max_r a_{n,r}
            order = idx[np.argsort(self.p[idx] / self.amax[idx])]
        # vectorised repair: evict the first j services of `order` such that the remainder fits
        A = self.An[:, order]
        cum = np.cumsum(A, axis=1)
        total = cum[:, -1:]
        suffix = total - cum + A          # suffix[:, j] = sum of A[:, j:]
        feas = (suffix <= 1 + 1e-9).all(0)
        j = int(np.argmax(feas)) if feas.any() else len(order)
        if j > 0:
            s[order[:j]] = False
        self.st["repairs"] += 1; self.st["repair_evictions"] += j

    def _landlord_repair_row(self, k, was_violation=False):
        """Algorithm 3 (updated Landlord-Repair) on sample path ``k``.

        The credit vector is synchronized on *every* refresh (Step 0), even when the
        sampled state is already feasible, so a service that enters the path receives
        its download cost as credit and a service that leaves the path loses it.
        """
        before = int(self.S[k].sum())
        evict_mode = {
            "landlord": "deficit",
            "landlord_pop": "popularity",
            "landlord_hybrid": "hybrid",
        }.get(self.repair_mode, "deficit")
        landlord_repair(self.S[k], self.p, self.A_raw, self.b_raw, self.cat.M,
                        self.credit[k], tol=1e-9, stats=self.landlord_stats,
                        evict_mode=evict_mode, pop_weight=self.repair_pop_weight)
        after = int(self.S[k].sum())
        if was_violation:
            self.st["repairs"] += 1
        ev = max(0, before - after)
        if ev:
            self.st["repair_evictions"] += ev

    def _topup_path(self, k):
        """Add the largest p_n/w_n services that still fit after repair.

        This is the optional "cache not filled" fix: repair may leave capacity unused, so
        greedily fill it by the same priority used by the stateless repair.
        """
        s = self.S[k]
        idx = np.flatnonzero(~s)
        if idx.size == 0:
            return 0
        prio = self.p[idx] / np.maximum(self.amax[idx], 1e-12)
        order = idx[np.argsort(-prio)]
        load = self.An @ s.astype(float)
        added = 0
        for n in order:
            col = self.An[:, n]
            if np.all(load + col <= 1.0 + 1e-9):
                s[n] = True
                load = load + col
                added += 1
        if added:
            self.st["topup_added"] += added
        return added

    def _refresh_path(self, k):
        """Refresh one sample path and repair it if it violates the constraints.

        With ``lazy=True`` the previous path is kept and the new rounding is added to it
        (evict only when the union is infeasible).  This reduces churn: a service that
        was cached last step is not dropped just because it missed the new sample.
        """
        new_sample = self.U[k] < self.p
        rounded_in = int(new_sample.sum())
        if self.lazy:
            sample = self.S[k] | new_sample
        else:
            sample = new_sample
        before_repair = int(sample.sum())
        self.S[k] = sample
        self.st["path_steps"] += 1; self.st["refreshes"] += 1
        viol = bool((self.S[k] @ self.An.T > 1 + 1e-9).any())
        if viol:
            self.st["raw_violations"] += 1
        if self.repair_mode in ("landlord", "landlord_pop", "landlord_hybrid"):
            # Always run Step 0 (credit synchronization); repair only if needed.
            self._landlord_repair_row(k, was_violation=viol)
        elif self.repair and viol:
            self._repair_row(k)
        after_repair = int(self.S[k].sum())
        evicted = max(0, before_repair - after_repair)
        added = self._topup_path(k) if self.topup else 0
        self.last_repair_stats[k] = {
            "rounded_in": rounded_in,
            "before_repair": before_repair,
            "after_repair": after_repair,
            "evicted": evicted,
            "fraction": evicted / max(1, before_repair),
            "topup_added": added,
            "final_count": int(self.S[k].sum()),
        }

    def _sample(self):
        if self.rounding_mode == "served":            # N7.2/N7.3: refresh + repair the served path
            self._refresh_path(self.kstar)
            if self.refresh_every and self.steps % self.refresh_every == 0:
                k2 = (self.kstar + 1) % self.K
                if k2 != self.kstar: self._refresh_path(k2)
        else:
            # Refresh every path through the same code path as the served mode so the
            # per-step repair statistics are complete and consistent.
            for k in range(self.K):
                self._refresh_path(k)
        drop = self.mask & ~self.S[self.kstar]
        for n in np.flatnonzero(drop):
            self.mask[n] = False; self.eng.reset_miss(n); self.st["dropped"] += 1
            self._close_residency(n)

    # ----- N1: close a residency and update the learned download probability -----
    def _close_residency(self, n):
        if not self.downloaded[n]:
            return
        b = float(self.benefit[n]); M = float(self.cat.M[n])
        self.st["residencies"] += 1
        if b < M: self.st["unprofitable"] += 1
        self.dprob[n] = float(np.clip(self.dprob[n] + self.eta_q * (b - M) / max(M, 1e-9), 0, 1))
        self.benefit[n] = 0.0
        self.downloaded[n] = False
        self.st["dprob_updates"] += 1

    def before_request(self, t, n, q, dm):
        self.steps += 1
        p_before = self.p.copy()
        s_before = self.S[self.kstar].copy()
        rec = None
        if self.log_steps:
            rec = {
                "t": float(t), "n": int(n), "q": float(q), "dm": float(dm),
                "p_sum": float(p_before.sum()),
                "resource_load_norm": (self.An[1:] @ p_before).tolist(),
                "resource_load_raw": (self.cat.a.T @ p_before).tolist(),
                "p_n_star_prev": float(p_before[n]),
                "x_bar_n_star_prev": float(s_before[n]),
                "p_minus_xbar": float(p_before[n] - s_before[n]),
                "q_ema_before": float(self.q_ema[n]),
                "q_ema_after": None,
                "grad": None,
                "l_n": None,
                "benefit": None,
                "p_n_star_after": None,
                "outcome": None,
                "hit_rank": None,
                "download_started": False,
                "c_t": 0.0,
                "repair_evicted_fraction": None,
                "repair_evicted": None,
                "topup_added": None,
                "p_sum_after": None,
            }
            if self.store_p:
                rec["p_before"] = p_before.tolist()
            self._current_rec = rec
        self.last_repair_stats = {}
        if self.peek:
            self._ogd(n, q, dm, t)
            if rec is not None:
                rec["grad"] = float(self._g)
                rec["l_n"] = float(self.cat.l[n])
                rec["benefit"] = float(self.cat.l[n] - self._g)
                rec["p_n_star_after"] = float(self.p[n])
        self._sample()
        if rec is not None:
            rs = self.last_repair_stats.get(self.kstar)
            if rs is not None:
                rec["repair_evicted_fraction"] = float(rs["fraction"])
                rec["repair_evicted"] = int(rs["evicted"])
                rec["topup_added"] = int(rs["topup_added"])
            rec["p_sum_after"] = float(self.p.sum())
            if self.store_p:
                rec["p_after"] = self.p.tolist()
        if self.peek:
            self._update_q_ema(n, q)
        if self.resample_every and self.steps % self.resample_every == 0:
            self.kstar = int(self.rng.integers(self.K)); self.st["kstar_redraws"] += 1
        if self.kstar_policy == "step":
            self.kstar = int(self.rng.integers(self.K)); self.st["kstar_redraws"] += 1
        elif self.kstar_policy == "miss" and (not self.mask[n]) and q == 0.0:
            self.kstar = int(self.rng.integers(self.K)); self.st["kstar_redraws"] += 1
        if self.record:
            self.P_hist.append(self.p.copy()); self.g_hist.append(self._g if self.peek else
                (1 - q) * self.cat.l[n] + q * min(self.cat.l[n], dm)); self.n_hist.append(n)

    def after_request(self, t, n, q, dm):
        if not self.peek:
            self._ogd(n, q, dm, t)
            self._update_q_ema(n, q)
        rec = self._current_rec
        if rec is not None:
            outcome = getattr(self.eng, "last_outcome", None)
            rec["outcome"] = outcome
            if outcome == "hit":
                rec["hit_rank"] = (int(self.popularity_rank[n])
                                   if self.popularity_rank is not None else None)
            rec["download_started"] = bool(q == 0.0 and n in getattr(self.eng, "pending", {}))
            rec["c_t"] = float(self.cat.M[n]) if rec["download_started"] else 0.0
            rec["q_ema_after"] = float(self.q_ema[n])
            if rec["grad"] is None and self._g is not None:
                rec["grad"] = float(self._g)
            if rec["l_n"] is None:
                rec["l_n"] = float(self.cat.l[n])
            if rec["benefit"] is None:
                rec["benefit"] = float(self.cat.l[n] - self._g)
            if rec["p_n_star_after"] is None:
                rec["p_n_star_after"] = float(self.p[n])
            self.step_log.append(rec)
            self._current_rec = None

    # ----- policy interface -----
    def set_popularity_rank(self, rank):
        """Set the service -> popularity-rank map used by the per-step hit log."""
        self.popularity_rank = np.asarray(rank, int)

    def cached(self, n): return bool(self.mask[n])
    def cached_list(self): return np.flatnonzero(self.mask)
    def allow_download(self, n, t): return (not self.gate) or bool(self.S[self.kstar, n])

    def download_prob(self, n, t, budget_left=None):
        """Probability of *starting* a download once the (random) threshold is met.

        1.0 = deterministic trigger (Online-DRL).  v1 clips the trigger only with the download
        budget (goal G2).  v2 (N1) also clips it with the learned download probability Q_n, and
        the expectation gradient (N2) makes the state smoother.
        """
        if self.gate and not self.S[self.kstar, n]:
            return 0.0
        base = 1.0
        if self.download_prob_mode == "learned":
            base = min(base, float(self.dprob[n]))
        if not self.budget_prob or budget_left is None or self.T is None:
            return float(np.clip(base, 0.0, 1.0))
        left = max(1, self.T - t)
        return float(np.clip(min(base, budget_left / (self.cat.M[n] * left)), 0.0, 1.0))

    def on_hit(self, n, saved=0.0):
        """A hit: the service was served from the cache, so `saved` latency was avoided."""
        self.benefit[n] += saved
        if self.popularity_rank is not None:
            self.hit_counts_by_rank[int(self.popularity_rank[n])] += 1

    def on_delayed(self, n, saved=0.0):
        """A delayed hit: the request was buffered; `saved` = l_n - remaining download time."""
        self.benefit[n] += saved

    def on_complete(self, n, t):
        if self.S[self.kstar, n]:
            self.mask[n] = True
            self.downloaded[n] = True      # a new residency starts (N1)
            self.benefit[n] = 0.0
            return True
        self.st["wasted_downloads"] += 1
        return False

    def diagnostics(self):
        d = dict(self.st)
        d["mean_proj_iters"] = self.st["proj_iters"] / max(1, self.st["proj_calls"])
        d["raw_violation_rate"] = self.st["raw_violations"] / max(1, self.st["path_steps"])
        d["mean_q_ema"] = self.st["q_ema_sum"] / max(1, self.steps)
        d["mean_dprob"] = float(self.dprob.mean())
        d["dprob_hist"] = self.dprob.copy()
        if self.record:
            d["P_hist"] = np.array(self.P_hist, np.float32); d["g_hist"] = np.array(self.g_hist)
            d["n_hist"] = np.array(self.n_hist)
        d["rho_final"] = self.rho.copy()
        d["repair_mode"] = self.repair_mode
        d["landlord_rent_rounds"] = self.landlord_stats.get("rent_rounds", 0)
        d["landlord_evictions"] = self.landlord_stats.get("evictions", 0)
        d["landlord_passes"] = self.landlord_stats.get("passes", 0)
        d["hit_counts_by_rank"] = self.hit_counts_by_rank.copy()
        d["topup_added"] = self.st["topup_added"]
        if self.log_steps and self.step_log:
            fr = [r["repair_evicted_fraction"] for r in self.step_log
                  if r.get("repair_evicted_fraction") is not None]
            d["mean_repair_evicted_fraction"] = float(np.mean(fr)) if fr else 0.0
            d["step_log"] = self.step_log
        else:
            d["mean_repair_evicted_fraction"] = 0.0
        return d
