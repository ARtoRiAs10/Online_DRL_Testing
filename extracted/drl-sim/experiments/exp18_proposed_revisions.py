"""Proposed revisions to the randomized DRL, measured.

The proposal leaves several parts unspecified and, in three places, states goals that the
simulation shows to be false as written.  This script measures the three revisions that are
proposed in `docs/Proposed_revisions_randomized_DRL.pdf`:

  R-B  horizon-free step size      eta_t = eta / sqrt(t) instead of eta = sqrt(N)/(l_max sqrt(T))
  R-C  a *correct* goal G3         the realized-latency regret is compared against a STATIC
                                   cache played with the same download dynamics ("G3'"):
                                       Reg_real(T) = L_randomized(T) - L_static(T)
                                   and it is checked that Reg_real(T)/T -> 0 (the real meaning
                                   of "regret = o(T)"), next to the download-charged inequality
                                       literal G3  <=  Reg_Prop1(T) + C_det(T)
  R-D  the serving path k*         "run" (sticky, our A3) vs "miss" vs "step" (the literal
                                   reading of the proposal) - the churn it causes is measured.
"""
from common import *
from sim.trace import synthetic_world
from sim.algos import rand_drl, online_drl
from sim.engine import Engine, DetThreshold, Policy
from sim.projection import normalize
from sim.regret import ogd_run, surrogate_regret, best_fixed_value

a = parse(__doc__, C=12)
Tmax = 8000 if not a.quick else 2000
Ts = [500, 1000, 2000, 4000, 8000] if not a.quick else [250, 500, 1000, 2000]
cat, trfull = synthetic_world(N=a.N, T=Tmax, seed=0)
N, lmax = cat.N, float(cat.l.max())
An = normalize(np.vstack([np.ones(N), cat.a.T]), np.r_[float(a.C), 1, 1, 1])


# ------------------------------------------------------------------ static baseline
class StaticSetPolicy(Policy):
    """A fixed cache: the set S is decided offline; a service of S is downloaded on its first
    miss and then kept (as long as it fits).  This is the apples-to-apples comparator for the
    realized-latency regret: it pays the same download delays as the randomized policy."""
    name = "static"

    def __init__(self, cat, C, S):
        self.cat, self.C, self.S = cat, C, set(int(s) for s in S)
        self.have = set()

    def cached(self, n): return n in self.have
    def cached_list(self): return list(self.have)
    def allow_download(self, n, t): return n in self.S
    def on_complete(self, n, t):
        used = self.cat.a[list(self.have)].sum(0) if self.have else np.zeros(3)
        if (used + self.cat.a[n] <= 1 + 1e-9).all() and len(self.have) < self.C:
            self.have.add(n); return True
        return False


def best_static_set(n_seq, g_seq, An, cat, C):
    """The best fixed fractional cache (LP) rounded to a feasible binary set: services are added
    in decreasing order of p_n / max_r a_n,r while the three resources and the cardinality allow
    it."""
    N = An.shape[1]
    G = np.bincount(n_seq, weights=g_seq, minlength=N)
    from scipy.optimize import linprog
    r = linprog(-G, A_ub=An, b_ub=np.ones(An.shape[0]), bounds=(0, 1), method="highs")
    p = r.x if r.x is not None else np.zeros(N)
    order = np.argsort(-(p / cat.a.max(1)))
    S, used = [], np.zeros(3)
    for n in order:
        if p[n] <= 1e-9: break
        if (used + cat.a[n] <= 1 + 1e-9).all() and len(S) < C:
            S.append(int(n)); used = used + cat.a[n]
    return S, p


# ------------------------------------------------------------------ R-B and R-C
print("### R-B (horizon-free step size) and R-C (corrected goal G3)")
rng = np.random.default_rng(1)
pop = (1 / np.arange(1, N + 1) ** 1.2); pop /= pop.sum()
rows = []
for T in Ts:
    tr = trfull.head(T)
    n_seq = tr.n.astype(int)
    # exogenous gradient sequence (as in Assumption 1) for the OGD runs
    g_ex = rng.uniform(0.5, 1.5, T) * cat.l[n_seq]
    eta_prop1 = np.sqrt(N) / (lmax * np.sqrt(T))
    P_prop1 = ogd_run(n_seq, g_ex, An, eta_prop1, peek=False)
    reg_prop1 = surrogate_regret(n_seq, g_ex, P_prop1, An)
    # horizon-free: eta_t = (1/l_max)/sqrt(t)  ->  bound 2 sqrt(N) l_max sqrt(T)
    P_ad = ogd_run(n_seq, g_ex, An, 1.0 / lmax, peek=False, eta_sched="1/sqrt(t)")
    reg_ad = surrogate_regret(n_seq, g_ex, P_ad, An)
    bound_ad = 2 * np.sqrt(N) * lmax * np.sqrt(T)
    # realized run of the randomized policy (endogenous gradients, recorded)
    rr = rand_drl(cat, tr, a.C, np.random.default_rng(0), K=4, record=True)
    ex = rr.extra
    reg_endog = surrogate_regret(ex["n_hist"], ex["g_hist"], ex["P_hist"], An)
    # literal G3 and the download-charged correction
    literal = rr.latency - (ex["g_hist"].sum() - best_fixed_value(ex["n_hist"], ex["g_hist"], An))
    det = online_drl(cat, tr, a.C)
    # G3': realized latency against the best static cache played with the same dynamics
    S, pstar = best_static_set(ex["n_hist"], ex["g_hist"], An, cat, a.C)
    st = Engine(cat, StaticSetPolicy(cat, a.C, S), DetThreshold(0.0)).run(tr)
    reg_real = rr.latency - st.latency
    row = dict(T=T, bound_prop1=float(np.sqrt(N) * lmax * np.sqrt(T)),
               regret_prop1_horizon_eta=float(reg_prop1),
               regret_adaptive_eta=float(reg_ad), bound_adaptive=float(bound_ad),
               regret_endogenous=float(reg_endog), literal_G3=float(literal),
               det_download_cost=float(det.cost), static_latency=float(st.latency),
               static_cost=float(st.cost),
               rand_latency=float(rr.latency), rand_cost=float(rr.cost),
               regret_real_G3prime=float(reg_real),
               regret_real_per_T=float(reg_real / T),
               regret_real_per_request_lat=float(reg_real / max(rr.latency, 1e-9)),
               n_static=len(S))
    rows.append(row); print(row)

print("\n### R-D (the serving path k*: sticky vs per-miss vs per-step)")
rows_d = []
for pol in ("run", "miss", "step"):
    rs = [rand_drl(cat, trfull, a.C, np.random.default_rng(s), K=4, kstar_policy=pol)
          for s in range(max(1, a.seeds // 2))]
    row = dict(policy=pol, latency=ms([r.latency for r in rs]), cost=ms([r.cost for r in rs]),
               hits=ms([r.hits for r in rs]), downloads=ms([r.downloads for r in rs]),
               dropped=ms([r.extra["dropped"] for r in rs]),
               redraws=ms([r.extra["kstar_redraws"] for r in rs]))
    rows_d.append(row); print(row)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ax[0].plot([r["T"] for r in rows], [r["regret_prop1_horizon_eta"] for r in rows], "o-",
           color="#1f3b73", label="eta = sqrt(N)/(l_max sqrt(T))")
ax[0].plot([r["T"] for r in rows], [r["regret_adaptive_eta"] for r in rows], "s-", color="#2e75b6",
           label="eta_t = 1/(l_max sqrt(t))")
ax[0].plot([r["T"] for r in rows], [r["bound_prop1"] for r in rows], "k--", label="bound sqrt(N) l_max sqrt(T)")
ax[0].plot([r["T"] for r in rows], [r["bound_adaptive"] for r in rows], ":", color="grey",
           label="bound 2 sqrt(N) l_max sqrt(T)")
ax[0].set_xlabel("T"); ax[0].set_ylabel("surrogate regret"); ax[0].legend(fontsize=7)
ax[0].set_title("R-B: horizon-free step size", fontsize=10)
ax[1].plot([r["T"] for r in rows], [r["literal_G3"] for r in rows], "^-", color="#9c2b2b",
           label="literal G3 (realized latency)")
ax[1].plot([r["T"] for r in rows], [r["bound_prop1"] + r["det_download_cost"] for r in rows], "o--",
           color="#1b6b3a", label="Reg_Prop1 + C_det (download-charged)")
ax[1].plot([r["T"] for r in rows], [r["bound_prop1"] for r in rows], "k:", label="Reg_Prop1")
ax[1].set_xlabel("T"); ax[1].set_ylabel("regret"); ax[1].legend(fontsize=7)
ax[1].set_title("R-C: the download-charged correction of G3", fontsize=10)
ax[2].plot([r["T"] for r in rows], [r["regret_real_per_T"] for r in rows], "o-", color="#1f3b3a",
           label="Reg_real(T)/T  (adaptive vs static cache)")
ax[2].axhline(0, color="k", lw=.8)
ax[2].set_xlabel("T"); ax[2].set_ylabel("regret per request")
ax[2].set_title("G3': the regret per request tends to 0", fontsize=10); ax[2].legend(fontsize=7)
for x in ax: x.grid(alpha=.3)
save(a, "exp18_proposed_revisions", dict(revisions=rows, serving_path=rows_d), fig)
