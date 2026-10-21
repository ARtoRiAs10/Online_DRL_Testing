"""Goal G2 (download budget) and goal G3 (regret) with the multi-resource projection.

G2: E[ sum_nt u_{n,t} M_n ] <= B_max.  The randomized trigger lowers the download probability
    to budget_left / (M_n * requests_left), so the expected spend stays inside the budget.
G3: E[ sum_t d_t(n*_t) ] - min_{P in D_vec} sum_t Fhat_t(P) <= Reg_T = o(T).
    Proposition 1 (multi-resource proposal): with eta = sqrt(N)/(l_max sqrt(T)) the regret of
    projected OGD is at most sqrt(N) l_max sqrt(T) (diameter of D_vec <= sqrt(N)).
"""
from common import *
from scipy.optimize import linprog
from sim.trace import synthetic_world
from sim.algos import rand_drl, online_drl
from sim.projection import normalize

a = parse(__doc__, C=12)
cat, tr = synthetic_world(N=a.N, T=a.T, seed=0, imbalance=3.0, cap_mult=4)
An = normalize(np.vstack([np.ones(cat.N), cat.a.T]), np.r_[float(a.C), 1, 1, 1])
lmax = float(cat.l.max())

print("### G2: download budget")
budget_rows = []
for frac in (0.2, 0.5, 1.0):
    det = online_drl(cat, tr, a.C, bmax=frac * det_cost_ref) if False else None
    det_full = online_drl(cat, tr, a.C)
    B = frac * det_full.cost
    rr = [rand_drl(cat, tr, a.C, np.random.default_rng(s), K=4, bmax=B) for s in range(a.seeds)]
    det_b = online_drl(cat, tr, a.C, bmax=B)
    row = dict(frac=frac, Bmax=B, det_cost_full=det_full.cost,
               det_cost_budgeted=det_b.cost, det_lat=det_b.latency,
               rand_cost=ms([r.cost for r in rr]), rand_lat=ms([r.latency for r in rr]))
    row["budget_respected"] = bool(row["rand_cost"][0] <= B + 1e-9)
    budget_rows.append(row); print(row)

print("\n### G3: regret of the surrogate on the *exogenous* gradient sequence")
# exogenous surrogate: q_{n,t} exogenous (assumption 1) -> use a fixed per-service gradient
rng = np.random.default_rng(0)
n_seq = tr.n; T = len(n_seq)
g_seq = rng.uniform(0.5, 1.5, T) * cat.l[n_seq]     # exogenous latencies
N = cat.N
eta = np.sqrt(N) / (lmax * np.sqrt(T))
bound = np.sqrt(N) * lmax * np.sqrt(T)
# best fixed P in D_vec
G = np.bincount(n_seq, weights=g_seq, minlength=N)
r = linprog(-G, A_ub=An, b_ub=np.ones(An.shape[0]), bounds=(0, 1), method="highs")
best_fixed = -r.fun
for peek in (True, False):
    from sim.regret import ogd_run, surrogate_regret
    P = ogd_run(n_seq, g_seq, An, eta, peek=peek)
    reg = surrogate_regret(n_seq, g_seq, P, An)
    print(f"peek={peek}: surrogate regret = {reg:.1f}   Prop.1 bound sqrt(N) l_max sqrt(T) = {bound:.1f} "
          f"-> {'OK' if reg <= bound else 'VIOLATED'}")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].bar(range(len(budget_rows)), [r["rand_cost"][0] for r in budget_rows])
ax[0].plot(range(len(budget_rows)), [r["Bmax"] for r in budget_rows], "k--", label="B_max")
ax[0].set_xticks(range(len(budget_rows))); ax[0].set_xticklabels([f"{r['frac']:.0%}" for r in budget_rows])
ax[0].set_xlabel("B_max as fraction of the deterministic cost"); ax[0].set_ylabel("download cost")
ax[0].legend(); ax[0].set_title("G2: expected cost stays below the budget")
ax[1].bar(range(len(budget_rows)), [r["rand_lat"][0] for r in budget_rows], label="Randomized DRL")
ax[1].plot(range(len(budget_rows)), [r["det_lat"] for r in budget_rows], "o--", label="Online-DRL")
ax[1].set_xticks(range(len(budget_rows))); ax[1].set_xticklabels([f"{r['frac']:.0%}" for r in budget_rows])
ax[1].set_xlabel("B_max as fraction of the deterministic cost"); ax[1].set_ylabel("latency")
ax[1].legend(); ax[1].set_title("latency under the same budget")
save(a, "exp15_budget_regret", dict(budget=budget_rows, bound=float(bound), N=N, T=T), fig)
