"""G3: regret of projected OGD on the surrogate.
(a) exogenous (q, dm) as in Assumption 1; (b) endogenous (q, dm from the real run);
(c) literal G3: realized latency of the full algorithm minus min_P sum F_t(P)."""
from common import *
from sim.trace import synthetic_world
from sim.algos import rand_drl
from sim.projection import normalize
from sim.regret import ogd_run, surrogate_regret, best_fixed_value

a = parse(__doc__, C=12)
Ts = [250, 500, 1000, 2000] if a.quick else [500, 1000, 2000, 4000, 8000]
cat, _ = synthetic_world(N=a.N, T=10, seed=0)
N, lmax = cat.N, cat.l.max()
An = normalize(np.vstack([np.ones(N), cat.a.T]), np.r_[float(a.C), 1, 1, 1])
rng = np.random.default_rng(1)
pop = lambda perm: (1 / np.arange(1, N + 1) ** 1.2)[np.argsort(perm)] / (1 / np.arange(1, N + 1) ** 1.2).sum()
p0, p1 = pop(rng.permutation(N)), pop(rng.permutation(N))
rows = []
for T in Ts:
    n_seq = np.r_[rng.choice(N, T // 2, p=p0), rng.choice(N, T - T // 2, p=p1)]       # popularity shift
    q = rng.random(T) < .3; dm = rng.uniform(0, 2 * cat.l[n_seq]); l = cat.l[n_seq]
    g = np.where(q, np.minimum(l, dm), l)
    eta = np.sqrt(N) / (lmax * np.sqrt(T))
    reg = {pk: surrogate_regret(n_seq, g, ogd_run(n_seq, g, An, eta, peek=pk), An) for pk in (True, False)}
    n_st = rng.choice(N, T, p=p0)                                                   # stationary sequence
    g_st = np.where(q, np.minimum(cat.l[n_st], rng.uniform(0, 2 * cat.l[n_st])), cat.l[n_st])
    reg_st = surrogate_regret(n_st, g_st, ogd_run(n_st, g_st, An, eta, peek=False), An)
    cat_, tr = synthetic_world(N=a.N, T=T, seed=0)
    rr = [rand_drl(cat_, tr, a.C, np.random.default_rng(s), K=4, record=True) for s in range(max(1, a.seeds // 2))]
    endo, lit = [], []
    for r in rr:
        ex = r.extra; endo.append(surrogate_regret(ex["n_hist"], ex["g_hist"], ex["P_hist"], An))
        lit.append(r.latency - (ex["g_hist"].sum() - best_fixed_value(ex["n_hist"], ex["g_hist"], An)))
    row = dict(T=T, bound=float(np.sqrt(N) * lmax * np.sqrt(T)), regret_exog_standard=reg[False], regret_exog_stationary=float(reg_st), regret_exog_peek=reg[True],
               regret_endog=ms(endo), literal_G3=ms(lit)); rows.append(row); print(row)
for r in rows:
    print(f"T={r['T']:5d} bound={r['bound']:8.0f}  stationary regret/bound={r['regret_exog_stationary']/r['bound']:+.3f}  "
          f"shifting regret/bound={r['regret_exog_standard']/r['bound']:+.3f}  literal-G3/bound={r['literal_G3'][0]/r['bound']:+.2f}")
print("Negative regret vs the best FIXED P is possible when popularity shifts (OGD adapts).")
print("literal G3 >> bound means the fractional comparator is unrealistically strong: it caches instantly, the real system must download first.")
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(Ts, [r["bound"] for r in rows], "k:", label="sqrt(N) lmax sqrt(T)")
ax.plot(Ts, [r["regret_exog_stationary"] for r in rows], "o-", label="exogenous, stationary")
ax.plot(Ts, [r["regret_exog_standard"] for r in rows], "o--", label="exogenous, popularity shift")
ax.plot(Ts, [r["regret_endog"][0] for r in rows], "s-", label="endogenous (real run)")
ax.plot(Ts, [r["literal_G3"][0] for r in rows], "^-", label="literal G3 (realized latency)")
ax.set_xlabel("T"); ax.set_ylabel("regret (ms)"); ax.legend(fontsize=7)
save(a, "exp09_regret", dict(rows=rows), fig)
