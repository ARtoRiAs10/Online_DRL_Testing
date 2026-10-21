"""Original paper: latency falls as the slot count C and the CPU/RAM/disk limits grow."""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, ll_rc

a = parse(__doc__)
Cs, mults = [1, 2, 4, 8, 16, 32], [2, 3, 4, 6, 8, 16]
outC, outM = [], []
cat, tr = synthetic_world(N=a.N, T=a.T, seed=0)
for C in Cs:
    r, b = online_drl(cat, tr, C), ll_rc(cat, tr, C)
    outC.append(dict(C=C, lat=r.latency, cost=r.cost, lat_llrc=b.latency, hits=r.hits)); print(outC[-1])
for m in mults:
    cat2, tr2 = synthetic_world(N=a.N, T=a.T, seed=0, cap_mult=m)
    r, b = online_drl(cat2, tr2, 10 ** 6), ll_rc(cat2, tr2, 10 ** 6)
    outM.append(dict(cap_mult=m, lat=r.latency, cost=r.cost, lat_llrc=b.latency, hits=r.hits)); print(outM[-1])
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].plot(Cs, [r["lat"] for r in outC], "o-", label="Online-DRL"); ax[0].plot(Cs, [r["lat_llrc"] for r in outC], "s--", label="LL-RC"); ax[0].set_xlabel("C"); ax[0].set_title("latency vs slots"); ax[0].legend()
ax[1].plot(mults, [r["lat"] for r in outM], "o-"); ax[1].plot(mults, [r["lat_llrc"] for r in outM], "s--"); ax[1].set_xlabel("resource limit (x mean demand)"); ax[1].set_title("latency vs resource limits (C unlimited)")
save(a, "exp04_capacity", dict(C=outC, resources=outM), fig)
