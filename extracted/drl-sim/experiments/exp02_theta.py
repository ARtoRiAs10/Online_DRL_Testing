"""Original paper: theta trades cost against latency; theta=0 equals LL-RC's download rule."""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, ll_rc

a = parse(__doc__, C=15)
thetas = [0, .25, .5, 1, 2, 4, 8, 16]
res = {k: [] for k in ("latency", "cost", "hits", "delayed_hits")}
cat, tr = synthetic_world(N=a.N, T=a.T, seed=0)
for th in thetas:
    r = online_drl(cat, tr, a.C, theta=th)
    for k in res: res[k].append(getattr(r, k))
assert abs(online_drl(cat, tr, a.C, 0).cost - ll_rc(cat, tr, a.C).cost) < 1e-9
print("theta=0 reproduces LL-RC: OK")
fig, ax = plt.subplots(1, 3, figsize=(11, 3))
for i, k in enumerate(("cost", "latency", "hits")):
    ax[i].plot(thetas, res[k], "o-"); ax[i].set_xlabel("theta"); ax[i].set_title(k)
for th, c, l in zip(thetas, res["cost"], res["latency"]): print(f"theta={th:5}  cost={c:12.0f}  latency={l:12.0f}")
save(a, "exp02_theta", dict(thetas=thetas, **res), fig)
