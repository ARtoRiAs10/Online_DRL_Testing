"""Experiment 2 -- the deterministic threshold theta trades download cost against latency.

Paper: theta = 0 reproduces LL-RC; increasing theta reduces downloads/cost but increases
the number of forwarded requests.  The updated randomized DRL replaces the fixed theta by
the randomized ski-rental threshold (R3); it is shown as a reference point.
"""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, ll_rc, rand_drl

a = parse(__doc__, C=15)
thetas = [0, .25, .5, 1, 2, 4, 8, 16]
res = {k: [] for k in ("latency", "cost", "hits", "delayed_hits", "downloads")}
cat, tr = synthetic_world(N=a.N, T=a.T, seed=0)
for th in thetas:
    r = online_drl(cat, tr, a.C, theta=th)
    for k in res:
        res[k].append(getattr(r, k))
assert abs(online_drl(cat, tr, a.C, 0).cost - ll_rc(cat, tr, a.C).cost) < 1e-9
print("theta=0 reproduces LL-RC: OK")

rr = rand_drl(cat, tr, a.C, np.random.default_rng(0), K=4)
res["randomized"] = dict(latency=rr.latency, cost=rr.cost, hits=rr.hits,
                         delayed_hits=rr.delayed_hits, downloads=rr.downloads)
print("updated randomized DRL (R3):", res["randomized"])

fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
for i, k in enumerate(("cost", "latency", "hits", "downloads")):
    ax[i].plot(thetas, res[k], "o-")
    ax[i].set_xlabel("theta"); ax[i].set_title(k)
for th, c, l in zip(thetas, res["cost"], res["latency"]):
    print(f"theta={th:5}  cost={c:12.0f}  latency={l:12.0f}")
save(a, "exp02_theta", dict(thetas=thetas, **res), fig)