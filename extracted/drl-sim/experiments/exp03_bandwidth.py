"""Original paper: delayed hits fall as uplink U rises and rise as downlink D rises."""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, ll_rc

a = parse(__doc__, C=15)
Us, Ds = [20, 50, 100, 200, 400], [250, 500, 1000, 2000, 4000]
out = {"U": [], "D": []}
for U in Us:
    cat, tr = synthetic_world(N=a.N, T=a.T, seed=0, U=U, req_kbit=2000)
    r, b = online_drl(cat, tr, a.C), ll_rc(cat, tr, a.C)
    out["U"].append(dict(U=U, dh=r.delayed_hits, lat=r.latency, cost=r.cost, lat_llrc=b.latency, cost_llrc=b.cost))
for D in Ds:
    cat, tr = synthetic_world(N=a.N, T=a.T, seed=0, D=D, req_kbit=2000)
    r, b = online_drl(cat, tr, a.C), ll_rc(cat, tr, a.C)
    out["D"].append(dict(D=D, dh=r.delayed_hits, lat=r.latency, cost=r.cost, lat_llrc=b.latency, cost_llrc=b.cost))
for k in out:
    for r in out[k]: print(k, r)
fig, ax = plt.subplots(1, 4, figsize=(14, 3))
ax[0].plot(Us, [r["dh"] for r in out["U"]], "o-"); ax[0].set_title("delayed hits vs U (Mbps)")
ax[1].plot(Us, [r["lat"] for r in out["U"]], "o-", label="Online-DRL"); ax[1].plot(Us, [r["lat_llrc"] for r in out["U"]], "s--", label="LL-RC"); ax[1].set_title("latency vs U"); ax[1].legend()
ax[2].plot(Ds, [r["dh"] for r in out["D"]], "o-"); ax[2].set_title("delayed hits vs D (Mbps)")
ax[3].plot(Ds, [r["cost"] for r in out["D"]], "o-", label="Online-DRL"); ax[3].plot(Ds, [r["cost_llrc"] for r in out["D"]], "s--", label="LL-RC"); ax[3].set_title("cost vs D"); ax[3].legend()
save(a, "exp03_bandwidth", out, fig)
