"""Experiment 3 -- delayed hits vs the uplink/downlink asymmetry.

Paper: delayed hits fall as the uplink U rises and rise as the downlink D rises; a delayed
hit needs remaining download time <= forwarding latency (slow uplink, fast downlink).
"""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, ll_rc, rand_drl
from paper_claims import PAPER_EXP03

a = parse(__doc__, C=15)
Us, Ds = [20, 50, 100, 200, 400], [250, 500, 1000, 2000, 4000]
out = {"U": [], "D": [], "paper": PAPER_EXP03}
for U in Us:
    cat, tr = synthetic_world(N=a.N, T=a.T, seed=0, U=U, req_kbit=2000)
    r, b = online_drl(cat, tr, a.C), ll_rc(cat, tr, a.C)
    q = rand_drl(cat, tr, a.C, np.random.default_rng(0), K=4)
    out["U"].append(dict(U=U, dh=r.delayed_hits, dh_rand=q.delayed_hits,
                         lat=r.latency, cost=r.cost, lat_llrc=b.latency, cost_llrc=b.cost))
for D in Ds:
    cat, tr = synthetic_world(N=a.N, T=a.T, seed=0, D=D, req_kbit=2000)
    r, b = online_drl(cat, tr, a.C), ll_rc(cat, tr, a.C)
    q = rand_drl(cat, tr, a.C, np.random.default_rng(0), K=4)
    out["D"].append(dict(D=D, dh=r.delayed_hits, dh_rand=q.delayed_hits,
                         lat=r.latency, cost=r.cost, lat_llrc=b.latency, cost_llrc=b.cost))
for k in ("U", "D"):
    for r in out[k]:
        print(k, r)
print("paper:", PAPER_EXP03)
fig, ax = plt.subplots(1, 4, figsize=(14, 3))
ax[0].plot(Us, [r["dh"] for r in out["U"]], "o-", label="Online-DRL")
ax[0].plot(Us, [r["dh_rand"] for r in out["U"]], "s--", label="Randomized")
ax[0].set_title("delayed hits vs U (Mbps)"); ax[0].legend(fontsize=7)
ax[1].plot(Us, [r["lat"] for r in out["U"]], "o-", label="Online-DRL")
ax[1].plot(Us, [r["lat_llrc"] for r in out["U"]], "s--", label="LL-RC")
ax[1].set_title("latency vs U"); ax[1].legend(fontsize=7)
ax[2].plot(Ds, [r["dh"] for r in out["D"]], "o-", label="Online-DRL")
ax[2].plot(Ds, [r["dh_rand"] for r in out["D"]], "s--", label="Randomized")
ax[2].set_title("delayed hits vs D (Mbps)"); ax[2].legend(fontsize=7)
ax[3].plot(Ds, [r["cost"] for r in out["D"]], "o-", label="Online-DRL")
ax[3].plot(Ds, [r["cost_llrc"] for r in out["D"]], "s--", label="LL-RC")
ax[3].set_title("cost vs D"); ax[3].legend(fontsize=7)
save(a, "exp03_bandwidth", out, fig)