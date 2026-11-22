"""Running-time ("time complexity") of the algorithms themselves.

The paper's "complexity" statement is about the *competitive ratio* (exp13).  Here we measure
the computational effort of each algorithm, because the randomized variant trades a
linear-in-k ratio for an extra O(N) projection + O(K N) rounding step per request:

  Online-DRL / LL-RC  : O(1) per request + O(k log k) per admission (Landlord eviction sort)
  R3 only            : same as above, plus one random draw per miss epoch
  Randomized DRL     : O(N) projection (dual ascent on 4 multipliers) + O(K N) rounding
                        + O(N) repair per violating sample path

We report micro-seconds per request as a function of the catalogue size N and of the cache
capacity k, and fit a power law (exponent) for each algorithm.
"""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, ll_rc, rand_landlord, rand_drl
from sim.analysis import power_fit

a = parse(__doc__, C=12)
ALGOS = {"Online-DRL": lambda c, t, C, s: online_drl(c, t, C),
         "LL-RC": lambda c, t, C, s: ll_rc(c, t, C),
         "R3 only": lambda c, t, C, s: rand_landlord(c, t, C, np.random.default_rng(s)),
         "Randomized DRL": lambda c, t, C, s: rand_drl(c, t, C, np.random.default_rng(s), K=4)}

print("### running time vs catalogue size N (k fixed)")
resN = []
for N in ([20, 40, 80] if a.quick else [20, 40, 80, 160, 320]):
    row = dict(N=N)
    cat, tr = synthetic_world(N=N, T=a.T, seed=0)
    for name, fn in ALGOS.items():
        ts = []
        for s in range(min(3, a.seeds)):
            R = fn(cat, tr, a.C, s); ts.append(1e6 * R.time_s / len(tr))
        row[name] = ms(ts)
    resN.append(row); print(row)

print("\n### running time vs cache capacity k (N fixed)")
resC = []
for C in ([4, 8, 16] if a.quick else [4, 8, 16, 32, 64]):
    row = dict(C=C)
    cat, tr = synthetic_world(N=a.N, T=a.T, seed=0)
    for name, fn in ALGOS.items():
        ts = []
        for s in range(min(3, a.seeds)):
            R = fn(cat, tr, C, s); ts.append(1e6 * R.time_s / len(tr))
        row[name] = ms(ts)
    resC.append(row); print(row)

fits = {}
for name in ALGOS:
    fits[name] = dict(
        in_N=power_fit([r["N"] for r in resN], [r[name][0] for r in resN]),
        in_C=power_fit([r["C"] for r in resC], [r[name][0] for r in resC]))
    print(f"{name:16s} us/req ~ N^{fits[name]['in_N']['b']:.2f} (R2={fits[name]['in_N']['R2']:.2f}), "
          f"~ C^{fits[name]['in_C']['b']:.2f} (R2={fits[name]['in_C']['R2']:.2f})")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
for name in ALGOS:
    ax[0].plot([r["N"] for r in resN], [r[name][0] for r in resN], "o-", label=name)
    ax[1].plot([r["C"] for r in resC], [r[name][0] for r in resC], "o-", label=name)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("N (services)"); ax[0].set_ylabel("us / request")
ax[0].set_title("cost vs catalogue size")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("k (cache capacity)"); ax[1].set_ylabel("us / request")
ax[1].set_title("cost vs cache capacity")
for x in ax: x.legend(fontsize=7)
save(a, "exp14_time_complexity", dict(by_N=resN, by_C=resC, fits=fits), fig)
