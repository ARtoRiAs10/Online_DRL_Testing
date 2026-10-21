"""The paper's own synthetic experiment (Exps. 15-16): how the result depends on the Zipf skew
and on the request rate.

The paper's synthetic dataset is built from the unique services of Google dataset2 with a Zipf
popularity (parameter rho) and Poisson arrivals (parameter xi); Exps. 15 and 16 vary C together
with rho and with xi.  We cannot rebuild that dataset without the Google trace, but we can ask the
question the paper's synthetic experiments are about: **how does the Online-DRL vs LL-RC trade-off
(and the behaviour of the randomized algorithm) depend on the popularity skew and on the
inter-arrival time?**  If the answer depends strongly on those parameters, then no conclusion
about the paper's headline numbers can be drawn from a synthetic workload alone - the real traces
are needed.

Outputs results/exp17_paper_synthetic_params.json and .png.
"""
from common import *
from sim.trace import synthetic_world
from sim.algos import online_drl, ll_rc, rand_drl
from sim.metrics import pct

a = parse(__doc__, C=15)
RHOS = [1.0, 1.2, 1.6, 2.0]        # the paper's default is rho = 1.6
RATES = [0.25, 0.5, 2.0]            # requests / ms  (the paper's xi is in its own time unit)

rows = []
for rho in RHOS:
    for rate in RATES:
        acc = []
        for seed in range(a.seeds):
            cat, tr = synthetic_world(N=a.N, T=a.T, seed=seed, zipf=rho, rate=rate)
            d, b = online_drl(cat, tr, a.C), ll_rc(cat, tr, a.C)
            r = rand_drl(cat, tr, a.C, np.random.default_rng(seed), K=4)
            acc.append(dict(lat_drl_pct=pct(d.latency, b.latency), cost_drl_pct=pct(d.cost, b.cost),
                            lat_rand_pct=pct(r.latency, b.latency), cost_rand_pct=pct(r.cost, b.cost),
                            dh_drl=d.delayed_hits, dh_llrc=b.delayed_hits))
        row = dict(rho=rho, rate=rate, n_seeds=len(acc))
        for k in ("lat_drl_pct", "cost_drl_pct", "lat_rand_pct", "cost_rand_pct", "dh_drl"):
            v = [x[k] for x in acc]
            row[k] = (float(np.mean(v)), float(np.std(v)))
        rows.append(row)
        print(f"rho={rho:.1f} rate={rate:4.2f}:  Online-DRL vs LL-RC  latency {row['lat_drl_pct'][0]:+7.2f}% "
              f"(sd {row['lat_drl_pct'][1]:.2f})   cost {row['cost_drl_pct'][0]:+7.2f}%   |  "
              f"Randomized vs LL-RC latency {row['lat_rand_pct'][0]:+7.2f}%  cost {row['cost_rand_pct'][0]:+7.2f}%")

pos = sum(1 for r in rows if r["lat_drl_pct"][0] > 0)
print(f"\nOnline-DRL has HIGHER latency than LL-RC in {pos}/{len(rows)} configurations "
      f"(paper, on Google traces: always, 0.016 %-7.41 %)")
print("mean latency difference of Online-DRL vs LL-RC over the sweep: "
      f"{np.mean([r['lat_drl_pct'][0] for r in rows]):+.2f} %")
print("mean cost difference of Online-DRL vs LL-RC over the sweep: "
      f"{np.mean([r['cost_drl_pct'][0] for r in rows]):+.2f} % (paper on Google: -77.57 % .. -92.54 %)")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
lbl = [f"r={r['rho']}\n$\\lambda$={r['rate']}" for r in rows]
xs = np.arange(len(rows))
ax[0].bar(xs, [r["lat_drl_pct"][0] for r in rows],
          yerr=[r["lat_drl_pct"][1] for r in rows],
          color=["#1b6b3a" if r["lat_drl_pct"][0] < 0 else "#9c2b2b" for r in rows])
ax[0].axhline(0, color="k", lw=.8)
ax[0].set_title("Online-DRL vs LL-RC: latency (%)\n(green = Online-DRL faster)", fontsize=9)
ax[1].bar(xs, [r["cost_drl_pct"][0] for r in rows],
          yerr=[r["cost_drl_pct"][1] for r in rows], color="#1f3b73")
ax[1].axhline(0, color="k", lw=.8)
ax[1].set_title("Online-DRL vs LL-RC: cost (%)", fontsize=9)
ax[2].bar(xs, [r["lat_rand_pct"][0] for r in rows], color="#1b6b3a")
ax[2].axhline(0, color="k", lw=.8)
ax[2].set_title("Randomized DRL vs LL-RC: latency (%)", fontsize=9)
for x in ax:
    x.set_xticks(xs); x.set_xticklabels(lbl, fontsize=6.5); x.grid(alpha=.3, axis="y")
save(a, "exp17_paper_synthetic_params", dict(rows=rows, n_positive=pos, n_configs=len(rows),
                                             C=a.C, N=a.N, T=a.T), fig)
