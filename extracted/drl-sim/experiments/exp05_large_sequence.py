"""Theorem 1 / G4: ratio on the paper's adversarial Large Sequence as k grows.
Compares deterministic Online-DRL with R3-only (random threshold + Landlord) and the full
randomized DRL (R1+R2+R3).  OPT = (k+2n)l (paper eq. 17)."""
from common import *
from sim.trace import large_sequence
from sim.algos import online_drl, ll_rc, rand_landlord, rand_drl

a = parse(__doc__)
ks = [3, 6, 9, 12] if a.quick else [3, 6, 9, 12, 15, 18, 21, 24]
rows = []
for k in ks:
    n = 2 * k
    cat, tr, opt = large_sequence(k, n)
    row = dict(k=k, n=n, opt=opt, det=online_drl(cat, tr, k).latency / opt, llrc=ll_rc(cat, tr, k).latency / opt)
    rl = [rand_landlord(cat, tr, k, np.random.default_rng(s)).latency / opt for s in range(a.seeds)]
    rf = [rand_drl(cat, tr, k, np.random.default_rng(s), K=4).latency / opt for s in range(a.seeds)]
    row["rand_landlord"], row["rand_landlord_sd"] = ms(rl); row["rand_drl"], row["rand_drl_sd"] = ms(rf)
    rows.append(row); print({k_: round(v, 3) if isinstance(v, float) else v for k_, v in row.items()})


def fit(x, y):
    x, y = np.array(x, float), np.array(y, float); out = {}
    for name, f in (("linear_k", x), ("log_k", np.log(x))):
        A = np.c_[f, np.ones_like(f)]; c, *_ = np.linalg.lstsq(A, y, rcond=None)
        ss = ((y - A @ c) ** 2).sum(); out[name] = dict(slope=float(c[0]), R2=float(1 - ss / ((y - y.mean()) ** 2).sum()))
    return out


fits = {m: fit([r["k"] for r in rows], [r[m] for r in rows]) for m in ("det", "llrc", "rand_landlord", "rand_drl")}
for m, f in fits.items(): print(m, f)
print("NOTE: this is one adversarial family; it shows scaling of this sequence only, not a general bound.")
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for m in ("det", "llrc", "rand_landlord", "rand_drl"): ax.plot(ks, [r[m] for r in rows], "o-", label=m)
ax.set_xlabel("k"); ax.set_ylabel("ALG / OPT"); ax.legend(); ax.set_title("Large Sequence")
save(a, "exp05_large_sequence", dict(rows=rows, fits=fits), fig)
