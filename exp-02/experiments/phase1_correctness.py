"""Phase 1 -- correctness tests (preconditions 2 and 3).

Checks the H0 reduction, the threshold mean, the pure ski-rental ratio, the
resource/cardinality/budget invariants and the marking invariant.
"""
from __future__ import annotations

import math

import numpy as np

from common import markdown_table, save_fig, save_json
import online_drl_r3 as m


def ski_rental_ratio(B=100.0, n_max=400, n_mc=20000, seed=0):
    """Worst-case randomized ski-rental ratio over horizon n.

    Rent costs 1 per step, buying costs ``B``.  OPT(n) = min(n, B).
    The randomized threshold is ``B*ln(1+u(e-1))``; the deterministic
    threshold is ``B``.
    """
    rng = np.random.default_rng(seed)
    u = rng.random(n_mc)
    theta = B * np.log1p((m.E - 1.0) * u)
    # cost if the buy happens at the first step with n >= theta
    buy_step = np.ceil(theta).astype(int)
    n = np.arange(1, n_max + 1)
    bought = n[None, :] >= buy_step[:, None]
    cost = np.where(bought, B + buy_step[:, None] - 1.0, n[None, :]).mean(0)
    opt = np.minimum(n, B)
    ratio = cost / opt
    det_step = int(math.ceil(B))
    det = np.where(n >= det_step, B + det_step - 1.0, n)
    det_ratio = det / opt
    return dict(random_worst=float(ratio.max()),
                random_argmax_n=int(n[int(ratio.argmax())]),
                deterministic_worst=float(det_ratio.max()),
                theory_random=float(m.E / (m.E - 1.0)),
                theory_deterministic=2.0)


def threshold_mean(n=400000, seed=0):
    rng = np.random.default_rng(seed)
    draws = m.RandThreshold(rng, lam=1.0)
    vals = np.array([draws.draw(0) for _ in range(n)])
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / math.sqrt(n))
    return dict(mean=mean, theory=1.0 / (m.E - 1.0), se=se,
                ci_low=mean - 1.96 * se, ci_high=mean + 1.96 * se,
                n=n)


def main():
    checks = {}

    # -- H0: deterministic seed independence -----------------------------
    cat, tr = m.synthetic_world(N=20, T=2000, seed=0)
    det = [m.online_drl(cat, tr, 10) for _ in range(4)]
    same = all((r.latency, r.cost, r.hits, r.delayed_hits, r.downloads)
               == (det[0].latency, det[0].cost, det[0].hits,
                   det[0].delayed_hits, det[0].downloads) for r in det)
    checks["h0_deterministic_seed_independent"] = bool(same)

    # -- lam=0 equals LL-RC ----------------------------------------------
    ll = m.ll_rc(cat, tr, 10)
    r0 = m.drl_r3(cat, tr, 10, lam=0.0, seed=12345)
    t0 = m.online_drl(cat, tr, 10, theta=0.0)
    checks["lam0_equals_llrc"] = bool(
        (r0.latency, r0.cost, r0.hits, r0.downloads)
        == (ll.latency, ll.cost, ll.hits, ll.downloads))
    checks["theta0_equals_lam0"] = bool(
        (t0.latency, t0.cost, t0.hits, t0.downloads)
        == (r0.latency, r0.cost, r0.hits, r0.downloads))
    checks["lam0_values"] = dict(llrc=[ll.latency, ll.cost, ll.hits, ll.downloads],
                                 r3=[r0.latency, r0.cost, r0.hits, r0.downloads],
                                 det=[t0.latency, t0.cost, t0.hits, t0.downloads])

    # -- threshold mean ---------------------------------------------------
    checks["threshold_mean"] = threshold_mean()

    # -- ski rental -------------------------------------------------------
    checks["ski_rental"] = ski_rental_ratio()

    # -- invariants -------------------------------------------------------
    inv_rows = []
    worlds = []
    for s in range(5):
        worlds.append((f"zipf_s{s}",) + m.synthetic_world(N=20, T=2000, seed=s))
    for k in (6, 12):
        worlds.append((f"large_k{k}",) + m.large_sequence(k, k)[:2])
    for k in (4, 8):
        worlds.append((f"cyclic_k{k}",) + m.cyclic_scan(k, 10)[:2])
    for name, c, t in worlds:
        C = 10 if not name.startswith("large") else (6 if "k6" in name else 12)
        if name.startswith("cyclic"):
            C = 4 if "k4" in name else 8
        for vname, fn in [
            ("online_drl", lambda: m.online_drl(c, t, C)),
            ("ll_rc", lambda: m.ll_rc(c, t, C)),
            ("r3", lambda: m.drl_r3(c, t, C, seed=0)),
            ("mg", lambda: m.drl_mg(c, t, C, seed=0)),
            ("r3mg", lambda: m.drl_r3_mg(c, t, C, seed=0)),
        ]:
            r = fn()
            v = r.extra.get("violations", {})
            inv_rows.append(dict(world=name, variant=vname,
                                 violations=v,
                                 marking_violations=r.extra.get("marking_violations", 0),
                                 evictions=r.extra.get("evictions", 0),
                                 phase_clears=r.extra.get("phase_clears", 0)))
    checks["invariant_rows"] = inv_rows
    checks["invariants_hold"] = bool(all(
        not row["violations"] and row["marking_violations"] == 0
        for row in inv_rows))

    # -- budget -----------------------------------------------------------
    bmax = 0.5 * m.online_drl(cat, tr, 10).cost
    rb = m.drl_r3_mg(cat, tr, 10, lam=1.0, seed=0, bmax=bmax)
    checks["budget"] = dict(bmax=float(bmax), cost=float(rb.cost),
                            holds=bool(rb.cost <= bmax + 1e-8),
                            violations=rb.extra.get("violations", {}))

    checks["preconditions"] = dict(
        baseline_reproduced=False,  # phase0
        reduction=bool(checks["h0_deterministic_seed_independent"]
                       and checks["lam0_equals_llrc"]
                       and checks["theta0_equals_lam0"]),
        invariants=bool(checks["invariants_hold"] and checks["budget"]["holds"]),
    )
    save_json("phase1_correctness", checks)

    # markdown
    lines = ["# Phase 1 -- correctness", "",
             f"H0 deterministic seed-independent: **{checks['h0_deterministic_seed_independent']}**",
             f"lam=0 equals LL-RC: **{checks['lam0_equals_llrc']}**",
             f"theta=0 equals lam=0: **{checks['theta0_equals_lam0']}**",
             f"invariants hold: **{checks['invariants_hold']}**",
             f"budget holds: **{checks['budget']['holds']}** "
             f"(cost {checks['budget']['cost']:.3f} <= Bmax {checks['budget']['bmax']:.3f})",
             "", "## Threshold mean", "",
             f"measured {checks['threshold_mean']['mean']:.6f}, "
             f"theory {checks['threshold_mean']['theory']:.6f}, "
             f"95% CI [{checks['threshold_mean']['ci_low']:.6f}, "
             f"{checks['threshold_mean']['ci_high']:.6f}]", "",
             "## Ski-rental ratio", "",
             f"randomized worst {checks['ski_rental']['random_worst']:.4f} "
             f"(theory {checks['ski_rental']['theory_random']:.4f}) at n="
             f"{checks['ski_rental']['random_argmax_n']}; deterministic "
             f"{checks['ski_rental']['deterministic_worst']:.4f}", ""]
    bad = [r for r in inv_rows if r["violations"] or r["marking_violations"]]
    lines += [f"Invariant rows: {len(inv_rows)}; violations: {len(bad)}", ""]
    if bad:
        lines.append(markdown_table(["world", "variant", "violations", "marking"],
                                    [[r["world"], r["variant"], r["violations"],
                                      r["marking_violations"]] for r in bad]))
    with open("docs/phase1_correctness.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote docs/phase1_correctness.md")
    print({k: checks[k] for k in ("h0_deterministic_seed_independent",
                                  "lam0_equals_llrc", "theta0_equals_lam0",
                                  "invariants_hold")})


if __name__ == "__main__":
    main()