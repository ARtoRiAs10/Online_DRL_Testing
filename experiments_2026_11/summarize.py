"""Print the headline tables from results/*.json."""
import json, os, glob
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def load(name):
    with open(os.path.join(R, name + ".json")) as f:
        return json.load(f)

print("=== Phase A audit")
a = load("phase_a_audit")
for k, s in a["summary"].items():
    print(f"{k:24s} cost={s['cost'][0]:.0f} strict={s['strict_cost'][0]:.0f} "
          f"charge_all_upper={s['charge_all_upper'][0]:.0f} viol={s['violations'][0]:.2f}")
print("\n=== Phase B frontier (randomized)")
b = load("phase_b_frontier")
for k, s in b["summary"].items():
    if "/rand/" in k:
        print(f"{k:34s} lat={s['latency'][0]:10.0f} cost={s['cost'][0]:10.0f}")
print("interpolated paired latency:")
for k, s in b["interp_summary"].items():
    print(f"  {k:24s} n={s['n']:3d} diff={s['mean']:+.0f} ±{s['ci_half']:.0f} win={s['win_rate']:.2f}")
print("\n=== Phase B equal cost")
e = load("phase_b_equal")
for k, s in e["summary"].items():
    print(f"{k:24s} dlat={s['paired_lat']['mean']:+.0f} ±{s['paired_lat']['ci_half']:.0f} "
          f"dcost={s['paired_cost']['mean']:+.0f} ±{s['paired_cost']['ci_half']:.0f}")
print("\n=== Phase C")
c = load("phase_c_horizon_opt")
for r in c["rows"]:
    if r["kind"] == "prefix":
        print(f"prefix {r['regime']:8s} T={r['T']:5d} dlat={r['paired_lat']['mean']:+.0f} "
              f"±{r['paired_lat']['ci_half']:.0f}")
for r in c["opt_rows"]:
    print(f"OPT k={r['k']:2d} M/l={r['M_over_l']:g} closed={r['closed']:.0f} "
          f"milp={r['milp']:.0f} det_ratio={r['det_ratio']:.3f} closed_ratio={r['closed_ratio']:.3f}")
print("\n=== Phase D")
d = load("phase_d_adversarial")
for fam, s in d["summary"].items():
    print(fam)
    for name, q in s.items():
        if name in ("online_drl", "r1r2_stateless", "r1r2r3_stateless", "r1r2r3_landlord", "r3_only"):
            print(f"  {name:24s} lat_ratio med={q['lat_ratio']['median']:.3f} "
                  f"cost_ratio med={q['cost_ratio']['median']:.3f}")
print("\n=== Phase E")
e = load("phase_e_real_data")
for r in e["rows"]:
    print(f"{r['trace']} {r['regime']:16s} M<l={r['frac_M_lt_l']:.3f} "
          f"det={r['det_lat']:.0f}/{r['det_cost']:.0f} "
          f"stateless={r['stateless_lat'][0]:.0f}/{r['stateless_cost'][0]:.0f} "
          f"landlord={r['landlord_lat'][0]:.0f}/{r['landlord_cost'][0]:.0f}")
print("\n=== Phase F")
f = load("phase_f_theory")
for mode, s in f["summary"].items():
    print(mode, "zeta_mae", s["zeta_mae"], "delta_round", s["delta_round"])
print("R3", f["r3_check"])
