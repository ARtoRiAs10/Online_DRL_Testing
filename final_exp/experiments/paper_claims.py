"""Reference values from the original paper.

Source: S. Deka and R. Sukapuram, "Edge Service Caching With Delayed Hits: Minimizing
Latency and Reducing Cost," IEEE Trans. Cloud Computing 13(4), 2025.

The paper evaluates Online-DRL on the Google cluster traces 2011 (dataset0..dataset10,
dataset2 in the paper) and on SHARCNet.  The numbers below are the headline claims used
for the comparison in ``final_exp``; the SHARCNet trace is not available in this
workspace, so the real-trace check uses the Google traces.
"""

# exp01: Online-DRL vs LL-RC on the Google traces.
PAPER_EXP01_LATENCY_PCT = (0.016, 7.41)      # Online-DRL is slightly slower
PAPER_EXP01_COST_PCT = (-92.54, -77.57)      # ... but much cheaper

# exp03: delayed hits vs uplink/downlink.
PAPER_EXP03 = ("delayed hits decrease as the uplink U increases and increase as the "
               "downlink D increases")

# exp05 / exp13: Theorem 1.
PAPER_THEOREM1 = {
    "O": "gamma in O(k (1 + 3 beta_up M_x))",
    "Omega": "gamma in Omega(k (1 + 3 beta_down M_min))",
    "OPT_large_sequence": "OPT = (k + 2 n) l",
}

# exp06: rent-or-buy.
PAPER_SKI_RENTAL = {"deterministic": 2.0, "randomized": 1.5819767068693265}  # e/(e-1)

# exp08 / exp15: goals G1/G2.
PAPER_G1 = "the repaired state satisfies the CPU/RAM/disk and cardinality constraints a.s."
PAPER_G2 = "E[sum u_{n,t} M_n] <= B_max"

# exp09: Proposition 2.
PAPER_REGRET_BOUND = "sqrt(N) l_max sqrt(T)"

# exp10 / exp13: expected ratio.
PAPER_G4 = "gamma_R = max_I E[ALG(I)]/OPT(I) below the deterministic gamma"

# exp11 / exp12: G5 slot reductions.
PAPER_G5 = ("a safe reduction needs phi(a_n) >= max_r a_{n,r}; unsafe (simplex) weights "
            "violate the true resource constraints")

# exp16: real traces.
PAPER_REAL_TRACES = {
    "datasets": ["Google cluster traces 2011 (dataset2)", "SHARCNet"],
    "google_note": "the paper's Google dataset2 has 1000 requests / 30 services; this "
                   "workspace regenerates it from clusterdata-2011-2 with data/prepare_google.py",
}

# exp17: synthetic dataset of the paper.
PAPER_EXP17 = {"rho_default": 1.6, "xi_default": 3e6,
               "note": "unique services of Google dataset2 + Zipf popularity + Poisson arrivals"}


def in_range(x, rng):
    return bool(rng[0] <= x <= rng[1])