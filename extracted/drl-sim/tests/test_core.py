import numpy as np, math
from sim.catalog import Catalog
from sim.trace import Trace, synthetic_world, large_sequence
from sim.engine import Engine, DetThreshold, RandThreshold
from sim.landlord import LandlordPolicy
from sim.algos import online_drl, ll_rc, rand_drl, rand_landlord
from sim.projection import project_dual, project_exact, normalize
from sim.opt import solve_opt


def tiny():
    cat = Catalog(l=np.array([5.0]), M=np.array([15.0]), a=np.array([[.3, .3, .3]]))
    return cat, Trace(np.array([0., 10, 20, 30]), np.zeros(4, int))


def test_ll_rc_latency_matches_hand_calc():
    cat, tr = tiny()
    r = ll_rc(cat, tr, C=1)
    # t=0 miss -> download (done t=15), t=10 in flight rem=5<=l -> delayed hit lat 5, t=20,30 hits
    assert r.latency == 10.0 and r.delayed_hits == 1 and r.hits == 2 and r.cost == 15.0


def test_opt_matches_hand_calc():
    cat, tr = tiny()
    o = solve_opt(cat, tr, C=1)
    assert abs(o["latency"] - 10.0) < 1e-6


def test_opt_budget_zero_forces_forwarding():
    cat, tr = tiny()
    assert abs(solve_opt(cat, tr, 1, bmax=0)["latency"] - 20.0) < 1e-6


def test_opt_is_lower_bound_on_online():
    cat, tr = synthetic_world(N=8, T=60, rate=0.5, seed=1)
    tr2 = Trace(tr.t, tr.n)
    d = online_drl(cat, tr2, C=3)
    o = solve_opt(cat, tr2, C=3, bmax=d.cost, time_limit=60)
    assert o["latency"] <= d.latency + 1e-6


def test_theta_zero_downloads_every_miss():
    cat, tr = synthetic_world(N=10, T=300, seed=2)
    a, b = ll_rc(cat, tr, 4), online_drl(cat, tr, 4, theta=1.0)
    assert a.cost >= b.cost


def test_landlord_feasible_always():
    cat, tr = synthetic_world(N=30, T=1500, seed=3, imbalance=3.0, cap_mult=4)
    r = online_drl(cat, tr, C=6)
    assert r.util.max() <= 1 + 1e-9


def test_projection_matches_qp():
    rng = np.random.default_rng(0)
    for _ in range(5):
        N = 25
        a = rng.uniform(.02, .5, (N, 3)); A = np.vstack([np.ones(N), a.T]); b = np.r_[5.0, 1, 1, 1]
        An = normalize(A, b); z = rng.uniform(-.3, 1.5, N)
        p, rho, _ = project_dual(z, An, tol=1e-9, max_iter=5000)
        q = project_exact(z, An)
        assert (An @ p).max() <= 1 + 1e-9
        assert np.linalg.norm(p - q) < 5e-3


def test_rand_drl_repair_gives_feasibility():
    cat, tr = synthetic_world(N=30, T=1500, seed=4, imbalance=3.0, cap_mult=4)
    r = rand_drl(cat, tr, 8, np.random.default_rng(0), K=4, repair=True)
    assert r.util.max() <= 1 + 1e-9


def test_random_threshold_cdf():
    rng = np.random.default_rng(0); s = RandThreshold(rng)
    x = np.array([s.draw(0) for _ in range(20000)])
    for q in (.2, .5, .8):  # F(x) = (e^x - 1)/(e-1)
        assert abs((x <= q).mean() - (math.exp(q) - 1) / (math.e - 1)) < .01


def test_large_sequence_opt_formula_vs_milp():
    cat, tr, opt = large_sequence(3, 2, M=3, Ml=6)
    o = solve_opt(cat, tr, C=3, time_limit=60)
    # the paper's (k+2n)l ignores an end-of-trace effect: in the last repetition OPT can download MS,
    # flush the cache (nothing is requested afterwards) and hit on MS's second request => OPT = (k+2n)l - l
    assert abs(o["latency"] - (opt - 1.0)) < 1e-6


# ---------------------------------------------------------------- revisions (N1, N2, N7)
def test_active_set_projection_equals_full_projection():
    """N7.1: projecting on the support of p is exact (the other services have p = 0 and
    contribute nothing to A p)."""
    from sim.projection import project_dual_active
    rng = np.random.default_rng(0)
    for _ in range(5):
        N = 30
        a = rng.uniform(.02, .5, (N, 3)); An = normalize(np.vstack([np.ones(N), a.T]), np.r_[5., 1, 1, 1])
        p0 = rng.uniform(0, 1, N); p0[p0 < .3] = 0.0
        z = p0.copy(); z[7] += .5
        active = np.flatnonzero(z > 1e-12)
        p_full, _, _ = project_dual(z, An, tol=1e-9, max_iter=5000)
        p_act, _, _ = project_dual_active(z, An, active, tol=1e-9, max_iter=5000)
        assert np.abs(p_full - p_act).max() < 1e-5
        assert (An @ p_act).max() <= 1 + 1e-9


def test_revised_policy_is_feasible_and_not_slower():
    """The revised policy (N1 + N2 + N7) must stay feasible and keep the resource limits."""
    cat, tr = synthetic_world(N=30, T=1500, seed=5, imbalance=3.0, cap_mult=4)
    r = rand_drl(cat, tr, 6, np.random.default_rng(0), K=4, download_prob_mode="learned",
                 grad_mode="expectation", proj_mode="active", rounding_mode="served", proj_tol=1e-4)
    assert r.util.max() <= 1 + 1e-9
    assert r.hits > 0 and r.downloads > 0


def test_learned_download_probability_suppresses_unprofitable_downloads():
    """N1: a service whose residency never pays for its download must end up with a small Q."""
    cat, tr = synthetic_world(N=20, T=1200, seed=6)
    r = rand_drl(cat, tr, 5, np.random.default_rng(0), K=4, download_prob_mode="learned")
    Q = r.extra["dprob_hist"]
    assert Q.min() <= 1.0 and (r.extra["residencies"] > 0)
    # the policy must still download (it is not degenerate)
    assert r.downloads > 0


def test_served_rounding_keeps_feasibility():
    """N7.2/N7.3: refreshing and repairing only the served path keeps the constraints."""
    cat, tr = synthetic_world(N=25, T=1000, seed=7, imbalance=4.0, cap_mult=3)
    r = rand_drl(cat, tr, 5, np.random.default_rng(0), K=4, rounding_mode="served", refresh_every=4)
    assert r.util.max() <= 1 + 1e-9
