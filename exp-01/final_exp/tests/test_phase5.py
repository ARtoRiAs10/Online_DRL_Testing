"""Tests for the phase-5 theory-check changes."""
import numpy as np
import pytest

from sim.trace import synthetic_world
from sim.algos import rand_drl
from sim.projection import normalize
from sim.regret import ogd_run, surrogate_regret
from sim.regret_bound import regret_bound_terms


def test_ogd_run_returns_pre_update_states():
    rng = np.random.default_rng(0)
    N, T = 6, 30
    An = normalize(np.vstack([np.ones(N), rng.random((2, N))]),
                   np.r_[3.0, 1.0, 1.0])
    n_seq = rng.integers(0, N, T)
    g_seq = rng.uniform(0.5, 1.5, T) * 2.0
    eta = np.sqrt(N) / (2.0 * np.sqrt(T))
    P = ogd_run(n_seq, g_seq, An, eta)
    # The first played state is the initial state (zero), not a post-update state.
    assert np.allclose(P[0], 0.0)
    # Reconstruct the pre-update states and compare.
    p = np.zeros(N)
    for i in range(T):
        assert np.allclose(P[i], p, atol=1e-5)
        z = p.copy(); z[n_seq[i]] += eta * g_seq[i]
        # The projection is deterministic; call the same routine indirectly by
        # checking the next row is feasible and different from the previous.
        from sim.projection import project_dual
        p, _, _ = project_dual(z, An, None, np.linalg.norm(An, 2) ** 2)
    # The regret of the returned sequence satisfies the Proposition-2(i) bound.
    reg = surrogate_regret(n_seq, g_seq, P, An)
    bound = np.sqrt(N) * 2.0 * np.sqrt(T)
    assert reg <= bound + 1e-8


def test_step_log_has_phase5_fields():
    cat, tr = synthetic_world(N=8, T=40, seed=0)
    R = rand_drl(cat, tr, C=4, rng=np.random.default_rng(0), K=2,
                 log_steps=True, repair_mode="stateless",
                 grad_mode="realized", download_time_mode="ema")
    log = R.extra["step_log"]
    required = {
        "c_realized", "c_pred", "zeta_pred", "zeta", "delta_l1",
        "delta_round", "delta_churn", "delta_churn_all", "p_minus_xbar_post",
        "repair_evicted_request", "repair_refetch", "kstar_step",
    }
    assert required <= set(log[0])
    assert all(np.isfinite(r["c_realized"]) for r in log)
    assert all(r["delta_l1"] is not None for r in log)
    assert all(r["delta_round"] is not None for r in log)
    assert all(r["delta_churn"] in (0, 1) for r in log)
    assert all(r["delta_churn_all"] >= 0 for r in log)
    assert R.extra["mean_abs_zeta"] == pytest.approx(0.0, abs=1e-12)


def test_zeta_is_signed_and_nonzero_for_expectation_mode():
    cat, tr = synthetic_world(N=8, T=200, seed=1)
    R = rand_drl(cat, tr, C=4, rng=np.random.default_rng(0), K=2,
                 log_steps=True, repair_mode="stateless",
                 grad_mode="expectation", download_time_mode="ema")
    log = R.extra["step_log"]
    zeta = np.array([r["zeta"] for r in log], float)
    # Realized cost minus the EMA prediction; not identically zero in expectation mode.
    assert np.any(np.abs(zeta) > 1e-9)
    assert R.extra["zeta_count"] == len(log)
    assert R.extra["mean_abs_zeta"] >= 0.0


def test_regret_bound_terms_exposes_prop2_fields():
    cat, tr = synthetic_world(N=8, T=60, seed=2)
    R = rand_drl(cat, tr, C=4, rng=np.random.default_rng(0), K=2,
                 log_steps=True, repair_mode="landlord",
                 grad_mode="expectation", download_time_mode="ema")
    terms = regret_bound_terms(R.extra["step_log"], cat, C=4)
    for key in ("surrogate_regret_pre", "realized_regret_pre", "prop2_bound_1",
                "prop2_bound_2", "zeta_signed", "zeta_mae", "zeta_pred_mae",
                "delta_l1_mean", "delta_round_mean", "delta_churn_rate"):
        assert key in terms, key
    assert terms["prop2_bound_1"] > 0.0
    assert terms["prop2_bound_2"] >= terms["prop2_bound_1"]
    # The deterministic surrogate bound is the clean OGD statement.
    assert terms["surrogate_regret_pre"] <= terms["prop2_bound_1"] + 1e-8


def test_repair_refetch_counters_are_consistent():
    cat, tr = synthetic_world(N=10, T=100, seed=3)
    R = rand_drl(cat, tr, C=4, rng=np.random.default_rng(0), K=2,
                 log_steps=True, repair_mode="stateless")
    d = R.extra
    # Counters must be non-negative and no larger than the number of evictions.
    assert d["repair_evictions"] >= 0
    assert 0 <= d["repair_evicted_requests"] <= d["repair_evictions"]
    assert 0 <= d["repair_refetches"] <= d["repair_evicted_requests"]
    assert d["repair_refetch_cost"] >= 0.0
    # Every refetch in the step log must be a download.
    for r in R.extra["step_log"]:
        if r["repair_refetch"]:
            assert r["download_started"]


def test_delta_churn_matches_realized_transitions():
    cat, tr = synthetic_world(N=8, T=80, seed=4)
    R = rand_drl(cat, tr, C=4, rng=np.random.default_rng(0), K=2,
                 log_steps=True, repair_mode="stateless")
    log = R.extra["step_log"]
    assert sum(r["delta_churn"] for r in log) == R.extra["delta_churn_sum"]
    assert sum(r["delta_churn_all"] for r in log) == R.extra["delta_churn_all_sum"]
    # A 0->1 transition cannot happen for a service that was already cached.
    for r in log:
        if r["delta_churn"]:
            assert r["x_bar_n_star_prev"] == 0