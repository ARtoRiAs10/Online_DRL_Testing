"""Tests for the phase-1 logging/top-up changes and the phase-2 bound utilities."""
import numpy as np
import pytest

from sim.trace import synthetic_world
from sim.algos import rand_drl
from sim.rand_drl import FracPolicy
from sim.engine import Engine, DetThreshold
from sim.regret_bound import (
    catalog_constraints,
    g3b_comparator_lp,
    regret_bound_terms,
    summarize_step_log,
)


def test_q_ema_gradient_uses_previous_q():
    cat, tr = synthetic_world(N=8, T=20, seed=0)
    pol = FracPolicy(cat, C=4, rng=np.random.default_rng(0), K=2,
                     grad_mode="expectation", q_ema_alpha=0.5, log_steps=True)
    res = Engine(cat, pol, DetThreshold(1.0), rng=np.random.default_rng(1)).run(tr)
    log = res.extra["step_log"]
    assert len(log) == len(tr)
    for r in log:
        n = r["n"]
        qe = r["q_ema_before"]
        l = cat.l[n]
        M = cat.M[n]
        expected_g = (1.0 - qe) * l + qe * min(l, M / 2.0)
        assert r["grad"] == pytest.approx(expected_g)
        # Q is updated only after the gradient, so the logged after-value is the EMA
        # of the old value and the current q.
        assert r["q_ema_after"] == pytest.approx(
            0.5 * r["q_ema_before"] + 0.5 * r["q"])


def test_step_log_contains_required_fields():
    cat, tr = synthetic_world(N=10, T=30, seed=1)
    R = rand_drl(cat, tr, C=4, rng=np.random.default_rng(0), K=2, log_steps=True)
    log = R.extra["step_log"]
    required = {
        "t", "n", "q", "dm", "p_sum", "resource_load_norm", "resource_load_raw",
        "p_n_star_prev", "x_bar_n_star_prev", "p_minus_xbar", "q_ema_before",
        "q_ema_after", "grad", "p_n_star_after", "outcome", "hit_rank",
        "download_started", "c_t", "repair_evicted_fraction", "p_sum_after",
    }
    assert required <= set(log[0])
    assert len(log[0]["resource_load_norm"]) == 3
    assert len(log[0]["resource_load_raw"]) == 3
    assert all(np.isfinite(log[i]["p_sum"]) for i in range(len(log)))
    assert all(log[i]["repair_evicted_fraction"] is not None for i in range(len(log)))


def test_topup_keeps_sample_paths_feasible():
    cat, tr = synthetic_world(N=12, T=100, seed=2)
    pol = FracPolicy(cat, C=5, rng=np.random.default_rng(0), K=3,
                     repair_mode="landlord", topup=True, log_steps=True)
    Engine(cat, pol, DetThreshold(1.0), rng=np.random.default_rng(1)).run(tr)
    # Every refreshed path must satisfy all constraints, including the cardinality row.
    assert np.all(pol.S @ pol.An.T <= 1.0 + 1e-7)
    assert pol.st["topup_added"] >= 0
    assert "topup_added" in pol.diagnostics()


def test_lazy_repair_keeps_previous_services_and_is_feasible():
    cat, tr = synthetic_world(N=10, T=80, seed=5)
    pol = FracPolicy(cat, C=4, rng=np.random.default_rng(0), K=2,
                     repair_mode="landlord", lazy=True, log_steps=True)
    Engine(cat, pol, DetThreshold(1.0), rng=np.random.default_rng(1)).run(tr)
    # Lazy repair starts from the previous path and only evicts when the union is
    # infeasible; the final paths must still satisfy every constraint.
    assert np.all(pol.S @ pol.An.T <= 1.0 + 1e-7)
    assert pol.st["refreshes"] > 0
    assert all(r["repair_evicted_fraction"] is not None
               for r in pol.step_log)


def test_g3b_comparator_lp_is_feasible_and_matches_objective():
    cat, _ = synthetic_world(N=6, T=10, seed=3)
    W = np.linspace(0.0, 2.0, cat.N)
    P, obj = g3b_comparator_lp(cat, C=3, W=W)
    An = catalog_constraints(cat, 3)
    assert np.all(An @ P <= 1.0 + 1e-7)
    assert np.all(P >= -1e-12) and np.all(P <= 1.0 + 1e-12)
    assert obj == pytest.approx(float(np.dot(cat.M - W, P)))


def test_regret_bound_terms_on_logged_run():
    cat, tr = synthetic_world(N=8, T=50, seed=4)
    R = rand_drl(cat, tr, C=3, rng=np.random.default_rng(0), K=2, log_steps=True)
    terms = regret_bound_terms(R.extra["step_log"], cat, C=3)
    assert terms["T"] == len(tr)
    assert terms["N"] == cat.N
    assert np.isfinite(terms["regret_bound_rhs"])
    assert terms["regret_bound_rhs"] >= terms["comparator_objective"]
    assert terms["sum_F_tilde_Pt"] == pytest.approx(
        sum(r["grad"] * r["p_n_star_prev"] for r in R.extra["step_log"]))
    summary = summarize_step_log(R.extra["step_log"])
    assert summary["n_steps"] == len(tr)
    assert summary["outcomes"]["hit"] + summary["outcomes"]["delayed"] + \
        summary["outcomes"]["forward"] == len(tr)