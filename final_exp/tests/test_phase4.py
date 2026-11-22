"""Tests for phase-4 EMA/churn/lazy changes."""
import numpy as np
import pytest

from sim.trace import synthetic_world
from sim.algos import rand_drl
from sim.regret_bound import regret_bound_terms, ema_tracking_error, optimal_ema_alpha


def test_download_time_ema_updates_and_is_logged():
    cat, tr = synthetic_world(N=8, T=200, seed=0)
    R = rand_drl(cat, tr, C=4, rng=np.random.default_rng(0), K=2,
                 log_steps=True, repair_mode="landlord",
                 download_time_mode="ema", dl_ema_alpha=0.2)
    log = R.extra["step_log"]
    updates = [r for r in log if r["q"] > 0 and r["dl_ema_before"] is not None]
    assert updates, "expected at least one in-flight observation"
    for r in updates:
        expected = (1 - 0.2) * r["dl_ema_before"] + 0.2 * r["dm"]
        assert r["dl_ema_after"] == pytest.approx(expected)
    assert "mean_dl_ema" in R.extra


def test_churn_charge_accounting():
    cat, tr = synthetic_world(N=10, T=200, seed=1)
    R = rand_drl(cat, tr, C=4, rng=np.random.default_rng(0), K=2,
                 log_steps=True, repair_mode="landlord", charge_churn="all")
    assert R.extra["churn_credit"] > 0
    assert R.extra["churn_events"] > 0
    # Churn charging is conservative: it can only reduce the download budget.
    R0 = rand_drl(cat, tr, C=4, rng=np.random.default_rng(0), K=2,
                  repair_mode="landlord", charge_churn="none")
    assert R.cost <= R0.cost + 1e-9


def test_ema_tracking_helpers():
    assert optimal_ema_alpha(sigma=1.0, drift=0.0) == 1.0
    alpha = optimal_ema_alpha(sigma=1.0, drift=1.0)
    assert alpha == pytest.approx(2.0 ** (2.0 / 3.0))
    err = ema_tracking_error(alpha, sigma=1.0, drift=1.0)
    assert err == pytest.approx(1.0 / alpha + np.sqrt(alpha))
    assert ema_tracking_error(0.0, sigma=1.0, drift=1.0) == float("inf")


def test_zeta_terms_include_download_time():
    cat, tr = synthetic_world(N=8, T=200, seed=2)
    R = rand_drl(cat, tr, C=4, rng=np.random.default_rng(0), K=2,
                 log_steps=True, repair_mode="landlord",
                 download_time_mode="ema")
    terms = regret_bound_terms(R.extra["step_log"], cat, C=4)
    assert "zeta_q_hat" in terms
    assert "zeta_dl_hat" in terms
    assert terms["zeta_hat"] == pytest.approx(
        terms["zeta_q_hat"] + terms["zeta_dl_hat"])