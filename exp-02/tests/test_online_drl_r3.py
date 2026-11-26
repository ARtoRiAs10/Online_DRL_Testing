"""Unit tests for online_drl_r3.py."""
from __future__ import annotations

import math

import numpy as np
import pytest

import online_drl_r3 as m


def test_llrc_equivalences():
    cat, tr = m.synthetic_world(N=15, T=1000, seed=0)
    ll = m.ll_rc(cat, tr, 5)
    r0 = m.drl_r3(cat, tr, 5, lam=0.0, seed=7)
    t0 = m.online_drl(cat, tr, 5, theta=0.0)
    key = lambda r: (r.latency, r.cost, r.hits, r.delayed_hits, r.downloads)
    assert key(ll) == key(r0) == key(t0)


def test_deterministic_seed_independent():
    cat, tr = m.synthetic_world(N=15, T=1000, seed=0)
    a = m.online_drl(cat, tr, 5)
    b = m.online_drl(cat, tr, 5)
    assert (a.latency, a.cost, a.hits) == (b.latency, b.cost, b.hits)


def test_threshold_mean():
    rng = np.random.default_rng(0)
    thr = m.RandThreshold(rng, lam=1.0)
    vals = np.array([thr.draw(0) for _ in range(200000)])
    assert abs(vals.mean() - 1.0 / (m.E - 1.0)) < 0.005


def test_ski_rental_ratio():
    # B=100, 20000 samples; the asymptotic ratio is e/(e-1).
    rng = np.random.default_rng(0)
    B, n_max, n_mc = 100.0, 400, 20000
    u = rng.random(n_mc)
    theta = B * np.log1p((m.E - 1.0) * u)
    buy = np.ceil(theta).astype(int)
    n = np.arange(1, n_max + 1)
    bought = n[None, :] >= buy[:, None]
    cost = np.where(bought, B + buy[:, None] - 1.0, n[None, :]).mean(0)
    ratio = cost / np.minimum(n, B)
    assert abs(ratio.max() - m.E / (m.E - 1.0)) < 0.02


def test_invariants_random_worlds():
    worlds = []
    for s in range(3):
        worlds.append(m.synthetic_world(N=15, T=800, seed=s))
    worlds.append(m.large_sequence(6, 6)[:2])
    worlds.append(m.cyclic_scan(8, 10)[:2])
    for cat, tr in worlds:
        C = min(8, max(1, cat.N // 2))
        for r in (m.online_drl(cat, tr, C),
                  m.drl_r3(cat, tr, C, seed=0),
                  m.drl_mg(cat, tr, C, seed=0),
                  m.drl_r3_mg(cat, tr, C, seed=0)):
            assert r.extra["violations"] == {}
            assert r.extra["marking_violations"] == 0


def test_budget():
    cat, tr = m.synthetic_world(N=15, T=1000, seed=0)
    bmax = 0.5 * m.online_drl(cat, tr, 5).cost
    r = m.drl_r3_mg(cat, tr, 5, seed=0, bmax=bmax)
    assert r.cost <= bmax + 1e-8
    assert r.extra["violations"] == {}


def test_marking_invariant():
    cat, tr = m.synthetic_world(N=20, T=2000, seed=1)
    r = m.drl_mg(cat, tr, 8, seed=0)
    assert r.extra["marking_violations"] == 0
    assert r.extra["phase_clears"] >= 0


def test_large_sequence_opt():
    cat, tr, opt = m.large_sequence(6, 6)
    assert opt == (6 + 2 * 6) * 1.0
    r = m.online_drl(cat, tr, 6)
    assert r.latency > 0


def test_matched_cost_curve_monotone():
    cat, tr = m.synthetic_world(N=15, T=1000, seed=0)
    thetas = [0.0, 1.0, 4.0, 16.0]
    costs = [m.online_drl(cat, tr, 5, theta=t).cost for t in thetas]
    assert all(a >= b for a, b in zip(costs, costs[1:]))


def test_soft_mg_invariants():
    cat, tr = m.synthetic_world(N=20, T=2000, seed=0)
    for tau in (0.1, 0.15, 0.2, 0.25):
        r = m.drl_mg(cat, tr, 10, seed=0, soft_tau=tau)
        assert r.extra["violations"] == {}
        assert r.extra["marking_violations"] == 0
        assert r.extra["phase_clears"] == 0


def test_soft_mg_zero_equals_hard():
    cat, tr = m.synthetic_world(N=20, T=2000, seed=0)
    a = m.drl_mg(cat, tr, 10, seed=0, soft_tau=0.0)
    b = m.drl_mg(cat, tr, 10, seed=0)
    assert (a.latency, a.cost) == (b.latency, b.cost)


def test_bayes_lam0_equals_llrc():
    cat, tr = m.synthetic_world(N=15, T=1000, seed=0)
    a = m.drl_bayes_r3(cat, tr, 5, lam=0.0, seed=0)
    b = m.ll_rc(cat, tr, 5)
    assert (a.latency, a.cost, a.downloads) == (b.latency, b.cost, b.downloads)


def test_result_summary_keys():
    cat, tr = m.synthetic_world(N=10, T=200, seed=0)
    s = m.online_drl(cat, tr, 3).summary()
    for k in ("latency", "cost", "hits", "delayed_hits", "downloads",
              "us_per_req", "evictions", "violations"):
        assert k in s