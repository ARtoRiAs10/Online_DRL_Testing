"""Tests for phase-3 utilities and trace generators."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from sim.trace import synthetic_world, cyclic_scan, deterministic_eviction_trace
from sim.landlord import LandlordPolicy
from sim.engine import Engine, DetThreshold
from common import mean_ci, paired_ci


def test_cyclic_scan_shape_and_capacity():
    k, n_reps = 8, 3
    cat, tr = cyclic_scan(k, n_reps)
    assert cat.N == k + 1
    assert len(tr) == (k + 1) * n_reps
    assert np.array_equal(tr.n[: k + 1], np.arange(k + 1))
    # k services fill each resource exactly; k+1 do not.
    assert np.isclose(cat.a[:k, 0].sum(), 1.0)
    assert cat.a[: k + 1, 0].sum() > 1.0


def test_landlord_logs_evictions():
    cat, tr = synthetic_world(N=12, T=200, seed=0)
    pol = LandlordPolicy(cat, C=5)
    Engine(cat, pol, DetThreshold(1.0), rng=np.random.default_rng(0)).run(tr)
    assert pol.step == len(tr)
    assert len(pol.evictions) == pol.n_evict
    assert all(0 <= f < cat.N for _, f in pol.evictions)


def test_deterministic_eviction_trace_is_fixed_and_valid():
    cat, base = synthetic_world(N=12, T=200, seed=1)
    tr1, base1, ev1 = deterministic_eviction_trace(cat, base, C=5)
    tr2, base2, ev2 = deterministic_eviction_trace(cat, base, C=5)
    assert len(tr1) == len(tr2)
    assert np.array_equal(tr1.n, tr2.n)
    assert np.array_equal(tr1.t, tr2.t)
    assert len(ev1) == len(ev2)
    assert len(tr1) >= len(base)
    assert tr1.n.min() >= 0 and tr1.n.max() < cat.N
    assert np.all(np.diff(tr1.t) > 0)


def test_mean_and_paired_ci():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 2.0, 4.0, 4.0])
    m, s, half = mean_ci(a)
    assert m == 2.5
    assert s > 0 and half > 0
    d = paired_ci(a, b)
    assert d["mean"] == -0.5
    assert d["ci_low"] < d["mean"] < d["ci_high"]
    assert 0.0 <= d["win_rate"] <= 1.0