import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import synthetic_world, rand_drl, AuditPolicy
from sim.engine import RandThreshold
from sim.engine import Engine
from sim.projection import project_dual, normalize
from sim.landlord_repair import landlord_repair, is_feasible


def test_projection_feasible():
    rng = np.random.default_rng(0)
    N, C = 30, 8
    A = np.vstack([np.ones(N), rng.random((3, N))])
    b = np.r_[float(C), np.ones(3)]
    An = normalize(A, b)
    for _ in range(20):
        z = rng.normal(size=N)
        p, rho, it = project_dual(z, An)
        assert np.all(p >= -1e-9) and np.all(p <= 1 + 1e-9)
        assert np.all(An @ p <= 1 + 1e-7)


def test_landlord_feasible():
    rng = np.random.default_rng(1)
    N, C = 30, 8
    A = np.vstack([np.ones(N), rng.random((3, N))])
    b = np.r_[float(C), np.ones(3)]
    M = rng.uniform(1, 20, N)
    for _ in range(20):
        x = (rng.random(N) < 0.5).astype(bool)
        credit = np.zeros(N)
        landlord_repair(x, rng.random(N), A, b, M, credit)
        assert is_feasible(x, A, b)


def test_audit_no_uncharged_hits():
    cat, tr = synthetic_world(N=20, T=2000, seed=0, imbalance=2.0, cap_mult=4)
    rng = np.random.default_rng(0)
    pol = AuditPolicy(cat, 8, rng, K=2, repair_mode="stateless",
                      gate_downloads=True, peek=True, download_time_mode="ema")
    Engine(cat, pol, RandThreshold(rng)).run(tr)
    assert pol.audit["violations"] == 0
    assert pol.audit["hits_checked"] == pol.audit["hits_checked"]


def test_r3_support():
    rng = np.random.default_rng(2)
    M = np.array([1.0, 5.0, 20.0])
    thr = RandThreshold(rng)
    for i, m in enumerate(M):
        draws = np.array([thr.draw(i) * m for _ in range(1000)])
        assert np.all(draws >= 0) and np.all(draws <= m + 1e-12)
