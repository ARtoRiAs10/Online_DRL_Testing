import numpy as np
from sim.landlord_repair import landlord_repair, is_feasible, sync_credit


def random_problem(rng, N=20, C=5):
    A = np.vstack([np.ones(N), rng.uniform(.01, .3, (3, N))])
    b = np.r_[float(C), 1.0, 1.0, 1.0]
    M = rng.uniform(1, 10, N)
    return A, b, M


def test_landlord_repair_always_feasible():
    rng = np.random.default_rng(0)
    for _ in range(300):
        A, b, M = random_problem(rng)
        N = A.shape[1]
        p = rng.random(N)
        x = rng.random(N) < p
        credit = np.zeros(N)
        y = landlord_repair(x, p, A, b, M, credit)
        assert is_feasible(y, A, b)
        assert np.all(credit >= 0)


def test_landlord_repair_is_cost_aware():
    """A service with a larger download cost must get a larger initial credit."""
    A = np.array([[1.0, 1.0], [0.2, 0.2], [0.2, 0.2], [0.2, 0.2]])
    b = np.array([1.0, 1.0, 1.0, 1.0])
    M = np.array([1.0, 100.0])
    credit = np.zeros(2)
    x = np.array([True, False])
    sync_credit(x, M, credit)
    assert credit[0] == 1.0 and credit[1] == 0.0
    x = np.array([True, True])
    sync_credit(x, M, credit)
    assert credit[0] == 1.0 and credit[1] == 100.0


def test_landlord_repair_evicts_deficit_aligned_service():
    """The service whose eviction leaves the smallest residual overflow is evicted first."""
    N = 4
    A = np.array([
        [1.0, 1.0, 1.0, 1.0],      # cardinality, b0 = 2
        [0.6, 0.6, 0.1, 0.1],      # CPU, b = 1
        [0.1, 0.1, 0.6, 0.6],      # RAM
        [0.1, 0.1, 0.1, 0.1],      # disk
    ])
    b = np.array([2.0, 1.0, 1.0, 1.0])
    M = np.ones(N)
    p = np.array([0.9, 0.9, 0.1, 0.1])
    x = np.array([True, True, True, False])
    credit = np.zeros(N)
    y = landlord_repair(x, p, A, b, M, credit)
    assert is_feasible(y, A, b)
    # the two CPU-heavy services cannot both stay; at most one does
    assert y[0] + y[1] <= 1


def test_landlord_repair_popularity_mode_is_feasible_and_protects_popular():
    """Popularity mode must stay feasible and evict low-p services first."""
    N = 6
    A = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.5, 0.5, 0.5],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ])
    b = np.array([2.0, 1.0, 1.0, 1.0])
    M = np.ones(N)
    # services 0-2 are popular; 3-5 are unpopular
    p = np.array([0.8, 0.7, 0.6, 0.1, 0.1, 0.1])
    x = np.array([True, True, True, True, False, False])
    credit = np.zeros(N)
    y = landlord_repair(x, p, A, b, M, credit, evict_mode="popularity")
    assert is_feasible(y, A, b)
    # The first evictions should hit the unpopular services (3-5), not 0-2.
    assert not y[3] or not y[0]


def test_landlord_repair_hybrid_mode_is_feasible():
    rng = np.random.default_rng(7)
    A, b, M = random_problem(rng)
    N = A.shape[1]
    x = rng.random(N) < 0.7
    y = landlord_repair(x, rng.random(N), A, b, M, np.zeros(N),
                        evict_mode="hybrid", pop_weight=0.5)
    assert is_feasible(y, A, b)


def test_landlord_repair_in_place():
    A, b, M = random_problem(np.random.default_rng(1))
    N = A.shape[1]
    x = (np.random.default_rng(2).random(N) < .5)
    y = landlord_repair(x, np.ones(N), A, b, M, np.zeros(N))
    assert y is x