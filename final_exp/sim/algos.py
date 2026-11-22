"""One-line constructors for every algorithm compared in the experiments."""
from __future__ import annotations
import numpy as np
from .engine import Engine, DetThreshold, RandThreshold
from .landlord import LandlordPolicy
from .rand_drl import FracPolicy


def online_drl(cat, trace, C, theta=1.0, bmax=None):
    """Deterministic Online-DRL (Algorithms 1+2).  theta=0 gives LL-RC."""
    return Engine(cat, LandlordPolicy(cat, C), DetThreshold(theta), bmax).run(trace)


def ll_rc(cat, trace, C, bmax=None):
    return online_drl(cat, trace, C, theta=0.0, bmax=bmax)


def rand_landlord(cat, trace, C, rng, bmax=None):
    """R3 only: Landlord eviction + randomized download threshold."""
    return Engine(cat, LandlordPolicy(cat, C), RandThreshold(rng), bmax).run(trace)


def rand_drl(cat, trace, C, rng, thr="random", bmax=None, **frac_kw):
    """R1+R2(+R3): fractional OGD state, randomized rounding with repair, random threshold."""
    pol = FracPolicy(cat, C, rng, **frac_kw)
    t = RandThreshold(rng) if thr == "random" else DetThreshold(1.0)
    return Engine(cat, pol, t, bmax).run(trace)
