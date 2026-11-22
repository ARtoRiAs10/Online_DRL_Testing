"""Request traces: synthetic (Zipf + Poisson), CSV loaders (Google / SHARCNet), and the
paper's adversarial 'Large Sequence' (Theorem 1)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .catalog import Catalog, from_physical


@dataclass
class Trace:
    t: np.ndarray   # (T,) arrival times (ms), strictly increasing
    n: np.ndarray   # (T,) service index

    def __len__(self):
        return len(self.t)

    def head(self, T):
        return Trace(self.t[:T], self.n[:T])


def synthetic_world(N=40, T=5000, zipf=1.2, rate=0.5, seed=0, imbalance=1.0, **phys):
    """Zipf popularity, Poisson arrivals (rate in requests/ms).

    imbalance>1 makes each service dominated by one random resource (demand x imbalance),
    which is what makes slot reductions wasteful.
    """
    rng = np.random.default_rng(seed)
    raw = np.c_[rng.lognormal(0, .6, N), rng.lognormal(1, .6, N), rng.lognormal(3.5, .8, N)]
    if imbalance != 1.0:
        dom = rng.integers(0, 3, N)
        raw[np.arange(N), dom] *= imbalance
    cat = from_physical(raw, **phys)
    pop = 1.0 / np.arange(1, N + 1) ** zipf
    pop /= pop.sum()
    perm = rng.permutation(N)
    svc = perm[rng.choice(N, size=T, p=pop)]
    t = np.cumsum(rng.exponential(1.0 / rate, T))
    return cat, Trace(t, svc.astype(int))


def load_trace_csv(path, id_col, time_col, cpu_col, ram_col, disk_col, time_scale_ms=1.0,
                   disk_unit_mb=1.0, max_rows=None, sep=",", disk_scale=1.0, **phys):
    """Generic loader for the preprocessed Google / SHARCNet CSVs (see data/README.md).

    Zeros in cpu/ram/disk are replaced by the column median (as in the paper); a service's
    resources are taken from its first occurrence; timestamps are made strictly increasing.
    """
    import pandas as pd
    df = pd.read_csv(path, sep=sep, nrows=max_rows)
    for c in (cpu_col, ram_col, disk_col):
        col = pd.to_numeric(df[c], errors="coerce")
        med = col.loc[col > 0].median()
        if not np.isfinite(med): med = 0.0
        df[c] = col.where(col > 0, med)   # zeros AND missing values -> column median
    df = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    ids, inv = np.unique(df[id_col].to_numpy(), return_inverse=True)
    first = df.groupby(id_col).first().loc[ids]
    raw = np.c_[first[cpu_col], first[ram_col],
                first[disk_col] * disk_unit_mb * float(disk_scale)]
    t = df[time_col].to_numpy(float) * time_scale_ms
    t = t + np.arange(len(t)) * 1e-6  # strictly increasing
    return from_physical(raw, **phys), Trace(t - t[0], inv.astype(int))


def large_sequence(k, n_reps, M=6.0, Ml=12.0, l=1.0, burst_frac=0.5):
    """Adversarial sequence from the proof of Theorem 1 (BS1, BS2, BS3, MS) x n_reps.

    BS_i = s 0^M s^{beta M} 0^M.  BS1/BS2/BS3 each hold k/3 services whose dominant resource is
    disk / RAM / CPU respectively.  MS needs (k/3) x the large demand of every resource, so caching
    it evicts the whole cache.  Time unit = one slot, D = 1 (download time = disk size).
    Returns (catalog, trace, opt_latency) with OPT = (k + 2 n) l (paper eq. 17).
    """
    assert k % 3 == 0 and k >= 3
    m = k // 3
    Rl, Cl, R, Cs = Ml, Ml, M, M
    caps = np.array([m * (Cl + 2 * Cs), m * (Rl + 2 * R), m * (Ml + 2 * M)], float)  # cpu, ram, disk
    raw, dl = [], []
    for _ in range(m): raw.append([Cs, R, Ml]); dl.append(Ml)   # BS1: disk-heavy
    for _ in range(m): raw.append([Cs, Rl, M]); dl.append(M)    # BS2: RAM-heavy
    for _ in range(m): raw.append([Cl, R, M]); dl.append(M)     # BS3: CPU-heavy
    raw.append([m * Cl, m * Rl, m * Ml]); dl.append(m * Ml)     # MS
    raw = np.array(raw, float)
    cat = Catalog(l=np.full(len(raw), float(l)), M=np.array(dl, float), a=raw / caps, raw=raw, caps=caps)
    ev, t = [], 0.0

    def basic(s):
        nonlocal t
        Mi = cat.M[s]
        ev.append((t, s)); t += Mi + 1
        for _ in range(max(1, int(round(burst_frac * Mi)))):
            ev.append((t, s)); t += 1
        t += Mi

    for _ in range(n_reps):
        for s in range(k):
            basic(s)
        Mm = cat.M[k]
        ev.append((t, k)); t += Mm + 1
        ev.append((t, k)); t += 1 + Mm
    tt = np.array([e[0] for e in ev]); ss = np.array([e[1] for e in ev], int)
    return cat, Trace(tt, ss), (k + 2 * n_reps) * l


def cyclic_scan(k, n_reps, l=1.0, M=1.0):
    """Classic cyclic scan over ``k+1`` services with cache capacity ``k``.

    Every service has unit latency and unit download time.  The demands are
    ``1/k`` per resource, so any ``k`` services fit and ``k+1`` do not.
    """
    n = k + 1
    cat = Catalog(l=np.full(n, float(l)), M=np.full(n, float(M)),
                  a=np.ones((n, 3), float) / float(k))
    t = np.arange(n_reps * n, dtype=float)
    svc = np.tile(np.arange(n), n_reps).astype(int)
    return cat, Trace(t, svc)


def deterministic_eviction_trace(cat, base_trace, C):
    """Build a trace that requests what deterministic Online-DRL just evicted.

    The trace is generated using only the deterministic Landlord policy, so it is
    fixed in advance and oblivious to the randomized algorithm's coin flips.  The
    returned trace has one extra request after each deterministic eviction.

    Returns ``(new_trace, base_trace, evictions)``.
    """
    from collections import defaultdict
    from .engine import Engine, DetThreshold
    from .landlord import LandlordPolicy

    pol = LandlordPolicy(cat, C)
    Engine(cat, pol, DetThreshold(1.0), rng=np.random.default_rng(0)).run(base_trace)

    by_step = defaultdict(list)
    for step, f in pol.evictions:
        by_step[step].append(f)

    svc = []
    for i, n in enumerate(base_trace.n):
        svc.append(int(n))
        for f in by_step.get(i + 1, []):
            svc.append(int(f))

    dt = float(base_trace.t[1] - base_trace.t[0]) if len(base_trace.t) > 1 else 1.0
    t = np.arange(len(svc), dtype=float) * dt
    return Trace(t, np.asarray(svc, int)), base_trace, pol.evictions
