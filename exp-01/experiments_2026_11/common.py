"""Shared helpers for the 2026-11-25 experimental turn.

The canonical simulator lives in ../final_exp.  This folder only adds the new
roadmap-driven experiments; it does not modify the simulator except for the
optional `bankrupt_sizes` diagnostic in `landlord_repair.py`.
"""
from __future__ import annotations
import json, os, sys, math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = os.path.join(ROOT, "final_exp")
if FINAL not in sys.path:
    sys.path.insert(0, FINAL)

from sim.trace import synthetic_world, load_trace_csv, large_sequence  # noqa: E402
from sim.algos import online_drl, ll_rc, rand_landlord, rand_drl  # noqa: E402
from sim.rand_drl import FracPolicy  # noqa: E402
from sim.opt import solve_opt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def save(name, data):
    path = os.path.join(RESULTS, name + ".json")
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=conv)
    print(f"[{name}] saved to {path}")
    return path


def mean_ci(x, z=1.96):
    x = np.asarray(x, float)
    n = len(x)
    m = float(x.mean()) if n else float("nan")
    s = float(x.std(ddof=1)) if n > 1 else 0.0
    half = float(z * s / math.sqrt(n)) if n else float("nan")
    return m, s, half


def paired_ci(a, b, z=1.96):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    m, s, half = mean_ci(d, z)
    return dict(mean=m, sd=s, ci_half=half, ci_low=m-half, ci_high=m+half,
                win_rate=float(np.mean(d < 0)) if len(d) else float("nan"))


def qstats(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return dict(mean=float("nan"), sd=0.0, median=float("nan"), q25=float("nan"),
                    q75=float("nan"), min=float("nan"), max=float("nan"))
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                median=float(np.median(x)),
                q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)),
                min=float(x.min()), max=float(x.max()))


def ms(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0]


def google_cfg(disk_scale=1.0, D=1000.0, disk_unit_mb=1.0/1024.0):
    return dict(id_col="JobID", time_col="Timestamp", cpu_col="CPU", ram_col="RAM",
                disk_col="Disk", time_scale_ms=1e-3, disk_unit_mb=disk_unit_mb,
                U=100.0, D=D, disk_scale=disk_scale)


def load_google(name="google.csv", max_rows=None, **kw):
    path = os.path.join(FINAL, "data", name)
    return load_trace_csv(path, max_rows=max_rows, **google_cfg(**kw))


def interp_log_cost(det_rows, cost):
    """Interpolate deterministic latency at `cost`, linear in log(cost).

    `det_rows` is a list of dicts with `cost` and `latency`.  Returns nan outside
    the observed range.
    """
    pts = sorted([(float(r["cost"]), float(r["latency"])) for r in det_rows
                  if r["cost"] > 0 and np.isfinite(r["latency"])])
    if len(pts) < 2:
        return float("nan")
    xs = np.log([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    if cost <= pts[0][0] or cost >= pts[-1][0]:
        return float("nan")
    return float(np.interp(math.log(cost), xs, ys))


class AuditPolicy(FracPolicy):
    """FracPolicy with the Phase-A accounting flag.

    `charged[k,n]` is true iff a download of n completed and was charged since the
    last time n entered sample path k.  A hit is valid only if the flag is true.
    The policy also counts every serving-path 0->1 transition (the charge-all
    accounting) and every violating hit (the strict accounting).
    """
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.charged = None
        self.audit = dict(violations=0, hits_checked=0, serving_entries=0,
                          all_entries=0, entry_cost_serving=0.0,
                          entry_cost_all=0.0, violation_cost=0.0,
                          violation_services=[])

    def set_horizon(self, T):
        super().set_horizon(T)
        self.charged = np.zeros((self.K, self.cat.N), bool)
        self.audit = dict(violations=0, hits_checked=0, serving_entries=0,
                          all_entries=0, entry_cost_serving=0.0,
                          entry_cost_all=0.0, violation_cost=0.0,
                          violation_services=[])

    def before_request(self, t, n, q, dm):
        old = self.S.copy()
        super().before_request(t, n, q, dm)
        k = int(self.kstar)
        ent = (~old[k]) & self.S[k]
        if ent.any():
            self.charged[k, ent] = False
            self.audit["serving_entries"] += int(ent.sum())
            self.audit["entry_cost_serving"] += float(self.cat.M[ent].sum())
        allent = (~old) & self.S
        if allent.any():
            self.audit["all_entries"] += int(allent.sum())
            self.audit["entry_cost_all"] += float((self.cat.M[None, :] * allent).sum())

    def on_complete(self, n, t):
        ok = super().on_complete(n, t)
        if ok and self.charged is not None:
            self.charged[int(self.kstar), n] = True
        return ok

    def on_hit(self, n, saved=0.0):
        self.audit["hits_checked"] += 1
        if self.charged is None or not self.charged[int(self.kstar), n]:
            self.audit["violations"] += 1
            self.audit["violation_cost"] += float(self.cat.M[n])
            self.audit["violation_services"].append(int(n))
        super().on_hit(n, saved)

    def diagnostics(self):
        d = super().diagnostics()
        d["audit"] = dict(self.audit)
        d["audit"]["strict_extra_cost"] = float(self.audit["violation_cost"])
        return d
