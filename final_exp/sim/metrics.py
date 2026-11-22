"""Resource metrics of Sec. 4.4 (replace the 'slots used' count)."""
from __future__ import annotations
import numpy as np


def resource_metrics(util, total_latency=None):
    Umax = util.max(1)
    imb = Umax - util.min(1)
    W = (Umax[:, None] - util).sum(1)
    out = dict(U_mean=util.mean(0).tolist(), Umax_mean=float(Umax.mean()), imbalance_mean=float(imb.mean()),
               stranded_mean=float(W.mean()), violation_rate=float((Umax > 1 + 1e-9).mean()),
               max_util=float(Umax.max()))
    if total_latency is not None:
        out["latency_per_unit_resource"] = float(total_latency / max(Umax.sum(), 1e-12))
    return out


def pct(a, b):
    """percentage by which a exceeds b"""
    return 100.0 * (a - b) / b
