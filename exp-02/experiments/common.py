"""Shared helpers for the exp-02 experiment scripts."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
DOCS = os.path.join(ROOT, "docs")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(DOCS, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_json(name, obj):
    path = os.path.join(RESULTS, name + ".json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_conv)
    print(f"[{name}] -> {path}")
    return path


def load_json(name):
    with open(os.path.join(RESULTS, name + ".json")) as f:
        return json.load(f)


def _conv(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save_fig(fig, name):
    path = os.path.join(RESULTS, name + ".png")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[{name}] -> {path}")
    return path


def mean_sd(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def mean_ci(x, z=1.96):
    x = np.asarray(x, float)
    n = len(x)
    m = float(x.mean()) if n else float("nan")
    s = float(x.std(ddof=1)) if n > 1 else 0.0
    half = float(z * s / np.sqrt(n)) if n else float("nan")
    return m, s, half


def bootstrap_ci(x, n_boot=2000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_ci(a, b, n_boot=2000, seed=0):
    """Paired difference a-b: mean, sd, bootstrap CI, win rate."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    m, lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    return dict(mean=m, ci_low=lo, ci_high=hi,
                sd=float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                win_rate=float(np.mean(d < 0)) if len(d) else float("nan"),
                n=len(d))


def wilcoxon(a, b):
    from scipy.stats import wilcoxon as _w
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0.0):
        return 0.0, 1.0
    try:
        stat, p = _w(a, b, zero_method="wilcox")
        return float(stat), float(p)
    except ValueError:
        return 0.0, 1.0


def holm(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def describe(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return dict(n=0)
    m, lo, hi = bootstrap_ci(x)
    return dict(n=int(len(x)), mean=m, ci_low=lo, ci_high=hi,
                sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                worst=float(x.max()), p95=float(np.quantile(x, 0.95)),
                min=float(x.min()))


def markdown_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)