import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse(desc, **extra):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--quick", action="store_true", help="small sizes for a fast smoke run")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=None)
    for k, v in extra.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args()
    a.N, a.T = (30, 2000) if a.quick else (40, 8000)
    if a.seeds is None:
        a.seeds = 3 if a.quick else 10
    os.makedirs(a.out, exist_ok=True)
    return a


def save(a, name, data, fig=None):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    with open(os.path.join(a.out, name + ".json"), "w") as f:
        json.dump(data, f, indent=2, default=conv)
    if fig is not None:
        fig.tight_layout(); fig.savefig(os.path.join(a.out, name + ".png"), dpi=130); plt.close(fig)
    print(f"[{name}] saved to {a.out}/")


def ms(x):
    x = np.asarray(x, float); return float(x.mean()), float(x.std())
