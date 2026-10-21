"""Landlord with resource checks (Algorithm 2 of the paper)."""
from __future__ import annotations
import numpy as np
from .engine import Policy


class LandlordPolicy(Policy):
    name = "landlord"

    def __init__(self, cat, C, size_mode="sum"):
        self.cat, self.C = cat, C
        self.size = cat.a.sum(1) if size_mode == "sum" else cat.a.max(1)
        self.credit: dict[int, float] = {}
        self.n_evict = 0

    def cached(self, n): return n in self.credit
    def cached_list(self): return list(self.credit.keys())
    def on_hit(self, n, saved=0.0):
        self.credit[n] = self.cat.M[n]      # line 25: reset credit to cost

    def on_complete(self, g, t):
        self._admit(g)
        return True

    def _decrease(self):
        d = min(self.credit[f] / self.size[f] for f in self.credit)
        for f in self.credit:
            self.credit[f] = max(0.0, self.credit[f] - d * self.size[f])
        return [f for f, c in self.credit.items() if c <= 1e-12]

    def _violated(self, g):
        used = self.cat.a[list(self.credit)].sum(0) if self.credit else np.zeros(3)
        need = np.maximum(used + self.cat.a[g] - 1.0, 0.0)
        return need, (need > 1e-12).any() or len(self.credit) + 1 > self.C

    def _admit(self, g):
        if g in self.credit:
            self.credit[g] = self.cat.M[g]; return
        while True:
            need, bad = self._violated(g)
            if not bad: break
            elig = self._decrease()
            # lines 8-10: prefer items whose resources beyond the unmet need are smallest
            elig.sort(key=lambda f: np.maximum(self.cat.a[f] - need, 0).sum())
            for f in elig:
                need, bad = self._violated(g)
                if not bad: break
                del self.credit[f]; self.n_evict += 1
                self.eng.reset_miss(f)
        self.credit[g] = self.cat.M[g]

    def diagnostics(self): return dict(evictions=self.n_evict)
