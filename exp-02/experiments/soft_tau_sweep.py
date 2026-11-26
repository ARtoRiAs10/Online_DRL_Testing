"""Soft-MG tau sweep (supports the choice tau=0.15).

For each tau in [0.1, 0.25] report
* cyclic-scan latency reduction vs Online-DRL (10 seeds),
* Google 50k calibrated latency/cost change (10 seeds),
* Zipf latency/cost change (10 seeds).
"""
from __future__ import annotations

import os

import numpy as np

from common import DATA, save_json
import online_drl_r3 as m

TAUS = [0.1, 0.15, 0.2, 0.25]
SEEDS = 10


def main():
    rows = []
    # cyclic
    cat_c, tr_c = m.cyclic_scan(16, 10)
    det_c = m.online_drl(cat_c, tr_c, 16)
    # google calibrated
    cat_g, tr_g = m.load_trace_csv(os.path.join(DATA, "google.csv"),
                                   id_col="JobID", time_col="Timestamp",
                                   cpu_col="CPU", ram_col="RAM",
                                   disk_col="Disk", time_scale_ms=1e-3,
                                   disk_unit_mb=1.0, D=1e-4, U=100.0,
                                   max_rows=50000)
    det_g = m.online_drl(cat_g, tr_g, 5)
    # zipf
    cat_z, tr_z = m.synthetic_world(N=20, T=2000, seed=0)
    det_z = m.online_drl(cat_z, tr_z, 10)
    for tau in TAUS:
        cyc = [m.drl_mg(cat_c, tr_c, 16, seed=s, soft_tau=tau).latency
               for s in range(SEEDS)]
        goog = [m.drl_mg(cat_g, tr_g, 5, seed=s, soft_tau=tau)
                for s in range(SEEDS)]
        zipf = [m.drl_mg(cat_z, tr_z, 10, seed=s, soft_tau=tau)
                for s in range(SEEDS)]
        row = dict(
            tau=tau,
            cyclic_reduction_pct=float(100 * (det_c.latency - np.mean(cyc)) / det_c.latency),
            google_lat_pct=float(100 * (np.mean([r.latency for r in goog]) - det_g.latency) / det_g.latency),
            google_cost_pct=float(100 * (np.mean([r.cost for r in goog]) - det_g.cost) / det_g.cost),
            zipf_lat_pct=float(100 * (np.mean([r.latency for r in zipf]) - det_z.latency) / det_z.latency),
            zipf_cost_pct=float(100 * (np.mean([r.cost for r in zipf]) - det_z.cost) / det_z.cost),
        )
        rows.append(row)
        print(row)
    save_json("soft_tau_sweep", dict(seeds=SEEDS, rows=rows,
                                     chosen_tau=0.15))


if __name__ == "__main__":
    main()