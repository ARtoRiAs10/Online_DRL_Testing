"""Run the full exp-02 pipeline: phases 0-7."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PHASES = [
    "phase0_baseline.py",
    "phase1_correctness.py",
    "phase2_synthetic.py",
    "phase3_real_traces.py",
    "phase4_matched_cost.py",
    "phase5_ablation.py",
    "phase6_stats_overhead.py",
    "phase7_report.py",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of phase file names")
    ap.add_argument("--many", action="store_true",
                    help="pass --many to phase 6 (100 seeds)")
    a = ap.parse_args()
    phases = a.only or PHASES
    for p in phases:
        if p not in PHASES:
            raise SystemExit(f"unknown phase {p}")
        cmd = [sys.executable, os.path.join(HERE, p)]
        if p == "phase6_stats_overhead.py" and a.many:
            cmd.append("--many")
        print("\n===", p, "===", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()