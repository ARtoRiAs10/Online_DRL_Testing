"""Run the 2026-11-25 roadmap experiments in order.

Usage: python run_all.py [--quick]
The scripts write JSON/PNGs to results/.
"""
import subprocess, sys, os, time
here = os.path.dirname(os.path.abspath(__file__))
scripts = ["phase_a_audit.py", "phase_b_frontier.py", "phase_b_equal.py",
           "phase_c_horizon_opt.py", "phase_d_adversarial.py",
           "phase_e_real_data.py", "phase_f_theory.py"]
extra = sys.argv[1:]
for s in scripts:
    print("=" * 70, "\n", s, flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, os.path.join(here, s)] + extra)
    print(f"-> exit {r.returncode} in {time.time()-t:.1f}s", flush=True)
