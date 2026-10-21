"""Run every experiment:  python experiments/run_all.py [--quick]"""
import subprocess, sys, glob, os, time
here = os.path.dirname(os.path.abspath(__file__))
extra = sys.argv[1:]
for f in sorted(glob.glob(os.path.join(here, "exp*.py"))):
    print("=" * 70, "\n", os.path.basename(f), flush=True); t = time.time()
    r = subprocess.run([sys.executable, f] + extra)
    print(f"-> exit {r.returncode} in {time.time() - t:.1f}s", flush=True)
