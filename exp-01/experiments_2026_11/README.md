# 2026-11-25 roadmap experiments

Run:

```bash
python run_all.py
python summarize.py
python -m pytest -q tests/test_invariants.py
```

Scripts write `results/*.json` (and `phase_b_frontier.png`). The canonical simulator
is `../final_exp`; this folder adds only the new Phase A–F drivers and the
`AuditPolicy` accounting wrapper. `final_exp/sim/landlord_repair.py` has one
diagnostic addition (`bankrupt_sizes`) used by Phase A4.

See `../docs_2026_11/FINAL_REPORT.md` for the write-up.
