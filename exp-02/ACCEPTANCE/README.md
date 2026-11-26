# Acceptance criteria — frozen before running

This folder contains the acceptance thresholds for the R3 + Marking-Gated
Landlord experiments.  The thresholds were fixed **before** any experiment was
run and must not be edited after the fact.  Any change requires a new commit
and a new run.

| file | content |
|---|---|
| [`PRECONDITIONS.md`](PRECONDITIONS.md) | baseline reproduction, reduction, invariants |
| [`CLAIMS.md`](CLAIMS.md) | claim-by-claim accept / reject table |
| [`STATISTICAL_STANDARD.md`](STATISTICAL_STANDARD.md) | seeds, CIs, tests, correction |
| [`EVIDENCE_STATUS.md`](EVIDENCE_STATUS.md) | what the evidence supports / does not support |

The machine-readable evaluation of these criteria is produced by
`experiments/phase7_report.py` into `results/acceptance.json` and
`docs/REPORT.md`.