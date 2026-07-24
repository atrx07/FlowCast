# NEXT_STEP.md

## Immediate Objective

Execute **Step 17 - Build the Inference and Reporting Services**. Create one
stable, validated service boundary that loads the frozen artifacts, produces
all required target/horizon forecasts without retraining, attaches Step 16
confidence and lineage, persists batch output, and exports a real-data report.

Do not begin Streamlit pages, upload handling, or retraining controls. Do not
change the Step 10-16 model selections, thresholds, calibrators, recurrent
architecture, test partitions, confidence widths, source predictions, or
reported metrics.

## Read Before Acting

1. `AGENTS.md`, `TECH_STACK.md`, `STATUS.md`, and this file.
2. `STEPS.md` - Step 17 plus the proven Steps 10 and 12-16 procedures.
3. Registry, recurrent, confidence, inference, reporting, lineage, performance,
   and dashboard boundaries in `PROJECT.md`, `ROADMAP.md`, and
   `ARCHITECTURE.md`.
4. The original PRD requirements for prediction inputs/outputs, confidence,
   model/data traceability, report generation, response time, and CPU support.
5. The Step 14 registry, Step 15 recurrent registry extension/model card, Step
   16 summary/calibrations/loaders, the current Git diff, and relevant tests.
6. The build-safety contract in `AGENTS.md` section 14.1 and the known
   challenge evidence in `STATUS.md`.

## Hardware Notice Before Acting

Before execution, obey the workstation-resource disclosure rule in
`AGENTS.md`. Step 17 is primarily model loading, feature preparation, small
batch inference, schema validation, and report rendering. It should normally
use the Intel Core Ultra 9 CPU, moderate system RAM, and light-to-moderate disk
I/O. The RTX 5070 Laptop GPU/VRAM should remain idle for ordinary full-corridor
inference unless a measured recurrent batch materially benefits from CUDA; the
public service must still pass forced-CPU tests. The Intel NPU remains outside
the approved v1 path.

## Single Best Next Action

Build and verify one inference/report service over the frozen artifacts:

1. Add an independent versioned inference/report YAML contract naming active
   artifact versions, supported request fields, output schema, report formats,
   runtime target, and device policy.
2. Define typed request/result contracts for road IDs, origin timestamp,
   horizons 1-4, optional batch scope, model/data/confidence versions, and
   validation errors.
3. Freeze active model routing without test-led selection. The recurrent model
   beats classical volume on validation at all four horizons; use that
   validation-only evidence if it becomes the active volume route, retain
   classical volume as the explicit fallback/comparator, and use the Step 14
   selected models for speed, travel time, congestion, and accident.
4. Build latest-origin feature preparation from the verified processed
   contract. Recurrent requests must prove 12 contiguous road-local half-hour
   rows; no request may cross road, cadence, or data boundaries.
5. Load every model through verified public loaders. Normal inference must
   never fit preprocessing, calibrators, thresholds, confidence widths, or
   model weights.
6. Generate volume, speed, travel time, congestion class/probabilities,
   accident probability/risk band, and confidence for every requested horizon,
   with origin/target timestamps and complete data/model/config lineage.
7. Apply Step 16 conformal widths and classification/risk semantics unchanged.
   Validate the output schema and physical/probability bounds before
   persistence.
8. Add deterministic batch Parquet plus JSON manifest output, and
   human-readable insights derived only from real aggregates/predictions.
9. Export at least CSV plus self-contained HTML (or an already approved
   equivalent) with no fabricated metrics or analytics.
10. Add CLI prediction/report commands, full-corridor CPU timing, verified
    loaders, unit/full-artifact contracts, repeatability checks, and tamper
    rejection.
11. Run every test through `scripts/run_tests.py`; writers must use temporary
    roots, and the session must finish with `FLOWCAST_PYTEST_EXIT=0` and no
    repository-mutation failure.

## Acceptance Gate

Step 17 is complete only when:

- One validated request returns all five required forecast targets at all four
  horizons plus interval/probability confidence, risk band, origin/target
  timestamps, and exact version lineage.
- Active routing is frozen from validation/config evidence and test metrics are
  reporting-only.
- Model loading and prediction occur without any retraining or learned-statistic
  fitting.
- Invalid roads, origins, horizons, missing sequence history, stale/tampered
  artifacts, and malformed requests fail clearly.
- Repeated seeded CPU inference is stable and persisted batch rows reconcile to
  the request manifest.
- CSV/HTML report content traces to real persisted inputs and predictions.
- Full-corridor inference is timed against the 30-second requirement and the
  measured result is reported honestly.
- Focused tests, the full suite, CLI/import smoke, dependency check,
  compilation, whitespace assurance, source-size checks, and artifact
  verification pass.
- Tests leave all canonical tracked files byte-identical; the repository guard
  must report no mutation.
- Required documentation and README progress are current before commit/push.

## Current Blockers

None. The unmet classifier targets and recurrent 120-minute test deficit are
known model limitations that must remain visible in inference/report outputs;
they do not authorize post-test retuning.
