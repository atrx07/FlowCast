# NEXT_STEP.md

## Immediate Objective

Execute **Step 18 - Build the Streamlit Dashboard**. Build the single
user-facing Streamlit surface over the verified Step 08-17 datasets, metrics,
predictions, models, confidence tables, and inference/reporting services.

Do not change the Step 10-17 model selections, thresholds, calibrators,
recurrent architecture, confidence widths, active routing, evaluation
partitions, source predictions, or reported metrics. Ordinary Streamlit reruns
must never train a model.

## Read Before Acting

1. `AGENTS.md`, `TECH_STACK.md`, `STATUS.md`, and this file.
2. `STEPS.md` - Step 18 plus the proven Steps 09 and 14-17 procedures.
3. Dashboard, inference, reporting, upload, retraining, lineage, navigation,
   and performance boundaries in `PROJECT.md`, `ROADMAP.md`, and
   `ARCHITECTURE.md`.
4. The original PRD dashboard views, supporting modules, roles, usability,
   transparency, and real-output requirements.
5. The Step 17 prediction/report manifests and public verified loaders.
6. The build-safety contract in `AGENTS.md` section 14.1.

## Hardware Notice Before Acting

Before execution, obey the workstation-resource disclosure rule in
`AGENTS.md`. Dashboard implementation, import tests, and ordinary page
rendering should use the Intel Core Ultra 9 CPU at low-to-moderate intensity,
moderate system RAM, and light disk I/O. The RTX 5070 Laptop GPU/VRAM and Intel
NPU should remain idle. Browser-based visual QA may briefly use normal desktop
graphics, but model CUDA execution is unnecessary. Give a separate notice
before any later full-suite or explicit retraining command.

## Single Best Next Action

Build and verify the complete Streamlit product surface:

1. Add `dashboard/app.py`, modular page/service boundaries, shared filters,
   version/status headers, severity colours, and hash/version-aware caching.
2. Implement all nine required real-data views:
   live predictions, historical trends, congestion heatmap, road comparison,
   model performance, feature importance, forecast visualisation, prediction
   confidence, and weather versus traffic.
3. Add the tenth support page for data upload/validation, explicit retraining,
   audit links, and report export without replacing a required view.
4. Route interactive forecasts through `Predictor`; load persisted batches and
   reports through the Step 17 verified loaders.
5. Keep every displayed value tied to a persisted real aggregate, metric, or
   prediction. Add no placeholders or fabricated analytics.
6. Keep every required view reachable within three clicks and use one
   congestion severity mapping everywhere.
7. Validate uploads before trusted use. Keep uploaded data separate from
   immutable source/reference files.
8. Add a `training_service` boundary with explicit confirmation, duplicate-run
   protection, new versioned outputs, and no automatic active-model switch.
   Ordinary page reruns must not invoke it.
9. Expose CSV/HTML report download from verified Step 17 output.
10. Add Streamlit import/page smoke tests, service tests, empty/error-state
    checks, and a manual visual/navigation walkthrough.
11. Run tests only through `scripts/run_tests.py`; writers and upload tests must
    use temporary roots and leave tracked files byte-identical.

## Acceptance Gate

Step 18 is complete only when:

- All nine required views render with real persisted data/model outputs.
- The support page provides validated upload, explicit retraining, prediction,
  and report-export controls without weakening source immutability.
- No model trains during an ordinary app launch, page change, filter change, or
  rerun.
- Shared road/time/horizon filters behave consistently and every view is
  reachable within three clicks.
- Live forecasts show all five targets, four horizons, confidence/risk, active
  and fallback volume lineage, and exact data/model versions.
- Model-performance and confidence views preserve the known unmet targets and
  120-minute recurrent limitation.
- Empty, missing, stale, and tampered artifact states fail clearly without
  fake values.
- Streamlit imports and every page pass smoke tests.
- A manual browser walkthrough verifies layout, navigation, readability, and
  report download.
- Focused tests, the full suite, dependency check, compilation, whitespace,
  source-size checks, and repository-mutation guard pass.
- Required documentation and README progress are current before commit/push.

## Current Blockers

None. The verified Step 17 batch/report service is ready for dashboard
integration. The unmet classifier targets and recurrent 120-minute test
deficit remain visible product limitations, not authorization to retune.
