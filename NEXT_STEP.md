# NEXT_STEP.md

## Immediate Objective

Execute **Step 16 - Add Confidence and Error Analysis**. Attach
validation-calibrated uncertainty to the persisted regression and
classification outputs, then explain where the frozen models fail by road,
time, weather, class/prevalence, and forecast horizon.

Do not begin inference services, report export, or Streamlit work. Do not
change the frozen Step 12-15 model selections, thresholds, calibrators,
architectures, sequence lengths, test partitions, or predictions to improve
visible test results.

## Read Before Acting

1. `AGENTS.md`, `TECH_STACK.md`, `STATUS.md`, and this file.
2. `STEPS.md` - Step 16 plus the proven Steps 10, 12, 13, 14, and 15 procedures.
3. Confidence, evaluation, registry, reporting, and lineage sections of
   `PROJECT.md`, `ROADMAP.md`, and `ARCHITECTURE.md`.
4. The original PRD requirements for quantified confidence, error analysis,
   prediction transparency, and honest hold-out reporting.
5. Step 12/13 classical cards and predictions, the Step 14 registry, Step 15
   recurrent card/predictions/comparison, the current Git diff, and relevant
   tests.

## Hardware Notice Before Acting

Before execution, obey the workstation-resource disclosure rule in
`AGENTS.md`. Step 16 should normally use the Intel Core Ultra 9 CPU, moderate
system RAM, and disk I/O for grouping/calibration/report artifacts. The
installed PyTorch 2.13.0 build is CPU-only, so the RTX 5070 Laptop GPU, VRAM,
and Intel NPU should not be claimed or used unless the approved environment
changes and is reverified.

## Single Best Next Action

Build and verify one confidence/error-analysis layer over frozen predictions:

1. Add an independent versioned confidence/error-analysis YAML contract so no
   Step 10-15 source hashes or selections change.
2. For volume, speed, and travel time, fit split-conformal or documented
   empirical residual intervals on validation residuals only, separately by
   target and horizon where coverage supports it.
3. Apply the frozen interval widths to validation/test predictions and report
   empirical coverage, average width, under/over-coverage, and zero leakage.
4. Apply the same comparable external interval method to recurrent volume
   predictions. Keep deep/classical comparison on identical origin rows.
5. For congestion, expose calibrated probabilities, maximum probability, and
   entropy; report reliability/calibration and uncertainty by class/horizon.
6. For accident risk, preserve the frozen validation-selected threshold,
   probability, calibration choice, and prevalence. Derive transparent risk
   bands from validation only and report PR-AUC/precision/recall limitations.
7. Break errors down by road, hour/peak period, weekday/weekend, weather,
   congestion severity, incident prevalence, and horizon using minimum-support
   rules fixed before test aggregation.
8. Diagnose the missed congestion and accident goals and the recurrent
   120-minute RMSE deficit without refitting or hiding unfavourable slices.
9. Persist dashboard-ready confidence/error Parquets plus JSON/CSV/Markdown
   summaries with complete model/data/prediction hashes.
10. Add verified loaders, CLI support, unit/full-artifact contracts, and
    tamper rejection.

## Acceptance Gate

Step 16 is complete only when:

- Every required forecast output has a documented confidence/uncertainty field
  derived without test fitting.
- Regression interval calibration uses validation residuals only and coverage/
  width are measured on the frozen test predictions.
- Classification uncertainty derives from the persisted ordered probabilities;
  no threshold or calibrator changes after test visibility.
- All subgroup tables reconcile exactly to their source prediction rows and
  enforce their documented minimum-support policy.
- The recurrent 120-minute deficit and weak congestion/accident outcomes remain
  visible with evidence-backed diagnoses.
- Artifacts are machine-readable, dashboard-ready, traceable, reloadable, and
  tamper checked.
- Focused tests, the full suite, CLI/import smoke, dependency check,
  compilation, whitespace assurance, source-size checks, and artifact
  verification pass.
- Required documentation and README progress are current before commit/push.

## Current Blockers

None. The low accident prevalence is a modelling limitation, not a reason to
change the frozen test protocol.
