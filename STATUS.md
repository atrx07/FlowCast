# STATUS.md

## Status Metadata

- **Project:** FlowCast v1.0
- **Last updated:** 2026-07-24
- **Current milestone:** M5 - Classical machine learning (in progress)
- **Current step:** Steps 00-13 complete; Step 14 next
- **Overall state:** Classical classification gate passed; formal classifier
  performance goals not met
- **Primary blocker:** None

## 1. Verified Current Position

FlowCast now has a reproducible Python 3.11 pipeline from immutable raw inputs
through validation, cleaning, merge, leakage-safe features, four-horizon
targets, EDA, frozen chronological evaluation, training-only preprocessing,
the NumPy regression proof, complete classical regression, and complete
classical classification.

Step 13 trains congestion and accident-risk classifiers at 30, 60, 90, and 120
minutes. All eight jobs have Decision Tree, Random Forest, XGBoost, and scaled
SVM evidence. Every selected artifact exposes finite, normalized probabilities
in a persisted fixed class order and has complete JSON/Markdown model cards.

All eight family choices, calibration decisions, and four accident thresholds
were serialized in a hashed selection manifest before the test partition was
loaded once. No model, threshold, calibrator, feature, or class order changed
after test access.

## 2. Step 13 Implementation

- Added the validated `classical_classification_v1` YAML contract: two tasks,
  four horizons, fixed class order, four required families, eight bounded
  configurations, five folds, seed 42, sigmoid calibration assessment, and
  validation-only accident-threshold selection.
- Added generated job/spec construction, fixed numeric target encoding,
  seeded estimator factories, fresh fold-local preprocessing, deterministic
  evenly spaced CV sampling, horizon leakage assertions, fold-local class
  balancing, runtime capture, and candidate-failure visibility.
- Added ordered multiclass/binary probability validation, Macro-F1 and
  per-class metrics, ROC-AUC/PR-AUC, operating precision/recall/F1, Brier
  score/log loss, confusion matrices, and deterministic threshold search.
- Split the validation window chronologically. The earlier half fits a sigmoid
  calibrator; the later half assesses Brier improvement and selects accident
  thresholds. LinearSVC calibration is mandatory for probability output;
  native-probability models calibrate only when the configured improvement
  gate passes.
- Added `flowcast train-classical-classification [--version VERSION]`.
- Added canonical fold/candidate/family/final scoreboards, ordered
  validation/test probabilities, calibration and threshold tables, confusion
  matrices, feature importance, hashes, generated Markdown, complete model
  cards, and verified Joblib loading.
- Added unit and full-data contracts for job/family/fold coverage, seeded
  determinism, horizon leakage, class order, class weights, metric correctness,
  threshold selection, calibration freeze, normalized probabilities, reload
  equality, card completeness, and tamper rejection.
- Refreshed the unchanged Step 10 evidence and rebuilt Step 12 against the
  expanded hashed model configuration. Step 12 selections and metrics remained
  identical.

## 3. Selection and Test-Sealing Evidence

- Generated classification jobs: 8.
- Configured candidates per job: 8.
- Expanding time-series folds per candidate: 5.
- Successful fold evaluations: 320 of 320.
- Required family/job validation comparisons: 32 of 32.
- CV training budget: 96 evenly spaced timestamps, retaining all roads and
  spanning every fold's available training interval.
- Final family fits: every eligible training row for that task/horizon.
- Congestion hyperparameter selection: mean CV Macro-F1.
- Accident hyperparameter selection: mean CV ROC-AUC.
- Family selection: matching full-validation primary metric.
- Calibration assessment: chronological later half of validation.
- Accident threshold selection: later-validation F1, then recall, precision,
  and lower threshold.
- Frozen selections before test: 8 of 8.
- Step 13 final-test loader calls: 1.
- Models refit after test access: 0.

The selection-manifest SHA-256 is
`af78be183276cd5d48c9d5eb63148457ebb641395693f8ec25925abf2255f835`.
It contains validation/CV/calibration/threshold evidence and explicitly
contains no test metrics.

## 4. Frozen Hold-Out Results

| Task | Horizon | Selected family | Calibrated | Threshold | Test primary | Test secondary |
|---|---:|---|---|---:|---:|---:|
| Congestion | 30 min | Random Forest | No | - | Macro-F1 0.7540 | Macro-recall 0.7659 |
| Congestion | 60 min | XGBoost | Yes | - | Macro-F1 0.7503 | Macro-recall 0.7736 |
| Congestion | 90 min | XGBoost | Yes | - | Macro-F1 0.7493 | Macro-recall 0.7762 |
| Congestion | 120 min | Random Forest | Yes | - | Macro-F1 0.7468 | Macro-recall 0.7751 |
| Accident risk | 30 min | SVM | Yes | 0.0133277 | ROC-AUC 0.6209 | PR-AUC 0.0209 |
| Accident risk | 60 min | SVM | Yes | 0.0189995 | ROC-AUC 0.6237 | PR-AUC 0.0182 |
| Accident risk | 90 min | SVM | Yes | 0.0132201 | ROC-AUC 0.5980 | PR-AUC 0.0161 |
| Accident risk | 120 min | SVM | Yes | 0.0102165 | ROC-AUC 0.5894 | PR-AUC 0.0165 |

The congestion Macro-F1 target of at least 0.80 and accident ROC-AUC target of
at least 0.75 were not met at any horizon. These are honest frozen outcomes;
the split and selected decisions were not changed to manufacture acceptance.

Accident operating-point test results are also weak and remain visible:

| Horizon | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 30 min | 0.0278 | 0.1776 | 0.0480 |
| 60 min | 0.0310 | 0.0426 | 0.0359 |
| 90 min | 0.0223 | 0.0504 | 0.0309 |
| 120 min | 0.0214 | 0.1705 | 0.0380 |

## 5. Probability and Imbalance Evidence

- Congestion probability order is fixed as Free-flow, Moderate, Heavy, Severe.
- Accident probability order is fixed as no-accident, accident.
- All persisted probability rows are finite, bounded to `[0, 1]`, and sum to
  one; reload comparison error was at most floating-point roundoff.
- Sigmoid calibration was applied to seven jobs. Congestion h1 retained raw
  Random Forest probabilities because later-validation Brier improvement was
  below the configured minimum.
- All four selected accident classifiers are scaled LinearSVC models and
  therefore require sigmoid calibration for probability output.
- Training-only accident positives range from 1,156 against about 122,500
  negatives per horizon; accuracy is never used as the headline metric.
- The threshold artifact contains 808 evaluated validation thresholds with
  exactly one selected row for each accident horizon.

## 6. Produced Artifacts

```text
artifacts/metrics/classical_classification_v1/
  accident_thresholds.csv
  calibration_metrics.csv
  confusion_matrices.csv
  cv_candidate_metrics.csv
  cv_fold_metrics.csv
  family_validation_metrics.csv
  feature_importance.csv
  scoreboard.csv
  selection_manifest.json
  summary.json
  summary.md

artifacts/model_cards/classical_classification_v1/
  {congestion,accident}_h{1,2,3,4}.json
  {congestion,accident}_h{1,2,3,4}.md

artifacts/models/classical_classification_v1/
  {congestion,accident}_h{1,2,3,4}.joblib

artifacts/predictions/classical_classification_v1/
  selected_predictions.parquet
```

The eight Joblib classifiers and combined prediction Parquet remain ignored by
Git as reproducibly generated binaries. Their hashes and sizes are recorded in
the tracked summary and model cards. The prediction artifact contains 428,257
validation/test rows.

## 7. Artifact and Runtime Evidence

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| CV fold metrics | 239,763 | `532e98369c5618d385a952d716d01039d5ca15cdbe1a99fa9e8c89d1baccad62` |
| CV candidate metrics | 17,729 | `fe1c9483bfd6e988d7e1435b1f5f62765704bf10399b7a6742aaff5a917a4145` |
| Family validation metrics | 23,621 | `15179977ef4773f0e21bfc6704b4a374fc76ab6961dbdcc17f470926de397813` |
| Final scoreboard | 14,150 | `c26af1d89fb833c082978ad059633566b7300e598f873eca2f6e825728d057d8` |
| Calibration metrics | 2,185 | `ba5059e3d902592990b000aeaa18a9deb6220e4eceb938f28a225fc53b3e4ecc` |
| Accident thresholds | 77,739 | `09096c2ad19ff1cda20ecde5bea8c67434230c784a64ca1a25737fe21d1286f7` |
| Confusion matrices | 9,543 | `bff0aeb9b2e71ab7a5a3c43b98e7619c1b43cff283c6fd3eb1a6154ba5dba62e` |
| Feature importance | 45,992 | `1ab34878ecc175b50d3cb0c61d841fa835b3561d858f08dadd2201ccc504cdc3` |
| Selected predictions | 8,808,475 | `304160c8331db6a63a77c942d2fdaf415f5100d154b4e5c8467dd1f63a30263e` |
| Generated report | 1,984 | `d3c8e86854ac56f23541950a6b77535528ac0b7af7798d8176730f152587059d` |

Canonical wall time was 196.750 seconds. Recorded component time was 22.375
seconds for CV fitting, 14.545 seconds for CV prediction, 106.369 seconds for
full-training family fits, 3.240 seconds for validation prediction, and 0.540
seconds for frozen test probability prediction.

## 8. Validation and Assurance Evidence

Executed with project-local CPython 3.11.9:

```text
.venv/Scripts/python.exe -m flowcast.cli prepare-modeling
.venv/Scripts/python.exe -m flowcast.cli train-classical-classification
.venv/Scripts/python.exe -m flowcast.cli train-classical-regression
.venv/Scripts/python.exe -m pytest -q tests/unit/test_classical_classification.py tests/data_contracts/test_classical_classification_contract.py
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m compileall -q src tests
.venv/Scripts/python.exe -m flowcast.cli train-classical-classification --help
git diff --check
```

Verified results:

- Focused Step 13 unit/full-data contracts: 12 passed in 207.31 seconds.
- Complete suite: 128 passed in 462.78 seconds.
- The full-data fixture independently retrained all eight classifier jobs under
  a temporary artifact root rather than trusting canonical outputs.
- Canonical model reload checks reproduced congestion and accident validation
  probabilities; all probability row sums were within floating-point roundoff.
- The Step 12 rebuild reproduced the previous volume test RMSE values exactly.
- Dependency consistency, byte compilation, CLI help, all 8 classifier and 12
  regressor verified-load paths, whitespace, read-only-path, and source-size
  checks passed.
- Every source file is below 400 lines; the largest remains
  `classical_regression.py` at 391 physical lines.
- `FlowCast-project_file/`, raw sources, split boundaries, dependency versions,
  and the product contract were not modified.

## 9. Decisions and Constraints

- LinearSVC is the required scaled SVM baseline. It scales to all eligible
  training rows without the quadratic cost of kernel SVC; its decision scores
  receive a validation-fitted sigmoid only after family selection.
- XGBoost uses fold-local balanced sample weights. Decision Tree, Random Forest,
  and SVM compute balancing from their supplied training labels only.
- Calibration is not fitted or assessed on training/test rows. Separating the
  validation halves prevents the calibrator from being rewarded on the same
  rows used to fit it.
- The 96-timestamp CV budget matches the proven Step 12 runtime policy and
  affects search only; every final family comparison uses all eligible
  training and validation rows.
- Step 13 reused the exact frozen 62-feature schema as required. No post-test
  feature engineering or target redefinition was attempted after the missed
  performance goals became visible.
- No new dependency, artifact format, service, source-data rule, or product
  scope was introduced.

## 10. Risks and Unresolved Work

- Congestion and accident-risk acceptance goals remain unmet. Step 16 must
  diagnose performance by road, time, weather, prevalence, and horizon without
  disguising the frozen Step 13 result.
- Very low accident prevalence produces low PR-AUC and weak thresholded
  precision/recall even where ROC-AUC is above random.
- The combined classical scoreboard/registry is not built yet.
- Regression confidence intervals and broader segmented error analysis remain
  Step 16 work.
- Model binaries and prediction Parquets are ignored and must be rebuilt with
  the documented CLIs after a clean clone.
- The later recurrent model must compare against the exact frozen classical
  volume test rows and RMSE values.

## 11. Next Gate

Proceed only to **Step 14 - Build the Classical Scoreboard and Registry**. The
bounded action and acceptance gate are maintained in `NEXT_STEP.md`.
