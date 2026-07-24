# STATUS.md

## Status Metadata

- **Project:** FlowCast v1.0
- **Last updated:** 2026-07-24
- **Current milestone:** M5 - Classical machine learning (in progress)
- **Current step:** Steps 00-12 complete; Step 13 next
- **Overall state:** Classical regression gate passed
- **Primary blocker:** None

## 1. Verified Current Position

FlowCast now has a reproducible Python 3.11 pipeline from immutable raw inputs
through validation, cleaning, merge, leakage-safe features, four-horizon
targets, EDA, frozen chronological evaluation, training-only preprocessing,
the direct NumPy regression proof, and the complete classical regression suite.

Step 12 trains volume, average-speed, and travel-time regressors at 30, 60, 90,
and 120 minutes. All 12 jobs have Linear Regression, Decision Tree, Random
Forest, and XGBoost comparison evidence. The selected pipeline for every job is
reloadable, traceable to the processed data/schema/split/configuration, and
paired with JSON and Markdown model cards.

All 12 choices were persisted in a hashed selection manifest before the test
partition was loaded once for final regression evaluation. No estimator was
refit and no model/hyperparameter changed after test access.

## 2. Step 12 Implementation

- Added the validated `classical_regression_v1` contract to
  `config/models.yaml`: three targets, four horizons, four required estimator
  families, seven bounded configurations, five folds, seed 42, and explicit
  selection/freeze rules.
- Added generated job/spec construction, seeded estimator factories, fresh
  fold-local preprocessing, deterministic evenly spaced CV sampling, fold
  leakage assertions, metric/runtime capture, failure visibility, and
  coefficient/tree importance extraction.
- Split cross-validation, estimator construction, per-job training, output
  assembly, reporting, verified loading, and orchestration by responsibility.
  The largest source file is 391 physical lines; no source file reaches 400.
- Added `flowcast train-classical-regression [--version VERSION]`.
- Added canonical candidate, fold, family-validation, and final scoreboards;
  selected validation/test predictions; feature importance; runtime/library
  evidence; hashes; generated Markdown; and complete model cards.
- Added verified Joblib loading that checks configuration, Step 10 lineage,
  every shared artifact, the requested model, and both model-card formats
  before returning a pipeline.
- Added unit and full-data contracts for job coverage, seeded determinism,
  time/horizon leakage, CV selection, complete family/fold evidence, selection
  freeze, metric finiteness, formal volume target, reload prediction equality,
  card completeness, and tamper rejection.

## 3. Selection and Test-Sealing Evidence

- Required generated jobs: 12.
- Configured candidates per job: 7.
- Expanding time-series folds per candidate: 5.
- Successful fold evaluations: 420 of 420.
- Required family/task validation comparisons: 48 of 48.
- CV training budget: 96 evenly spaced timestamps, retaining every road and
  spanning the complete available training interval in each fold.
- Final family fits: every eligible training row for that target/horizon.
- Hyperparameter selection: mean CV RMSE within each family.
- Family selection: validation RMSE with deterministic tie-breakers.
- Frozen selections before test: 12 of 12.
- Step 12 final-test loader calls: 1.
- Models refit after test access: 0.
- Selected family: Random Forest for all 12 jobs.

The selection manifest contains validation/CV evidence and explicitly contains
no test metrics. Its SHA-256 is
`84c153160c105f73cf49f9e133e9ade982101d355c40f3d51f1567df082828f1`.

## 4. Frozen Hold-Out Results

| Target | Horizon | Validation RMSE | Test RMSE | Test MAE | Test MAPE | Test R-squared |
|---|---:|---:|---:|---:|---:|---:|
| Volume | 30 min | 61.1926 | 63.4595 | 42.2054 | 10.218% | 0.9514 |
| Volume | 60 min | 61.1236 | 62.8626 | 42.0109 | 10.263% | 0.9522 |
| Volume | 90 min | 65.6565 | 65.3058 | 44.0438 | 10.952% | 0.9483 |
| Volume | 120 min | 60.5924 | 62.0092 | 41.6263 | 10.295% | 0.9533 |
| Speed | 30 min | 3.7230 | 3.7400 | 2.8214 | 9.029% | 0.8980 |
| Speed | 60 min | 3.7591 | 3.7683 | 2.8538 | 9.056% | 0.8960 |
| Speed | 90 min | 3.8544 | 3.7940 | 2.8715 | 9.145% | 0.8944 |
| Speed | 120 min | 3.8419 | 3.7923 | 2.8731 | 9.085% | 0.8945 |
| Travel time | 30 min | 1.1005 | 1.1426 | 0.4610 | 9.753% | 0.8065 |
| Travel time | 60 min | 1.0853 | 1.0949 | 0.4393 | 9.316% | 0.8210 |
| Travel time | 90 min | 1.0870 | 1.0822 | 0.4291 | 9.012% | 0.8247 |
| Travel time | 120 min | 1.0823 | 1.1016 | 0.4381 | 9.203% | 0.8184 |

All regression metrics are finite and derive from persisted predictions. Every
volume horizon meets the formal hold-out MAPE target of at most 12%; observed
volume MAPE ranges from 10.218% to 10.952%.

## 5. Produced Artifacts

```text
artifacts/metrics/classical_regression_v1/
  cv_candidate_metrics.csv
  cv_fold_metrics.csv
  family_validation_metrics.csv
  feature_importance.csv
  scoreboard.csv
  selection_manifest.json
  summary.json
  summary.md

artifacts/model_cards/classical_regression_v1/
  {volume,speed,travel_time}_h{1,2,3,4}.json
  {volume,speed,travel_time}_h{1,2,3,4}.md

artifacts/models/classical_regression_v1/
  {volume,speed,travel_time}_h{1,2,3,4}.joblib

artifacts/predictions/classical_regression_v1/
  selected_predictions.parquet
```

The 12 Joblib pipelines and combined prediction Parquet remain ignored by Git
as reproducibly generated binaries. Their hashes and sizes are recorded in the
tracked canonical summary and model cards. The prediction artifact contains
650,700 validation/test rows.

## 6. Artifact and Runtime Evidence

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| CV fold metrics | 130,231 | `d8bc856876c10f5d7cdea4252bd4b4f71cce229b83c5c69d66522cb241fc3ea5` |
| CV candidate metrics | 23,859 | `d4ced6aa3f1d0129437190fc36dd030b1427b5ff2dc48f3af6e6fde273566a67` |
| Family validation metrics | 12,423 | `7232f7b78c9c65425efb7aaab82a0724e3f93cb3845dc7b709e59e729928aace` |
| Final scoreboard | 4,143 | `ed99964fd713f37b6488fa1c15a6b540e233ed449e72e0227a0334d82573db85` |
| Feature importance | 69,738 | `03c51c8b8830111bfd88bc0d052ee38b7a5934992e01f3c217a02b2d8c0a6f1a` |
| Selected predictions | 11,879,451 | `ce53fda7ebaae58b5f198d3c323fd4001d6456eee849e92670c6b5c959cebc26` |
| Generated report | 2,401 | `73b17366097be3f76245c8c1145f902f49c4edc7b6d4371fe31b71e9eb59a40b` |

Canonical wall time was 147.5 seconds. Recorded component time was 20.700
seconds for CV fitting, 10.906 seconds for CV prediction, 64.099 seconds for
full-training family fits, 2.714 seconds for validation prediction, and 0.851
seconds for frozen test prediction.

## 7. Validation and Assurance Evidence

Executed with project-local CPython 3.11.9:

```text
.venv/Scripts/python.exe -m flowcast.cli prepare-modeling
.venv/Scripts/python.exe -m flowcast.cli train-classical-regression
.venv/Scripts/python.exe -m pytest -q tests/unit/test_classical_regression.py tests/data_contracts/test_classical_regression_contract.py
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m compileall -q src tests
.venv/Scripts/python.exe -m flowcast.cli train-classical-regression --help
.venv/Scripts/python.exe -c "from flowcast.modelling.classical_artifacts import load_classical_regression_model"
git diff --check
```

Verified results:

- Focused Step 12 unit/full-data contracts: 10 passed in 155.07 seconds.
- Final complete suite: 116 passed in 272.80 seconds.
- The full-data fixture independently retrained all 12 jobs in a temporary
  artifact root rather than trusting the canonical outputs.
- Dependency consistency, byte compilation, CLI help, persisted pipeline load
  (`Pipeline volume_h1`, 650,700 prediction rows), and whitespace passed.
- Every source file is below 400 lines; the largest is
  `classical_regression.py` at 391 physical lines after responsibility splits.
- `FlowCast-project_file/`, raw source files, split boundaries, dependency
  versions, and the product contract were not modified.

## 8. Decisions and Constraints

- The 96-timestamp fold budget was chosen after measured larger probes exceeded
  a practical repeatable runtime. Sampling is deterministic, retains all roads,
  spans every fold's full training history, and affects CV search only.
- Every final family comparison uses all eligible training rows and all
  eligible validation rows; final metrics use all eligible test rows.
- Speed remains a required regression target because the approved PRD product
  objectives explicitly require multi-horizon average-speed forecasts.
- Random Forest honestly won all validation comparisons. Linear, Decision
  Tree, and XGBoost evidence remains visible rather than being discarded.
- Step 10 remains the frozen split/preprocessing authority. Its canonical
  summary was refreshed only because it hashes the extended model config.
- No new dependency, artifact format, service, source-data rule, or dashboard
  scope was introduced.

## 9. Risks and Unresolved Work

- Congestion and accident-risk classifiers are not trained yet; Macro-F1 and
  ROC-AUC acceptance targets remain unmeasured.
- Accident positives remain rare, so Step 13 must use fold-local class
  weighting, PR-AUC, validation threshold selection, and honest recall/precision
  evidence rather than accuracy.
- Regression confidence intervals and segmented error analysis remain Step 16
  work.
- Selected regression binaries are ignored and must be rebuilt with the
  documented CLI after a clean clone.
- The later recurrent model must compare against these exact classical volume
  test rows and RMSE values.

## 10. Next Gate

Proceed only to **Step 13 - Train Congestion and Accident Classifiers**. The
bounded action and acceptance gate are maintained in `NEXT_STEP.md`.
