# STATUS.md

## Status Metadata

- **Project:** FlowCast v1.0
- **Last updated:** 2026-07-22
- **Current milestone:** M5 - Classical machine learning (in progress)
- **Current step:** Steps 00-10 complete; Step 11 next
- **Overall state:** Frozen evaluation and preprocessing gate passed
- **Primary blocker:** None

## 1. Verified Current Position

FlowCast now has a reproducible Python 3.11 pipeline from immutable raw inputs
through validation, cleaning, cardinality-safe merging, leakage-safe features,
multi-horizon targets, EDA/reporting, and a frozen modelling-data protocol.
Milestones M0 through M4 are complete; M5 has begun.

All 181,200 origins have one deterministic chronological partition. Every
target/horizon combines its original availability mask with a persisted
same-partition target flag, preventing labels from crossing train, validation,
or test boundaries. Five expanding CV folds remain entirely inside training.
Four model-family preprocessors are fit from training rows only, while test
access is sealed by default. No forecast model has been trained or selected and
no test metric has been inspected.

## 2. Step 10 Implementation

- Added `config/models.yaml` and `split_preprocessing_v1` settings containing
  exact time boundaries, ratio/count evidence, horizon-boundary policy, CV
  geometry, sealed-test purposes, feature grouping, and scaling policies.
- Added verified Step 09/processed/feature-manifest loading. Configs, EDA
  outputs, processed Parquet, target/schema manifest, quality summary, and
  explanatory-feature manifest are hash-checked before use.
- Added largest-remainder chronological allocation across the 7,248 unique
  corridor timestamps while retaining all 25 roads at every timestamp.
- Added `target_within_split_h1` through `target_within_split_h4`; a target is
  usable only when this boundary flag and its target-specific availability mask
  are both true.
- Added five expanding-window CV folds inside training, each with a four-window
  gap covering the maximum 120-minute horizon and a 336-window/seven-day
  validation period.
- Added default-sealed test access. Tuning loaders expose train/validation;
  test requires the explicit `final_evaluation` purpose.
- Added linear, tree, SVM, and recurrent `ColumnTransformer` pipelines. All use
  training-fitted imputation and one-hot encoding; linear/SVM standardize
  numeric inputs, trees keep numeric inputs unscaled, and recurrent processing
  uses Min-Max scaling for documented bounded fields plus standardization for
  the remainder.
- Persisted exact feature order, output order, imputer/scaler/category
  statistics, library versions, training-only class weights, and Joblib hashes.
- Added `flowcast prepare-modeling [--version VERSION]` and a generated split/
  preprocessing report.

## 3. Produced Artifacts

```text
data/processed/split_preprocessing_v1/
  assignments.parquet

artifacts/features/split_preprocessing_v1/
  cv_folds.json
  feature_schema.json
  summary.json
  summary.md

artifacts/models/split_preprocessing_v1/
  linear.joblib
  tree.joblib
  svm.joblib
  recurrent.joblib
```

The assignment Parquet and fitted Joblib files are reproducibly generated and
remain ignored by Git. Their byte counts and SHA-256 hashes are recorded in the
tracked canonical JSON artifacts.

## 4. Frozen Split and Target Evidence

| Partition | Exact coverage (Asia/Kolkata) | Timestamps | Rows | Share |
|---|---|---:|---:|---:|
| Train | 2025-01-01 00:00 to 2025-04-16 16:30 | 5,074 | 126,850 | 70.0055% |
| Validation | 2025-04-16 17:00 to 2025-05-09 08:00 | 1,087 | 27,175 | 14.9972% |
| Test | 2025-05-09 08:30 to 2025-05-31 23:30 | 1,087 | 27,175 | 14.9972% |

Every road has exactly the same timestamp boundaries and one assignment per
origin. For a standard complete target, boundary-safe eligible rows decrease by
25 rows per horizon in each partition:

| Horizon | Train eligible | Validation eligible | Test eligible |
|---:|---:|---:|---:|
| h1 / 30 min | 126,825 | 27,150 | 27,150 |
| h2 / 60 min | 126,800 | 27,125 | 27,125 |
| h3 / 90 min | 126,775 | 27,100 | 27,100 |
| h4 / 120 min | 126,750 | 27,075 | 27,075 |

Accident labels additionally exclude reconstructed unknown windows. Eligible
accident rows range from 123,716 (train h1) to 123,644 (train h4), 26,455 to
26,383 on validation, and 26,456 to 26,384 on test. Training h1 contains 1,156
positives and 122,560 negatives; its persisted `scale_pos_weight` is 106.0208.
No validation/test label contributed to that weight.

## 5. Preprocessing and CV Evidence

| Family | Input features | Output features | Ordinary numeric | Bounded numeric |
|---|---:|---:|---|---|
| Linear | 62 | 64 | StandardScaler | StandardScaler |
| Tree | 62 | 64 | Unscaled | Unscaled |
| SVM | 62 | 64 | StandardScaler | StandardScaler |
| Recurrent | 62 | 64 | StandardScaler | MinMaxScaler |

The 62 inputs come directly from the Step 07 `known_at_origin` manifest: 27
ordinary numeric, 9 bounded numeric, 25 binary, and 1 categorical feature.
Keys, identifiers, timestamps, raw lineage strings, targets, and availability
masks are excluded. All families produce 64 dense numeric columns after
training-fitted temperature-band one-hot encoding.

CV fold training endpoints expand from 2025-03-12 14:30 through 2025-04-09
14:30. Each four-window gap ends immediately before a seven-day validation
window; fold 5 validation ends exactly at the frozen training boundary on
2025-04-16 16:30.

## 6. Artifact Evidence

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Split assignments Parquet | 354,856 | `dd30fe7a475c049f9f981374ee3c97533a7b2270b7ef8336ea5be37cf2f55892` |
| CV folds JSON | 2,436 | `ce36264dbe3824c4d119ea7e801471b94bdeeee6b4bcff250cf8f9f1c06488bb` |
| Feature schema JSON | 73,319 | `204d2fc3ab00e18a452e4ef2898826cf9dc0bd05dbaf795f8f203ca26f71f453` |
| Canonical summary JSON | 76,872 | `14c373c3627243e6718660e18feead722447210ec44c51b8e60665969569ad46` |
| Generated Markdown | 3,865 | `e03a2dd4b91f607ee8d4f12b4b2cf9e94460ad0855dd7e215a14da85ada5fea0` |
| Linear/SVM Joblib (each) | 3,418 | `8550eba7e08b44b14e6df1d48afac22494da94746a3ff352c389ff76a23ccd92` |
| Tree Joblib | 2,307 | `75a791fbd4aac2be00b9dece31a554ac768f0e2a1310bcb13cd116ede4a9be36` |
| Recurrent Joblib | 3,519 | `88f77f64081940bf787bc92702362033f37507f03ced1f95bc14b82ccf27e7d3` |

Linear and SVM preprocessors are byte-identical because their Step 10 policies
are intentionally identical. Repeated full-data runs reproduced every listed
artifact byte-for-byte.

## 7. Validation and Assurance Evidence

Executed with project-local CPython 3.11.9:

```text
.venv/Scripts/python.exe -m pip install -e ".[classical,eda,test]"
.venv/Scripts/python.exe -c "import joblib, sklearn, xgboost"
.venv/Scripts/python.exe -m flowcast.cli merge-sources
.venv/Scripts/python.exe -m flowcast.cli engineer-features
.venv/Scripts/python.exe -m flowcast.cli prepare-data
.venv/Scripts/python.exe -m flowcast.cli eda
.venv/Scripts/python.exe -m flowcast.cli prepare-modeling
.venv/Scripts/python.exe -m pytest -q tests/unit/test_modelling_split.py tests/unit/test_preprocessing.py tests/data_contracts/test_modelling_contract.py
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m compileall -q src tests
.venv/Scripts/python.exe -m flowcast.cli prepare-modeling --help
.venv/Scripts/python.exe -c "from flowcast.modelling.inputs import load_modeling_partition, load_preprocessor"
git diff --check
```

Verified results:

- Approved classical dependencies installed; Joblib 1.5.2, scikit-learn 1.9.0,
  and XGBoost 3.2.0 import successfully.
- The refreshed merge, feature, processed, EDA, and modelling-preparation CLI
  stages exited 0; upstream data hashes remain content-stable while base-config
  lineage now includes the modelling contract.
- Focused Step 10 unit/full-data contract tests: 10 passed in 9.27 seconds.
- Final full suite: 98 passed in 66.39 seconds.
- Tests cover exact boundaries/counts, road coverage, no target crossing, CV
  gaps, training-only statistics/weights, unseen categories, test sealing,
  preprocessor reload/transform, deterministic hashes, and tamper rejection.
- Dependency consistency, byte compilation, CLI-help, whitespace, canonical
  validation loading (27,175 rows), and fitted linear-preprocessor loading (64
  outputs) passed.
- Largest source module remains `src/flowcast/data/audit.py` at 366 physical
  lines; every source file remains below 400 lines.

## 8. Decisions and Constraints

- Largest-remainder allocation is used so 70/15/15 ratios consume all 7,248
  timestamps without dropping or duplicating an origin.
- Partition assignment belongs to the origin; target eligibility is separate by
  horizon and target family. This retains traceability without leakage.
- A four-window CV gap is the smallest gap that covers the maximum target
  horizon before each fold's validation window.
- Training/validation/test boundaries are shared by classical and later deep
  models. Deep sequences will inherit the same origin boundary contract.
- Validation may tune hyperparameters, calibration, and thresholds. Test remains
  inaccessible to tuning loaders unless `purpose="final_evaluation"` is passed
  explicitly after selection is frozen.
- Full-data EDA redundancy flags did not remove features. The frozen schema
  keeps all 62 approved inputs; future selection must occur within training.
- The classical dependency group is now active. No dependency version or
  approved technology changed.

## 9. Risks and Unresolved Work

- No trained model exists yet, so formal accuracy targets remain unmeasured.
- Accident training remains severely imbalanced despite explicit weights;
  threshold selection and probability calibration still belong to later steps.
- The scratch linear baseline must prove its gradients and convergence before
  library regression training begins.
- Final-test access is guarded by explicit software intent, not an external
  authorization service; tests and operating rules enforce the v1 boundary.
- Generated assignment/preprocessor binaries are ignored by Git and must be
  rebuilt with `prepare-modeling` after a clean clone.
- The documented Matplotlib/Pillow workstation fallback remains in effect for
  static figures; it does not affect modelling dependencies.

## 10. Next Gate

Proceed only to **Step 11 - Implement NumPy Linear Regression**. The bounded
action and acceptance gate are maintained in `NEXT_STEP.md`.
