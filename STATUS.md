# STATUS.md

## Status Metadata

- **Project:** FlowCast v1.0
- **Last updated:** 2026-07-22
- **Current milestone:** M5 - Classical machine learning (in progress)
- **Current step:** Steps 00-11 complete; Step 12 next
- **Overall state:** NumPy regression mathematics gate passed
- **Primary blocker:** None

## 1. Verified Current Position

FlowCast has a reproducible Python 3.11 pipeline from immutable raw inputs
through validated and cleaned sources, a cardinality-safe merge, leakage-safe
features, four-horizon targets, EDA/reporting, frozen chronological evaluation,
and training-only preprocessing. Step 11 now adds the PRD-required direct NumPy
linear-regression proof before the wider classical model family is trained.

Prediction `X @ w + b`, MSE, analytical weight/bias gradients, central finite
differences, seeded initialization, and full-batch gradient descent are visibly
implemented without delegating the scratch fit to a modelling library. A
controlled synthetic problem proves the gradients and known parameter recovery.
The FlowCast slice compares scratch and scikit-learn LinearRegression on the
same frozen training/validation matrices. No final-test row or metric was
loaded, and no production model was selected.

## 2. Step 11 Implementation

- Added a validated `scratch_linear_v1` contract to `config/models.yaml` for
  next-window volume, the earliest chronological training subset, optimizer,
  numerical-gradient tolerances, and synthetic proof.
- Added pure NumPy prediction, MSE, analytical gradients, central finite
  differences, a seeded gradient-descent loop, convergence history, and JSON
  coefficient reload in `flowcast.modelling.scratch_linear`.
- Split synthetic proof, verified artifact loading, generated reporting, and
  real-data orchestration into focused modules; the largest new source module
  is 356 lines.
- Added reusable RMSE, MAE, MAPE, and R-squared calculations with explicit MAPE
  denominator counts under `flowcast.evaluation`.
- Added `flowcast train-scratch-linear [--version VERSION]`.
- Hash-verifies Step 10, loads train/validation only, deliberately proves the
  default test loader rejects access, and uses the persisted linear
  preprocessor for both estimators.
- Persists convergence, coefficients, model JSON, validation predictions,
  canonical metrics/lineage JSON, and generated Markdown with hashes.
- Added unit and full-data contracts for gradients, convergence,
  reproducibility, exact row sharing, sealed test, reload/prediction equality,
  deterministic artifacts, and tamper rejection.

## 3. Produced Artifacts

```text
artifacts/metrics/scratch_linear_v1/
  coefficients.csv
  convergence.csv
  summary.json
  summary.md

artifacts/models/scratch_linear_v1/
  model.json

artifacts/predictions/scratch_linear_v1/
  validation.parquet
```

The model and validation prediction artifacts are reproducibly generated and
remain ignored by Git. Their hashes and sizes are recorded in the tracked
canonical metrics summary.

## 4. Mathematical Proof Evidence

| Check | Verified result |
|---|---:|
| Central finite-difference parameters checked | 6 (5 weights + bias) |
| Maximum absolute gradient error | `2.6421437260e-09` |
| Maximum relative gradient error | `3.3878532351e-09` |
| Synthetic initial MSE | `10.5942952212` |
| Synthetic final MSE | `7.9389663111e-16` |
| Synthetic updates | 107 |
| Maximum synthetic coefficient error | `1.9887123948e-08` |
| Synthetic bias error | `6.8495747407e-09` |

Every analytical gradient passed the configured `atol=1e-6`, `rtol=1e-5`
comparison. Synthetic coefficients and bias passed the `1e-5` recovery gate.

## 5. Frozen FlowCast Comparison

- Target: `target_volume_h1` (30 minutes).
- Eligible frozen training population: 126,825 rows.
- Demonstration subset: earliest 25,000 eligible rows in timestamp/road order,
  spanning 2025-01-01 00:00 through 2025-01-21 19:30 Asia/Kolkata.
- Validation: all 27,150 eligible rows spanning 2025-04-16 17:00 through
  2025-05-09 07:30.
- Inputs: the same 62 manifest features and same 64-column fitted linear
  preprocessing output for both estimators.
- Test rows loaded: 0; test metrics calculated: false.

The configured scratch optimizer converged after 4,153 updates under its
relative-loss tolerance and patience rule. Training MSE decreased from
278,398.0056 to 6,716.1160.

| Estimator | Validation RMSE | MAE | MAPE | R-squared |
|---|---:|---:|---:|---:|
| NumPy gradient descent | 86.9331 | 62.9818 | 18.2032% | 0.902682 |
| scikit-learn LinearRegression | 80.8723 | 59.1634 | 16.5849% | 0.915778 |

The library baseline is better on this bounded validation demonstration. This
is retained as honest evidence and is not a formal test result or a production
selection. Step 12 owns full classical regression tuning and selection.

## 6. Artifact Evidence

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Coefficients CSV | 3,448 | `6f64a0f6e060190d69add70215ce79fb60c46092112cbcd5d6c631454e2ed72d` |
| Convergence CSV | 137,529 | `f674728b98451ef1f73200fda7bd70c29e944fff40c6acf3f99cc20989736ab2` |
| Scratch model JSON | 4,016 | `a0af8e3979ce6ddc836d70245445fcd52b51581c1c2fa06b4edd49c5bf8cce39` |
| Generated report | 2,152 | `5a473fe6b0c9459584a63c983af0883a1e25d19f9f098c3992b81e58d8d24d67` |
| Validation predictions | 605,805 | `73f6785f2f9f499b1c56378e8f84df74f8a36c21a9ab83ac1106e3a1a29fc944` |

## 7. Validation and Assurance Evidence

Executed with project-local CPython 3.11.9:

```text
.venv/Scripts/python.exe -m flowcast.cli prepare-modeling
.venv/Scripts/python.exe -m flowcast.cli train-scratch-linear
.venv/Scripts/python.exe -m pytest -q tests/unit/test_scratch_linear.py tests/data_contracts/test_scratch_linear_contract.py
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m compileall -q src tests
.venv/Scripts/python.exe -m flowcast.cli train-scratch-linear --help
.venv/Scripts/python.exe -c "from flowcast.modelling.scratch_inputs import load_scratch_linear_model"
git diff --check
```

Verified results:

- Focused Step 11 mathematics/full-data contracts: 8 passed in 35.43 seconds.
- Final full suite: 106 passed in 101.26 seconds.
- Gradient, convergence, real-data comparison, model reload, prediction
  reproduction, determinism, tamper evidence, and sealed-test checks passed.
- Dependency consistency, byte compilation, CLI help, persisted model loading
  (64 weights, 27,150 validation rows, zero test rows), and whitespace passed.
- All source files remain below 400 lines; the largest remains
  `src/flowcast/data/audit.py` at 366 physical lines.
- The full-suite EDA rerun used the now-available preferred Matplotlib renderer.
  Six tracked PNGs and their recorded hashes were refreshed; underlying EDA
  findings, contextual tables, processed data, and model inputs did not change.

## 8. Decisions and Constraints

- Step 11 intentionally uses a bounded earliest-chronological subset so the
  explicit full-batch loop remains fast, inspectable, and reproducible.
- Both estimators use identical rows and preprocessing. The comparison does not
  change features, tune on validation, or open test to improve the scratch
  result.
- Fixed-step gradient descent is stopped by configured relative-loss tolerance
  and patience. The coefficient difference from scikit-learn is preserved as a
  limitation of this mathematical demonstration.
- The scratch JSON coefficient artifact is reloadable without joblib or a
  modelling-library estimator; all learned values and lineage are auditable.
- Step 10 remains the frozen split/preprocessing authority. Its canonical
  summary was refreshed only for the extended model-config and EDA hashes.
- No dependency version, raw file, original source document, product contract,
  or final-test boundary changed.

## 9. Risks and Unresolved Work

- No full classical target/horizon family has been trained, tuned, selected, or
  evaluated on final test yet; formal accuracy targets remain unmeasured.
- The Step 11 scratch validation MAPE is above the formal volume target, but it
  is a bounded mathematical baseline rather than the selected final model.
- Step 12 must generate 48 regression fits across three targets, four horizons,
  and four required families efficiently while preserving test sealing until
  validation choices are frozen.
- Accident training remains severely imbalanced; classification threshold and
  calibration work remains in Step 13.
- Generated split/preprocessor/model/prediction binaries remain ignored and
  must be rebuilt by their documented CLI commands after a clean clone.

## 10. Next Gate

Proceed only to **Step 12 - Train Classical Regression Models**. The bounded
action and acceptance gate are maintained in `NEXT_STEP.md`.
