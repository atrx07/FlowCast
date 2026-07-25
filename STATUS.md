# STATUS.md

## Status Metadata

- **Project:** FlowCast v1.0
- **Last updated:** 2026-07-25
- **Current milestone:** M7 - Dashboard and service layer (in progress)
- **Current step:** Steps 00-17 complete; Step 18 next
- **Overall state:** Frozen-model inference and reporting services are verified;
  the Streamlit dashboard is the next gate
- **Primary blocker:** None

## 1. Verified Current Position

FlowCast has a reproducible Python 3.11 pipeline from immutable raw inputs
through validation, cleaning, merge, leakage-safe features, four-horizon
targets, EDA, frozen chronological evaluation, training-only preprocessing,
NumPy regression proof, complete classical regression/classification, a
combined classical registry, a from-scratch recurrent volume forecaster,
validation-calibrated confidence/error analysis, and frozen-model
inference/reporting.

Step 17 loads only verified Step 08/10/14/15/16 artifacts. It does not fit a
preprocessor, model, calibrator, threshold, or confidence width. The recurrent
volume route is frozen from lower validation RMSE at all four horizons; the
classical volume model remains an explicit comparator. Speed, travel time,
congestion, and accident risk resolve through the frozen Step 14 registry.

`FlowCast-project_file/`, `data/raw/`, the delivered CSV/DOCX sources, and all
Step 10-16 source artifacts remain unchanged.

## 2. Step 17 Implementation

- Added standalone `config/inference.yaml` with frozen upstream versions,
  validation-led active routing, explicit classical-volume fallback,
  request/cadence/sequence rules, CPU-default device policy, output schemas,
  report formats, and the 30-second full-corridor target.
- Added typed `PredictionRequest`, `PredictionResult`, and `Predictor`
  interfaces under `flowcast.inference`.
- Latest-origin preparation verifies road coverage, 30-minute alignment, and
  twelve contiguous road-local recurrent rows before preprocessing.
- The service returns volume, speed, travel time, four-class congestion
  probabilities, accident probability/decision/risk band, regression
  intervals, classifier confidence/entropy, origin/target timestamps, and exact
  data/model/config lineage for every requested road and horizon.
- Active recurrent volume inference uses the portable state dictionary with
  explicit `map_location`; ordinary inference defaults to CPU and guarded CUDA
  remains optional.
- Classical estimators load only through verified source loaders after the
  registry is recursively verified. Loaded entries are checked back against
  their registry model/card records.
- Step 16 conformal widths, classifier confidence bands, accident thresholds,
  and risk-band multipliers are applied unchanged.
- Added deterministic Parquet prediction batches plus JSON manifests, verified
  batch reload, real-data insight aggregation, full CSV export, self-contained
  HTML export, and report-manifest verification.
- Added `flowcast predict` and `flowcast build-reports`. Normal prediction and
  report generation contain no training path.

## 3. Canonical Inference and Report Evidence

Canonical CPU requests used the latest common origin
`2025-05-31T23:30:00+05:30`.

### Full-corridor one-horizon benchmark

- Request ID: `3e0585348e1f7f4b`.
- Roads/horizons/rows: 25 / 1 / 25.
- Service initialization: 0.844 seconds.
- Prediction execution including first-use model loading: 1.506 seconds.
- Cold total: 2.350 seconds.
- Result: meets the PRD target of at most 30 seconds.

### Full five-target, four-horizon request

- Request ID: `0f02bc6449c56a75`.
- Roads/horizons/rows: 25 / 4 / 100.
- Service initialization: 0.817 seconds.
- Prediction execution: 3.404 seconds.
- Cold total: 4.222 seconds.
- Verified reload reconciles all 100 rows, 25 roads, four horizons, schema,
  model records, upstream hashes, and the deterministic request identifier.
- CSV and self-contained HTML reports were generated and verified from the
  persisted batch.

The RTX 5070 Laptop GPU/VRAM and Intel NPU were not used. Step 17 ran on the
Intel Core Ultra 9 CPU.

## 4. Produced Artifacts

```text
config/inference.yaml

src/flowcast/inference/
  artifacts.py
  confidence.py
  config.py
  feature_prep.py
  inputs.py
  model_router.py
  predictor.py
  schemas.py

src/flowcast/reports/
  export.py
  insights.py

artifacts/predictions/inference_reporting_v1/
  3e0585348e1f7f4b/               # ignored, reproducible benchmark
  0f02bc6449c56a75/               # ignored, reproducible complete batch

artifacts/reports/inference_reporting_v1/0f02bc6449c56a75/
  predictions.csv                 # ignored, reproducible
  report.html                     # ignored, reproducible
  manifest.json                   # ignored, reproducible
```

Canonical complete-batch sizes:

| Artifact | Bytes |
|---|---:|
| Prediction manifest | 20,687 |
| Prediction Parquet | 50,977 |
| Report manifest | 4,110 |
| Report CSV | 83,628 |
| Self-contained HTML | 3,211 |

## 5. Validation and Assurance Evidence

Executed with project-local CPython 3.11.9:

```text
.venv/Scripts/python.exe -m flowcast.cli predict --horizons 1
.venv/Scripts/python.exe -m flowcast.cli predict --horizons 1 2 3 4 --export-reports
.venv/Scripts/python.exe -m flowcast.cli build-reports --manifest <manifest>
.venv/Scripts/python.exe scripts/run_tests.py -q tests/unit/test_inference_reporting.py tests/data_contracts/test_inference_reporting_contract.py
.venv/Scripts/python.exe scripts/run_tests.py -q tests/unit/test_confidence_analysis.py tests/data_contracts/test_confidence_analysis_contract.py tests/data_contracts/test_recurrent_volume_contract.py tests/unit/test_package.py tests/unit/test_build_safety.py
.venv/Scripts/python.exe scripts/run_tests.py -q
.venv/Scripts/python.exe -m compileall -q src tests scripts
.venv/Scripts/python.exe -m flowcast.cli predict --help
.venv/Scripts/python.exe -m flowcast.cli build-reports --help
.venv/Scripts/python.exe -m pip check
git diff --check
```

Verified results:

- Focused Step 17 unit/full-artifact contracts: 9 passed in 14.54 seconds with
  `FLOWCAST_PYTEST_EXIT=0`.
- Affected confidence/recurrent/package/build-safety regression set: 21 passed
  in 22.82 seconds with `FLOWCAST_PYTEST_EXIT=0`.
- Complete repository suite: 173 passed in 504.65 seconds with
  `FLOWCAST_PYTEST_EXIT=0`.
- Repeated seeded CPU inference returns exactly equal prediction frames.
- Invalid roads, horizons, cadence, origins, and insufficient sequence history
  fail clearly.
- Prediction and report byte tampering is rejected before data loads.
- Report CSV rows/schema reconcile to the verified prediction batch; HTML
  identifies its exact request and contains only real aggregates/evidence.
- CLI help, dependency consistency, source/test byte compilation, whitespace,
  and source-size assurance pass.
- Every source file remains below 400 physical lines.
- The full test session completed its repository comparison without a tracked
  file mutation.

## 6. Decisions and Constraints

- Recurrent volume is active at all horizons because validation RMSE is lower
  at all four; the 120-minute test deficit remains reported and cannot alter
  this validation-led policy.
- Classical volume predictions remain visible as a comparator/fallback.
- Observed origin weather is used; no future-weather values are fabricated.
- Inference accepts only origins present in the verified processed dataset.
- Report insights are deterministic aggregates over persisted prediction rows.
- Runtime evidence records both service initialization and prediction time.
- CPU is the default and required reproduction path.

## 7. Risks and Unresolved Work

- Congestion Macro-F1 and accident ROC-AUC formal targets remain unmet and are
  included in exported report evidence.
- Low accident prevalence makes probability ranking and high/critical risk
  groups uncertain.
- The recurrent model still trails classical volume slightly at the
  120-minute test horizon, especially in late-night slices.
- The Streamlit views, upload validation, explicit retraining control, and
  final clean reproduction remain.
- Generated models, predictions, and reports are ignored by Git and must be
  rebuilt through documented CLI commands after a clean clone.
- The Matplotlib default Windows cache path is sandbox-restricted on this
  workstation; commands/tests use a writable `MPLCONFIGDIR` when needed.

## 8. Next Gate

Proceed only to **Step 18 - Build the Streamlit Dashboard**. The bounded action
and evidence gate are maintained in `NEXT_STEP.md`.
