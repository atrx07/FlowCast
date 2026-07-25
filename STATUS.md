# STATUS.md

## Status Metadata

- **Project:** FlowCast v1.0
- **Last updated:** 2026-07-25
- **Current milestone:** M8 - Reproducibility and delivery (in progress)
- **Current step:** Steps 00-18 complete; Step 19 next
- **Overall state:** The complete frozen-model pipeline and ten-page Streamlit
  product surface are verified; clean reproduction and final delivery remain
- **Primary blocker:** None

## 1. Verified Current Position

FlowCast has a reproducible Python 3.11 pipeline from immutable raw inputs
through validation, cleaning, merge, leakage-safe features, four-horizon
targets, EDA, frozen chronological evaluation, training-only preprocessing,
NumPy regression proof, complete classical regression/classification, a
combined classical registry, a from-scratch recurrent volume forecaster,
validation-calibrated confidence/error analysis, and frozen-model
inference/reporting. Step 18 adds the complete Streamlit product surface over
those verified artifacts without changing their frozen decisions.

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

## 3. Step 18 Dashboard Implementation

- Added `dashboard/app.py` with native grouped `st.navigation`, shared
  road/date/horizon filters, verified artifact status, and directly addressable
  routes.
- Implemented all nine required views: live predictions, historical trends,
  congestion heatmap, road comparison, model performance, feature importance,
  forecast visualization, prediction confidence, and weather versus traffic.
- Added the tenth Data and training page for exact-schema upload validation,
  isolated staging, verified CSV/HTML report downloads, audit evidence, and
  explicit versioned retraining.
- Added `src/flowcast/dashboard/` service boundaries for recursively verified
  loading, metadata-aware caching, CPU predictor reuse, deterministic
  analytics, Plotly figures, shared state, upload staging, design components,
  and duplicate-safe retraining.
- Retraining requires the exact `RETRAIN` confirmation, writes a new versioned
  run/log manifest, prevents concurrent dashboard runs, and records that active
  routing was not switched. Ordinary reruns have no training path.
- Added `.streamlit/config.toml` and checked in `DESIGN.md`. The resulting
  dark, editorial interface follows the provided Linear-derived token system
  while retaining native Streamlit controls and accessibility semantics.
- Reworked the shared desktop opener from marketing-scale hero treatment to a
  compact operational header. At 1280 x 720 the live opener measures 122.3px
  instead of 372.0px, a 67.1% reduction, and the status strip now begins below
  Streamlit's top toolbar.
- Added reusable data-backed evidence briefs to all ten pages. Each brief states
  the current filtered reading and explains how to interpret its chart or
  table; values come from the same verified frame or persisted metric shown on
  that page, and weather language remains explicitly non-causal.
- Live predictions now defaults to the first horizon present in the latest
  verified request, so a legitimate user-generated subset does not open on an
  unavailable horizon. The dashboard contract test now reconciles the latest
  batch to its own verified request/coverage manifest instead of assuming the
  canonical 25-road/four-horizon batch is always newest.
- Replaced the arbitrary 336-timestamp, seven-day prediction-origin dropdown
  with native date and time controls over the complete verified history. The
  selector derives 7,237 full-corridor origins from the configured twelve-step,
  30-minute recurrent contract, covering 1 January 2025 at 05:30 through
  31 May 2025 at 23:30, and revalidates the combined value before inference.
- A real empty-state interpretation now preserves the live page's evidence-brief
  contract when the latest persisted batch does not match session filters.
- All displayed metrics, aggregates, predictions, intervals, probabilities,
  limitations, and lineage come from verified Step 08-17 artifacts. No
  placeholder analytics or fabricated future weather were introduced.

## 4. Canonical Inference and Report Evidence

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

## 5. Produced Artifacts

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

Step 18 source surface:

```text
.streamlit/config.toml
DESIGN.md
dashboard/app.py
dashboard/app_pages/                 # ten directly addressable page scripts
src/flowcast/dashboard/              # verified services and presentation layer
tests/unit/test_dashboard.py
tests/data_contracts/test_dashboard_contract.py
tests/smoke/test_dashboard_app.py
```

## 6. Validation and Assurance Evidence

Executed with project-local CPython 3.11.9:

```text
.venv/Scripts/python.exe -m flowcast.cli predict --horizons 1
.venv/Scripts/python.exe -m flowcast.cli predict --horizons 1 2 3 4 --export-reports
.venv/Scripts/python.exe -m flowcast.cli build-reports --manifest <manifest>
.venv/Scripts/python.exe scripts/run_tests.py -q tests/unit/test_inference_reporting.py tests/data_contracts/test_inference_reporting_contract.py
.venv/Scripts/python.exe scripts/run_tests.py -q tests/unit/test_confidence_analysis.py tests/data_contracts/test_confidence_analysis_contract.py tests/data_contracts/test_recurrent_volume_contract.py tests/unit/test_package.py tests/unit/test_build_safety.py
.venv/Scripts/python.exe scripts/run_tests.py -q
.venv/Scripts/python.exe -m compileall -q src dashboard tests scripts
.venv/Scripts/python.exe -m flowcast.cli predict --help
.venv/Scripts/python.exe -m flowcast.cli build-reports --help
.venv/Scripts/python.exe scripts/run_tests.py -q tests/unit/test_dashboard.py tests/data_contracts/test_dashboard_contract.py tests/smoke/test_dashboard_app.py
.venv/Scripts/python.exe -m pip check
git diff --check
```

Verified results:

- Focused Step 17 unit/full-artifact contracts: 9 passed in 14.54 seconds with
  `FLOWCAST_PYTEST_EXIT=0`.
- Affected confidence/recurrent/package/build-safety regression set: 21 passed
  in 22.82 seconds with `FLOWCAST_PYTEST_EXIT=0`.
- Focused dashboard unit/data-contract/all-page smoke suite: 7 passed in 12.29
  seconds with `FLOWCAST_PYTEST_EXIT=0`; it verifies the 7,237 eligible origin
  contract, native date/time widgets, and chart-reading guidance on every
  route.
- Affected feature/processed contracts: 10 passed in 7.08 seconds.
- Isolated classical regression/classification/scratch contracts: 15 passed in
  380.40 seconds without changing the canonical processed-data hash.
- Isolated confidence plus dashboard contracts: 7 passed in 43.63 seconds
  without changing canonical confidence artifacts.
- Complete repository suite: 180 passed in 510.45 seconds with
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
- Canonical processed and confidence Parquet hashes were identical before and
  after the complete suite.
- Browser QA loaded all ten routes at 1280 x 720, 1440 x 900, and
  1920 x 1080. The final settled 1280 pass found no exceptions, horizontal
  overflow, top-toolbar overlap, or missing briefs; maximum opener height was
  182.9px and the status strip began at 63.8px. Representative live,
  model-performance, confidence, and system-control pages were also visually
  inspected.
- Follow-up browser QA verified the origin calendar, 48-choice half-hour time
  menu, visible January-May eligibility range, side-by-side desktop geometry,
  and zero horizontal overflow at 1280 x 720, 1440 x 900, and 1920 x 1080;
  the browser console reported no errors.

## 7. Decisions and Constraints

- Recurrent volume is active at all horizons because validation RMSE is lower
  at all four; the 120-minute test deficit remains reported and cannot alter
  this validation-led policy.
- Classical volume predictions remain visible as a comparator/fallback.
- Observed origin weather is used; no future-weather values are fabricated.
- Inference accepts only origins present in the verified processed dataset.
- The dashboard exposes every full-corridor origin that satisfies the frozen
  twelve-row recurrent history requirement; this excludes only the first
  eleven dataset timestamps rather than hiding an arbitrary recent window.
- Report insights are deterministic aggregates over persisted prediction rows.
- Runtime evidence records both service initialization and prediction time.
- CPU is the default and required reproduction path.
- Streamlit `1.59.2` requires PyArrow below 25, so the approved direct pin is
  PyArrow `24.0.0`. Existing frozen artifacts retain their original verified
  PyArrow 25 bytes; tests now stage every writable processed/model/confidence
  path under temporary roots and never mix byte identities across versions.
- The provided `gpt-taste` visual direction was adapted to the mandatory
  single-app Streamlit architecture. React, GSAP, FastAPI, a database, and a
  second frontend were not added.

## 8. Risks and Unresolved Work

- Congestion Macro-F1 and accident ROC-AUC formal targets remain unmet and are
  included in exported report evidence.
- Low accident prevalence makes probability ranking and high/critical risk
  groups uncertain.
- The recurrent model still trails classical volume slightly at the
  120-minute test horizon, especially in late-night slices.
- Final clean reproduction, cross-platform reviewer verification, the final
  technical report, and delivery packaging remain.
- Step 19 must rebuild one internally consistent artifact lineage under the
  pinned PyArrow 24 environment; it must not combine newly serialized Parquet
  files with frozen PyArrow 25 manifests.
- Generated models, predictions, and reports are ignored by Git and must be
  rebuilt through documented CLI commands after a clean clone.
- The Matplotlib default Windows cache path is sandbox-restricted on this
  workstation; commands/tests use a writable `MPLCONFIGDIR` when needed.

## 9. Next Gate

Proceed only to **Step 19 - Reproducibility, Documentation, and Final
Acceptance**. The bounded action
and evidence gate are maintained in `NEXT_STEP.md`.

## 10. Step 18 Environment and Build Evidence

- Installed the pinned dashboard stack: Streamlit `1.59.2`, Plotly `6.9.0`,
  Seaborn `0.13.2`, and their transitive dependencies.
- Streamlit `1.59.2` requires PyArrow below version 25, so the direct PyArrow
  pin was changed from `25.0.0` to `24.0.0`, the highest compatible release.
  The Parquet artifact format and project interfaces are unchanged.
- Installed Streamlit's official `developing-with-streamlit` agent skill with
  `streamlit skills --yes`. Windows project symlinks were unavailable, so the
  command used its supported global fallback under
  `~/.agents/skills/developing-with-streamlit`.
- Verified Python `3.11.9`, Streamlit `1.59.2`, Plotly `6.9.0`, Seaborn
  `0.13.2`, Matplotlib `3.11.1`, and PyArrow `24.0.0` imports.
- `python -m pip check` reports no broken requirements, a temporary
  pandas/PyArrow Parquet write/read round-trip passed, and Streamlit's CLI
  reports version `1.59.2`.
- Focused package/build-safety validation passed 7 tests in 0.83 seconds with
  `FLOWCAST_PYTEST_EXIT=0`; the repository guard reported no tracked mutation.
- The dashboard and all assurance work ran on the Intel Core Ultra 9 CPU. The
  NVIDIA GeForce RTX 5070 Laptop GPU/VRAM and Intel NPU were not used.
