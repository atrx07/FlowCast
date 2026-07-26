# ARCHITECTURE.md
## 1. Architectural Goal
FlowCast uses one user-facing Streamlit web application backed by a reusable Python package. The same package powers CLI workflows, tests, model training, batch inference, report generation, and dashboard views.

The architecture must remain modular, reproducible, time-series safe, cross-platform, and small enough for a single engineer to maintain.

## 2. System Context
```text
Delivered CSV files + reference documents
                |
                v
       Ingestion and validation
                |
       quarantine + validated tables
                |
                v
       Cleaning and temporal alignment
                |
                v
       Feature and target generation
                |
       processed/versioned datasets
           |                   |
           v                   v
   Classical ML engine   Deep-learning engine
           |                   |
           +---------+---------+
                     v
      Evaluation, calibration, registry
                     |
       predictions + metrics + model cards
                     |
                     v
            Streamlit dashboard
                     |
        forecasts, analytics, reports
```

## 3. Runtime Surfaces
### 3.1 CLI
The canonical automation surface. It must support individual stages and an end-to-end run.

Proposed commands:

```bash
python -m flowcast.cli audit
python -m flowcast.cli validate
python -m flowcast.cli clean-context
python -m flowcast.cli clean-traffic
python -m flowcast.cli merge-sources
python -m flowcast.cli engineer-features
python -m flowcast.cli prepare-data
python -m flowcast.cli eda
python -m flowcast.cli prepare-modeling
python -m flowcast.cli train-scratch-linear
python -m flowcast.cli train-classical-regression
python -m flowcast.cli train-classical-classification
python -m flowcast.cli build-classical-registry
python -m flowcast.cli train-recurrent-volume
python -m flowcast.cli analyze-confidence
python -m flowcast.cli predict --horizons 1 2 3 4
python -m flowcast.cli build-reports
python -m flowcast.cli run-all
```

Command names may be refined during implementation, but one stable `run-all` equivalent is mandatory.

### 3.2 Streamlit
```bash
streamlit run dashboard/app.py
```

The dashboard imports services from the `flowcast` package. It does not duplicate data/model logic inside pages.

### 3.3 Notebooks
Notebooks explain EDA and experiments. They call package functions and never become the only implementation of a required pipeline stage.

## 4. Proposed Repository Structure
```text
flowcast-repository/
├── AGENTS.md
├── PROJECT.md
├── ROADMAP.md
├── ARCHITECTURE.md
├── STEPS.md
├── STATUS.md
├── NEXT_STEP.md
├── README.md
├── FlowCast-project_file/          # original reference directory, read-only
├── config/
│   ├── base.yaml
│   ├── data_contracts.yaml
│   ├── cleaning.yaml
│   ├── features.yaml
│   ├── models.yaml
│   └── registry.yaml
├── data/
│   ├── raw/                        # immutable CSV copies
│   ├── interim/                    # validated/cleaned source-level outputs
│   ├── processed/                  # analysis-ready/versioned Parquet files
│   └── quarantine/                 # invalid rows and rejection reasons
├── artifacts/
│   ├── audits/
│   ├── quality/
│   ├── features/
│   ├── predictions/
│   ├── metrics/
│   ├── models/
│   │   ├── classical/
│   │   └── deep/
│   ├── model_cards/
│   └── reports/
├── dashboard/
│   ├── app.py
│   ├── components/
│   │   ├── filters.py
│   │   ├── cards.py
│   │   └── charts.py
│   └── pages/
│       ├── 01_live_predictions.py
│       ├── 02_historical_trends.py
│       ├── 03_congestion_heatmap.py
│       ├── 04_road_comparison.py
│       ├── 05_model_performance.py
│       ├── 06_feature_importance.py
│       ├── 07_forecast_visualisation.py
│       ├── 08_prediction_confidence.py
│       ├── 09_weather_vs_traffic.py
│       └── 10_data_and_training.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_analysis.ipynb
├── src/
│   └── flowcast/
│       ├── __init__.py
│       ├── cli.py
│       ├── settings.py
│       ├── logging_config.py
│       ├── data/
│       │   ├── contracts.py
│       │   ├── ingest.py
│       │   ├── validation.py
│       │   ├── validation_state.py
│       │   ├── audit.py
│       │   ├── artifacts.py
│       │   ├── cleaning_types.py
│       │   ├── clean_traffic.py
│       │   ├── traffic_recovery.py
│       │   ├── traffic_pipeline.py
│       │   ├── clean_weather.py
│       │   ├── clean_calendar.py
│       │   ├── clean_context.py
│       │   ├── validated_inputs.py
│       │   ├── cleaned_inputs.py
│       │   ├── merge.py
│       │   ├── merge_pipeline.py
│       │   ├── quarantine.py
│       │   └── quality_report.py
│       ├── features/
│       │   ├── temporal.py
│       │   ├── traffic.py
│       │   ├── weather.py
│       │   ├── calendar.py
│       │   ├── targets.py
│       │   └── pipeline.py
│       ├── modelling/
│       │   ├── split.py
│       │   ├── preprocessing.py
│       │   ├── scratch_linear.py
│       │   ├── regression.py
│       │   ├── classification.py
│       │   ├── imbalance.py
│       │   ├── sequence_data.py
│       │   ├── recurrent.py
│       │   ├── training.py
│       │   ├── calibration.py
│       │   └── registry.py
│       ├── evaluation/
│       │   ├── regression.py
│       │   ├── classification.py
│       │   ├── error_analysis.py
│       │   ├── scoreboard.py
│       │   └── model_cards.py
│       ├── inference/
│       │   ├── schemas.py
│       │   ├── predictor.py
│       │   ├── batch.py
│       │   └── confidence.py
│       ├── reports/
│       │   ├── insights.py
│       │   └── export.py
│       └── services/
│           ├── dashboard_data.py
│           ├── upload_service.py
│           └── training_service.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── data_contracts/
│   └── smoke/
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

The implemented Step 11 modelling boundary splits pure mathematics
(`scratch_linear.py`), synthetic proof (`scratch_proof.py`), verified artifact
loading (`scratch_inputs.py`), generated reporting (`scratch_report.py`), and
real-data orchestration (`regression.py`) by responsibility. Shared regression
metrics live in `flowcast/evaluation/regression.py` for reuse in Step 12.

This is a target structure, not permission to create empty files unnecessarily. Create modules when their step begins. Keep each source file below 500 lines unless documented otherwise.

## 5. Configuration Architecture
### `config/base.yaml`
- Directory paths.
- Global seed.
- timezone policy.
- logging level.
- validation, cleaning, merge, feature, target, EDA, and modelling-preparation
  artifact versions.

### `config/data_contracts.yaml`
- Required columns and types.
- uniqueness keys.
- allowed categorical values.
- physical range constraints.
- null policy.
- weather normalization map.

### `config/cleaning.yaml`
- Context- and traffic-cleaning contract versions.
- Complete traffic-grid bounds and expected road count.
- Field-specific causal imputation hierarchy and maximum supported gap lengths.
- Vehicle-share key, tolerance, naming, and normalization policy.
- Calendar flag/name relationship checks.

### `config/features.yaml`
- Explanatory-feature contract and output version.
- Lag windows 1, 2, and 48; rolling windows 4 and 8.
- Forecast horizons 1-4 reserved for target construction.
- Named half-open peak periods.
- Capacity denominator, rain/visibility thresholds, weather categories, and
  temperature bands.
- Event-proximity window.

### `config/eda.yaml`
- EDA/report contract and artifact version.
- Descriptive fields and required road/time/weather/calendar context slices.
- Safe origin-time correlation candidates and target-association definition.
- Redundancy threshold, congestion order, and figure settings.

### `config/models.yaml`
- Exact chronological split dates, ratios, timestamp counts, and target-boundary
  policy.
- Expanding-window CV fold count, validation length, and maximum-horizon gap.
- Default-sealed test-access purpose.
- Feature grouping, imputation, encoding, and per-family scaling policies.
- Step 11 scratch-linear target, chronological row budget, optimizer,
  finite-difference tolerances, and synthetic parameter-recovery contract.
- Step 12 direct-regression targets/horizons, four required estimator families,
  bounded two-candidate search spaces, deterministic CV timestamp budget,
  validation selection rule, and pre-test freeze rule.
- Step 13 congestion/accident targets and class order, four classifier
  families, bounded two-candidate search spaces, chronological sigmoid
  calibration assessment, accident-threshold selection, and pre-test freeze.
- `config/registry.yaml` independently defines the Step 14 frozen source
  versions, target/horizon order, task-aware metrics, acceptance rules,
  prediction-index policy, and deterministic key template. Keeping registry
  settings separate prevents reporting changes from invalidating frozen
  training hashes.
- `config/recurrent.yaml` independently defines the Step 15 multi-horizon
  volume target, sequence isolation policy, bounded LSTM candidates, Adam
  settings, learning-rate schedule, early stopping, device policy, validation
  selection rule, and exact-row classical comparison. Keeping it independent
  prevents recurrent work from invalidating the frozen Step 10-14 lineage.
- `config/confidence.yaml` independently defines the Step 16 validation-only
  interval method, probability diagnostics, accident risk-band multipliers,
  reliability bins, subgroup dimensions, and minimum-support policy. It names
  every frozen upstream version and cannot alter a model-selection hash.
- `config/inference.yaml` independently defines the Step 17 active routing,
  upstream versions, request/cadence/sequence contract, CPU-default device
  policy, output schema, report formats, and full-corridor runtime target.
  Reporting or dashboard changes therefore cannot silently change a frozen
  training, selection, calibration, or confidence hash.

## 6. Data Contracts
### 6.1 Raw traffic contract
Key: `road_id + timestamp` after timestamp construction.

Required source fields are the 17 delivered columns. Original values are preserved in raw copies. Validation output adds metadata such as:

- `_source_file`
- `_source_row`
- `_validation_status`
- `_rejection_reason`

### 6.2 Validated traffic contract
- Parsed timestamp.
- One preferred row per road/timestamp after deterministic duplicate resolution.
- Physically invalid cells set to missing with flags, not silently accepted.
- Source filename and physical CSV row retained on every output row.
- Missing full windows are represented later during segment-wise reindexing.
- Validated JSON remains serialized until vehicle shares are expanded during
  traffic cleaning.

### 6.3 Weather contract
Key: `station_id + weather_hour`.

- Date parsed with day-first semantics.
- Controlled vocabulary: Clear, Cloudy, Overcast, Rain, and Fog.
- Complete hourly grid per station.
- Missing temperature and visibility values use a causal station-local forward
  fill limited to two consecutive hours.
- Every imputed value carries a missingness flag, method, and donor source-row
  identity; unsupported leading or longer gaps fail closed.

### 6.4 Calendar contract
Key: normalized date.

- Boolean flags validated as 0/1.
- Empty names allowed only when the corresponding flag is zero.

### 6.5 Validation artifact boundary

The `validate` command writes one versioned Parquet file per retained source to
`data/interim/<validation_version>/`. It writes rejected source rows, a unified
cell/row issue ledger, and `summary.json` to
`data/quarantine/<validation_version>/`.

The summary records source hashes, row accounting, reason/disposition counts,
artifact hashes, and dataset-level schema failure. Recoverable invalid cells are
set to missing and labelled `valid_with_issues`; invalid structural rows and
non-retained duplicate keys are quarantined. A schema failure still serializes
the affected rows and causes the CLI to return a non-zero exit status.

### 6.6 Cleaned context artifact boundary

The `clean-context` command verifies the hashes of the `validated_v1` calendar
and weather Parquet inputs. It writes trusted outputs to
`data/interim/<cleaning_version>/` and writes canonical JSON plus generated
Markdown quality evidence to `artifacts/quality/<cleaning_version>/`.

The cleaned calendar preserves one row per normalized date. The cleaned weather
table preserves one row per station/hour and retains validation/source lineage
alongside normalization and imputation metadata.

### 6.7 Cleaned traffic artifact boundary

The `clean-traffic` command verifies the validated traffic Parquet and unified
issue-ledger hashes. It reconstructs the complete 25-road by 7,248-window grid,
writes `traffic.parquet` beneath `data/interim/<cleaning_version>/`, and writes
canonical JSON plus generated Markdown evidence beneath
`artifacts/quality/<cleaning_version>/`.

Every repaired measurement stores original-missing/physical-invalid state,
method, donor timestamp, and donor source-row lineage. Inserted windows are
explicit. Vehicle JSON is preserved and expanded to four normalized shares.
Blank congestion is derived from exact half-hour V/C bands while existing
labels are preserved and audited. Accident count remains unknown on inserted
windows and `_accident_observed` prevents those rows from becoming fabricated
negative targets.

### 6.8 Merged contract
Key: `road_id + timestamp`.

- `merge-sources` hash-verifies all three cleaned Parquet files and both cleaning
  summaries before reading them.
- Weather joins by station and floored local hour with explicit many-to-one
  validation; calendar joins by normalized local date with the same validation.
- `weather_join_status` and `calendar_join_status`, matched/missing counts,
  source hashes, and prefixed weather/calendar source-row lineage are persisted.
- The output is `data/interim/<merge_version>/merged.parquet`; canonical JSON
  and generated Markdown live under `artifacts/quality/<merge_version>/`.
- Any duplicate right key, unexpected miss, row-count change, or duplicate
  traffic output key fails closed.
- `weather_station_id` may remain for lineage but is excluded where redundant
  for modelling.

### 6.9 Processed feature contract
- Sorted by `road_id, timestamp`.
- Stable feature names and dtypes.
- Step 07 writes `data/interim/engineered_features_v1/features.parquet`; Step
  08 preserves its 144 columns exactly and writes the 188-column
  `data/processed/processed_targets_v1/dataset.parquet`.
- Rows without enough history are retained and marked by `history_available`;
  each of the 20 target/horizon pairs has its own nullable target and explicit
  boolean availability mask.
- `artifacts/features/<feature_version>/manifest.json` records source columns,
  transforms, dtype, version, and leakage classification for every
  model-candidate feature.
- Canonical quality JSON and generated Markdown under
  `artifacts/quality/<feature_version>/` record feature null/range counts and
  input/output/manifest hashes.
- `prepare-data` verifies the Step 07 summary, manifest, configuration hashes,
  Parquet hash, cardinality, and feature dtypes before target construction.
- `artifacts/features/processed_targets_v1/manifest.json` records the complete
  188-column schema, 20 target definitions, source/transform/horizon/dtype/mask
  metadata, and input/output lineage. Coverage JSON and generated Markdown live
  under `artifacts/quality/processed_targets_v1/`.

### 6.10 EDA and quality-report artifact boundary

- `eda` hash-verifies the processed Parquet, schema manifest, Step 08 summary,
  current configuration files, complete upstream quality-summary chain, and
  immutable raw-copy hashes before analysis.
- `src/flowcast/analysis/` owns descriptive calculations, contextual
  aggregates, correlation/covariance, quality reconciliation, figure rendering,
  generated Markdown, and pipeline orchestration. The EDA notebook calls this
  package boundary and contains no unique transformations.
- Canonical machine-readable results live under
  `artifacts/reports/<eda_version>/`; deterministic PNGs live under
  `artifacts/figures/<eda_version>/`. Every artifact path, byte count, and
  SHA-256 is recorded in the canonical summary.
- Matplotlib is the preferred renderer. A Pillow renderer provides the approved
  deterministic fallback when platform application control blocks a compiled
  Matplotlib extension; the output contract remains PNG.
- Identifiers, lineage strings, timestamps, future targets, and target masks are
  excluded from origin-feature correlation candidates. Full-data redundancy
  flags are descriptive and cannot make feature-selection decisions outside a
  training fold.

### 6.11 Frozen split and preprocessing artifact boundary

- `prepare-modeling` verifies the full Step 09 and processed-feature lineage
  before reading model inputs.
- `data/processed/split_preprocessing_v1/assignments.parquet` assigns every
  `road_id + timestamp` origin to train, validation, or test and records four
  horizon-within-partition flags. Target-specific availability remains in the
  processed dataset and combines with these flags at model-load time.
- `artifacts/features/split_preprocessing_v1/` stores canonical split/CV
  evidence, the exact 62-feature input schema, learned training statistics,
  training-only class weights, artifact hashes, and generated Markdown.
- Four fitted Joblib preprocessors live under
  `artifacts/models/split_preprocessing_v1/`: linear and SVM standardize numeric
  fields, trees retain unscaled numeric fields, and recurrent preprocessing
  applies Min-Max scaling to documented bounded fields and standardization to
  the remainder. All families use training-fitted imputation and one-hot
  encoding.
- Tuning access can load train and validation only. Loading test requires the
  explicit `final_evaluation` purpose after model selection is frozen.

### 6.12 NumPy regression proof artifact boundary

- `train-scratch-linear` verifies the complete Step 10 summary, assignments,
  feature schema, and fitted linear preprocessor before loading data.
- It selects the earliest 25,000 boundary-safe `target_volume_h1` training
  rows in timestamp/road order and uses every eligible validation row; the test
  loader is called only to prove default access is rejected before any test row
  can load.
- `scratch_linear.py` owns the visible `X @ w + b` prediction, MSE, analytical
  gradients, central finite differences, and seeded full-batch update loop.
  Synthetic proof helpers, verified loading, generated reporting, and pipeline
  orchestration remain separate modules.
- Canonical JSON, generated Markdown, convergence CSV, and coefficient CSV live
  under `artifacts/metrics/scratch_linear_v1/`. The reloadable JSON coefficient
  model and validation prediction Parquet are reproducible ignored artifacts
  under `artifacts/models/` and `artifacts/predictions/`; all paths and hashes
  are recorded in the canonical summary.
- The comparison against scikit-learn uses identical preprocessed training and
  validation matrices and reports RMSE, MAE, MAPE, and R-squared. It is a
  mathematical verification, not final model selection or test evaluation.

### 6.13 Classical regression artifact boundary

- `train-classical-regression` generates 12 direct jobs from the processed
  target manifest: volume, speed, and travel time at horizons 1-4.
- Linear Regression, Decision Tree, Random Forest, and XGBoost candidates use
  all five frozen horizon-gapped folds. Each fold fits a fresh training-only
  preprocessor; the deterministic timestamp budget spans the fold's complete
  expanding training interval while each final family fit uses all eligible
  training rows.
- Mean CV RMSE selects hyperparameters within a family. Validation RMSE then
  selects one family per target/horizon, with documented deterministic
  tie-breakers.
- All 12 selected pipelines and `selection_manifest.json` are persisted before
  one explicit `final_evaluation` test-partition load. Models are never refit
  after that load.
- Canonical candidate/fold/family/final scoreboards and feature importance live
  under `artifacts/metrics/classical_regression_v1/`. Selected Joblib pipelines,
  combined validation/test predictions, and JSON plus Markdown model cards live
  under the corresponding versioned model, prediction, and model-card roots.
- The canonical summary records input/config hashes, the selection-manifest
  hash, all output hashes, runtime, library versions, model lineage, and the
  single test-access evidence. Reloading verifies the complete chain before
  returning a selected pipeline.

### 6.14 Classical classification artifact boundary

- `train-classical-classification` generates eight direct jobs from the
  processed target manifest: four-class congestion and binary accident risk at
  horizons 1-4.
- Decision Tree, Random Forest, XGBoost, and scaled linear SVM candidates use
  all five frozen horizon-gapped folds. Each fold fits fresh preprocessing and
  derives balancing weights from only that fold's sampled training labels.
- Mean CV Macro-F1 selects congestion hyperparameters; mean CV ROC-AUC selects
  accident hyperparameters. The corresponding full-validation metric selects
  one family per target/horizon.
- Each selected training-fitted model uses the earlier half of validation to
  fit a sigmoid calibrator and the later half to assess Brier improvement. SVM
  calibration is mandatory because LinearSVC has no native probabilities;
  other families retain raw probabilities unless the configured Brier
  improvement gate passes.
- Accident operating thresholds maximize F1 on only the later validation
  assessment slice, with recall, precision, and lower threshold as deterministic
  tie-breakers.
- All eight probability classifiers, calibration decisions, four accident
  thresholds, and `selection_manifest.json` are persisted before one explicit
  `final_evaluation` test load. No estimator, threshold, calibrator, or feature
  changes after that load.
- Canonical fold/candidate/family/final scoreboards, ordered probabilities,
  per-class metrics, confusion matrices, threshold tables, calibration
  evidence, feature importance, hashes, JSON/Markdown model cards, and a
  verified Joblib loader live under `classical_classification_v1`.

### 6.15 Recurrent volume artifact boundary

- `train-recurrent-volume` recursively verifies the Step 10 modelling summary,
  recurrent preprocessor, processed lineage, classical regression outputs, and
  Step 14 registry before training.
- `sequence_data.py` owns lazy road-local sequence views, contiguous-cadence
  checks, common validation/test origin eligibility, and training-only
  per-horizon target scaling. A sequence cannot cross a road, chronological
  partition, 30-minute gap, or target boundary.
- `recurrent_model.py` owns the from-scratch PyTorch LSTM/GRU plus four-value
  volume head. `recurrent_training.py` owns seeding, Adam, mini-batches,
  gradient clipping, learning-rate reduction, validation-led early stopping,
  best-weight restoration, and deterministic inference.
- Candidate metrics, epoch curves, feature/scaler manifests, sequence
  evidence, selection manifest, pre-test card, and the selected state-dictionary
  checkpoint are persisted before one explicit `final_evaluation` test load.
- Canonical metrics, exact-row classical comparison, JSON/Markdown card,
  environment snapshot, four-entry registry extension, and long-form
  validation/test predictions live under `recurrent_volume_v1`. Verified
  loading checks every recorded dependency before reconstructing the network.

### 6.16 Confidence and error-analysis artifact boundary

- `analyze-confidence` recursively verifies the processed-data, classical
  registry/regression/classification, and recurrent summary chains before
  reading their frozen validation/test prediction Parquets.
- `flowcast/evaluation/confidence_metrics.py` owns finite-sample conformal
  calibration, interval application, context joins, probability validation,
  entropy, confidence bands, reliability bins, and aggregate coverage.
- `confidence_slices.py` owns minimum-support regression/classification slices;
  `confidence_pairing.py` owns exact-row recurrent/classical comparison;
  `confidence_report.py` owns deterministic diagnoses and Markdown.
- Dashboard-ready row-level Parquets live under
  `artifacts/predictions/confidence_error_v1/`. Calibration, coverage,
  reliability, risk-band, slice, confusion, paired-comparison, JSON, and
  Markdown evidence lives under `artifacts/metrics/confidence_error_v1/`.
- The public confidence loader recursively re-verifies upstream lineage and
  every recorded output hash before returning the three row-level tables.
- The confidence layer is immutable-model evaluation: it cannot refit, select,
  recalibrate, rethreshold, or replace a Step 12-15 prediction.

### 6.17 Inference and reporting artifact boundary

- `flowcast.inference.Predictor` is the single public forecast interface.
  Initialization verifies the Step 08 processed chain, Step 10 preprocessing
  chain, Step 14 registry, Step 15 recurrent checkpoint, and Step 16
  calibration before model use.
- Active recurrent volume routing is frozen from validation-only evidence at
  all four horizons. The classical registry volume winner remains an explicit
  comparator/fallback; speed, travel time, congestion, and accident models
  resolve through the frozen registry.
- Feature preparation accepts only a verified processed origin, checks
  half-hour cadence and road coverage, and requires twelve contiguous
  road-local rows for the recurrent input. It never constructs future observed
  weather or target features.
- Every output row contains five target predictions, conformal interval or
  probability confidence, accident threshold/risk band, origin and target
  timestamps, active/fallback model versions, data/feature/preprocessing
  versions, registry/confidence/inference versions, and the processed-data
  SHA-256.
- `flowcast.inference.artifacts` writes one request-scoped Parquet plus JSON
  manifest under `artifacts/predictions/inference_reporting_v1/<request_id>/`.
  Reload recursively verifies configs, upstream summaries, model/card records,
  output hash, deterministic request identity, schema, probabilities, physical
  bounds, and coverage.
- `flowcast.reports` derives operational findings only from verified
  prediction rows, exports full CSV plus self-contained HTML under the matching
  report directory, and persists a tamper-checked report manifest.
- No Step 17 module exposes model fitting, recalibration, rethresholding, or
  active-route mutation.

## 7. Cleaning Strategy
### 7.1 Duplicate policy
- Exact duplicates and key duplicates are identified separately.
- For key duplicates, keep the row with the greatest number of non-null trusted fields.
- If completeness ties, choose deterministically by original source-row order.
- Record removed source rows and retained row identity.

### 7.2 Invalid numeric policy
- Negative traffic volume: mark invalid; recover from valid `vehicle_count` only if the data audit proves semantic equivalence for that row, otherwise impute.
- Speed above 200 km/h: mark invalid and impute using segment/time context.
- Occupancy outside 0-100: mark invalid and impute/cap only according to documented rule; physical impossibilities should not be treated as genuine extremes.
- Statistical outliers that are physically possible are flagged and investigated before capping.

### 7.3 Missing-value policy
- Reindex each segment to the complete 30-minute grid.
- Add missing-window, source-null, physical-invalid, method, and donor flags.
- Recover missing/invalid volume from valid same-row `vehicle_count`, whose
  release-wide equality was verified before the rule was enabled.
- Then use the same road/window from the previous day, followed by same-road
  causal forward fill limited to four windows.
- Only leading speed/occupancy values may use the same-timestamp station median;
  all contributing source rows are recorded.
- Preserve unknown accident targets for inserted windows rather than imputing a
  no-incident label.
- Fail closed outside the configured hierarchy and preserve every decision in
  the quality report.

## 8. Feature and Target Architecture
### 8.1 Feature timestamp rule
At prediction origin `t`, every feature must be known at or before `t`.

- Lag: `groupby(road_id).shift(k)`.
- Rolling: shift first, then roll.
- Weather/calendar: only observations available at `t`; no future weather unless explicitly labelled as a forecast input, which the delivered data is not.
- Static road metadata may be used directly.

The implemented Step 07 pipeline validates the merged summary and artifact hash
before reading data. It preserves all source and repair lineage, sorts by road
and timestamp, uses full-width shifted rolling windows, and retains the expected
leading nulls rather than dropping origins. Scheduled calendar-event proximity
is treated as known at origin; it never reads future traffic measurements.

### 8.2 Horizon targets
For each horizon `h in {1,2,3,4}`:

```text
target_timestamp_h{h} = timestamp shifted by -h within road
target_volume_h{h} = traffic_volume shifted by -h within road
target_speed_h{h} = avg_speed shifted by -h within road
target_travel_time_h{h} = travel_time shifted by -h within road
target_congestion_h{h} = congestion_level shifted by -h within road
target_accident_h{h} = (accident_count shifted by -h) > 0
target_{name}_h{h}_available = target-specific future-label availability
```

Every shifted timestamp must equal `t + (30 minutes x h)` within the same road.
Accident availability additionally requires the shifted `_accident_observed`
flag; unknown reconstructed windows remain null rather than negative. The
common base retains trailing origins instead of globally dropping them.

### 8.3 Multi-horizon strategy
- Classical models use a shared training loop that produces one model per target and horizon.
- A multi-output wrapper may be used only when metrics and persistence remain separately traceable by horizon.
- The recurrent volume model outputs four horizons in one forward pass.
- Evaluation always reports each horizon and an aggregate.

## 9. Split and Preprocessing Architecture
- One frozen chronological split is shared by all model families: train is
  2025-01-01 00:00 through 2025-04-16 16:30 (5,074 timestamps), validation is
  2025-04-16 17:00 through 2025-05-09 08:00 (1,087), and test is 2025-05-09
  08:30 through 2025-05-31 23:30 (1,087), all in Asia/Kolkata.
- Every origin is retained. A target/horizon is usable only when its availability
  mask is true and its target timestamp remains inside the origin partition.
- Five expanding time-series CV folds are restricted to training. Each uses a
  four-window maximum-horizon gap and a seven-day validation window.
- Validation selects hyperparameters, calibration, and thresholds.
- Test is sealed by default and opened only through explicit final-evaluation
  access after model choices are frozen.
- Preprocessors that learn imputation, category, scaling, or weighting
  statistics are fit on training data only and persisted with their feature
  schema.
- Random seeds and library versions are recorded.

## 10. Model Architecture
### 10.1 Scratch linear-regression proof

- The Step 11 baseline is implemented directly in NumPy with seeded
  initialization, a fixed configured learning rate, bounded iterations,
  relative-loss convergence tolerance, and patience.
- Every analytical weight and bias gradient is checked against a central
  finite difference before real data is used; a noiseless synthetic problem
  must recover its known weights and bias within configured tolerance.
- The real-data demonstration uses the frozen linear preprocessor and next-
  window volume target. Persisted coefficients can be reloaded without a
  modelling-library estimator and must reproduce stored validation predictions.

### 10.2 Classical regression selection

- The three continuous targets use direct horizon-specific pipelines.
- Seven configured candidates cover all four required estimator families.
- All five expanding-window folds contribute to hyperparameter selection.
- Mean CV RMSE selects a configuration within each family; validation RMSE
  selects the family for each target/horizon.
- The final test is prediction-only and cannot alter selections.
- Selected pipelines include their fitted preprocessor and expose coefficients
  or tree feature importance through a common persisted table.

### 10.3 Classical classification selection

- Congestion and accident risk use direct horizon-specific classifiers and the
  exact 62-feature Step 10 schema.
- Eight configured candidates cover Decision Tree, Random Forest, XGBoost, and
  scaled LinearSVC across all five expanding folds.
- Congestion uses the fixed numeric/report order Free-flow, Moderate, Heavy,
  Severe and selects by Macro-F1. Accident selects by ROC-AUC, with PR-AUC and
  operating-point precision/recall/F1 always visible.
- Tree/XGBoost probabilities are checked for finiteness, bounds, fixed class
  order, and row normalization. Selected SVM decision scores are converted to
  probabilities by the validation-fitted sigmoid calibrator.
- A chronological validation sub-split separates calibration fitting from
  calibration assessment and accident threshold selection. The final test is
  prediction-only.

### 10.4 Classical registry key
```text
{target}/{horizon}/{model_name}/{version}
```

Each registry entry contains:

- serialized estimator/pipeline record and SHA-256.
- model-card and source-prediction records and SHA-256 values.
- feature manifest, processed-data, preprocessing, and selection lineage.
- train/validation/test boundaries and eligible row counts.
- hyperparameters, seed, task-aware primary metric, and direction.
- frozen CV/validation evidence plus validation/test metrics.
- calibration/class order/threshold metadata where relevant.
- runtime, interpretability context, rationale, acceptance state, and
  limitations.

The canonical Step 14 output under
`artifacts/metrics/classical_registry_v1/` contains `registry.json`,
`scoreboard.csv`, `prediction_index.json`, `summary.json`, and generated
`summary.md`. The prediction index maps the two existing source Parquets in
place instead of duplicating values.

Registry construction never loads a source test partition or retrains a model;
it consumes frozen artifacts only. The verified loader recursively checks the
independent registry configuration and outputs, both source summaries, their
configuration/input/artifact chains, and every model/card before an entry
resolves through its source loader.

### 10.5 Accident-risk classifier
- Binary label: future `accident_count > 0`.
- Use class weights or XGBoost `scale_pos_weight` derived from training only.
- Do not select by accuracy.
- Tune threshold on validation data according to operational trade-off and report precision/recall.
- Calibrate probability when the selected model requires it.

### 10.6 Recurrent volume model
Preferred v1 design:

```text
Input: [batch, sequence_length, feature_count]
 -> one or two LSTM/GRU layers
 -> dropout
 -> dense hidden layer
 -> four-value volume head for h1-h4
```

- PyTorch implementation.
- MSE-based training loss, optionally horizon-weighted only if documented.
- Early stopping on validation RMSE/loss.
- Device policy supports `auto`, explicit `cpu`, and guarded `cuda`. Material
  full-candidate recurrent training may use a verified CUDA device; small
  probes/tests and non-tensor stages normally use CPU to avoid transfer and
  startup overhead.
- CPU is a required execution and reproduction path. Model code must not contain
  CUDA-only operators, datasets remain lazy, batch sizes and workers remain
  configurable, pinned/non-blocking transfers are enabled only for CUDA, and
  checkpoints are portable state dictionaries loaded with `map_location`.
- The approved local CUDA distribution is PyTorch `2.13.0+cu130`; the portable
  dependency identity remains `torch==2.13.0`.
- Checkpoint stores model state, architecture config, scaler/feature manifest, split, seed, and metrics.

The verified v1 implementation tunes two predeclared unidirectional LSTM
candidates on common validation origins. `lstm_s12_h32` was selected at epoch
8. It consumes 12 half-hour steps of 64 transformed known-at-origin features
and produces all four volume horizons. On 26,500 exact shared hold-out origins
per horizon it beats the frozen classical model at 30, 60, and 90 minutes but
trails by 0.0471 RMSE at 120 minutes. Test results cannot trigger a refit or
architecture change.

## 11. Confidence Architecture
### Regression
- The implemented method is finite-sample split conformal over absolute
  validation residuals, separately by model version, target, and horizon.
- Quantile rank is `ceil((n + 1) * confidence_level)`, capped at `n`, using the
  higher order statistic. Test residuals never fit interval width.
- The same comparable external layer covers classical volume/speed/travel-time
  and recurrent volume. Row-level lower/upper bounds, coverage, and width are
  persisted; lower bounds are clipped at zero for physical outputs.

### Classification
- Persisted ordered probabilities remain the source of truth.
- Congestion and accident rows expose maximum probability, entropy, normalized
  entropy, and a configured confidence band.
- Fixed ten-bin reliability tables compare congestion maximum probability with
  correctness and accident probability with observed events.
- Accident risk output preserves the frozen validation-selected operating
  threshold. Low/elevated/high/critical bands use configured threshold
  multiples and never retune on test.
- Subgroup metrics enforce row and rare-positive support thresholds. Low-support
  rows remain visible with counts and blank metrics.

## 12. Inference Architecture
`Predictor` is the single public inference interface.

Conceptual request:

```python
PredictionRequest(
    road_ids=["NL-001"],
    origin_timestamp="2025-05-31T23:30:00",
    horizons=[1, 2, 3, 4],
)
```

Conceptual output fields:

- road ID/name.
- origin and target timestamp.
- horizon.
- volume/speed/travel-time predictions.
- congestion class and probabilities.
- accident-risk probability and risk band.
- confidence interval or confidence score.
- model versions and data version.

The implemented interface defaults to the latest common origin and full
25-road scope when these values are omitted. Requests accept only horizons
1-4, exact half-hour origins present in the processed dataset, and `cpu` or
guarded `cuda` execution. CPU is the default and reproduction path.

Batch inference writes Parquet plus a JSON manifest. The manifest contains the
normalized request, row/road/horizon coverage, measured initialization and
prediction time, verified upstream/model records, schema version, and output
hash. CSV/HTML reports load only through this verified boundary.

## 13. Dashboard Architecture
### App shell
- Project title, data version, model version, and last successful pipeline run.
- Shared road, date/time, and horizon filters.
- Cached artifact loading with cache invalidation based on file/version hash.

### Required pages
1. Live predictions - segment table/map, future horizons, confidence, risk ranking.
2. Historical trends - volume/speed timeline and peak profiles.
3. Congestion heatmap - segment x time severity matrix.
4. Road comparison - selected segment metrics and reliability.
5. Model performance - target/horizon scoreboard and deep-vs-classical table.
6. Feature importance - tree importance and optional permutation/SHAP-lite analysis.
7. Forecast visualisation - predicted vs actual and multi-horizon overlays.
8. Prediction confidence - intervals, coverage, classifier confidence/calibration.
9. Weather vs traffic - rainfall/visibility/condition relationships.
10. Data and training - upload validation, explicit retraining, audit links, export.

The tenth page groups support modules and does not replace any required view.

### Implemented dashboard boundary

- `dashboard/app.py` is the only Streamlit entry point. It owns page config,
  grouped `st.navigation`, shared filters, and the verified status strip.
- `dashboard/app_pages/` contains ten thin page scripts. Required analytics do
  not live in those scripts; they call the package-level dashboard services.
- `src/flowcast/dashboard/data.py` recursively verifies processed data,
  registry, recurrent comparison, confidence/error artifacts, the latest
  prediction batch, and its report before exposing a read-only
  `DashboardBundle`.
- `src/flowcast/dashboard/cache.py` uses `st.cache_resource` for the verified
  bundle and CPU predictor. The cache fingerprint covers data, registry,
  confidence, inference configuration, latest prediction, and report manifest
  metadata so new persisted output invalidates stale UI state.
- `analytics.py` and `charts.py` contain deterministic real-data aggregation and
  Plotly rendering. `state.py` owns session-local shared road/date/horizon
  filters. `ui.py` owns the scoped design layer and reusable status/metric/error
  components. Its compact page banner keeps the status strip below Streamlit's
  desktop toolbar, preserves title/subtitle visibility, and responsively hides
  only secondary context below 1400px. Its reusable evidence-brief component
  pairs a data-derived current reading with chart-reading guidance. Page
  scripts format those briefs only from the same verified frames and aggregates
  already shown.
- The live page's presentation order is header, displayed horizon, KPI cards,
  frozen-model request, current reading, then the aligned Corridor
  Signal/Priority Queue output row. This is a presentation-only boundary: the
  form retains the same widget values, validation, frozen predictor, batch
  persistence, report generation, session state, and rerun path.
- Live prediction origins are derived from the complete verified processed
  history and the inference configuration. `analytics.py` offers only
  full-corridor timestamps with the configured twelve contiguous 30-minute
  rows; the page presents those 7,237 eligible January-May origins as separate
  native date and half-hour time controls and revalidates the combined value
  before inference.
- `uploads.py` validates an exact traffic/weather/calendar source contract and
  stages accepted content by hash under `artifacts/uploads/`; immutable source
  and raw directories are never write targets.
- `training_service.py` is synchronous and explicit for v1. It requires
  `RETRAIN`, prevents concurrent dashboard runs, passes a new version to an
  approved CLI training command, persists log/manifest evidence, and never
  changes active routing.
- Confidence regeneration accepts an optional test-owned output artifact root.
  Contract tests use it for two deterministic writes while continuing to read
  verified canonical inputs; model-training fixtures likewise copy the complete
  processed contract into temporary roots. No test may rewrite canonical
  ignored Parquet artifacts.
- `.streamlit/config.toml` is the runtime theme source. `DESIGN.md` is the
  checked-in visual contract adapted to Streamlit without adding React,
  FastAPI, a database, or another deployment surface.
- Desktop visual acceptance covers all ten routes at 1280 x 720, 1440 x 900,
  and 1920 x 1080. The status strip must clear the top toolbar, the page opener
  must stay below 190px after layout settles, every page must expose at least
  one plain-language evidence brief, and the document must not overflow
  horizontally. On the live route, every forecast-request control must fit
  above the 1280 x 720 fold and the Corridor Signal/Priority Queue card
  boundaries must remain aligned.

## 14. Training-Service Boundary
The dashboard may trigger retraining only through `training_service`:

- Require explicit confirmation.
- Validate that required raw/processed data exists.
- Prevent duplicate concurrent runs.
- Show progress/log path.
- Write to a new versioned artifact directory.
- Switch the active model only after successful evaluation and artifact validation.
- Never retrain because Streamlit reran a page.

A simple synchronous implementation is acceptable for v1.0 if it is explicit and safe.

## 15. Logging, Lineage, and Failure Handling
Each pipeline run receives a `run_id` and records:

- command/config/seed.
- source hashes.
- input/output row counts.
- quarantine counts by reason.
- join counts.
- feature version.
- split boundaries.
- model versions and metrics.
- runtime and failure traceback.

Critical failures stop the pipeline. Recoverable row-level problems enter quarantine. No exception should be converted into a silent empty result.

## 16. Testing Architecture
### Build safety

- `scripts/run_tests.py` is the canonical pytest entry point. It calls pytest
  in-process, returns the exact pytest status, and prints an explicit
  `FLOWCAST_PYTEST_EXIT` marker so a timing shell cannot be mistaken for a
  passing test run.
- `tests/conftest.py` snapshots every Git-tracked file at session start. At
  teardown it restores any changed/deleted tracked file to its exact pre-test
  bytes and fails with the offending paths. Existing dirty user files are
  preserved as captured; the guard does not assume `HEAD` is the desired state.
- Tests that invoke writers, including notebook execution, must replace all
  writable settings with test-owned temporary roots. Canonical artifacts are
  read-only test inputs; determinism is verified by comparing isolated output
  rather than regenerating frozen evidence in place.

### Unit
- datetime parsing.
- weather normalization.
- congestion boundaries.
- JSON parsing.
- duplicate selection.
- invalid-value policy.
- lag/rolling construction.
- target shifting.
- descriptive statistics and imbalance denominators.
- correlation-candidate safety and quality reconciliation.
- largest-remainder split allocation and sealed-test access.
- model-family preprocessing and training-only learned statistics.
- scratch gradient calculation.
- metric functions.
- ordered multiclass/binary probability metrics and validation-only threshold
  selection.

### Data-contract
- required columns and dtypes.
- uniqueness.
- range and controlled vocabulary.
- join cardinality.
- no unexpected nulls in trusted modelling fields.
- deterministic EDA artifacts, hashes, figure dimensions, and tamper rejection.
- chronological split order, horizon boundary isolation, CV containment,
  deterministic preprocessing artifacts, reloadability, and tamper rejection.
- complete classifier family/fold coverage, pre-test decision freeze, fixed
  class order, probability normalization, threshold/calibration evidence,
  model-card completeness, probability equality after reload, and tamper
  rejection.
- exact 20-entry classical-registry coverage, task-aware metric schemas,
  frozen-source equality, complete lineage, deterministic generation,
  one-to-one prediction indexing, all-model reloadability, and
  registry/upstream tamper rejection.
- recurrent road/split/gap/target isolation, training-only target scaling,
  four-output shapes, seeded initialization, validation-only selection,
  pre-test checkpoint freeze, best-weight reload equality, exact classical
  origin mapping, metric correctness, registry-extension coverage, and
  checkpoint tamper rejection.
- inference request/cadence/sequence rejection, five-target/four-horizon
  coverage, frozen routing, repeated CPU equality, physical/probability schema
  bounds, prediction/report reconciliation, runtime measurement, and
  prediction/report tamper rejection.

### Integration
- raw -> interim.
- interim -> processed.
- processed -> selected model.
- saved model -> batch predictions.
- predictions/metrics -> dashboard service.

### Smoke
- CLI imports/help.
- small-sample end-to-end run.
- EDA notebook top-to-bottom kernel execution.
- Streamlit app imports all pages.
- model registry loads active artifacts.

## 17. Architectural Decision Rules
- Prefer package services over page-local logic.
- Prefer generated loops/config over copied model code for four horizons.
- Prefer Parquet/JSON artifacts over a database in v1.
- Prefer explicit pipeline stages over hidden automation.
- Any architecture change must update this file, `STATUS.md`, and `NEXT_STEP.md` in the same turn.
