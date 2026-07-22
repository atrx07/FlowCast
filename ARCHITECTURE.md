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
python -m flowcast.cli prepare-data
python -m flowcast.cli eda
python -m flowcast.cli train-classical
python -m flowcast.cli train-deep
python -m flowcast.cli evaluate
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
│   └── models.yaml
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
│       │   ├── clean_weather.py
│       │   ├── clean_calendar.py
│       │   ├── clean_context.py
│       │   ├── align.py
│       │   ├── merge.py
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

This is a target structure, not permission to create empty files unnecessarily. Create modules when their step begins. Keep each source file below 500 lines unless documented otherwise.

## 5. Configuration Architecture
### `config/base.yaml`
- Directory paths.
- Global seed.
- timezone policy.
- logging level.
- artifact version/tag.

### `config/data_contracts.yaml`
- Required columns and types.
- uniqueness keys.
- allowed categorical values.
- physical range constraints.
- null policy.
- weather normalization map.

### `config/cleaning.yaml`
- Context-cleaning contract version.
- Causal numeric imputation methods and maximum supported gap lengths.
- Calendar flag/name relationship checks.

### `config/features.yaml`
- lag windows: 1, 2, 48.
- rolling windows: 4, 8.
- forecast horizons: 1, 2, 3, 4.
- peak periods.
- low-visibility threshold.
- selected feature groups.

### `config/models.yaml`
- split dates/ratios.
- model grids and search budgets.
- class weights.
- calibration method.
- recurrent architecture and training budget.
- early-stopping settings.

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

### 6.7 Merged contract
Key: `road_id + timestamp`.

- Many-to-one joins only.
- Weather joined by station and floored hour.
- Calendar joined by date.
- Join indicators and missing join counts recorded during pipeline execution.
- `weather_station_id` may remain for lineage but is excluded where redundant for modelling.

### 6.8 Processed feature contract
- Sorted by `road_id, timestamp`.
- Stable feature names and dtypes.
- Explicit target columns per horizon.
- Rows without enough history/future for a selected horizon flagged and filtered only in target-specific modelling views.
- A feature manifest records source columns, transforms, version, and leakage classification.

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
- Add missing-window and imputation flags.
- Short internal gaps: segment-wise time interpolation when both sides are available and policy allows.
- Longer gaps: training-derived segment x time-of-day medians or fallback hierarchy.
- Do not impute future targets using future data during model evaluation.
- Preserve all imputation decisions in the quality report.

## 8. Feature and Target Architecture
### 8.1 Feature timestamp rule
At prediction origin `t`, every feature must be known at or before `t`.

- Lag: `groupby(road_id).shift(k)`.
- Rolling: shift first, then roll.
- Weather/calendar: only observations available at `t`; no future weather unless explicitly labelled as a forecast input, which the delivered data is not.
- Static road metadata may be used directly.

### 8.2 Horizon targets
For each horizon `h in {1,2,3,4}`:

```text
target_volume_h = traffic_volume shifted by -h within road
 target_speed_h = avg_speed shifted by -h within road
 target_travel_time_h = travel_time shifted by -h within road
 target_congestion_h = congestion_level shifted by -h within road
 target_accident_h = (accident_count shifted by -h) > 0
```

The leading space in the example is visual only; actual column names must be normalized.

### 8.3 Multi-horizon strategy
- Classical models use a shared training loop that produces one model per target and horizon.
- A multi-output wrapper may be used only when metrics and persistence remain separately traceable by horizon.
- The recurrent volume model outputs four horizons in one forward pass.
- Evaluation always reports each horizon and an aggregate.

## 9. Split and Preprocessing Architecture
- Use one frozen chronological split shared by model families.
- Recommended starting point: earliest 70% train, next 15% validation, latest 15% test, converted into exact timestamp boundaries and recorded.
- Time-series cross-validation folds are restricted to the training period.
- Validation selects hyperparameters, calibration, and thresholds.
- Test is opened only after model choices are frozen.
- Preprocessors that learn statistics are fit on training data only and persisted with the model.
- Random seeds and library versions are recorded.

Exact split boundaries must be finalized after the processed data coverage is verified and then stored in config/model cards.

## 10. Model Architecture
### 10.1 Classical registry key
```text
{target}/{horizon}/{model_name}/{version}
```

Each registry entry contains:

- serialized estimator/pipeline.
- feature manifest/hash.
- split boundaries.
- hyperparameters and seed.
- validation/test metrics.
- calibration/threshold artifact where relevant.
- model card.

### 10.2 Accident-risk classifier
- Binary label: future `accident_count > 0`.
- Use class weights or XGBoost `scale_pos_weight` derived from training only.
- Do not select by accuracy.
- Tune threshold on validation data according to operational trade-off and report precision/recall.
- Calibrate probability when the selected model requires it.

### 10.3 Recurrent volume model
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
- CPU-compatible; CUDA used when available.
- Checkpoint stores model state, architecture config, scaler/feature manifest, split, seed, and metrics.

## 11. Confidence Architecture
### Regression
Preferred minimum:

- Use validation residuals for split-conformal or empirical residual intervals by target/horizon.
- Report interval bounds and empirical coverage.
- Deep models may additionally use MC dropout, but conformal/residual calibration remains the comparable external layer.

### Classification
- Return class probabilities.
- Calibrate selected probabilities when needed.
- Confidence may be displayed as maximum probability plus entropy/uncertainty.
- Accident-risk output is probability plus the chosen operating threshold and risk band.

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

Batch inference writes Parquet plus a JSON manifest.

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
### Unit
- datetime parsing.
- weather normalization.
- congestion boundaries.
- JSON parsing.
- duplicate selection.
- invalid-value policy.
- lag/rolling construction.
- target shifting.
- scratch gradient calculation.
- metric functions.

### Data-contract
- required columns and dtypes.
- uniqueness.
- range and controlled vocabulary.
- join cardinality.
- no unexpected nulls in trusted modelling fields.

### Integration
- raw -> interim.
- interim -> processed.
- processed -> selected model.
- saved model -> batch predictions.
- predictions/metrics -> dashboard service.

### Smoke
- CLI imports/help.
- small-sample end-to-end run.
- Streamlit app imports all pages.
- model registry loads active artifacts.

## 17. Architectural Decision Rules
- Prefer package services over page-local logic.
- Prefer generated loops/config over copied model code for four horizons.
- Prefer Parquet/JSON artifacts over a database in v1.
- Prefer explicit pipeline stages over hidden automation.
- Any architecture change must update this file, `STATUS.md`, and `NEXT_STEP.md` in the same turn.
