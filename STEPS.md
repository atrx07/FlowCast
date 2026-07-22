# STEPS.md
## How to Use This File
Execute steps in order unless `NEXT_STEP.md` explicitly points to a justified branch. Before a step, read its goal, tasks, outputs, checks, and exit gate. After the step, update `STATUS.md` and `NEXT_STEP.md` with evidence.

Do not start modelling before the data and leakage gates are satisfied. Do not start dashboard polish before persisted outputs exist.
---
## Step 00 - Confirm References and Governance
### Goal
Establish the documents and original source material as the project contract.

### Tasks
1. Confirm `FlowCast-project_file/` contains both reference DOCX files and all three CSV files.
2. Read `AGENTS.md`, `PROJECT.md`, `ROADMAP.md`, `ARCHITECTURE.md`, `STATUS.md`, and `NEXT_STEP.md`.
3. Confirm the PRD success metrics and nine dashboard views.
4. Confirm the data dictionary joins and congestion rule.
5. Record any discrepancy; do not silently choose one interpretation.

### Outputs
- Governance documents present.
- Reference directory confirmed.

### Exit gate
- All project instructions can be located from the repository root.
- No unresolved conflict blocks bootstrap.
---
## Step 01 - Bootstrap the Repository
### Goal
Create a cross-platform Python 3.11 project skeleton without unnecessary frameworks.

### Tasks
1. Initialize Git if needed.
2. Create the minimal folders from `ARCHITECTURE.md` required for Steps 01-04.
3. Create `pyproject.toml`, `requirements.txt`, `.gitignore`, and package `src/flowcast/`.
4. Add configuration loading and path resolution using `pathlib`.
5. Add deterministic seed configuration.
6. Add structured console/file logging.
7. Add pytest and a basic package import test.
8. Add CLI help with placeholder commands that fail clearly if not implemented; do not add fake success.
9. Document environment setup in the initial README.

### Outputs
- Installable/importable package.
- Basic CLI.
- Test scaffold.
- Config scaffold.

### Checks
```bash
python -m pip install -e .
python -m flowcast.cli --help
pytest -q
```

### Exit gate
- Package imports on the current platform.
- Tests run successfully.
- No code file exceeds 500 lines.
---
## Step 02 - Preserve and Audit Raw Inputs
### Goal
Create immutable raw copies, hashes, and an automated baseline audit.

### Tasks
1. Copy the three delivered CSV files from `FlowCast-project_file/` into `data/raw/` without modifying bytes.
2. Calculate SHA-256 hashes for delivered and copied files; assert equality.
3. Load only enough metadata to report shape, columns, dtypes, date ranges, unique keys, nulls, duplicates, and numeric ranges.
4. Profile weather label variants and accident prevalence.
5. Compute the expected traffic grid and missing full windows.
6. Save audit results as JSON and Markdown under `artifacts/audits/<version>/`.
7. Add tests for hashes and expected source schemas.

### Expected baseline evidence
- Traffic: 178,468 rows; 1,767 key duplicates.
- Weather: 10,872 rows; complete station-hour grid.
- Calendar: 151 rows.
- Unique traffic keys: 176,701 of 181,200 expected; 4,499 missing windows.

### Exit gate
- Raw-copy hashes match.
- Automated audit reproduces known counts or documents a precise reason for variance.
- Raw source files remain untouched.
---
## Step 03 - Define Data Contracts and Quarantine
### Goal
Convert the dictionary into executable validation rules.

### Tasks
1. Define required columns, parsers, allowed categories, keys, and numeric physical limits.
2. Build validation result structures containing valid rows, quarantined rows, and summary counts.
3. Preserve source file and source row number.
4. Implement reason codes, such as:
   - missing_required_column
   - invalid_timestamp
   - duplicate_key
   - negative_traffic_volume
   - excessive_speed
   - invalid_occupancy
   - invalid_json
   - invalid_flag
5. Distinguish row-level rejection from recoverable cell-level invalidation.
6. Add tests for every reason code and boundary.

### Outputs
- Executable contracts.
- Versioned validated Parquet under `data/interim/<version>/`.
- Rejected-row Parquet, unified issue ledger, and summary JSON under
  `data/quarantine/<version>/`.

### Checks
```bash
python -m flowcast.cli validate
pytest -q
```

### Exit gate
- No invalid input can disappear without a reason record.
- Tests prove deterministic validation.
---
## Step 04 - Clean Calendar and Weather
### Goal
Produce trusted calendar and hourly weather tables before touching the large merge.

### Calendar tasks
1. Parse `YYYY-MM-DD` dates.
2. Validate unique date.
3. Validate flags as 0/1.
4. Validate names against flags.
5. Preserve holiday, event, and roadwork indicators.

### Weather tasks
1. Parse date with day-first semantics and combine with hourly time.
2. Normalize whitespace and casing.
3. Map label variants into a controlled vocabulary such as Clear, Cloudy, Overcast, Rain, and Fog.
4. Verify unique `station_id + weather_hour`.
5. Add missingness flags for temperature and visibility.
6. Impute missing numeric weather values using a time-aware station-local method defined in config.
7. Validate non-negative rainfall and visibility.
8. Write cleaned Parquet and quality summary.

### Checks
- Three stations.
- 3,624 hours per station.
- 10,872 final station-hour keys.
- Zero uncontrolled weather labels.

```bash
python -m flowcast.cli clean-context
pytest -q
```

### Proven missing-weather policy

- Sort strictly by `station_id + weather_hour`.
- Add missingness, imputation-method, and donor-source-row columns.
- Forward-fill within the same station for at most two hourly gaps.
- Fail closed on leading gaps, longer gaps, incomplete hourly grids, or unknown
  labels; never use a future or cross-station observation.
- Persist cleaned Parquet plus canonical JSON and generated Markdown quality
  evidence.

### Exit gate
- Calendar and weather contracts pass.
- Normalization and imputation counts are reported.
---
## Step 05 - Clean Traffic and Reconstruct the Grid
### Goal
Produce one trusted row per road and 30-minute timestamp, including explicit missing-window records.

### Tasks
1. Parse traffic date/time into timestamp.
2. Validate road metadata consistency per `road_id`.
3. Resolve duplicates using completeness then source-row order.
4. Parse `vehicle_type_dist` JSON into four numeric shares.
5. Validate share keys, ranges, and near-unit sum; normalize only within documented tolerance.
6. Mark physical invalids:
   - volume < 0
   - speed > 200 or speed <= 0 if physically impossible by policy
   - occupancy outside 0-100
7. Reindex each road to the complete 30-minute grid.
8. Add flags for original missing windows, null cells, physical invalids, and imputation method.
9. Recover missing/invalid volume from the same-row `vehicle_count` after
   proving release-wide equality; then use previous-day same-window values and
   same-road causal forward fill limited to four windows.
10. For unresolved leading speed/occupancy values only, use a concurrent
    same-station median and record every contributing source row. Fail closed
    outside this hierarchy.
11. Derive blank congestion labels from V/C after valid/imputed volume is available.
12. Preserve original congestion where present, but audit disagreement with the derivation.
13. Write cleaned traffic Parquet and quarantine/quality outputs.
14. Keep accident count unknown on inserted windows and mark target
    availability explicitly; do not fabricate zero-incident labels.

### Checks
- Exactly 25 roads.
- Unique road/timestamp keys.
- Full expected grid represented or any exclusion explicitly documented.
- No invalid numeric ranges in trusted fields.
- Congestion boundary unit tests at 0.50, 0.80, and 1.00.

### Exit gate
- Traffic cleaning is deterministic and audited.
- All 1,767 duplicates and 4,499 missing windows are accounted for.
---
## Step 06 - Align and Merge Sources
### Goal
Create one segment-window table without cardinality errors.

### Tasks
1. Add `weather_hour = timestamp.floor("h")` to traffic.
2. Join traffic to weather on station mapping and weather hour with many-to-one validation.
3. Join calendar on normalized date with many-to-one validation.
4. Record join indicators and missing matches.
5. Assert row count remains equal to the trusted traffic grid.
6. Verify no duplicate key was introduced.
7. Preserve key lineage fields.
8. Write versioned merged interim Parquet.

### Exit gate
- Zero unexpected join misses.
- Zero row multiplication.
- One row per road/timestamp.
---
## Step 07 - Engineer Features
### Goal
Generate reproducible, leakage-safe explanatory features.

### Tasks
1. Add hour/day cyclical sine and cosine features.
2. Add weekday, weekend, and configured peak-period flags.
3. Add parsed vehicle-class shares.
4. Add volume/speed lag features at 1, 2, and 48 windows.
5. Add shifted rolling mean/std over 4 and 8 prior windows.
6. Add half-hour capacity, V/C ratio, and capacity headroom.
7. Add rain, low-visibility, weather-category, and temperature-band features.
8. Add holiday x peak, event, event proximity, and roadwork features.
9. Add missingness/imputation flags where useful.
10. Emit a feature manifest with name, dtype, source, transform, and leakage status.

### Leakage checks
- For a selected road and time, manually verify each lag against the raw preceding row.
- Mutating a future row must not change features at earlier timestamps.
- Rolling values must exclude the current row.

### Exit gate
- Feature tests pass.
- Feature manifest exists.
- No feature uses future information.
---
## Step 08 - Build Multi-Horizon Targets and Processed Data
### Goal
Create prediction targets for 30, 60, 90, and 120 minutes.

### Tasks
1. Within each road, shift future volume, speed, travel time, congestion, and accident labels by horizons 1-4.
2. Define accident target as future `accident_count > 0`.
3. Add target timestamps.
4. Keep a common base dataset; create target-specific valid-row masks rather than deleting rows globally.
5. Store target schema and horizon definitions in the manifest.
6. Persist versioned processed Parquet and metadata JSON.
7. Add contract tests for target alignment.

### Exit gate
- A row at time `t` maps to the correct `t+h` target within the same road.
- No road boundary contamination.
- Dataset version and hash recorded.
---
## Step 09 - Produce Data-Quality Report and EDA
### Goal
Explain the data and justify modelling decisions.

### Tasks
1. Build a report from pipeline counters, not manually typed estimates.
2. Include source/processed row counts, missingness, duplicate resolution, invalid values, imputation, label normalization, and join results.
3. Explore volume, speed, occupancy, travel time, congestion, and accidents.
4. Analyze by road, hour, weekday, weather, holidays, events, and roadworks.
5. Inspect correlations/covariance and feature redundancy.
6. Examine accident imbalance and congestion distribution.
7. Export figures into versioned report folders.
8. Record modelling implications and potential bias/limitations.

### Exit gate
- EDA notebook runs top-to-bottom using package functions.
- Data-quality report is reproducible.
- Findings lead to explicit model/preprocessing decisions.
---
## Step 10 - Freeze Splits and Preprocessing
### Goal
Create one defensible evaluation protocol shared across models.

### Tasks
1. Inspect final timestamp coverage.
2. Convert the planned 70/15/15 chronological split into exact timestamp boundaries.
3. Store boundaries in config and prevent accidental test access during tuning.
4. Define time-series CV folds inside training only.
5. Build preprocessing pipelines by model family.
6. Fit scalers/encoders/imputers on training only.
7. Persist feature order and preprocessing metadata.
8. Add split and leakage assertions.

### Exit gate
- Train < validation < test chronologically.
- Every horizon uses compatible target availability.
- Test period remains untouched by hyperparameter selection.
---
## Step 11 - Implement NumPy Linear Regression
### Goal
Demonstrate the mathematics behind regression.

### Tasks
1. Implement prediction `X @ w + b`.
2. Implement MSE loss.
3. Implement analytical gradients for weights and bias.
4. Implement gradient descent with history and stopping criteria.
5. Add numerical gradient checks on synthetic data.
6. Train on a controlled FlowCast subset.
7. Compare against scikit-learn Linear Regression.
8. Document convergence, scaling, and limitations.

### Exit gate
- Gradient check passes within tolerance.
- Loss decreases on synthetic and real subset.
- Results are reproducible.
---
## Step 12 - Train Classical Regression Models
### Goal
Forecast volume, average speed, and travel time for all four horizons.

### Models
- Linear Regression.
- Decision Tree Regressor.
- Random Forest Regressor.
- XGBoost Regressor.

### Tasks
1. Implement a config-driven target/horizon training loop.
2. Tune through time-series CV with bounded search budgets.
3. Evaluate validation metrics and runtime.
4. Freeze the selected model per target/horizon.
5. Evaluate frozen choices once on test.
6. Persist pipelines, predictions, metrics, and feature importance where available.
7. Build model cards.

### Exit gate
- Required model families are represented.
- Metrics include RMSE, MAE, MAPE, and R².
- Artifacts reload and reproduce predictions.
---
## Step 13 - Train Congestion and Accident Classifiers
### Goal
Predict future congestion classes and accident-risk probabilities.

### Models
- Decision Tree Classifier.
- Random Forest Classifier.
- XGBoost Classifier.
- SVM baseline with scaled inputs.
- Optional simple linear/logistic baseline for context.

### Congestion tasks
1. Preserve four-class ordering in reports.
2. Tune using Macro-F1.
3. Inspect per-class recall and confusion matrix.
4. Calibrate selected probabilities if needed.

### Accident tasks
1. Use binary `accident_count > 0` targets.
2. Compute class weights from training only.
3. Tune by ROC-AUC with PR-AUC and recall constraints visible.
4. Select operating threshold using validation data.
5. Report precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrix on test.
6. Never use accuracy as the headline result.

### Exit gate
- Class probabilities are available for confidence/risk ranking.
- Threshold and calibration decisions are persisted.
- Model cards and load tests pass.
---
## Step 14 - Build the Classical Scoreboard and Registry
### Goal
Create one traceable selection and comparison layer.

### Tasks
1. Consolidate metrics by target, horizon, model, split, and version.
2. Rank by primary metric with runtime and interpretability as secondary factors.
3. Record selected model and rationale per target/horizon.
4. Register artifact paths and feature/preprocessing versions.
5. Generate model cards from metadata.
6. Export dashboard-ready metrics and predictions.

### Exit gate
- Every selected prediction maps to exactly one registry entry.
- Scoreboard is machine-readable and human-readable.
---
## Step 15 - Build and Train the Recurrent Model
### Goal
Train a from-scratch LSTM/GRU that predicts future volume across four horizons.

### Tasks
1. Select features and scale using training-only statistics.
2. Generate fixed-length sequences within each road.
3. Ensure sequences never cross road or split boundaries.
4. Implement PyTorch dataset/dataloader.
5. Implement one/two-layer LSTM or GRU, dropout, and four-value output head.
6. Train with Adam, early stopping, learning-rate reduction, and best checkpoint restoration.
7. Tune a bounded set of sequence lengths and architecture parameters.
8. Save curves, checkpoint, config, scaler, and feature manifest.
9. Evaluate on the identical test rows/horizons used for classical comparison.
10. Optionally add a congestion head only after the volume model is stable.

### Exit gate
- No pretrained weights.
- Training/validation curves and best epoch are saved.
- Checkpoint reload reproduces inference.
- Head-to-head benchmark is fair and documented.
---
## Step 16 - Add Confidence and Error Analysis
### Goal
Attach trustworthy uncertainty information and explain failure modes.

### Tasks
1. Build validation-residual/conformal intervals for regression by target/horizon.
2. Measure interval coverage and width.
3. Use calibrated class probabilities and uncertainty measures for classification.
4. Add accident risk bands derived from validation decisions.
5. Analyze errors by road, hour, weekday, weather, congestion, and horizon.
6. Compare deep and classical volume errors on identical rows.
7. Export dashboard-ready confidence and error tables.

### Exit gate
- Every forecast output has confidence/uncertainty.
- Coverage/calibration evidence exists.
- Limitations are recorded honestly.
---
## Step 17 - Build the Inference and Reporting Services
### Goal
Create one stable interface between artifacts and the dashboard.

### Tasks
1. Implement model registry loading and active-version selection.
2. Validate prediction requests.
3. Build latest-origin feature preparation.
4. Generate all target/horizon outputs in one request.
5. Attach model/data versions and confidence.
6. Implement batch prediction persistence.
7. Implement human-readable insight generation from actual aggregates.
8. Implement CSV/HTML or equivalent report export.
9. Benchmark full-corridor inference.

### Exit gate
- CLI prediction works without retraining.
- Output schema is validated.
- Full-corridor runtime is measured.
---
## Step 18 - Build the Streamlit Dashboard
### Goal
Deliver the interactive product surface.

### Tasks
1. Build app shell, navigation, status header, shared filters, and artifact caches.
2. Implement all nine required views.
3. Add upload validation and preview.
4. Add selected-segment/horizon prediction interface.
5. Add explicit retraining control through `training_service` with confirmation.
6. Add report export.
7. Show data/model versions and last successful run.
8. Ensure severity colours are consistent.
9. Add graceful empty/error states without fake values.
10. Add import and page smoke tests.

### Exit gate
- All nine views use real outputs.
- No training occurs on ordinary reruns.
- Navigation and core actions pass manual walkthrough.
---
## Step 19 - Reproducibility, Documentation, and Final Acceptance
### Goal
Prove that another reviewer can reproduce and audit FlowCast.

### Tasks
1. Pin compatible direct dependencies.
2. Complete README setup and command reference.
3. Run the full pipeline from a clean artifact state.
4. Run all tests.
5. Launch dashboard and perform acceptance walkthrough.
6. Verify all model cards, metrics, figures, and reports exist.
7. Measure runtime and memory-sensitive stages.
8. Write final technical report covering mathematics, data quality, models, deep benchmark, confidence, errors, limitations, and recommendations.
9. Compare results against every success criterion.
10. Record unmet targets without hiding them.
11. Update all dynamic Markdown files to final state.

### Final checks
```bash
pytest -q
python -m flowcast.cli run-all
streamlit run dashboard/app.py
```

### Definition of done
A fresh reviewer can reproduce the pipeline, metrics, persisted models, predictions, reports, and nine-view dashboard from the delivered raw files through documented commands.
