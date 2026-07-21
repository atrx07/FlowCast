# PROJECT.md

## 1. Project Identity

**Name:** FlowCast - Smart Traffic Flow Prediction System

**Release:** v1.0

**Delivery model:** Single-engineer capstone/internship project

**Planned build window:** Four weeks

**Primary interface:** Streamlit web application

**Reference source:** `FlowCast-project_file/`

FlowCast is an end-to-end urban traffic intelligence system for the Northline Corridor. It converts raw road-sensor, weather, and calendar data into short-horizon forecasts and operational analytics for 25 road segments.

## 2. Business Problem

The corridor operations team currently reacts after congestion or incidents emerge. FlowCast must answer:

> Given all information available up to the current 30-minute window, what will each road segment look like 30-120 minutes ahead, and where is operational risk concentrating?

The system is a decision-support layer. It does not directly control traffic signals.

## 3. Intended Users

| User | Main need | Primary product surface |
|---|---|---|
| Traffic Operations Analyst | Anticipate congestion before it forms | Live forecasts, heatmaps, confidence, segment ranking |
| Incident Response Coordinator | Identify rising incident risk | Accident-risk probability and ranked roads |
| Transport Planner | Understand historical and structural patterns | Trends, road comparison, weather impact |
| System Owner / Reviewer | Verify trust, quality, and reproducibility | Metrics, model cards, feature importance, data-quality report |

## 4. Product Outputs

For each road segment and each future horizon of 1-4 windows (30, 60, 90, and 120 minutes), FlowCast should produce:

1. Predicted traffic volume.
2. Predicted average speed.
3. Predicted congestion class:
   - Free-flow
   - Moderate
   - Heavy
   - Severe
4. Predicted/estimated travel time.
5. Accident-risk probability.
6. Prediction confidence or uncertainty.

The dashboard also exposes historical analytics, model diagnostics, data quality, and downloadable reports.

## 5. Source Data

The original files are stored in `FlowCast-project_file/` and must remain unchanged.

### 5.1 Delivered files

| File | Grain | Observed size |
|---|---|---:|
| `traffic_sensor_log.csv` | Road segment x 30-minute window | 178,468 rows, 17 columns |
| `weather_observations.csv` | Weather station x hour | 10,872 rows, 7 columns |
| `calendar_events.csv` | Calendar date | 151 rows, 6 columns |
| `FlowCast_PRD.docx` | Product requirements | Formal v1.0 requirements |
| `FlowCast_Data_Dictionary.docx` | Field and join definitions | Formal data reference |

### 5.2 Coverage

- Date range: 1 January 2025 through 31 May 2025.
- Traffic roads: 25 (`NL-001` through `NL-025`).
- Traffic times: 48 half-hour windows per day.
- Weather stations: `WS-NORTH`, `WS-CENTRAL`, `WS-SOUTH`.
- Expected complete traffic grid: 181,200 road-window rows.
- Unique delivered road-window keys after duplicate collapse: 176,701.
- Entirely missing road windows: 4,499.

### 5.3 Confirmed raw-data defects

Traffic data:

- 1,767 exact/key duplicate rows.
- 4,387 null `traffic_volume` values.
- 4,382 null `avg_speed` values.
- 4,383 null `occupancy` values.
- 26,883 blank `congestion_level` values.
- 241 negative traffic-volume values.
- Speeds reach 319.6 km/h, including invalid values above 200.
- Occupancy reaches 259.9%, including invalid values above 100.
- 1,669 rows have at least one recorded accident, about 0.94% of rows.
- `vehicle_type_dist` is a JSON string with keys `2W`, `Car`, `LCV`, and `HCV`.

Weather data:

- Complete station-hour grid of 10,872 rows.
- 167 null temperatures.
- 111 null visibility values.
- Inconsistent weather labels such as `Clear`, `clear`, `CLEAR`, `Rain`, `rainy`, `RAIN`, and `rain `.
- Weather dates use `DD/MM/YYYY`, unlike traffic and calendar.

Calendar data:

- Complete 151-day date coverage.
- Six holidays, six event days, and eleven roadwork days.
- Blank holiday/event names are expected where their flag is zero.

## 6. Join and Derivation Contract

### 6.1 Timestamp alignment

- Traffic timestamp = traffic `date` (`YYYY-MM-DD`) + traffic `time`.
- Weather timestamp = weather `date` (`DD/MM/YYYY`) + weather `time`.
- Weather is hourly; each observation applies to the two traffic windows inside that hour.

### 6.2 Joins

- Traffic to weather: `traffic.weather_station_id = weather.station_id` plus aligned hour.
- Traffic to calendar: normalized traffic date = calendar date.
- The final modelling grain remains one row per `road_id + timestamp`.

### 6.3 Missing congestion label

For blank congestion labels:

```text
half_hour_capacity = road_capacity / 2
volume_capacity_ratio = traffic_volume / half_hour_capacity
```

Banding:

| Ratio | Congestion class |
|---:|---|
| `< 0.50` | Free-flow |
| `0.50-0.79` | Moderate |
| `0.80-0.99` | Heavy |
| `>= 1.00` | Severe |

The implementation must define and test exact boundary behaviour using numeric comparisons, not rounded display values.

## 7. Functional Scope

### 7.1 Data platform

- Load and validate all three sources.
- Quarantine invalid records with reasons.
- Remove duplicates deterministically.
- Repair or impute recoverable missing values.
- Normalize weather labels and dates.
- Parse vehicle-distribution JSON.
- Align and merge sources.
- Generate a versioned analysis-ready dataset.
- Generate a machine-readable and human-readable data-quality report.

### 7.2 Feature platform

Required feature families:

- Hour/day cyclical encodings.
- Weekend and peak-period flags.
- Volume and speed lags at `t-1`, `t-2`, and `t-48`.
- Rolling mean and standard deviation over previous 4 and 8 windows.
- Volume-to-capacity ratio and capacity headroom.
- Rain and low-visibility flags.
- Temperature bands.
- Holiday x peak interactions.
- Event proximity and roadwork signals.
- Parsed vehicle-class proportions.

### 7.3 Classical modelling

Required baselines and candidates:

- From-scratch NumPy linear regression with explicit loss and gradients.
- scikit-learn Linear Regression.
- Decision Tree.
- Random Forest.
- XGBoost.
- SVM classification baseline.

Required prediction tasks:

- Volume regression.
- Average-speed regression.
- Travel-time regression.
- Four-class congestion classification.
- Binary accident-risk probability classification.

### 7.4 Deep learning

- Primary model: LSTM or GRU sequence forecaster built with no pretrained weights.
- Framework: PyTorch by default.
- Time-ordered sequence generation and splitting.
- Dropout, early stopping, learning-rate scheduling, and best-weight restoration.
- Multi-horizon volume output is the preferred v1 design.
- Optional congestion head only after the volume model is stable.
- Benchmark against classical models on the identical test window.

### 7.5 Dashboard

The Streamlit application must contain:

1. Live/near-term predictions.
2. Historical trends.
3. Congestion heatmap.
4. Road comparison.
5. Model performance.
6. Feature importance.
7. Forecast visualisation.
8. Prediction confidence.
9. Weather versus traffic.

Supporting functions:

- Data upload and validation.
- Explicit user-triggered retraining.
- Segment and horizon prediction controls.
- Insights/report export.

## 8. Non-Functional Contract

| Area | Requirement |
|---|---|
| Performance | Full-corridor batch inference for one horizon should complete in 30 seconds or less on a workstation |
| Memory | Full multi-month pipeline should run on a 16 GB system |
| Reproducibility | Seed randomness; reproduce reported results from raw inputs |
| Maintainability | Modular code, PEP 8, type hints, docstrings, source files normally under 500 lines |
| Usability | Any dashboard view reachable within three clicks |
| Reliability | Invalid rows quarantined; zero silent drops |
| Portability | Windows, macOS, and Linux through a pinned environment |
| Transparency | Predictions traceable to model, feature, and data versions |

## 9. Success Criteria

| Dimension | Target |
|---|---:|
| Volume forecast | Hold-out MAPE <= 12% |
| Congestion classification | Hold-out Macro-F1 >= 0.80 |
| Accident-risk ranking | Hold-out ROC-AUC >= 0.75 |
| Deep vs classical | Deep sequence model beats best classical volume RMSE |
| Reproducibility | Full raw-to-dashboard pipeline runs with one command |
| Dashboard | All nine required views render using real outputs |

These are honest evaluation targets. Failure to reach a target must be documented with error analysis; the test split must never be adjusted to manufacture success.

## 10. Technical Blueprint

- Python 3.11.
- NumPy and Pandas for numerical/data work.
- scikit-learn and XGBoost for classical models.
- PyTorch for the recurrent model.
- Plotly and Matplotlib for charts; Seaborn may be used only in notebooks/reports if useful.
- Streamlit for the web interface.
- Jupyter for EDA narrative, not production pipeline logic.
- YAML configuration.
- Parquet for processed datasets.
- Joblib for scikit-learn artifacts; framework-native files for deep models.
- Git for source history.

## 11. Key Product Decisions

1. There is one user-facing Streamlit web application, not separate frontend and backend deployments.
2. Backend functionality is a modular Python package shared by CLI, tests, and dashboard.
3. Dashboard pages read persisted artifacts by default; training occurs only through an explicit action or CLI command.
4. The processed-data and model pipelines are versioned and traceable.
5. Evaluation is strictly time-aware.
6. Classical models use direct per-horizon prediction unless a tested multi-output design proves simpler and equivalent.
7. Regression uncertainty uses validation-calibrated residual/conformal intervals where practical.
8. Classification confidence uses calibrated probabilities; accident-risk threshold is selected on validation data.
9. Raw files and original reference documents are immutable.
10. No separate API or database is required for v1.0.

## 12. Expected Deliverables

- Reproducible repository.
- Immutable raw source copies and checksums.
- Versioned processed dataset.
- Quarantine datasets and data-quality report.
- EDA notebook and exported figures.
- Classical model artifacts and scoreboard.
- Deep model, curves, and benchmark.
- Model cards for every selected model.
- Streamlit dashboard with nine views and supporting modules.
- Final technical report.
- README with setup, commands, architecture, and reproduction instructions.

## 13. Definition of Done

FlowCast v1.0 is done only when a reviewer can set up the project, rebuild from raw files, reproduce the documented hold-out metrics, load models without retraining, run the Streamlit application, use all required views, request predictions for 30-120 minutes, validate an uploaded file, and export a report while maintaining complete lineage to the source data and model version.
