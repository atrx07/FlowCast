# ROADMAP.md

## 1. Roadmap Purpose

This roadmap converts the FlowCast PRD into a controlled four-week implementation plan. It defines scope, milestone gates, priorities, and release acceptance. Detailed procedures live in `STEPS.md`; current progress lives in `STATUS.md`; the immediate action lives in `NEXT_STEP.md`.

## 2. Release Goal

Deliver FlowCast v1.0 as a reproducible Streamlit traffic-intelligence web application for the 25-segment Northline Corridor, backed by:

- Deterministic ingestion and data-quality handling.
- Analysis-ready, leakage-safe features.
- Classical regression and classification models.
- A recurrent deep-learning volume forecaster.
- Time-aware evaluation and confidence estimates.
- Persisted predictions, metrics, model cards, and reports.
- Nine required dashboard views plus upload, predict, retrain, and export controls.

## 3. Priority Model

### Must have

- FR-01 through FR-05 and FR-07 through FR-10, FR-12 through FR-15.
- Immutable raw data and full data lineage.
- Time-aware train/validation/test split.
- Volume, speed, travel-time, congestion, and accident-risk outputs.
- NumPy gradient-descent linear regression.
- Linear Regression, Decision Tree, Random Forest, XGBoost, and classification SVM.
- LSTM/GRU built and trained from scratch.
- Persisted models and reproducible inference.
- Nine dashboard views on real data/model outputs.
- Tests for critical data, leakage, model-loading, and dashboard paths.
- One-command pipeline and one-command dashboard launch.

### Should have

- Versioned processed dataset and formal data-quality report.
- Prediction confidence/uncertainty.
- Weather-versus-traffic view.
- Upload validation and prediction refresh.
- Explicit user-triggered retraining workflow.
- Calibrated accident-risk probabilities and validation-selected operating threshold.
- Cross-platform scripts and clear failure messages.

### Could have

- Downloadable insights summary for selected filters.
- Bidirectional LSTM for historical comparison only.
- Rich map view using existing latitude/longitude.
- HTML/PDF report styling beyond the minimum export.
- Optional TCN comparison after all acceptance gates pass.

### Explicitly out of scope for v1.0

- Live streaming ingestion.
- Traffic-signal control.
- Multi-city or multi-corridor deployment.
- Graph neural networks.
- Transformers/attention models.
- Cloud infrastructure or paid services.
- Separate React frontend or FastAPI deployment.
- Full MLOps automation, drift monitoring, or scheduled retraining.
- Mobile/native applications.

## 4. Milestone Map

| Milestone | Target week | Exit artifact | Current state |
|---|---:|---|---|
| M0 - Governance and plan | Before Week 1 | Reference Markdown pack and approved architecture | Complete |
| M1 - Ingestion and validation | Week 1 | Validated raw loads, quarantine logs, schema tests | Complete - Step 03 gate passed with deterministic artifacts and 26 tests |
| M2 - Cleaning and merge | Week 1 | Cleaned source tables and merged interim table | Complete - Step 06 gate passed with zero misses or row multiplication |
| M3 - Features and targets | Week 1 | Analysis-ready versioned dataset | Complete - Step 08 gate passed with 181,200 origins and 20 masked targets |
| M4 - EDA and quality report | Week 1 | EDA notebook, figures, data-quality report | Complete - Step 09 gate passed with nine reconciliations and six figures |
| M5 - Classical ML | Week 2 | Scoreboard, selected models, model cards | Complete - Step 14 verified 20 entries and 1,078,957 indexed predictions |
| M6 - Deep learning | Week 3 | Sequence model, curves, benchmark | Complete - Steps 15-16 verified with confidence/error artifacts |
| M7 - Dashboard | Week 4 | Nine-view Streamlit app and support controls | In progress - Step 17 inference/reporting service next |
| M8 - Reproducibility and delivery | Week 4 | Clean rerun, final report, README, acceptance evidence | Not started |

## 5. Week 1 - Data Engineering and EDA

### Objectives

- Establish repository, environment, configuration, logging, and tests.
- Copy delivered CSVs into immutable `data/raw/` and record checksums.
- Build schema validation and quarantine behaviour.
- Clean traffic, weather, and calendar data deterministically.
- Reconstruct the expected road-window grid and identify missing sensor windows.
- Merge data at one row per road segment and 30-minute timestamp.
- Engineer leakage-safe features and future targets.
- Produce a versioned processed dataset, EDA, and quality report.

### Week 1 acceptance gate

- All raw files are unchanged and checksummed.
- Duplicate, invalid, missing, and normalized counts are reported.
- No silent row loss occurs.
- Final modelling keys are unique.
- Join cardinality is verified and row explosion is impossible.
- Modelling columns meet documented null/range rules.
- Lags and rolling features pass leakage tests.
- Processed output can be rebuilt from raw data through a CLI command.
- EDA observations lead to documented modelling choices.

## 6. Week 2 - Classical Machine Learning

### Objectives

- Define frozen time boundaries for train, validation, and test.
- Build reusable preprocessing without leakage.
- Implement NumPy linear regression and gradient checks/tests.
- Train regression models for volume, speed, and travel time.
- Train congestion and accident-risk classifiers.
- Use time-series cross-validation within the training period.
- Handle accident class imbalance using training-only class weights and threshold tuning.
- Calibrate selected classifier probabilities where feasible.
- Build one scoreboard and select the winning model per target/horizon.
- Persist artifacts, feature schemas, metrics, and model cards.

### Week 2 acceptance gate

- Final test period remained untouched until model selection was frozen.
- Scratch linear regression converges on a controlled test and real training subset.
- Every required model family was evaluated or a documented technical blocker exists.
- Metrics are calculated consistently from persisted predictions.
- Congestion uses Macro-F1 as primary metric.
- Accident risk uses ROC-AUC and includes PR-AUC/threshold analysis.
- Model loading reproduces stored predictions within tolerance.
- Selected models have model cards and lineage metadata.

## 7. Week 3 - Deep Learning and Confidence

### Objectives

- Build segment-wise sequence windows with no cross-road contamination.
- Train a PyTorch LSTM or GRU for multi-horizon volume forecasting.
- Tune sequence length, hidden size, layers, dropout, learning rate, and batch size within a controlled budget.
- Use the verified CUDA path for material recurrent training when it provides a
  practical benefit, while retaining configurable CPU execution and portable
  checkpoints as acceptance requirements.
- Apply early stopping and best-weight restoration.
- Persist train/validation curves and the best checkpoint.
- Evaluate on the exact classical hold-out.
- Add prediction intervals/confidence for regression.
- Compare deep and classical volume models honestly.

### Week 3 acceptance gate

- Sequence split is time-safe and segment-safe.
- No pretrained weights or transfer learning are used.
- Repeated seeded inference is stable.
- Training curves show the selected epoch and overfitting controls.
- Deep and classical comparison uses identical rows, horizons, and metrics.
- The report states whether the deep model earned its complexity.
- Confidence estimates are validated for coverage/calibration on validation/test data.

## 8. Week 4 - Dashboard, Reporting, and Packaging

### Objectives

- Build Streamlit shell, navigation, cache boundaries, and shared filters.
- Implement all nine required views.
- Connect views to persisted datasets, predictions, metrics, and models.
- Add validated file upload and prediction refresh.
- Add an explicit retraining action that does not run on normal page refresh.
- Add segment/horizon prediction controls.
- Add report export.
- Complete README and final technical report.
- Run clean-environment reproduction and acceptance walkthrough.

### Week 4 acceptance gate

- All nine views render with real values.
- No model trains during ordinary Streamlit reruns.
- Invalid uploads fail clearly and are not merged into trusted data.
- Full-corridor inference meets or meaningfully approaches the 30-second target; measured results are reported.
- Every displayed model result includes version and confidence information.
- Any view is reachable within three clicks.
- One command rebuilds the pipeline; one command launches the dashboard.
- Fresh-environment acceptance walkthrough is documented.

## 9. Requirement Traceability

| Requirement group | Main implementation milestone | Main verification |
|---|---|---|
| FR-01 to FR-06 | M1-M4 | Schema tests, quarantine report, processed data contract |
| FR-07 to FR-12 | M5-M6 | Scoreboard, hold-out metrics, persistence tests, model cards |
| FR-13 to FR-18 | M7-M8 | Dashboard smoke tests and acceptance walkthrough |
| Performance | M7-M8 | Timed batch inference benchmark |
| Reproducibility | Every milestone | Seeded runs, hashes, clean rerun |
| Reliability | M1-M4 | Quarantine counts and zero silent drops |
| Transparency | M5-M8 | Data version, feature version, model version, prediction lineage |

## 10. Release Metrics

### Formal targets

- Volume MAPE <= 12% on hold-out.
- Congestion Macro-F1 >= 0.80.
- Accident-risk ROC-AUC >= 0.75.
- Deep volume RMSE lower than the best classical volume RMSE.
- Full-corridor inference for one horizon <= 30 seconds.

### Quality targets

- Zero duplicate `road_id + timestamp` keys in trusted processed data.
- Zero unreported row drops.
- Zero future leakage in features.
- Zero placeholder values in dashboard analytics.
- 100% selected models with model cards and load tests.
- 100% required views included in acceptance walkthrough.

## 11. Major Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Scope too large for four weeks | Incomplete dashboard or weak models | Enforce must/should/could order; do not add extra frameworks |
| Leakage from lags/rolling/scalers | Inflated metrics and invalid project | Shift-before-roll tests; fit learned preprocessing on train only |
| Missing full sensor windows | Biased sequences and broken lags | Reindex per segment; preserve missingness flags; apply explicit gap policy |
| Rare accidents (~0.94%) | Misleading accuracy and unstable classifier | Class weights, PR-AUC, calibrated probability, threshold analysis |
| Classifier acceptance goals missed | Congestion/risk forecasts are weaker than the formal target | Step 16 preserves sealed-test evidence, probability reliability, confusion/slice diagnostics, and rare-event support flags; carry these limitations into inference and dashboard views |
| Deep model fails to beat XGBoost | Formal target missed | Tune within budget; report honestly; keep strong classical fallback |
| Deep model trails classical at the 120-minute horizon | All-horizon comparison goal is missed despite wins at 30-90 minutes | Step 16 preserves exact-row evidence and identifies the largest deficit in late-night 120-minute slices; retain the classical fallback and expose the limitation |
| Reporting config invalidates frozen model hashes | Unnecessary retraining or stale lineage | Keep the Step 14 registry contract independent of frozen training config and recursively verify both chains |
| Multi-horizon complexity | Too many model artifacts | Use generated horizon loops and shared interfaces/configuration |
| Streamlit rerun retrains models | Slow and unsafe UI | Persist artifacts; explicit retraining service only |
| Cross-platform environment issues | Reviewer cannot run project | Python module CLI, pathlib, pinned dependencies, CPU fallback |
| CUDA path becomes a hidden requirement | CPU-only reviewers cannot reproduce training or load artifacts | Keep device-agnostic code, explicit CPU override, bounded batches, portable state dictionaries, and CPU smoke coverage |
| GPU used for trivial work | Startup/transfer overhead wastes time and power | Reserve CUDA for material tensor workloads; keep small tests, probes, tabular processing, and lightweight inference on CPU |
| Large files committed to Git | Bloated repository | Gitignore generated artifacts; document artifact generation |

## 12. Change Control

A proposed change must be classified before implementation:

- **Clarification:** no scope change; update relevant docs and continue.
- **Architecture change:** update `ARCHITECTURE.md`, `STATUS.md`, and `NEXT_STEP.md` before or with code.
- **Roadmap change:** update milestone/risk sections and explain why.
- **Scope expansion:** requires explicit user approval.
- **Acceptance reduction:** requires explicit user approval and must remain visible in the final report.

## 13. Current Roadmap Position

Milestones M0 through M6 are complete and M7 is in progress. Step 10 froze the
shared chronological evaluation and preprocessing contract, and Step 11
verified the required NumPy regression mathematics. Step 12 now provides 12
direct volume/speed/travel-time models across horizons 1-4, complete
Linear/Tree/Forest/XGBoost evidence, a pre-test selection freeze, persisted
predictions and model cards, and honest final hold-out metrics. Random Forest
won all 12 validation comparisons, and all four volume horizons met the formal
hold-out MAPE target. Step 13 now adds eight direct congestion/accident
classifiers, all four required families, chronological probability-calibration
assessment, validation-only accident thresholds, ordered persisted
probabilities, model cards, and a pre-test freeze. The final congestion
Macro-F1 range of 0.7468-0.7540 and accident ROC-AUC range of 0.5894-0.6237
miss the formal goals and remain visible for later error analysis. Step 14 now
adds a deterministic 20-entry combined registry, task-aware scoreboard,
complete hash lineage, verified loading, selection rationales, and an in-place
index of all 1,078,957 persisted validation/test predictions without
retraining. Step 15 now adds a from-scratch, four-output PyTorch LSTM selected
on common validation origins, with zero road/split/gap/target-boundary
violations, persisted curves/checkpoint/model card, verified reload, and
212,000 validation/test prediction rows. On 26,500 exact shared test origins
per horizon it beats the frozen classical volume model at 30, 60, and 90
minutes, while trailing by 0.0471 RMSE at 120 minutes. This misses the formal
all-horizon comparison goal without invalidating the implementation gate.
Step 16 now adds 16 validation-only finite-sample conformal calibrations,
862,700 regression-confidence rows, 428,257 classification-confidence rows,
212,000 exact paired volume rows, fixed-bin reliability, threshold-relative
accident risk bands, and 3,408 minimum-support error slices. Test interval
coverage ranges from 0.8924 to 0.9055 around the nominal 0.90 level. The weak
classifier goals and the recurrent 120-minute deficit remain explicit rather
than triggering post-test changes. Step 17 inference and reporting services
are next. See `STATUS.md` and `NEXT_STEP.md` for the live state.
