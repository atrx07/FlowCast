# FlowCast

FlowCast is an end-to-end traffic forecasting and congestion-intelligence
project for the 25-segment Northline Corridor. It is being built as a
reproducible Python 3.11 pipeline with a Streamlit dashboard.

For each road segment, FlowCast will forecast the next 30, 60, 90, and 120
minutes of traffic volume, average speed, congestion class, travel time, and
accident risk, together with prediction confidence. The finished system will
combine validated traffic, weather, and calendar data with classical machine
learning and a from-scratch recurrent neural network.

## Current status

Last verified on 24 July 2026. Milestones M0-M5 are complete and M6 is next.
Steps 00-14 now
provide immutable raw preservation, executable data contracts,
reason-preserving quarantine, trusted cleaning, a cardinality-safe 181,200-row
merged table, 62 leakage-safe explanatory features, and a versioned processed
dataset with 20 targets plus explicit availability masks across the four
forecast horizons. The reproducible EDA layer adds nine passing quality
reconciliations, 67 contextual slices, six inspected figures, machine-readable
statistics, a generated report, and a notebook that runs top-to-bottom. The
evaluation layer freezes 5,074/1,087/1,087 train/validation/test timestamps,
five horizon-gapped training CV folds, four training-only preprocessors, and a
default-sealed test partition. Step 11, the from-scratch NumPy linear-regression
baseline, now includes explicit prediction, MSE,
analytical gradients, central finite-difference checks, deterministic synthetic
recovery, and a real-data comparison on 27,150 validation rows. The scratch
model converged after 4,153 updates with validation RMSE 86.93 versus 80.87 for
scikit-learn on identical inputs; test remained sealed during that proof.
Step 12 adds 12 direct regression pipelines across volume, speed, and travel
time at four horizons, with seven configurations spanning Linear Regression,
Decision Tree, Random Forest, and XGBoost over all five frozen CV folds. All 12
choices were persisted before one final test evaluation; Random Forest won each
validation comparison. Volume hold-out MAPE ranges from 10.218% to 10.952%, so
all four horizons meet the formal 12% target. The run persists 650,700
validation/test predictions, 12 model cards, candidate/family scoreboards, and
feature importance. Step 13 adds eight direct congestion and accident-risk
classifiers with Decision Tree, Random Forest, XGBoost, and scaled SVM evidence
across all five folds. All family, probability-calibration, and
accident-threshold decisions were persisted before one final test load. The
428,257 validation/test prediction rows contain finite normalized
probabilities, and all eight selected models have verified model cards.
Congestion hold-out Macro-F1 is 0.7468-0.7540 and accident ROC-AUC is
0.5894-0.6237, so the formal classifier goals remain honestly unmet. Step 14
adds a deterministic, tamper-checked 20-entry classical registry and task-aware
scoreboard while preserving every frozen choice and test result. Its indexed
manifest maps all 1,078,957 existing validation/test prediction rows without
copying or fabricating values, and verified loading resolves every registered
model/card lineage. All 137 tests pass. Step 15 from-scratch recurrent volume
forecasting is the
immediate gate; confidence, inference services, report export, and the
dashboard remain ahead.

## Delivery timeline

| Planned phase | Milestones | Current state |
|---|---|---|
| Before Week 1 | M0 governance and plan | Complete |
| Week 1 | M1 ingestion, M2 cleaning/merge, M3 features/targets, M4 EDA | Complete - Steps 01-09 verified |
| Week 2 | M5 classical machine learning | Complete - Steps 10-14 verified |
| Week 3 | M6 recurrent deep learning and confidence | Next - Step 15 recurrent model |
| Week 4 | M7 dashboard, M8 reproducibility and delivery | Not started |

## Quick start

Python 3.11 is required.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[classical,eda,test]"
.\.venv\Scripts\python.exe -m flowcast.cli audit
.\.venv\Scripts\python.exe -m flowcast.cli validate
.\.venv\Scripts\python.exe -m flowcast.cli clean-context
.\.venv\Scripts\python.exe -m flowcast.cli clean-traffic
.\.venv\Scripts\python.exe -m flowcast.cli merge-sources
.\.venv\Scripts\python.exe -m flowcast.cli engineer-features
.\.venv\Scripts\python.exe -m flowcast.cli prepare-data
.\.venv\Scripts\python.exe -m flowcast.cli eda
.\.venv\Scripts\python.exe -m flowcast.cli prepare-modeling
.\.venv\Scripts\python.exe -m flowcast.cli train-scratch-linear
.\.venv\Scripts\python.exe -m flowcast.cli train-classical-regression
.\.venv\Scripts\python.exe -m flowcast.cli train-classical-classification
.\.venv\Scripts\python.exe -m flowcast.cli build-classical-registry
.\.venv\Scripts\python.exe -m pytest -q
```

On macOS or Linux, use `.venv/bin/python` in place of the Windows interpreter
path.

The source documents and delivered CSV files remain read-only in
`FlowCast-project_file/`. Byte-identical working copies belong in `data/raw/`;
versioned validated data and issue records are written beneath `data/interim/`
and `data/quarantine/`.

See `PROJECT.md` for the product contract, `ARCHITECTURE.md` for the system
design, `TECH_STACK.md` for approved technologies, and `STATUS.md` /
`NEXT_STEP.md` for verified progress and the immediate build step.
