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

Last verified on 26 July 2026. Milestones M0-M7 are complete and M8 is in
progress. Steps 00-18 now
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
model/card lineage. Step 15 adds a from-scratch PyTorch LSTM that consumes 12
road-local half-hour steps and produces all four volume horizons in one pass.
The selected epoch-8 checkpoint uses 126,475 training sequences and 26,500
validation/test sequences with zero road, split, cadence, or target-boundary
violations. Hold-out RMSE is 60.1443, 60.8154, 61.2014, and 61.8966 for
30-120 minutes. On the exact shared test origins, it beats the frozen classical
model at the first three horizons and trails by 0.0471 RMSE at 120 minutes, so
the formal all-horizon goal remains honestly unmet. Its state dictionary,
training curves, model card, environment snapshot, 212,000 persisted
predictions, and four-entry registry extension pass verified reload and
tamper-rejection tests. Step 16 adds validation-only 90% split-conformal
intervals for every classical regression output and recurrent volume,
maximum-probability/entropy confidence for classifiers, fixed-bin reliability,
frozen-threshold accident risk bands, and minimum-support error slices by road,
time, weekday, peak period, weather, congestion, and horizon. Its dashboard-ready
outputs contain 862,700 regression rows, 428,257 classification rows, and
212,000 exact deep/classical paired rows. Test interval coverage across the 16
groups is 0.8924-0.9055; the weak classifier goals and 120-minute recurrent
deficit remain visible. Step 17 adds one verified frozen-model `Predictor` with
validation-led recurrent volume routing, an explicit classical volume
comparator, speed/travel-time/congestion/accident outputs, unchanged Step 16
confidence, and full data/model/config lineage. The latest-origin 25-road
one-horizon CPU request completes in 2.350 seconds cold against the 30-second
target; the complete four-horizon request produces 100 validated rows in 4.222
seconds cold. Request-scoped Parquet/JSON batches, verified reload, real-data
insights, full CSV, and self-contained HTML reports are available through the
CLI. Step 18 adds the complete ten-route Streamlit product surface: all nine
required real-output views, shared corridor filters, verified lineage,
frozen-model prediction controls, schema-validated upload staging, verified
report downloads, audit evidence, and explicit duplicate-safe versioned
retraining that never changes active routing on a rerun. The supplied
`DESIGN.md` is implemented as a dark, editorial Streamlit-native design system.
Every route now uses a compact operational opener and a data-derived
plain-language reading with chart guidance; no brief invents values or causal
claims. The live route now reads header, displayed horizon, KPI cards,
frozen-model request, current reading, then aligned Corridor Signal/Priority
Queue outputs. At 1280 x 720 the final 60px banner is 50.9% shorter than the
preceding compact version and 83.9% shorter than the original hero; all
road/date/time/horizon controls plus Run prediction fit above the fold. The
verified status strip clears Streamlit's top toolbar. The request still uses a
calendar plus a 30-minute time field for all 7,237 model-eligible origins from
1 January through 31 May instead of a seven-day timestamp dropdown.
The complete assurance suite passes all 180 tests, including deterministic
inference, report/prediction tamper rejection, an isolated notebook smoke, an
exact-exit pytest runner, and a session guard that restores and rejects
tracked-file mutation. Browser QA verified all ten routes at 1280 x 720,
1440 x 900, and 1920 x 1080 with no page exceptions, horizontal overflow, or
top-chrome overlap. Step 19 clean reproduction, final technical reporting, and
delivery acceptance remain ahead.

## Delivery timeline

| Planned phase | Milestones | Current state |
|---|---|---|
| Before Week 1 | M0 governance and plan | Complete |
| Week 1 | M1 ingestion, M2 cleaning/merge, M3 features/targets, M4 EDA | Complete - Steps 01-09 verified |
| Week 2 | M5 classical machine learning | Complete - Steps 10-14 verified |
| Week 3 | M6 recurrent deep learning and confidence | Complete - Steps 15-16 verified |
| Week 4 | M7 dashboard, M8 reproducibility and delivery | In progress - Step 18 plus desktop readability pass complete; Step 19 next |

## Quick start

Python 3.11 is required.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[classical,deep,eda,dashboard,test]"
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
.\.venv\Scripts\python.exe -m flowcast.cli train-recurrent-volume
.\.venv\Scripts\python.exe -m flowcast.cli analyze-confidence
.\.venv\Scripts\python.exe -m flowcast.cli predict --horizons 1 2 3 4 --export-reports
.\.venv\Scripts\python.exe scripts\run_tests.py -q
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

To rebuild reports from an existing verified prediction batch:

```powershell
.\.venv\Scripts\python.exe -m flowcast.cli build-reports --manifest <path-to-manifest.json>
```

The test runner prints `FLOWCAST_PYTEST_EXIT=0` only for a successful suite and
faithfully returns pytest's status. Tests that generate outputs use temporary
roots; a session guard restores and fails any tracked repository mutation. This
prevents notebook or environment-snapshot smoke tests from invalidating frozen
artifact hashes.

On a compatible NVIDIA Windows/Linux workstation, install the optional official
CUDA wheel after the portable environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall -r requirements-cuda.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

FlowCast selects devices through configuration. Use CUDA for material recurrent
training when available; CPU remains supported for training, tests, inference,
small jobs, and machines without an NVIDIA GPU. A separate system-wide CUDA
Toolkit is not required by the approved wheel.

On macOS or Linux, use `.venv/bin/python` in place of the Windows interpreter
path.

The source documents and delivered CSV files remain read-only in
`FlowCast-project_file/`. Byte-identical working copies belong in `data/raw/`;
versioned validated data and issue records are written beneath `data/interim/`
and `data/quarantine/`.

See `PROJECT.md` for the product contract, `ARCHITECTURE.md` for the system
design, `TECH_STACK.md` for approved technologies, and `STATUS.md` /
`NEXT_STEP.md` for verified progress and the immediate build step.
