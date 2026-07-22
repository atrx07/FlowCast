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

Last verified on 22 July 2026. Milestones M0-M4 are complete and M5 is in
progress. Steps 00-10 now
provide immutable raw preservation, executable data contracts,
reason-preserving quarantine, trusted cleaning, a cardinality-safe 181,200-row
merged table, 62 leakage-safe explanatory features, and a versioned processed
dataset with 20 targets plus explicit availability masks across the four
forecast horizons. The reproducible EDA layer adds nine passing quality
reconciliations, 67 contextual slices, six inspected figures, machine-readable
statistics, a generated report, and a notebook that runs top-to-bottom. The
evaluation layer freezes 5,074/1,087/1,087 train/validation/test timestamps,
five horizon-gapped training CV folds, four training-only preprocessors, and a
default-sealed test partition. All 98 tests pass. Step 11, the from-scratch
NumPy linear-regression baseline, is the immediate next gate. Trained forecast
models, inference, confidence, report export services, and the dashboard have
not begun.

## Delivery timeline

| Planned phase | Milestones | Current state |
|---|---|---|
| Before Week 1 | M0 governance and plan | Complete |
| Week 1 | M1 ingestion, M2 cleaning/merge, M3 features/targets, M4 EDA | Complete - Steps 01-09 verified |
| Week 2 | M5 classical machine learning | In progress - Step 10 complete; Step 11 next |
| Week 3 | M6 recurrent deep learning and confidence | Not started |
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
