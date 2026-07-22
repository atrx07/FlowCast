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

Milestone M1 is complete and M2 is in progress. The repository preserves and
verifies raw files, enforces executable data contracts, resolves duplicate
keys, and persists validation lineage. Calendar, weather, and traffic cleaning
are complete. Traffic now has a traceable 181,200-row half-hour grid with
causal repairs and parsed vehicle shares. Source merging, feature engineering,
modelling, and the dashboard remain; current outputs are not model-ready.

## Quick start

Python 3.11 is required.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m flowcast.cli audit
.\.venv\Scripts\python.exe -m flowcast.cli validate
.\.venv\Scripts\python.exe -m flowcast.cli clean-context
.\.venv\Scripts\python.exe -m flowcast.cli clean-traffic
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
