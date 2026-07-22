# STATUS.md

## Status Metadata

- **Project:** FlowCast v1.0
- **Last updated:** 2026-07-22
- **Current milestone:** M4 - EDA and quality report (complete)
- **Current step:** Steps 00-09 complete; Step 10 next
- **Overall state:** Week 1 data-engineering and EDA gate passed
- **Primary blocker:** None

## 1. Verified Current Position

FlowCast now has a reproducible Python 3.11 pipeline from immutable raw inputs
through validation, cleaning, cardinality-safe merging, leakage-safe features,
multi-horizon targets, and a versioned EDA/data-quality layer. Milestones M0
through M4 are complete.

The analysis covers all 181,200 road/half-hour prediction origins, all 25 road
segments, the 62 model-candidate explanatory features, and the 20 masked target
definitions. The EDA notebook delegates to tested package functions and runs
top-to-bottom. Classical/deep model training, inference, confidence, report
export services, and the Streamlit dashboard have not begun.

## 2. Step 09 Implementation

- Added `config/eda.yaml` and `eda_v1` settings for descriptive fields,
  contextual slices, correlation candidates, target association, redundancy
  threshold, class order, and figure settings.
- Added a fail-closed processed-input loader that verifies current config
  hashes, Step 08 summary/manifest lineage, Parquet bytes/SHA-256, row/key
  cardinality, exact column order, and dtypes before analysis.
- Added package-backed descriptive statistics, target distributions,
  road/time/weather/calendar aggregates, correlation/covariance, redundancy
  checks, findings, modelling decisions, and limitations.
- Added a consolidated quality reconciliation that verifies every persisted
  upstream summary and immutable raw-copy hash, then checks source, validation,
  reconstruction, merge, feature, and target counters without handwritten
  estimates.
- Added `flowcast eda [--version VERSION]`, a generated Markdown report,
  machine-readable JSON/CSV outputs, an environment snapshot, and six
  deterministic PNG figures.
- Added `notebooks/01_eda.ipynb`; all calculations remain in the package and
  the notebook executes in a clean temporary Jupyter environment.
- Added unit, full-data contract, determinism, tamper-rejection, figure, and
  notebook smoke tests.

## 3. Produced Artifacts

```text
artifacts/reports/eda_v1/
  summary.json
  data_quality.md
  context_aggregates.csv
  correlation.csv
  covariance.csv
  environment.txt

artifacts/figures/eda_v1/
  traffic_distributions.png
  hourly_profiles.png
  road_comparison.png
  class_balance.png
  weather_traffic.png
  correlation_heatmap.png

notebooks/
  01_eda.ipynb
```

## 4. Step 09 Data Evidence

| Check | Verified result |
|---|---:|
| Dataset rows / unique keys | 181,200 / 181,200 |
| Columns / roads | 188 / 25 |
| Time coverage | 2025-01-01 00:00 to 2025-05-31 23:30 IST |
| Model-candidate features / targets | 62 / 20 |
| Context slices | 67 |
| Reconciliation checks | 9 of 9 passed |
| Exported figures | 6 |

The quality chain reconciles 189,491 delivered source rows (178,468 traffic,
10,872 weather, and 151 calendar), 187,724 validation-retained rows, 1,767
rejected rows, 42,792 issue records, and the reconstructed 181,200-row traffic
grid with 4,499 explicitly inserted windows. Merge coverage remains complete:
zero weather/calendar misses, row loss, row multiplication, or duplicate keys.

Key descriptive evidence:

- Traffic volume: mean 431.52, median 362, range 41-2,090, skew 1.005.
- Average speed: mean 42.10 km/h, median 44.9, range 6.7-64.8.
- Travel time: mean 3.57 minutes, median 3.12, range 0.80-29.91.
- Congestion: Free-flow 61.43%, Moderate 23.82%, Heavy 9.23%, Severe 5.52%.
- Accident risk: 1,652 positives among 176,701 observed labels, a 0.935%
  positive rate and approximately 106 negatives per positive; 4,499 inserted
  windows remain unknown rather than fabricated negatives.
- `NL-006` has the highest mean volume at 565.69; local hour 08 has the
  highest mean volume at 886.01; Rain has the lowest mean speed at 40.27 km/h.
- Current traffic volume has the strongest inspected linear association with
  next-window volume (`r = 0.929527`).
- At the configured absolute-correlation threshold of 0.95, the three flagged
  redundancy pairs are occupancy/V-C ratio, volume lag-2/rolling-mean-4, and
  speed lag-2/rolling-mean-4. These are review signals, not full-data feature
  elimination decisions.

## 5. Artifact Evidence

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| EDA summary JSON | 30,917 | `986ef2fbacce048cb866e5beac106d63979133f992e1af44fc8686c9ba4f4bf7` |
| Data-quality Markdown | 6,273 | `a5c82377be4831a38532a46f4e4be1cf0d80fb01da96cf78a2e3d7b23b3affcf` |
| Context aggregates CSV | 9,068 | `f6a7ad7ac6c58ca7435e00b88be374358e626f356754479d5f0a627616123348` |
| Correlation CSV | 9,175 | `49b848b62b0cfa58ea247006e6504d88617cc236f7f6ce4528f6ef4cd58bd183` |
| Covariance CSV | 10,116 | `7aef5bb1cd5f9ef5c28220d2e9d0f78a5f0404703eed3fcd2c9fcb1326e66b6f` |
| Environment snapshot | 1,994 | `1e4507b1a24dd60ed32851ee9af58f178c772e6f3e97c69054090d6f7ede861c` |

All six figure bytes/hashes are recorded inside the canonical EDA summary.
Repeated runs produced byte-identical CSV, JSON, Markdown, environment, and PNG
artifacts. The six figures were also inspected at original resolution and had
legible labels with no clipping or overlap.

## 6. Validation and Assurance Evidence

Executed with project-local CPython 3.11.9:

```text
.venv/Scripts/python.exe -m flowcast.cli merge-sources
.venv/Scripts/python.exe -m flowcast.cli engineer-features
.venv/Scripts/python.exe -m flowcast.cli prepare-data
.venv/Scripts/python.exe -m flowcast.cli eda
.venv/Scripts/python.exe -m pytest -q tests/unit/test_eda_statistics.py tests/data_contracts/test_eda_contract.py tests/smoke/test_eda_notebook.py
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m compileall -q src tests
.venv/Scripts/python.exe -m flowcast.cli eda --help
git diff --check
```

Verified results:

- All four pipeline commands exited 0; `eda` reported 181,200 rows, 67 context
  slices, and six figures.
- Focused calculation, artifact-contract, and notebook tests: 11 passed in
  9.00 seconds.
- Final full suite: 88 passed in 55.22 seconds.
- Exact findings/counts, nine reconciliation checks, candidate-column safety,
  deterministic reruns, PNG validity, and pre-read tamper rejection passed.
- Notebook execution through a fresh kernel passed top-to-bottom.
- Dependency consistency, byte compilation, CLI-help smoke, and whitespace
  checks passed.
- Largest source module is `src/flowcast/data/audit.py` at 366 physical lines;
  all source files remain below 400 lines.

## 7. Decisions and Constraints

- Evaluation must remain chronological. Step 10 will freeze exact 70/15/15
  timestamp boundaries and time-series CV folds within training only.
- Learned imputers, encoders, scalers, class weights, thresholds, and feature
  selection decisions must use training/validation data only.
- Linear/SVM/recurrent families require training-fitted scaling; tree families
  retain unscaled numeric inputs unless evidence supports a change.
- Congestion evaluation must lead with Macro-F1 and per-class metrics.
- Accident modelling must use observed labels only, training-only class
  weighting, ROC-AUC plus PR-AUC, and validation-based threshold selection.
- Model-specific target/history masks will preserve the common prediction-origin
  table without treating unavailable labels as negatives.
- Full-data redundancy is descriptive only; removal decisions belong inside the
  training protocol.
- Matplotlib remains the preferred plotting library, but Windows Application
  Control blocked its compiled `_path` extension in this environment. The
  documented Pillow fallback generated deterministic PNGs without reducing the
  required EDA/report contract.

## 8. Risks and Unresolved Work

- Severe accident imbalance can make accuracy misleading and may destabilize
  minority recall; Step 10 must isolate all learned handling to training.
- Congestion is dominated by Free-flow observations, so stratified reporting
  and per-class evaluation remain mandatory even with chronological splits.
- Correlations are observational, not causal, and reflect one corridor over 151
  days; conclusions should not be generalized to other corridors without data.
- Causal traffic recovery may smooth some extremes; inserted accident windows
  are explicitly unknown; hourly weather is shared across two half-hour rows.
- Exact split boundaries, preprocessing artifacts, and sealed-test controls are
  not yet implemented.

## 9. Next Gate

Proceed only to **Step 10 - Freeze Splits and Preprocessing**. The bounded action
and acceptance gate are maintained in `NEXT_STEP.md`.
