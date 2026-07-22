# STATUS.md

## Status Metadata

- **Project:** FlowCast v1.0
- **Last updated:** 2026-07-22
- **Current milestone:** M3 - Features and targets (in progress)
- **Current step:** Steps 01-07 complete; Step 08 not started
- **Overall state:** Step 07 leakage-safe explanatory feature gate passed
- **Primary blocker:** None

## 1. Verified Current Position

FlowCast has a reproducible Python 3.11 data pipeline from immutable raw inputs
through a hash-verified, cardinality-safe merged table and a deterministic
explanatory-feature artifact. Milestones M0, M1, and M2 are complete. M3 is in
progress with its feature slice complete and future target construction next.

The feature table retains all 181,200 road/half-hour keys and all source,
validation, cleaning, imputation, and join lineage. It adds 62 documented
model-candidate features without constructing a future target or freezing a
split. EDA, models, inference, confidence, reporting services, and the
Streamlit dashboard have not begun.

## 2. Step 07 Implementation

- Added versioned `explanatory_features_v1` configuration and
  `engineered_features_v1` artifacts.
- Added a hash-verified merged-input boundary that checks the current base and
  cleaning configurations, merge summary, Parquet byte count/SHA-256, row
  count, and unique key contract before reading data.
- Added local-hour and day-of-week cyclical encodings, weekend flags, named
  morning/evening peak flags, and a combined peak flag.
- Added within-road volume and speed lags at 1, 2, and 48 windows.
- Added four/eight-window rolling means and sample standard deviations after a
  one-window shift, with full-width minimum history.
- Added half-hour capacity, V/C ratio, capacity headroom, rain and
  low-visibility flags, weather-category indicators, and temperature bands.
- Added holiday x peak, days to the nearest scheduled event, one-day event
  proximity, and preserved event/roadwork inputs.
- Retained vehicle shares and selected missingness, repair, inserted-window,
  and normalization lineage as model-candidate features.
- Added `history_available`; no origin is discarded because history is
  unavailable.
- Added `flowcast engineer-features [--version VERSION]`, deterministic
  Parquet/JSON/Markdown artifacts, and unit/full-source leakage tests.
- Updated README status/timeline and added the user-approved governance rule
  requiring README progress to be current before every push.

## 3. Produced Artifacts

```text
data/interim/engineered_features_v1/
  features.parquet

artifacts/features/engineered_features_v1/
  manifest.json

artifacts/quality/engineered_features_v1/
  summary.json
  summary.md
```

Generated Parquet remains ignored by Git. The feature manifest, canonical
quality JSON, and generated Markdown report are tracked.

## 4. Step 07 Data Evidence

| Check | Verified result |
|---|---:|
| Input rows / keys | 181,200 / 181,200 |
| Output rows / keys | 181,200 / 181,200 |
| Row-count change / duplicate keys | 0 / 0 |
| Model-candidate features | 62 |
| History-available rows | 180,000 |
| History-unavailable rows | 1,200 |
| History-unavailable origins per road | 48 |
| Peak rows | 45,300 |
| Rain rows | 7,166 |
| Low-visibility rows | 954 |
| Event-proximity rows | 21,600 |
| Holiday x peak rows | 1,800 |
| Cool / mild / warm rows | 48,878 / 122,362 / 9,960 |

Expected history nulls are fully accounted for: lag-1 has 25 nulls, lag-2 has
50, lag-48 has 1,200, four-window rolling features have 100, and eight-window
rolling features have 200. Volume and speed follow the same counts. All other
manifest features have zero nulls.

## 5. Artifact Evidence

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Feature Parquet | 17,198,757 | `0ae1bb32a9a53b05f0ac61d393f005785d798490fd33463ba5ffec43da100753` |
| Feature manifest | 19,835 | `9f8efd6283d095320c339460e0f774b70d1d22c7e1d08dcb0ce9a43c6fe2851f` |
| Quality JSON | 13,068 | `f4b706215beb02f3f5db4574954b49fc2bad06efed1fec55d40bd2ffe223275f` |
| Quality Markdown | 1,390 | `488fce040c6566d7181b1a912faac11a9f1d577ed3a48833b8f750979565e1a2` |

Repeated Step 07 runs produced byte-identical Parquet, manifest, quality JSON,
and generated Markdown.

## 6. Validation and Assurance Evidence

Executed with project-local CPython 3.11.9:

```text
.venv/Scripts/python.exe -m flowcast.cli merge-sources
.venv/Scripts/python.exe -m flowcast.cli engineer-features
.venv/Scripts/python.exe -m pytest -q tests/unit/test_features.py tests/unit/test_package.py
.venv/Scripts/python.exe -m pytest -q tests/data_contracts/test_feature_contract.py
.venv/Scripts/python.exe -m pytest -q
```

Verified results:

- Both CLI commands exited 0; Step 07 reported 181,200 rows/keys, 62 features,
  and 1,200 explicitly marked history-unavailable origins.
- Focused unit/config tests: 7 passed.
- Full-source feature contract tests: 4 passed.
- Full suite: 66 passed in 43.06 seconds.
- Future-row mutation left all earlier model-candidate features unchanged.
- Segment-boundary, shift-before-roll, formula/boundary, manifest completeness,
  input-hash rejection, and deterministic rerun tests passed.
- Largest source module remains `src/flowcast/data/audit.py` at 366 physical
  lines; the new engineering module is 350 lines and every source file remains
  below 400 lines.

## 7. Decisions and Constraints

- Peak periods are half-open 07:00-10:00 and 17:00-20:00 local time.
- Rain means positive rainfall or the controlled `Rain` condition;
  low visibility means strictly below 1,000 metres.
- Temperature bands are left-closed: cool below 15 C, mild from 15 to below
  25 C, and warm from 25 C.
- Capacity features use `road_capacity / 2` for the half-hour denominator.
- Scheduled events are treated as known exogenous calendar inputs. Event
  proximity is the absolute distance to the nearest scheduled event and does
  not read future traffic measurements.
- Rolling standard deviation uses Pandas sample standard deviation (`ddof=1`)
  over a complete shifted window.
- Forecast horizons remain reserved configuration only; target columns and
  availability masks belong exclusively to Step 08.
- No dependency or technology change was required.

## 8. Risks and Unresolved Work

- Step 08 must preserve the common base table and add target-specific
  availability masks rather than globally deleting leading-history or
  trailing-future rows.
- Accident targets must remain unavailable when the future window has
  `_accident_observed = false`; reconstructed windows must not become fake
  negative incidents.
- Exact train/validation/test boundaries remain intentionally unfrozen until
  target coverage is measured.
- Modelling, deep-learning, and dashboard dependency groups remain deferred.

## 9. Next Gate

Proceed only to **Step 08 - Build Multi-Horizon Targets and Processed Data**.
The bounded action and acceptance gate are maintained in `NEXT_STEP.md`.
