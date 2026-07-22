# STATUS.md

## Status Metadata

- **Project:** FlowCast v1.0
- **Last updated:** 2026-07-22
- **Current milestone:** M3 - Features and targets (not started)
- **Current step:** Steps 01-06 complete; Step 07 not started
- **Overall state:** M2 complete; merged source table passed its gate
- **Primary blocker:** None

## 1. Verified Current Position

FlowCast has a reproducible Python 3.11 package with immutable raw preservation,
SHA-256 audit, executable source contracts, reason-preserving quarantine,
trusted calendar/weather/traffic cleaning, and a cardinality-safe merged table.
Milestones M0, M1, and M2 are complete.

The current merged artifact preserves exactly one row per road/half-hour window,
all traffic repair lineage, and prefixed weather/calendar source lineage. Feature
engineering, future targets, EDA, models, inference, reporting services, and the
Streamlit dashboard have not begun.

## 2. Step 06 Implementation

- Added `source_merge_v1`, `merged_sources_v1`, and configuration-backed join
  keys/versioning.
- Added generic artifact-record verification plus a cleaned-input boundary that
  validates the current cleaning configuration, both quality summaries, and all
  three cleaned Parquet hashes before reading data.
- Added a pure source merge with pre-join right-key uniqueness checks, explicit
  Pandas `many_to_one` validation, join indicators, and fail-closed coverage,
  row-count, output-key, alignment, and trusted-context-null checks.
- Added prefixed weather and calendar source/validation/cleaning lineage while
  preserving traffic columns unchanged.
- Added `flowcast merge-sources [--version VERSION]`.
- Added canonical JSON, generated Markdown, and deterministic Parquet artifacts.
- Added unit and full-source tests for hourly broadcasting, date alignment,
  duplicate right keys, missing matches, cardinality, lineage, and reruns.

## 3. Produced Artifacts

```text
data/interim/cleaned_sources_v1/
  calendar.parquet
  weather.parquet
  traffic.parquet

data/interim/merged_sources_v1/
  merged.parquet

artifacts/quality/cleaned_sources_v1/
  summary.json
  summary.md
  traffic_summary.json
  traffic_summary.md

artifacts/quality/merged_sources_v1/
  summary.json
  summary.md
```

Generated Parquet remains ignored by Git. Canonical JSON evidence and generated
Markdown reports are tracked.

## 4. Step 06 Data Evidence

| Check | Verified result |
|---|---:|
| Traffic input rows / unique keys | 181,200 / 181,200 |
| Weather rows / unique station-hours | 10,872 / 10,872 |
| Calendar rows / unique dates | 151 / 151 |
| Merged output rows / unique keys | 181,200 / 181,200 |
| Weather matches / misses | 181,200 / 0 |
| Calendar matches / misses | 181,200 / 0 |
| Row-count change | 0 |
| Duplicate output keys | 0 |
| Trusted joined-context nulls | 0 |

Each traffic timestamp is paired with its floored local `weather_hour`; the
00:00 and 00:30 windows, for example, share the same station/hour observation.
Each row also carries its normalized local `calendar_date`. Traffic `_source_row`,
`_inserted_window`, `_accident_observed`, repaired values, and methods are
unchanged by the merge.

## 5. Traffic Cleaning Baseline Retained

- Exactly 25 roads x 7,248 windows = 181,200 traffic keys.
- All 1,767 duplicate rows and 4,499 inserted windows remain accounted for.
- Trusted volume, speed, occupancy, travel time, and congestion contain no nulls
  or invalid physical values.
- Existing/derived congestion labels remain 150,077 / 31,123 with zero existing
  label disagreements.
- Four vehicle shares remain within 0-1 and sum to one.
- Accident count remains unknown on 4,499 inserted windows; those rows retain
  `_accident_observed = false`.

## 6. Validation and Assurance Evidence

Executed with project-local CPython 3.11.9:

```text
.venv/Scripts/python.exe -m flowcast.cli clean-context
.venv/Scripts/python.exe -m flowcast.cli clean-traffic
.venv/Scripts/python.exe -m flowcast.cli merge-sources
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m compileall -q src
git diff --check
```

Verified results:

- All three CLI commands exited 0; merge reported 181,200 rows/keys and zero
  weather/calendar misses.
- Tests: 57 passed, including full-source deterministic artifact reruns.
- Dependency integrity, byte-compilation, and patch whitespace checks passed.
- Repeated merge runs produced byte-identical Parquet, JSON, and Markdown.
- Largest source module remains `src/flowcast/data/audit.py` at 366 physical
  lines; every source file is below 400 lines.
- Raw and validated source artifacts remain unchanged.

## 7. Decisions and Constraints

- Local Asia/Kolkata timestamps are floored before the hourly weather join; no
  timezone is dropped for weather alignment.
- Calendar joins use a timezone-naive normalized local date because the source
  calendar contract is date-only.
- Right-side key uniqueness is established before each merge and reinforced by
  `validate="many_to_one"`; no deduplication is attempted during merging.
- Cleaning summaries were regenerated after merge configuration was added so
  their recorded configuration hashes remain current.
- No dependency or technology change was required.

## 8. Risks and Unresolved Work

- Imputation and inserted-window flags must remain available to EDA and models
  for sensitivity/error analysis.
- Step 07 must shift before every rolling calculation and prove that future-row
  mutation cannot change earlier features.
- Exact train/validation/test boundaries remain intentionally unfrozen until
  processed coverage and target availability are measured.
- Modelling, deep-learning, and dashboard dependency groups remain deferred.

## 9. Next Gate

Proceed only to **Step 07 - Engineer Features**. The bounded action and
acceptance gate are maintained in `NEXT_STEP.md`.
