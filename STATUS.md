# STATUS.md

## Status Metadata

- **Project:** FlowCast v1.0
- **Last updated:** 2026-07-22
- **Current milestone:** M2 - Cleaning and merge (in progress)
- **Current step:** Steps 01-04 complete; Step 05 not started
- **Overall state:** Trusted calendar and hourly weather gate passed
- **Primary blocker:** None

## 1. Verified Current Position

FlowCast has a reproducible Python 3.11 package with immutable raw preservation,
SHA-256 audit, executable source contracts, reason-preserving quarantine, and
versioned validation artifacts. Milestone M1 is complete.

Step 04 now adds trusted calendar and hourly weather tables. The pipeline
verifies validated-input hashes, normalizes weather labels, applies causal
station-local imputation, preserves donor-row lineage, and emits canonical JSON
plus generated Markdown quality evidence.

Traffic cleaning, grid reconstruction, source merging, features, EDA, models,
inference, reporting services, and the Streamlit dashboard have not begun.

## 2. Step 04 Implementation

- Added `config/cleaning.yaml` as `context_cleaning_v1` and configured the
  versioned output `cleaned_sources_v1`.
- Added shared deterministic Parquet/JSON artifact helpers.
- Added independent calendar and weather cleaners plus a bounded context
  pipeline and generated quality-report renderer.
- Added `flowcast clean-context [--version VERSION]`.
- Calendar cleaning normalizes dates, enforces unique keys and 0/1 flags,
  validates flag/name relationships, and preserves source lineage.
- Weather cleaning enforces a complete unique hourly grid, maps all delivered
  label variants to `Clear`, `Cloudy`, `Overcast`, `Rain`, or `Fog`, and
  validates trusted numeric fields.
- Temperature and visibility use station-local forward fill for at most two
  consecutive hours. Each output row records original missingness, method, and
  donor source row. Leading/longer gaps fail closed.
- Added unit tests for boundaries, invalid inputs, donor lineage, and future
  mutation; added full-source and byte-deterministic artifact tests.

## 3. Produced Artifacts

```text
data/interim/cleaned_sources_v1/
  calendar.parquet
  weather.parquet

artifacts/quality/cleaned_sources_v1/
  summary.json
  summary.md
```

The Parquet files are reproducible generated data and remain ignored by Git.
The compact quality report and canonical JSON evidence are tracked.

## 4. Step 04 Data Evidence

| Check | Verified result |
|---|---:|
| Calendar rows / unique dates | 151 / 151 |
| Calendar range | 2025-01-01 to 2025-05-31 |
| Holiday / event / roadwork days | 6 / 6 / 11 |
| Weather rows / unique station-hours | 10,872 / 10,872 |
| Stations / rows per station | 3 / 3,624 |
| Controlled weather labels | 5 |
| Temperature values imputed | 167 |
| Visibility values imputed | 111 |
| Maximum observed missing run | 2 hours |
| Remaining trusted weather nulls | 0 |
| Negative rainfall / visibility | 0 / 0 |

Controlled label counts are Clear 8,168; Cloudy 1,844; Fog 76; Overcast 358;
and Rain 426. All 278 fills use an earlier value from the same station and store
the donor `_source_row`.

## 5. Validation and Assurance Evidence

Executed with project-local CPython 3.11.9:

```text
.venv/Scripts/python.exe -m flowcast.cli clean-context
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m compileall -q src
git diff --check
```

Verified results:

- Context CLI: exit code 0; calendar 151 rows, weather 10,872 rows.
- Tests: 42 passed, including deterministic full-source reruns.
- Dependency integrity: no broken requirements.
- Package byte-compilation and patch whitespace checks succeeded.
- Repeated runs produced byte-identical cleaned Parquet, JSON, and Markdown.
- Largest source module remains `src/flowcast/data/audit.py` at 366 physical
  lines; every source file is below 400 lines.
- Raw and validated input hashes remain unchanged.

## 6. Source Validation Baseline

| Dataset | Input | Retained after validation | Quarantined rows | Issues |
|---|---:|---:|---:|---:|
| Calendar | 151 | 151 | 0 | 0 |
| Weather | 10,872 | 10,872 | 0 | 278 |
| Traffic | 178,468 | 176,701 | 1,767 | 42,514 |
| **Total** | **189,491** | **187,724** | **1,767** | **42,792** |

## 7. Decisions and Constraints

- The source documents permit forward/interpolated weather fill. FlowCast uses
  forward-only fill because weather at time `t` must not depend on `t+1`.
- The actual 167 temperature and 111 visibility gaps are all internal runs of
  one or two hours with an earlier same-station observation, so no fallback or
  learned global statistic is needed.
- Known normalization variants pass raw validation but are changed only in the
  versioned cleaned weather artifact; raw and validated artifacts remain intact.
- The cleaning report is generated from `summary.json` and is not edited by
  hand.
- No dependency or technology change was required.

## 8. Risks and Unresolved Work

- Traffic still contains missing/invalid measurements, blank congestion labels,
  and 4,499 absent road/time windows. Step 05 must choose field-specific,
  leakage-safe recovery rules from observed gap structure.
- M2 remains open until traffic is cleaned and all three trusted sources are
  merged without row multiplication.
- Modelling, deep-learning, and dashboard dependency groups remain deferred.

## 9. Next Gate

Proceed only to **Step 05 - Clean Traffic and Reconstruct the Grid**. The bounded
action and acceptance gate are maintained in `NEXT_STEP.md`.
