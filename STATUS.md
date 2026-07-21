# STATUS.md

## Status Metadata

- **Project:** FlowCast v1.0
- **Last updated:** 2026-07-21
- **Current milestone:** M1 - Ingestion and validation (complete)
- **Current step:** Steps 01-03 complete; Step 04 not started
- **Overall state:** Deterministic raw validation and quarantine gate passed
- **Primary blocker:** None

## 1. Verified Current Position

FlowCast now has an installable Python 3.11 `src/` package, versioned YAML
configuration, immutable raw-data preservation, SHA-256 audit, executable source
contracts, deterministic duplicate resolution, complete issue lineage, and a
versioned validation CLI.

Milestone M1 is complete. No source cleaning, traffic-grid reconstruction,
cross-source merge, feature engineering, EDA, model training, inference,
reporting service, or Streamlit dashboard has begun.

## 2. Step 03 Implementation

- Expanded `config/data_contracts.yaml` into the executable
  `raw_contract_v1` bundle for traffic, weather, and calendar sources.
- Added typed validation issues and results with source filename, physical CSV
  row, field, rejected value, stable reason code, disposition, and retained-row
  identity for duplicates.
- Added exact timestamp/frequency parsing, categorical and ID checks, numeric
  coercion and physical boundaries, vehicle-distribution JSON checks, calendar
  flag/name checks, uniqueness checks, and schema failure handling.
- Added deterministic duplicate retention by greatest post-validation
  completeness, then earliest source row.
- Added cell invalidation for recoverable values and row quarantine for invalid
  structure, keys, timestamps, flags, and non-retained duplicates.
- Added `flowcast validate [--version VERSION]`. Complete schema failure returns
  exit code 2 after persisting evidence; successful source validation returns 0.
- Added versioned Parquet writers and a JSON summary containing row accounting,
  source hashes, artifact hashes, and issue counts.
- Rewrote `README.md` as a concise product overview, current-state explanation,
  and quick-start guide.

## 3. Produced Artifacts

The successful default run produced ignored, reproducible local artifacts:

```text
data/interim/validated_v1/
  calendar.parquet
  weather.parquet
  traffic.parquet

data/quarantine/validated_v1/
  calendar_rejected.parquet
  weather_rejected.parquet
  traffic_rejected.parquet
  issues.parquet
  summary.json
```

The validated tables are a contract boundary, not cleaned/model-ready data.
Weather label normalization and numeric imputation belong to Step 04; traffic
JSON expansion, grid reconstruction, and imputation belong to Step 05.

## 4. Validation Evidence

Executed from the repository root with project-local CPython 3.11.9:

```text
.venv/Scripts/python.exe -m flowcast.cli validate
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m compileall -q src
git diff --check
```

Results:

- CLI validation: exit code 0; no dataset-level schema failure.
- Tests: `26 passed in 10.24s`, including two full deterministic validation runs.
- Dependency integrity: `No broken requirements found.`
- Package byte-compilation: succeeded.
- Patch whitespace check: succeeded.
- Largest source module: `src/flowcast/data/validation.py`, 331 physical lines;
  every
  source file remains below the 500-line limit.
- Repeated validation produced byte-identical validated, rejected, issue, and
  summary artifacts.

## 5. Source and Row Accounting

| Dataset | Input | Retained | Quarantined rows | Issues |
|---|---:|---:|---:|---:|
| Calendar | 151 | 151 | 0 | 0 |
| Weather | 10,872 | 10,872 | 0 | 278 |
| Traffic | 178,468 | 176,701 | 1,767 | 42,514 |
| **Total** | **189,491** | **187,724** | **1,767** | **42,792** |

Traffic reason counts:

- `duplicate_key`: 1,767
- `missing_value`: 40,035
- `negative_traffic_volume`: 241
- `excessive_speed`: 237
- `invalid_occupancy`: 234

Weather has 278 `missing_value` findings: 167 temperature cells and 111
visibility cells. Calendar has no source findings.

All three `data/raw/` SHA-256 values still match the delivered reference files
and `raw_contract_v1`. No raw or reference file was rewritten.

## 6. Decisions and Constraints

- Physical numeric violations are set to nullable values with issue lineage so
  later cleaning can make an explicit recovery decision; they are not silently
  capped or imputed during validation.
- Duplicate rows are the only delivered rows quarantined in Step 03. Findings
  on a row later removed as a duplicate remain in the unified issue ledger.
- Known weather spelling/casing/whitespace variants pass vocabulary validation
  but remain unchanged until Step 04.
- The configured vehicle-share tolerance is an absolute 0.02, covering the
  delivered rounding range of 0.99-1.01 without normalizing the raw JSON.
- Empty rejected-row Parquet files are intentionally written for calendar and
  weather to make row accounting and downstream automation uniform.
- Generated validation data remains outside Git; configuration, code, tests,
  and reproducibility evidence are tracked.

## 7. Risks and Unresolved Work

- Step 04 must define and test a causal, station-local weather imputation policy
  before it mutates any validated values.
- Traffic still contains missing and invalid measurements, blank congestion
  labels, and 4,499 absent road/time windows; these belong to Step 05.
- Deferred modelling, deep-learning, and dashboard dependencies remain
  uninstalled until their milestones.

## 8. Next Gate

Proceed only to **Step 04 - Clean Calendar and Weather**. The bounded action and
acceptance gate are maintained in `NEXT_STEP.md`.
