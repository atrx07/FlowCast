# STATUS.md

## Status Metadata

- **Project:** FlowCast v1.0
- **Last updated:** 2026-07-22
- **Current milestone:** M3 - Features and targets (complete)
- **Current step:** Steps 01-08 complete; Step 09 next
- **Overall state:** Analysis-ready processed-data gate passed
- **Primary blocker:** None

## 1. Verified Current Position

FlowCast now has a reproducible Python 3.11 data pipeline from immutable raw
inputs through validation, cleaning, cardinality-safe merging, leakage-safe
features, and exact multi-horizon future targets. Milestones M0 through M3 are
complete.

The versioned processed table retains all 181,200 road/half-hour origins and
all 144 input feature/source/lineage columns. It appends four exact future
timestamps, 20 target columns, and 20 target-specific availability masks for a
total of 188 columns. EDA, modelling, inference, confidence, reporting
services, and the Streamlit dashboard have not begun.

## 2. Step 08 Implementation

- Added the versioned `multi_horizon_targets_v1` contract and
  `processed_targets_v1` dataset version.
- Added hash-verified Step 07 input loading that validates current base and
  feature configurations, quality summary, feature manifest, Parquet byte
  count/SHA-256, row/key cardinality, and manifest dtypes before reading.
- Added same-road future timestamps and volume, speed, travel-time, congestion,
  and accident targets for 30, 60, 90, and 120 minutes.
- Added one availability mask per target and horizon while retaining every
  prediction origin.
- Defined accident risk as shifted future `accident_count > 0` only when the
  shifted `_accident_observed` flag is true. Reconstructed unobserved windows
  remain null and are never converted to negative labels.
- Preserved all explanatory features and lineage exactly; the processed runner
  fails if target creation changes an input column.
- Added a complete column/target schema manifest, coverage summary, generated
  Markdown report, and `flowcast prepare-data [--version VERSION]`.
- Added focused and full-source tests for exact timestamps/shifts, road
  isolation, availability, accident unknowns, manifest coverage, determinism,
  and hash-tamper rejection.

## 3. Produced Artifacts

```text
data/processed/processed_targets_v1/
  dataset.parquet

artifacts/features/processed_targets_v1/
  manifest.json

artifacts/quality/processed_targets_v1/
  summary.json
  summary.md
```

Generated Parquet remains ignored by Git. The target/schema manifest,
canonical quality JSON, and generated Markdown report are tracked.

## 4. Step 08 Data Evidence

| Check | Verified result |
|---|---:|
| Input rows / keys | 181,200 / 181,200 |
| Output rows / keys | 181,200 / 181,200 |
| Row-count change / duplicate keys | 0 / 0 |
| Roads | 25 |
| Preserved input columns | 144 |
| Model-candidate features | 62 |
| Future timestamp columns | 4 |
| Target definitions / masks | 20 / 20 |
| Total processed columns | 188 |

| Horizon | Minutes | Standard targets available | Tail unavailable | Accident available | Accident unavailable | Accident positive |
|---:|---:|---:|---:|---:|---:|---:|
| h1 | 30 | 181,175 | 25 | 176,676 | 4,524 | 1,652 |
| h2 | 60 | 181,150 | 50 | 176,651 | 4,549 | 1,652 |
| h3 | 90 | 181,125 | 75 | 176,628 | 4,572 | 1,652 |
| h4 | 120 | 181,100 | 100 | 176,604 | 4,596 | 1,652 |

Volume, speed, travel time, and congestion have only the exact 25 x horizon
corridor-tail unavailable rows. Accident availability additionally respects
the 4,499 source windows whose incident status is unknown; overlap with the
road tails is accounted per horizon.

## 5. Artifact Evidence

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Processed Parquet | 21,813,624 | `f5377b7f8969d6b74e850d71a803c91f252ec236d5bceeaa02e3e31dedfa81a4` |
| Target/schema manifest | 31,747 | `d5651a8a4354f3f352dbda20009605299e5df9ce8433341da4197540e7493eb8` |
| Quality JSON | 6,140 | `e5588965ce8e1ac667004e87175a34e2f330fd05c62cae463241ee8a8c6c32df` |
| Quality Markdown | 2,017 | `3619126de26a06983160a2898c3384f59b364642d3e0839b85958dae1e613a9f` |

Repeated Step 08 runs produced byte-identical Parquet, manifest, quality JSON,
and generated Markdown. The merge and Step 07 lineage summaries were refreshed
after the version contracts changed; Step 07 feature Parquet content and hash
remain unchanged.

## 6. Validation and Assurance Evidence

Executed with project-local CPython 3.11.9:

```text
.venv/Scripts/python.exe -m flowcast.cli merge-sources
.venv/Scripts/python.exe -m flowcast.cli engineer-features
.venv/Scripts/python.exe -m flowcast.cli prepare-data
.venv/Scripts/python.exe -m pytest -q tests/unit/test_targets.py tests/unit/test_features.py tests/unit/test_package.py
.venv/Scripts/python.exe -m pytest -q tests/data_contracts/test_processed_contract.py
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m compileall -q src tests
.venv/Scripts/python.exe -m flowcast.cli prepare-data --help
git diff --check
```

Verified results:

- All three CLI commands exited 0; `prepare-data` reported 181,200 rows/keys
  and 20 target definitions.
- Focused target/feature/config tests: 12 passed.
- Full-source processed-data contract tests: 6 passed.
- Full suite: 77 passed in 46.99 seconds.
- Dependency check, byte compilation, CLI help smoke, and whitespace checks
  passed.
- Exact same-road target timestamp/value alignment passed for all horizons.
- Input-column equality, manifest completeness, deterministic rerun, and
  pre-read hash rejection checks passed.
- Largest source module remains `src/flowcast/data/audit.py` at 366 physical
  lines; every source file remains below 400 lines.

## 7. Decisions and Constraints

- A common 181,200-row base is retained; model-specific filtering must use the
  applicable target/horizon mask.
- `target_timestamp_h1` through `target_timestamp_h4` are shared across target
  families and advance by exactly 30 minutes per horizon within a road.
- Classical model targets use normalized names such as `target_volume_h1` and
  the corresponding `target_volume_h1_available` mask.
- Accident targets are nullable booleans; null means label unavailable, not no
  accident.
- Exact split boundaries remain intentionally unfrozen until Step 09 findings
  inform the modelling plan.
- No dependency or technology change was required.

## 8. Risks and Unresolved Work

- Accident positives are rare: 1,652 available positives at every horizon.
  Step 09 must characterize imbalance and Step 10 must use training-only class
  weighting and validation-based threshold selection.
- Congestion is imbalanced, with Free-flow dominant; EDA must quantify this by
  road and time before preprocessing/model choices are frozen.
- The processed Parquet is reproducible but ignored by Git, so clean rebuild
  evidence remains required before delivery.
- Modelling, deep learning, confidence, and dashboard dependency groups remain
  deferred.

## 9. Next Gate

Proceed only to **Step 09 - Produce Data-Quality Report and EDA**. The bounded
action and acceptance gate are maintained in `NEXT_STEP.md`.
