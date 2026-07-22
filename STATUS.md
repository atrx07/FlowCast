# STATUS.md

## Status Metadata

- **Project:** FlowCast v1.0
- **Last updated:** 2026-07-22
- **Current milestone:** M2 - Cleaning and merge (in progress)
- **Current step:** Steps 01-05 complete; Step 06 not started
- **Overall state:** All three trusted source tables are ready to merge
- **Primary blocker:** None

## 1. Verified Current Position

FlowCast has a reproducible Python 3.11 package with immutable raw preservation,
SHA-256 audit, executable source contracts, reason-preserving quarantine, and
versioned validation artifacts. Milestone M1 is complete.

Steps 04 and 05 now provide trusted calendar, hourly weather, and complete-grid
traffic tables. Every cleaning command verifies its validated-input hashes and
emits canonical JSON plus generated Markdown quality evidence. Source merging,
features, EDA, models, inference, reporting services, and the Streamlit
dashboard have not begun.

## 2. Step 05 Implementation

- Added `traffic_cleaning_v1` policy to `config/cleaning.yaml`, including the
  fixed date grid, expected road count, field-specific causal hierarchy,
  vehicle-share contract, and physical ranges.
- Added shared hash-verified access to validated tables and the issue ledger.
- Added separate traffic grid/contract, causal recovery, artifact pipeline, and
  generated-report responsibilities; every source module remains under 400
  physical lines.
- Added `flowcast clean-traffic [--version VERSION]`.
- Verified static road metadata is constant per `road_id`, then reconstructed
  exactly 25 roads x 7,248 half-hour windows.
- Preserved source JSON and expanded it to `share_2w`, `share_car`, `share_lcv`,
  and `share_hcv`, normalizing only source sums within the documented 0.02
  tolerance.
- Added inserted-window, original-null, physical-invalid, method, donor
  timestamp, single-donor, and complete donor-row lineage fields.
- Derived only missing congestion labels using exact V/C boundaries and audited
  all existing labels without overwriting them.
- Preserved accident count as unknown on inserted windows and added
  `_accident_observed`; no synthetic zero-incident targets were created.

## 3. Produced Artifacts

```text
data/interim/cleaned_sources_v1/
  calendar.parquet
  weather.parquet
  traffic.parquet

artifacts/quality/cleaned_sources_v1/
  summary.json
  summary.md
  traffic_summary.json
  traffic_summary.md
```

The Parquet files are reproducible generated data and remain ignored by Git.
The compact quality reports and canonical JSON evidence are tracked.

## 4. Step 05 Data Evidence

| Check | Verified result |
|---|---:|
| Validated traffic rows | 176,701 |
| Output rows / unique keys | 181,200 / 181,200 |
| Roads / windows per road | 25 / 7,248 |
| Duplicate rows already accounted | 1,767 |
| Explicitly inserted windows | 4,499 |
| Road metadata discrepancies | 0 |
| Maximum trusted-measurement gap | 4 windows |
| Existing / derived congestion labels | 150,077 / 31,123 |
| Existing congestion disagreements | 0 |
| Vehicle-share rows normalized | 60,347 |
| Unknown accident targets on inserted windows | 4,499 |
| Remaining trusted-field nulls | 0 |

Retained validation lineage contains 4,348 original missing and 239 physically
invalid volume values, 4,343 missing and 236 invalid speeds, and 4,344 missing
and 229 invalid occupancies.

## 5. Causal Recovery Evidence

| Field | Total repaired | Same row | Prior day | Causal forward | Station median |
|---|---:|---:|---:|---:|---:|
| Traffic volume | 9,086 | 4,587 | 4,337 | 162 | 0 |
| Vehicle count | 4,499 | 0 | 4,337 | 162 | 0 |
| Average speed | 9,078 | 0 | 8,539 | 536 | 3 |
| Occupancy | 9,072 | 0 | 8,540 | 529 | 3 |
| Travel time | 4,499 | 0 | 4,337 | 162 | 0 |
| Signal timing | 4,499 | 0 | 4,337 | 162 | 0 |
| Vehicle distribution | 4,499 | 0 | 4,337 | 162 | 0 |

All 172,114 retained rows with both observed volume and count agree exactly.
Every retained invalid/missing volume therefore uses its valid same-row count.
Remaining gaps use the prior-day same window, then a same-road forward fill
bounded at four windows. Only three leading speed and three leading occupancy
values require a concurrent same-station median; their contributing source-row
lists are persisted. Future-value mutation tests leave earlier repairs unchanged.

## 6. Validation and Assurance Evidence

Executed with project-local CPython 3.11.9:

```text
.venv/Scripts/python.exe -m flowcast.cli clean-context
.venv/Scripts/python.exe -m flowcast.cli clean-traffic
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m compileall -q src
git diff --check
```

Verified results:

- Context and traffic CLI commands exited 0 and regenerated internally
  consistent artifacts after the shared cleaning configuration changed.
- Tests: 50 passed, including exact congestion boundaries, future mutation,
  full-grid contracts, donor lineage, and byte-deterministic reruns.
- Dependency integrity, byte-compilation, and patch whitespace checks passed.
- Repeated traffic runs produced byte-identical Parquet, JSON, and Markdown.
- Largest source module remains `src/flowcast/data/audit.py` at 366 physical
  lines; every source file is below 400 lines.
- Raw and validated source artifacts remain unchanged.

## 7. Decisions and Constraints

- The original PRD permits interpolation for short gaps, but FlowCast uses only
  past or concurrent observations so a value at `t` cannot depend on `t+1`.
- The observed maximum measurement gap is four windows; the bounded fallback is
  encoded in configuration and fails closed beyond it.
- A same-timestamp station median is permitted only for leading speed and
  occupancy, where no earlier road observation exists. It uses no future data.
- Vehicle share sums range from 0.99 to 1.01 and all pass the dictionary's 0.02
  tolerance before unit-sum normalization.
- Reconstructed accident labels remain unavailable; later target construction
  must exclude them instead of treating missing sensor windows as non-events.
- No dependency or technology change was required.

## 8. Risks and Unresolved Work

- Imputed sensor values must remain identifiable in EDA and modelling; model
  evaluation should include sensitivity/error analysis by imputation state.
- Step 06 must prove both joins are many-to-one and preserve all 181,200 traffic
  keys without multiplication or unexpected misses.
- M2 remains open until the three trusted sources are merged.
- Modelling, deep-learning, and dashboard dependency groups remain deferred.

## 9. Next Gate

Proceed only to **Step 06 - Align and Merge Sources**. The bounded action and
acceptance gate are maintained in `NEXT_STEP.md`.
