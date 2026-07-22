# NEXT_STEP.md

## Immediate Objective

Execute **Step 05 - Clean Traffic and Reconstruct the Grid**. Consume the
verified `validated_v1` traffic table and produce one trusted row for every
Northline road/30-minute window, with explicit source, missing-window, invalid,
and imputation lineage.

Do not merge calendar/weather, engineer modelling features or targets, run EDA,
train models, or begin dashboard work.

## Read Before Acting

1. `AGENTS.md`, `TECH_STACK.md`, `STATUS.md`, and this file.
2. `STEPS.md` - Step 05.
3. Traffic cleaning, congestion, leakage, artifact, and lineage sections of
   `PROJECT.md`, `ROADMAP.md`, and `ARCHITECTURE.md`.
4. Traffic definitions and quality rules in both original DOCX references.
5. `data/interim/validated_v1/traffic.parquet`, its issue ledger/summary, the
   actual gap patterns, current Git diff, and relevant tests.

## Single Best Next Action

Build one configuration-backed traffic-cleaning slice:

1. Verify the validated traffic artifact hash and the 176,701 unique retained
   road/timestamp keys.
2. Audit road metadata consistency, numeric gap-run lengths, zero/non-positive
   speeds, and whether `vehicle_count` safely recovers each invalid volume.
3. Parse the vehicle-distribution JSON into `share_2w`, `share_car`,
   `share_lcv`, and `share_hcv`, preserving the original JSON and validation
   lineage.
4. Reindex all 25 roads to the complete 2025-01-01 through 2025-05-31
   half-hour grid of 181,200 rows; flag all 4,499 inserted windows.
5. Encode field-specific causal recovery limits and fail-closed fallbacks in
   `config/cleaning.yaml` before applying any fill.
6. Add original-null, physical-invalid, inserted-window, imputation-method, and
   donor lineage fields for every repaired measurement.
7. Derive only blank congestion labels from exact half-hour V/C boundaries and
   report disagreement with existing labels without silently overwriting them.
8. Persist versioned traffic Parquet plus canonical JSON and generated Markdown
   quality evidence; expose a precise CLI command.
9. Add boundary, leakage, full-grid, contract, and deterministic-rerun tests.

## Acceptance Gate

Step 05 is complete only when:

- Exactly 25 roads and 181,200 unique `road_id + timestamp` keys exist.
- All 1,767 duplicate rows remain accounted for by Step 03 lineage and all
  4,499 missing windows are explicitly represented.
- Road metadata is internally consistent or every discrepancy is documented.
- Vehicle shares contain the four required classes, remain within 0-1, and sum
  to one within the documented normalization policy.
- Trusted volume, speed, occupancy, travel-time, and congestion fields meet
  their contracts with every repair traceable to a method and source state.
- Congestion tests prove exact behaviour at V/C 0.50, 0.80, and 1.00.
- Future mutation cannot change an earlier causal fill.
- Input/output hashes, row counts, recovery counts, and remaining failures are
  persisted and reproducible.
- Focused tests, full tests, CLI smoke, dependency check, compilation, and
  whitespace assurance pass.
- Project-state documents are current and no source file reaches 400 lines
  without a responsibility-based split.

## Current Blockers

None. Field-specific traffic recovery limits must be selected from measured
gap structure before implementation mutates validated values.
