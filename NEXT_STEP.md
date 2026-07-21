# NEXT_STEP.md

## Immediate Objective

Execute **Step 03 - Define Data Contracts and Quarantine** in the next
implementation turn. The repository baseline is published first as a separately
verified atomic commit.

Do not begin cleaning, merging, feature engineering, EDA, model training, or
dashboard work. The next gate is deterministic source validation in which every
invalid row or recoverable cell has an explicit reason and lineage.

## Read Before Acting

1. `AGENTS.md`
2. `TECH_STACK.md`
3. `STATUS.md`
4. `STEPS.md` - Step 03
5. `PROJECT.md`, `ROADMAP.md`, and `ARCHITECTURE.md` sections governing ingestion,
   validation, data contracts, quarantine, and artifacts
6. `FlowCast-project_file/FlowCast_Data_Dictionary.docx`
7. Relevant PRD acceptance requirements and the actual source CSV edge cases
8. Current Git diff and all tests

## Single Best Next Action

Implement an executable, configuration-backed validation boundary for all three raw
tables:

1. Add typed validation result and issue structures that preserve source filename,
   source row number, field, rejected value, reason code, and disposition.
2. Convert the required columns, exact date/time formats, keys, categories, flags,
   JSON structure, and physical numeric limits into executable checks.
3. Distinguish fatal row-level rejection from recoverable cell-level invalidation.
4. Detect exact/key duplicates before any merge and record every affected row.
5. Write versioned quarantine Parquet plus a machine-readable JSON summary without
   changing `data/raw/`.
6. Add focused boundary tests for every required reason code and deterministic
   ordering/reproducibility.
7. Expose validation through a real CLI command only when it has honest failure and
   exit semantics.

Required reason-code coverage begins with:

- `missing_required_column`
- `invalid_timestamp`
- `duplicate_key`
- `negative_traffic_volume`
- `excessive_speed`
- `invalid_occupancy`
- `invalid_json`
- `invalid_flag`

## Acceptance Gate

Step 03 is complete only when:

- Valid and quarantined outputs account for every input row/cell disposition.
- No invalid input disappears without a reason record.
- Source row lineage survives validation and serialization.
- Duplicate resolution occurs before merge preparation.
- The three raw hashes still match the manifest and original contracts.
- Tests prove each boundary/reason code and deterministic reruns.
- The smallest relevant CLI smoke command succeeds.
- No source-code file exceeds 500 lines.
- `STATUS.md` and `NEXT_STEP.md` are updated with commands, counts, artifacts, and
  unresolved findings.

## Current Blockers

None known. Step 03 is the next approved implementation action.
