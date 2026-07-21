# NEXT_STEP.md

## Immediate Objective

Execute **Step 04 - Clean Calendar and Weather**. Consume the versioned Step 03
validated tables and produce trusted source-level calendar and hourly weather
artifacts with an auditable normalization and imputation summary.

Do not begin traffic cleaning, grid reconstruction, merging, feature
engineering, EDA, modelling, or dashboard work.

## Read Before Acting

1. `AGENTS.md`, `TECH_STACK.md`, `STATUS.md`, and this file.
2. `STEPS.md` - Step 04.
3. Relevant cleaning, contract, artifact, and lineage sections of
   `PROJECT.md`, `ROADMAP.md`, and `ARCHITECTURE.md`.
4. Weather and calendar definitions in
   `FlowCast-project_file/FlowCast_Data_Dictionary.docx` and the related PRD
   requirements.
5. The actual validated Parquet, issue summary, current Git diff, and tests.

## Single Best Next Action

Build one configuration-backed cleaning slice for calendar and weather:

1. Load `data/interim/validated_v1/calendar.parquet` and
   `weather.parquet`, verifying their recorded hashes and contracts.
2. Preserve the 151 unique calendar dates and validated holiday, event, and
   roadwork semantics.
3. Normalize weather whitespace/casing/spelling into exactly `Clear`, `Cloudy`,
   `Overcast`, `Rain`, and `Fog`, while recording source-to-canonical counts.
4. Inspect missing-value gap structure, choose the smallest causal
   station-local imputation policy supported by the source data, and encode all
   thresholds/fallbacks in configuration before applying it.
5. Add temperature and visibility missingness/imputation flags; preserve
   rainfall and verify all physical constraints.
6. Persist versioned cleaned Parquet plus a machine-readable quality summary
   with input/output hashes and transformation counts.
7. Add unit, contract, deterministic-rerun, and CLI tests.

## Acceptance Gate

Step 04 is complete only when:

- Calendar has 151 unique normalized dates with valid flag/name relationships.
- Weather has three stations, 3,624 hourly rows per station, and 10,872 unique
  `station_id + weather_hour` keys.
- No uncontrolled weather labels remain.
- All 167 missing temperatures and 111 missing visibility values are accounted
  for by explicit flags and documented methods; no value is silently filled.
- Rainfall and visibility are non-negative after cleaning.
- Input and output hashes, row counts, normalization counts, and imputation
  counts are persisted and reproducible.
- Focused tests, full tests, CLI smoke, dependency check, compilation, and
  whitespace assurance pass.
- No source file exceeds 500 lines and project-state documents are current.

## Current Blockers

None. The precise weather fallback policy must be selected from observed gap
structure during Step 04, before data mutation.
