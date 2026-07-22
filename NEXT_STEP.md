# NEXT_STEP.md

## Immediate Objective

Execute **Step 06 - Align and Merge Sources**. Join the three hash-verified
`cleaned_sources_v1` tables into one versioned interim table while preserving
exactly one row per traffic road/half-hour key.

Do not engineer modelling features or targets, run EDA, train models, or begin
dashboard work.

## Read Before Acting

1. `AGENTS.md`, `TECH_STACK.md`, `STATUS.md`, and this file.
2. `STEPS.md` - Step 06.
3. Join, timestamp, lineage, artifact, and cardinality sections of `PROJECT.md`,
   `ROADMAP.md`, and `ARCHITECTURE.md`.
4. Join definitions in both original DOCX references.
5. All three cleaned Parquet tables and their quality summaries, current Git
   diff, and relevant tests.

## Single Best Next Action

Build one configuration-backed, cardinality-safe source-merging slice:

1. Verify hashes for cleaned traffic, weather, and calendar artifacts against
   their current quality summaries before reading them.
2. Add `weather_hour = timestamp.floor("h")` to traffic and normalize a calendar
   join date without changing the traffic key.
3. Assert uniqueness of `station_id + weather_hour` and calendar `date` before
   either merge.
4. Join weather through `weather_station_id -> station_id` plus aligned hour
   using explicit many-to-one validation.
5. Join calendar on normalized date using explicit many-to-one validation.
6. Persist join indicators, matched/missing counts, source versions/hashes, and
   retained lineage in canonical JSON plus generated Markdown.
7. Write one versioned merged Parquet artifact and expose a precise CLI command.
8. Add boundary/alignment, cardinality, no-row-multiplication, contract, and
   deterministic-rerun tests.

## Acceptance Gate

Step 06 is complete only when:

- Input hashes match the tracked Step 04/05 summaries.
- Output has exactly 181,200 rows and 181,200 unique `road_id + timestamp` keys.
- Both right-side keys are unique before merging and Pandas validates each join
  as many-to-one.
- Every traffic row has exactly one weather match and one calendar match, or an
  unexpected miss fails closed with persisted diagnostics.
- Weather alignment broadcasts each hourly observation only to its two
  corresponding half-hour windows.
- Traffic repair/source lineage and `_accident_observed` survive unchanged.
- Input/output hashes, join counts, null counts, and remaining failures are
  persisted and reproducible.
- Focused tests, full tests, CLI smoke, dependency check, compilation, and
  whitespace assurance pass.
- Project-state documents are current and every source file remains below 400
  lines.

## Current Blockers

None.
