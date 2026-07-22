# NEXT_STEP.md

## Immediate Objective

Execute **Step 07 - Engineer Features**. Consume the hash-verified
`merged_sources_v1` table and produce a deterministic, leakage-safe explanatory
feature table plus a machine-readable feature manifest.

Do not construct future targets or split data, run EDA, train models, or begin
dashboard work.

## Read Before Acting

1. `AGENTS.md`, `TECH_STACK.md`, `STATUS.md`, and this file.
2. `STEPS.md` - Step 07.
3. Feature, leakage, timestamp, artifact, and lineage sections of `PROJECT.md`,
   `ROADMAP.md`, and `ARCHITECTURE.md`.
4. Feature requirements in the original PRD and traffic definitions in the
   original data dictionary.
5. The merged Parquet/summary, actual field distributions, current Git diff,
   and relevant tests.

## Single Best Next Action

Build one configuration-backed explanatory-feature slice:

1. Add `config/features.yaml` with a versioned feature contract, lags 1/2/48,
   shifted rolling windows 4/8, forecast horizons reserved for Step 08, peak
   periods, visibility/rain thresholds, and temperature bands.
2. Verify the merged input hash and its 181,200 unique keys before reading it.
3. Add hour/day cyclical encodings, weekday/weekend, and configured peak flags.
4. Compute volume and speed lags within road in strict timestamp order.
5. Shift first, then compute four/eight-window rolling mean and standard
   deviation so the current and future values cannot leak.
6. Add half-hour capacity, V/C ratio, capacity headroom, rain/low-visibility,
   temperature-band, holiday x peak, event, event-proximity, roadwork, and
   existing vehicle-share features.
7. Preserve source/imputation lineage and explicitly mark history-unavailable
   feature rows instead of silently dropping them.
8. Persist versioned feature Parquet, a JSON feature manifest, canonical quality
   JSON, generated Markdown, and a precise CLI command.
9. Add formula, boundary, segment-isolation, shift-before-roll, future-mutation,
   contract, and deterministic-rerun tests.

## Acceptance Gate

Step 07 is complete only when:

- Output retains exactly 181,200 unique `road_id + timestamp` keys.
- Every configured feature has a stable name, dtype, source, transform, version,
  and leakage classification in the manifest.
- Lag values never cross road boundaries and rolling windows exclude the current
  row.
- Mutating any future measurement cannot change an earlier feature row.
- Expected leading history nulls are flagged and exactly accounted for; no row
  is silently discarded.
- Capacity, weather, calendar, and vehicle-share feature formulas pass exact
  boundary tests.
- Input/output/manifest hashes and feature null/range counts are persisted and
  reproducible.
- Focused tests, full tests, CLI smoke, dependency check, compilation, and
  whitespace assurance pass.
- Project-state documents are current and every source file remains below 400
  lines.

## Current Blockers

None.
