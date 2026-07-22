# NEXT_STEP.md

## Immediate Objective

Execute **Step 08 - Build Multi-Horizon Targets and Processed Data**. Consume
the hash-verified `engineered_features_v1` table and construct traceable future
targets for horizons 1-4, equivalent to 30, 60, 90, and 120 minutes.

Do not freeze train/validation/test boundaries, run EDA, train models, or begin
dashboard work.

## Read Before Acting

1. `AGENTS.md`, `TECH_STACK.md`, `STATUS.md`, and this file.
2. `STEPS.md` - Step 08.
3. Target, timestamp, accident-availability, artifact, and processed-data
   sections of `PROJECT.md`, `ROADMAP.md`, and `ARCHITECTURE.md`.
4. Target definitions in the original PRD and accident/congestion definitions
   in the original data dictionary.
5. The feature Parquet, manifest, quality summary, actual target coverage,
   current Git diff, and relevant tests.

## Single Best Next Action

Build one configuration-backed multi-horizon target slice:

1. Add a versioned processed-data/target contract while retaining horizons
   1, 2, 3, and 4 from `config/features.yaml` as the shared authority.
2. Verify the feature configuration, feature manifest, quality summary, and
   feature Parquet hashes before reading data.
3. Within each road in strict timestamp order, add future target timestamps and
   volume, speed, travel-time, congestion, and accident targets for every
   horizon.
4. Define accident risk as future `accident_count > 0` only where the shifted
   `_accident_observed` flag is true; never turn an unobserved reconstructed
   window into a negative label.
5. Add per-target/per-horizon availability masks and keep the common 181,200-row
   base table instead of dropping rows globally.
6. Preserve explanatory features and all source/imputation/history lineage.
7. Persist versioned processed Parquet, a target/schema manifest, canonical
   quality JSON, generated Markdown, and a precise CLI command.
8. Report valid/unavailable counts by target and horizon, including the expected
   trailing rows and accident-specific unobserved windows.
9. Add exact alignment, road-boundary isolation, timestamp, accident-unknown,
   contract, hash-verification, and deterministic-rerun tests.

## Acceptance Gate

Step 08 is complete only when:

- Output retains exactly 181,200 unique `road_id + timestamp` keys.
- For every road and horizon, a row at `t` maps to the exact same-road `t+h`
  target and target timestamp.
- No target crosses a road boundary and no explanatory feature changes.
- Trailing target unavailability is explicitly masked and exactly accounted
  for: at least `25 x h` unavailable origins for each horizon before
  target-specific accident availability is considered.
- Accident labels exist only when the future source window was observed.
- The manifest records every target name, source, horizon/window minutes,
  dtype, availability mask, version, and transform.
- Input/output/manifest hashes and target coverage counts are persisted and
  reproducible.
- Focused tests, full tests, CLI smoke, dependency check, compilation, and
  whitespace assurance pass.
- Project-state documents and README timeline are current and every source file
  remains below 400 lines.

## Current Blockers

None.
