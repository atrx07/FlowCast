# NEXT_STEP.md

## Immediate Objective

Execute **Step 09 - Produce Data-Quality Report and EDA**. Build a reproducible
quality/analysis layer from `processed_targets_v1` and the canonical pipeline
counters, then convert observed patterns into explicit preprocessing and
modelling decisions.

Do not train or select models, open the final test period, build inference or
confidence services, or begin dashboard work.

## Read Before Acting

1. `AGENTS.md`, `TECH_STACK.md`, `STATUS.md`, and this file.
2. `STEPS.md` - Step 09.
3. EDA, reporting, split, metrics, artifact, and processed-data sections of
   `PROJECT.md`, `ROADMAP.md`, and `ARCHITECTURE.md`.
4. The relevant requirements in the original PRD and field definitions in the
   original data dictionary.
5. The processed Parquet, target/schema manifest, all quality summaries,
   actual distributions, current Git diff, and relevant tests.

## Single Best Next Action

Build one configuration-backed EDA and data-quality slice:

1. Hash-verify `processed_targets_v1`, its manifest, quality summary, and
   current configurations before analysis.
2. Generate the consolidated data-quality report from persisted pipeline
   counters rather than manually copied estimates.
3. Analyze volume, speed, occupancy, travel time, congestion, and observed
   accident labels overall and by road, hour, weekday, weather, holidays,
   events, and roadworks.
4. Quantify missingness/availability, duplicate resolution, invalid-value
   recovery, imputation, normalization, join coverage, target tails, accident
   imbalance, and congestion imbalance.
5. Inspect numeric correlation/covariance and likely feature redundancy without
   using target-horizon information as an origin feature.
6. Export deterministic, versioned figures and machine-readable analysis
   summaries plus a human-readable report.
7. Add an EDA notebook that delegates calculations to tested package functions
   and runs top-to-bottom against persisted artifacts.
8. Record concrete preprocessing, split, metric, imbalance, and bias/limitation
   implications for Step 10 without training a model.
9. Add calculation, artifact, hash-verification, determinism, and notebook smoke
   tests.

## Acceptance Gate

Step 09 is complete only when:

- The EDA notebook runs top-to-bottom using package functions and no hidden
  notebook-only transformation.
- The consolidated quality report reconciles all source-to-processed row,
  issue, cleaning, merge, feature, and target counters.
- Required target/traffic distributions and contextual slices are persisted
  from real data with denominators and availability stated.
- Correlation/redundancy results are reproducible and exclude identifiers,
  lineage strings, and future target columns from feature recommendations.
- Accident and congestion imbalance are quantified with explicit modelling
  implications and limitations.
- Versioned figures, machine-readable summaries, report artifacts, and hashes
  are reproducible.
- Focused tests, full tests, CLI/notebook smoke, dependency check, compilation,
  and whitespace assurance pass.
- Project-state documents and README timeline are current and every source file
  remains below 400 lines.

## Current Blockers

None.
