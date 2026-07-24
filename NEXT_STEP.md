# NEXT_STEP.md

## Immediate Objective

Execute **Step 15 - Build and Train the Recurrent Model**. Build a from-scratch
PyTorch LSTM or GRU that forecasts traffic volume at 30, 60, 90, and 120
minutes, using time-ordered, road-isolated sequences and the same frozen
hold-out contract as the classical volume models.

Do not begin confidence/error analysis, inference services, report export, or
dashboard work. Do not change the classical winners or use the test split for
architecture, sequence-length, epoch, or hyperparameter selection.

## Read Before Acting

1. `AGENTS.md`, `TECH_STACK.md`, `STATUS.md`, and this file.
2. `STEPS.md` - Step 15 plus the proven Steps 08, 10, 12, and 14 procedures.
3. Deep-model, sequence, split, registry, persistence, hardware, and
   reproducibility sections of `PROJECT.md`, `ROADMAP.md`, and
   `ARCHITECTURE.md`.
4. The original PRD requirements for LSTM/GRU training from scratch,
   multi-horizon volume output, curves, early stopping, comparison, and
   persistence.
5. Step 10 feature/split manifests, Step 12 classical volume cards and
   predictions, Step 14 registry, current Git diff, and relevant tests.

## Hardware Notice Before Acting

Before implementation or execution, obey the hardware-resource disclosure rule
in `AGENTS.md`. Identify whether Step 15 will use the Intel Core Ultra 9 CPU,
RTX 5070 Laptop GPU/CUDA, Intel NPU, system RAM, VRAM, and disk. Verify the
installed PyTorch/CUDA capability before promising GPU use. Notify the user
again before the first resource-heavy sequence build, tuning run, or full test
run if usage changes.

## Single Best Next Action

Build and verify the primary recurrent volume forecaster:

1. Add a versioned recurrent-model YAML contract without changing the hashes of
   frozen classical training configuration. Record sequence features/length,
   architecture candidates, seed, optimizer, learning-rate schedule, batch
   size, epoch budget, early-stopping rule, and device policy.
2. Construct fixed-length sequences within each `road_id`, in timestamp order,
   using only origin-time-known features. A sequence may not cross roads, split
   boundaries, time gaps, or a target boundary.
3. Fit any learned feature/target scaling on training rows only and persist the
   scaler, feature order, split/data hashes, and sequence eligibility manifest.
4. Implement a PyTorch Dataset/DataLoader and a one- or two-layer LSTM or GRU
   with dropout and a four-value volume head. Use seeded initialization and no
   pretrained weights.
5. Tune a small predeclared set of sequence/architecture candidates using
   training and validation only. Use Adam, validation-led early stopping,
   learning-rate reduction, and best-checkpoint restoration.
6. Persist candidate evidence, train/validation curves, best epoch,
   checkpoint, configuration, scaler, feature manifest, runtime/device
   metadata, and a complete model card before test access.
7. Load the test split once for final evaluation. Produce per-horizon RMSE,
   MAE, MAPE, and R-squared on rows that map exactly to the corresponding
   frozen classical volume predictions.
8. Generate a head-to-head classical/deep comparison that proves row identity
   per horizon and states honestly whether the recurrent model beats the best
   classical test RMSE at every required horizon.
9. Add a verified checkpoint loader and unit/full-data contracts for road/split
   isolation, gap policy, train-only scaling, shapes, seed reproducibility,
   early stopping, best-weight restoration, exact row mapping, metric
   correctness, reload equality, and tamper rejection.
10. Add a CLI command, generated JSON/CSV/Markdown reports, and only then extend
    the registry with the verified deep-model entry or entries.

## Acceptance Gate

Step 15 is complete only when:

- Sequence windows never cross roads, chronological partitions, target
  boundaries, or disallowed time gaps.
- The recurrent network is built from scratch and outputs all four volume
  horizons with no pretrained weights.
- Architecture selection, early stopping, and checkpoint choice use no test
  information.
- Training/validation curves, best epoch, config, scaler, feature manifest,
  checkpoint, predictions, metrics, and model card are persisted with hashes.
- Reloaded best-checkpoint inference reproduces persisted predictions.
- Each deep test prediction maps to the exact comparable frozen classical
  volume row for its horizon.
- The benchmark reports honestly whether deep test RMSE beats the classical
  baseline; unfavourable results remain visible.
- Focused tests, the full suite, CLI/import smoke, dependency check,
  compilation, whitespace assurance, source-size checks, and artifact
  verification pass.
- Required documentation and README progress are current before commit/push.

## Current Blockers

None. CUDA availability and memory capacity must be measured before selecting a
training device; CPU fallback is allowed by the architecture and technology
contract.
