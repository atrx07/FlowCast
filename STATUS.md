# STATUS.md

## Status Metadata

- **Project:** FlowCast v1.0
- **Last updated:** 2026-07-24
- **Current milestone:** M5 - Classical machine learning (complete)
- **Current step:** Steps 00-14 complete; Step 15 next
- **Overall state:** Classical model registry verified; recurrent volume modelling
  is the next gate
- **Primary blocker:** None

## 1. Verified Current Position

FlowCast has a reproducible Python 3.11 pipeline from immutable raw inputs
through validation, cleaning, merge, leakage-safe features, four-horizon
targets, EDA, frozen chronological evaluation, training-only preprocessing,
the NumPy regression proof, complete classical regression/classification, and a
combined classical model registry.

Step 14 consolidates all 20 selected classical target/horizon jobs without
retraining, changing a selection, or loading the sealed source test partition.
The registry recursively verifies the frozen Step 12/13 summaries, selection
manifests, source scoreboards, model/card artifacts, prediction Parquets,
Step 10 feature schema, processed-data lineage, and configuration hashes.

The requested workstation-resource disclosure rule is now part of `AGENTS.md`.
Before hardware-using work, future agents must name the expected CPU, GPU, NPU,
system-RAM, VRAM, and disk usage, including resources that will not be used.

## 2. Step 14 Implementation

- Added independent `config/registry.yaml` with the
  `classical_registry_v1` contract, five required targets, four horizons,
  task-aware primary metrics/directions, acceptance gates, frozen upstream
  versions, indexed prediction mapping, and registry-key template.
- Kept registry configuration separate from `config/models.yaml`. This
  preserves the hashes of the already-frozen Step 10/12/13 training artifacts
  and avoids unnecessary retraining.
- Added deterministic registry construction, normalized scoreboard generation,
  selection rationales, runtime/interpretability context, honest acceptance
  results, and an indexed source-prediction manifest.
- Added recursive verified loading. The loader rejects missing, stale, or
  tampered registry outputs or upstream artifacts before resolving a model
  through the existing regression/classification loaders.
- Added `flowcast build-classical-registry [--version VERSION]`.
- Added unit/full-artifact contracts for configuration coverage, task-aware
  metrics, exact lineage, deterministic bytes, prediction mapping, all-model
  Joblib reloads, public loader resolution, and registry/upstream tampering.

## 3. Registry Coverage and Governance Evidence

- Required targets: volume, speed, travel time, congestion, accident risk.
- Required horizons: 1-4 windows (30, 60, 90, and 120 minutes).
- Registry entries: 20 unique entries from 20 unique jobs.
- Regression entries: 12; classification entries: 8.
- Prediction sources: 2 immutable Parquets.
- Indexed validation/test predictions: 1,078,957 rows.
- Unmapped source prediction rows: 0.
- Registry entries without prediction rows: 0.
- Source selections changed: 0.
- Models retrained in Step 14: 0.
- Direct source test-partition loads in Step 14: 0.
- Test metrics used for selection: no.

Registry keys use:

```text
{target}/h{horizon}/{family}/{model_version}
```

Each entry records model/card/prediction/selection artifacts, hashes,
preprocessing and feature versions, processed-data lineage, seed,
hyperparameters, train/validation/test windows and row counts, validation/test
metrics, class/probability information where relevant, runtime,
interpretability, selection rationale, acceptance state, and limitations.

## 4. Frozen Hold-Out Results

Step 14 preserves the Step 12/13 outcomes exactly:

| Target | Primary metric | Test range across h1-h4 | Formal target | Result |
|---|---|---:|---:|---|
| Volume | RMSE | 62.0092-65.3058 | MAPE <= 12% | Met at 4/4 horizons |
| Speed | RMSE | 3.73998-3.79402 | Not specified | Reported |
| Travel time | RMSE | 1.08217-1.14263 | Not specified | Reported |
| Congestion | Macro-F1 | 0.7468-0.7540 | >= 0.80 | Met at 0/4 horizons |
| Accident risk | ROC-AUC | 0.5894-0.6237 | >= 0.75 | Met at 0/4 horizons |

The missed classifier goals remain explicit model-risk findings. No split,
feature, family, calibrator, threshold, prediction, or score was changed after
the frozen test results became visible.

## 5. Produced Artifacts

```text
artifacts/metrics/classical_registry_v1/
  prediction_index.json
  registry.json
  scoreboard.csv
  summary.json
  summary.md
```

The prediction index references existing versioned prediction Parquets rather
than duplicating roughly 1.08 million real prediction rows.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Registry JSON | 128,915 | `9b80859ba4b20999f7c31645e95474aa72bb6e8e5a082266e64f0cd144ae7138` |
| Combined scoreboard | 17,629 | `0feaf46b8c1cffd1a574af6fe7414c393078180dcf2551befa055acd586018b7` |
| Prediction index | 5,915 | `457383a72d2e6583a9af0911f331cf00ac920a078c75efbc97e47b27bd9a15fe` |
| Generated report | 11,426 | `01a837d0c01bc7707669795c8e590855c53f663d8c9e9b5ea9ffa5ce482b0d9f` |

The independent registry-config SHA-256 is
`dc65cc1a7dc230c0b719ac5189131508d8a7f67f590f6e3eb93cec96ee120d0c`.
The verified regression/classification summary hashes remain
`519b91265fbec38370f0d2821a3cf08cc22058fc845828d5f175f90bf103a382`
and
`61a4d8a1e26e4297068bf2b7826945198c067a8981cbf593b08a1dbac539f6f5`.

## 6. Validation and Assurance Evidence

Executed with project-local CPython 3.11.9:

```text
.venv/Scripts/python.exe -m flowcast.cli build-classical-registry
.venv/Scripts/python.exe -m pytest -q tests/unit/test_classical_registry.py tests/data_contracts/test_classical_registry_contract.py
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m compileall -q src tests
.venv/Scripts/python.exe -m flowcast.cli build-classical-registry --help
git diff --check
```

Verified results:

- Canonical registry build: passed in about 3 seconds.
- Focused Step 14 unit/full-artifact contracts: 9 passed in 3.32 seconds.
- Complete suite: 137 passed in 471.56 seconds.
- The complete suite independently retrained the Step 12 and Step 13 model
  families beneath temporary artifact roots and then rebuilt the registry.
- Registry generation is byte-deterministic across repeated runs.
- All 20 selected model files deserialize; public verified loading resolves
  both regression and probability-classification entries.
- Registry and upstream-summary byte tampering are rejected.
- Dependency consistency, CLI help, byte compilation, whitespace, final
  20-entry artifact verification, and read-only source/reference checks pass.
- Every source file is below 400 lines; the largest is
  `registry_outputs.py` at 395 physical lines.

## 7. Decisions and Constraints

- A separate registry YAML is an artifact-boundary decision, not a new product
  dependency. It prevents reporting-only configuration from invalidating
  frozen model-training lineage.
- The combined scoreboard does not rank unlike tasks against one another.
  Regression remains RMSE-led, congestion remains Macro-F1-led, and accident
  risk remains ROC-AUC-led.
- Runtime and interpretability are stored as operating context only. They do
  not override the already-frozen validation winner.
- Model cards remain the detailed target/horizon governance record; registry
  entries normalize and index them rather than replacing them.
- `FlowCast-project_file/`, `data/raw/`, source data, test boundaries, model
  binaries, source predictions, and dependency versions were not modified.

## 8. Risks and Unresolved Work

- Congestion and accident-risk goals remain unmet. Step 16 must diagnose
  segment/time/weather/prevalence failure modes without concealing the baseline.
- Very low accident prevalence produces weak PR-AUC and operating precision/F1.
- Deep sequence modelling, confidence intervals, inference/report services,
  Streamlit views, upload/retraining controls, and final reproduction remain.
- The Step 15 recurrent model must compare with the exact frozen classical
  volume rows and RMSE values and must not earn a favourable comparison through
  different test coverage.
- Generated model binaries and prediction Parquets remain ignored by Git and
  must be rebuilt with documented commands after a clean clone.

## 9. Next Gate

Proceed only to **Step 15 - Build and Train the Recurrent Model**. The bounded
action and evidence gate are maintained in `NEXT_STEP.md`.
