# STATUS.md

## Status Metadata

- **Project:** FlowCast v1.0
- **Last updated:** 2026-07-24
- **Current milestone:** M6 - Deep learning and confidence (in progress)
- **Current step:** Steps 00-15 complete; Step 16 next
- **Overall state:** Recurrent multi-horizon volume model verified; confidence
  and error analysis is the next gate
- **Primary blocker:** None

## 1. Verified Current Position

FlowCast has a reproducible Python 3.11 pipeline from immutable raw inputs
through validation, cleaning, merge, leakage-safe features, four-horizon
targets, EDA, frozen chronological evaluation, training-only preprocessing,
NumPy regression proof, complete classical regression/classification, a
combined classical registry, and a from-scratch recurrent volume forecaster.

Step 15 consumes the exact frozen Step 10 recurrent preprocessing contract and
does not change the Step 10-14 configuration hashes, model choices, test
boundaries, or persisted classical predictions. It persists selection and the
best state-dictionary checkpoint before the single explicit test load.

`FlowCast-project_file/`, `data/raw/`, the delivered CSV/DOCX sources, and all
frozen classical source artifacts remain unchanged.

## 2. Step 15 Implementation

- Added independent `config/recurrent.yaml` with the
  `recurrent_volume_v1` contract, two bounded LSTM candidates, target columns
  h1-h4, sequence/cadence isolation, Adam, training-only per-horizon target
  scaling, learning-rate reduction, early stopping, device policy, validation
  selection, and exact-row classical comparison.
- Added lazy PyTorch datasets over transformed partitions; sequences cannot
  cross roads, chronological partitions, 30-minute gaps, or target boundaries.
- Restricted candidate validation to the exact origins eligible for the
  longest configured sequence so sequence length cannot win through different
  validation coverage.
- Added a unidirectional LSTM/GRU implementation with dropout and a four-value
  volume head. All weights are seeded and initialized from scratch.
- Added candidate training, gradient clipping, scheduler, early stopping,
  best-state restoration, deterministic inference, and CPU/CUDA policy
  resolution.
- Post-Step-15 CUDA hardening adds GPU name, total VRAM, peak allocated VRAM,
  CUDA/PyTorch version, selected policy, and explicit CPU-fallback evidence to
  future recurrent run summaries. Unit coverage now protects explicit CPU,
  unavailable-CUDA auto fallback, and guarded CUDA failure.
- Persisted candidate metrics, epoch curves, feature/scaler manifests,
  pre-test sequence/card/selection evidence, state dictionary, validation/test
  predictions, metrics, exact-row comparison, JSON/Markdown model card,
  environment snapshot, report, and a four-entry recurrent registry extension.
- Added `flowcast train-recurrent-volume`.
- Added unit/full-artifact contracts for config coverage, sequence isolation,
  target boundaries, training-only scaling, shapes/seeding, exact comparison,
  freeze order, metric/artifact coverage, reload equality, and tamper
  rejection.

## 3. Sequence, Training, and Resource Evidence

- Selected candidate: `lstm_s12_h32`.
- Recurrent type: unidirectional LSTM; sequence length: 12 windows.
- Transformed inputs: 64 from the frozen 62-feature origin schema.
- Best epoch: 8; stopped epoch: 11 through validation-led early stopping.
- Training sequences: 126,475 across 25 roads.
- Validation sequences: 26,500 across 25 roads.
- Test sequences: 26,500 across 25 roads.
- Cross-road sequences: 0.
- Cross-partition sequences: 0.
- Non-contiguous sequences: 0.
- Target-boundary violations: 0.
- Candidate fit time: 69.19 seconds; total canonical run: 76.72 seconds.
- Frozen Step 15 training framework: PyTorch `2.13.0+cpu`.
- Frozen Step 15 execution device: Intel Core Ultra 9 CPU, 8 PyTorch threads.
- RTX 5070 Laptop GPU/VRAM used for the frozen Step 15 run: no; CUDA was
  unavailable in that recorded environment.
- Intel NPU used: no.

Post-Step-15 environment update, verified 2026-07-24:

- Installed framework: PyTorch `2.13.0+cu130` from the official PyTorch CUDA
  13.0 wheel index.
- NVIDIA driver `610.74` exposes CUDA UMD compatibility 13.3.
- `torch.cuda.is_available()` is true for the NVIDIA GeForce RTX 5070 Laptop
  GPU with compute capability 12.0 and 8,151 MiB VRAM.
- CUDA LSTM forward/backward smoke: passed with output shape `[32, 4]`, cuDNN
  9.2, 64.19 MiB allocated VRAM, and 0.213 seconds measured kernel/test time.
- Forced-CPU LSTM forward/backward smoke: passed with finite output shape
  `[8, 4]`.
- The existing Step 15 checkpoint, predictions, and hold-out metrics were not
  retrained or rewritten; their original CPU environment evidence remains
  immutable.

## 4. Frozen Hold-Out Results

The recurrent model was evaluated once after candidate/checkpoint freeze.
Each comparison uses the exact same 26,500 road/timestamp origins and actual
values as the corresponding frozen classical prediction.

| Horizon | Deep RMSE | Classical RMSE | Delta deep-classical | Deep wins | Deep MAPE |
|---:|---:|---:|---:|---|---:|
| 30 min | 60.1443 | 63.2354 | -3.0910 | Yes | 10.210% |
| 60 min | 60.8154 | 62.6833 | -1.8678 | Yes | 10.388% |
| 90 min | 61.2014 | 65.0565 | -3.8551 | Yes | 10.979% |
| 120 min | 61.8966 | 61.8495 | +0.0471 | No | 11.535% |

- Mean deep test RMSE: 61.0144.
- The volume MAPE target remains met at all four horizons.
- Deep RMSE beats classical at 3 of 4 horizons.
- The formal all-horizon deep-vs-classical target is therefore not met.
- No post-test architecture, sequence-length, epoch, or prediction change was
  made to chase the 120-minute result.

## 5. Produced Artifacts

```text
artifacts/metrics/recurrent_volume_v1/
  candidate_metrics.csv
  classical_comparison.csv
  environment.txt
  horizon_metrics.csv
  pretest_model_card.json
  pretest_sequence_manifest.json
  registry_extension.json
  selection_manifest.json
  sequence_manifest.json
  summary.json
  summary.md
  training_curves.csv

artifacts/model_cards/recurrent_volume_v1/
  volume_multi_horizon.json
  volume_multi_horizon.md

artifacts/models/recurrent_volume_v1/       # ignored, reproducible
  best_checkpoint.pt
  feature_manifest.json
  target_scaler.json

artifacts/predictions/recurrent_volume_v1/  # ignored, reproducible
  predictions.parquet
```

Key canonical artifacts:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Checkpoint | 64,125 | `c23ed0580af9ea39a68dfda60b79011e2d32f4564f9303235655fe7bdb5b90dd` |
| Predictions | 4,414,323 | `ba38dcf66bc793e3a5a2abcbbc98cfe9ce01432afc91625e9fd906e5d9917e82` |
| Model-card JSON | 17,681 | `8022c70873e072167faccc48da70da16b1c7f55bd7df81fb57af5d614ff84f4c` |
| Selection manifest | 1,758 | `63c8a615bd73312942ac2569f64593dfbc86f883943e131b500a77dfc9240bfa` |
| Comparison CSV | 484 | `aec0dd2feb36566cf12b51db4bd82b5f148501d07f2e832fab60be8315f35c20` |

## 6. Validation and Assurance Evidence

Executed with project-local CPython 3.11.9:

```text
.venv/Scripts/python.exe -m pip install "torch==2.13.0"
.venv/Scripts/python.exe -m flowcast.cli train-recurrent-volume
.venv/Scripts/python.exe -m pytest -q tests/unit/test_recurrent_volume.py tests/data_contracts/test_recurrent_volume_contract.py
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m compileall -q src tests
.venv/Scripts/python.exe -m flowcast.cli train-recurrent-volume --help
.venv/Scripts/python.exe -m pip check
git diff --check
```

Post-Step-15 CUDA enablement:

```text
.venv/Scripts/python.exe -m pip install --force-reinstall "torch==2.13.0+cu130" --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/python.exe -c "<CUDA LSTM forward/backward smoke>"
.venv/Scripts/python.exe -c "<forced-CPU LSTM forward/backward smoke>"
.venv/Scripts/python.exe -m pip check
nvidia-smi
```

CUDA enablement verification results:

- Optional `requirements-cuda.txt` reinstall path resolves the already installed
  official `torch==2.13.0+cu130` build.
- Focused recurrent unit/full-artifact contracts after CUDA enablement:
  12 passed.
- Existing recurrent checkpoint loading and persisted-prediction equality pass
  under the CUDA-capable environment without retraining.
- `pip check`, source/test byte compilation, `git diff --check`, and the
  under-400-line source-size assurance pass.

Verified results:

- Canonical Step 15 build: passed in 76.72 seconds.
- Focused Step 15 unit/full-artifact contracts: 10 passed in 6.07 seconds.
- Complete suite: 147 passed in 464.79 seconds.
- The complete suite independently retrained the frozen classical families in
  temporary artifact roots, verified every prior pipeline layer, and then
  validated the canonical recurrent chain.
- Checkpoint reconstruction reproduces all persisted validation predictions.
- Deliberate checkpoint byte tampering is rejected before deserialization.
- All 212,000 long-form validation/test prediction rows are finite and
  horizon-traceable.
- Exact classical origin, target timestamp, and actual-value mapping passes at
  all horizons.
- Dependency consistency, CLI help, byte compilation, and whitespace checks
  pass.
- Every source file remains below 400 physical lines.

## 7. Decisions and Constraints

- The recurrent YAML and registry extension are separate from the classical
  registry contract. This prevents deep-model reporting from invalidating
  frozen Step 10-14 hashes or rewriting the classical-only registry.
- Validation mean RMSE selects the candidate and epoch. Test RMSE is reporting
  evidence only.
- The state dictionary is the primary PyTorch artifact; no opaque whole-model
  object or pretrained weights are persisted.
- Validation/test comparisons use the longest-candidate eligible-origin
  intersection. This gives every candidate the same selection origins and
  gives deep/classical models identical comparison rows.
- The CPU-only PyTorch build satisfies the approved CPU acceptance baseline.
  The current local environment now adds the approved CUDA wheel for
  workload-aware acceleration, while CPU execution remains mandatory and was
  reverified after installation.
- Full recurrent/deep candidate training should normally use the verified RTX
  5070 when accelerator overhead is justified. Unit tests, small training
  probes, tabular confidence/error analysis, and lightweight inference should
  normally stay on CPU.
- The portable dependency remains `torch==2.13.0`; the optional local NVIDIA
  distribution is pinned separately as `torch==2.13.0+cu130` in
  `requirements-cuda.txt`.

## 8. Risks and Unresolved Work

- The recurrent model trails the classical model by 0.0471 RMSE at 120 minutes,
  so the formal all-horizon deep comparison goal is missed.
- Congestion Macro-F1 and accident ROC-AUC targets remain unmet.
- Step 16 must diagnose the 120-minute, congestion, and accident failure modes
  without refitting on or concealing the sealed-test results.
- Regression intervals, unified confidence tables, inference/report services,
  Streamlit views, upload/retraining controls, and final reproduction remain.
- Generated checkpoint/scaler/prediction artifacts are ignored by Git and must
  be rebuilt with the documented CLI after a clean clone.

## 9. Next Gate

Proceed only to **Step 16 - Add Confidence and Error Analysis**. The bounded
action and evidence gate are maintained in `NEXT_STEP.md`.
