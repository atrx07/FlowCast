# STATUS.md

## Status Metadata

- **Project:** FlowCast v1.0
- **Last updated:** 2026-07-24
- **Current milestone:** M7 - Dashboard and service layer (in progress)
- **Current step:** Steps 00-16 complete; Step 17 next
- **Overall state:** Confidence/error evidence is verified and dashboard-ready;
  inference and reporting services are the next gate
- **Primary blocker:** None

## 1. Verified Current Position

FlowCast has a reproducible Python 3.11 pipeline from immutable raw inputs
through validation, cleaning, merge, leakage-safe features, four-horizon
targets, EDA, frozen chronological evaluation, training-only preprocessing,
NumPy regression proof, complete classical regression/classification, a
combined classical registry, a from-scratch recurrent volume forecaster, and
validation-calibrated confidence/error analysis.

Step 16 consumes only the frozen Step 08 and Step 12-15 datasets, summaries,
registries, model cards, and prediction tables. It does not alter a model,
candidate, threshold, calibrator, architecture, sequence length, split, source
prediction, or upstream hash.

`FlowCast-project_file/`, `data/raw/`, the delivered CSV/DOCX sources, and all
frozen Step 10-15 source artifacts remain unchanged.

## 2. Step 16 Implementation

- Added independent `config/confidence.yaml` with the
  `confidence_error_v1` contract, exact upstream versions, validation-only
  split-conformal calibration, fixed reliability bins, confidence/risk bands,
  subgroup dimensions, and minimum-support rules.
- Added modular confidence configuration, verified input loading, artifact
  loading, interval/probability metrics, supported error slices, exact-row
  deep/classical pairing, diagnostics, reporting, and orchestration under
  `flowcast.evaluation`.
- Regression intervals use the finite-sample higher absolute-residual
  quantile at `ceil((n + 1) * 0.90)` for each model/target/horizon validation
  group. Test residuals never fit the width.
- Classification rows preserve frozen ordered probabilities and add maximum
  probability, entropy, normalized entropy, and confidence band.
- Accident rows preserve the validation-selected operating threshold and add
  low/elevated/high/critical bands at 0.5x/1x/2x threshold boundaries. Empty
  bands remain present in the aggregate table.
- Error analysis covers road, origin hour, weekday, weekday/weekend, peak
  period, weather, actual congestion, and horizon. Unsupported slices remain
  visible with counts, `sufficient_support=false`, and blank metrics.
- Added `flowcast analyze-confidence`. Modeling/evaluation command
  registration and dispatch now live in `cli_model_commands.py`, keeping the
  root CLI small without changing existing command behavior.
- Added unit/full-artifact contracts for validation-only fitting, probability
  and risk-band semantics, minimum support, exact pairing, row reconciliation,
  determinism, recursive loading, and tamper rejection.
- Corrected the EDA notebook smoke test to use isolated temporary
  artifacts. With CUDA packages installed, its previous canonical rerun
  changed only the EDA environment snapshot and invalidated frozen downstream
  hashes; the isolated smoke now passes while preserving the canonical EDA
  SHA-256.

## 3. Canonical Confidence and Error Evidence

- Canonical CPU run: 18.35 seconds.
- Regression confidence rows: 862,700.
- Classification confidence rows: 428,257.
- Exact recurrent/classical paired volume rows: 212,000.
- Conformal calibration groups: 16.
- Reliability rows: 160 (ten bins x task/horizon/split).
- Accident risk-band rows: 32, including configured zero-count bands.
- Error slices: 3,408 total; 3,394 meet support and 14 remain visible as
  unsupported.
- RTX 5070 Laptop GPU/VRAM used: no. Step 16 is tabular/statistical and stayed
  on the Intel Core Ultra 9 CPU.
- Intel NPU used: no.

### Regression test intervals

The nominal coverage is 0.90. Across the 16 model/target/horizon test groups:

- Minimum empirical coverage: 0.8924 (recurrent volume, 90 minutes).
- Maximum empirical coverage: 0.9055 (classical travel time, 90 minutes).
- Nine groups are slightly below nominal and seven are at/above nominal.
- Recurrent volume mean interval widths are 185.45, 187.00, 186.37, and
  190.19 vehicles for 30-120 minutes.
- Classical volume mean interval widths are 198.39, 197.19, 209.37, and
  198.78 vehicles.

The observed range is close to nominal, but remains hold-out evidence rather
than a guarantee under future distribution shift.

### Classification calibration and failure modes

- Congestion test Macro-F1 remains 0.7540, 0.7503, 0.7493, and 0.7468.
- Congestion expected calibration error is 0.0029 at 30 minutes and
  0.0515-0.0602 at 60-120 minutes.
- The largest overall off-diagonal congestion confusion is actual Free-flow
  predicted Moderate at 60 minutes (1,263 rows).
- Accident ROC-AUC remains 0.6209, 0.6237, 0.5980, and 0.5894; PR-AUC remains
  0.0209, 0.0182, 0.0161, and 0.0165 against roughly 0.98% prevalence.
- Accident probability expected calibration error is low
  (0.00087-0.00124), but low calibration error under extreme imbalance does
  not compensate for weak ranking and precision.
- Where populated, higher configured accident risk bands show higher observed
  event rates. Very small high/critical groups remain visibly count-qualified.

### Exact deep/classical diagnosis

- The recurrent model retains test RMSE wins at 30, 60, and 90 minutes and
  trails classical by 0.0471 overall at 120 minutes.
- On validation-only paired evidence, recurrent RMSE is lower at all four
  horizons, so a future active-volume policy can be frozen without consulting
  test outcomes.
- The largest supported test deficit is at origin hour 22 for the 120-minute
  horizon: recurrent minus classical RMSE is +10.0931 across 1,100 exact rows.
- Other large deficits cluster around origin hours 23 and 0. These are
  descriptive associations, not causal conclusions.

## 4. Produced Artifacts

```text
config/confidence.yaml

artifacts/metrics/confidence_error_v1/
  accident_risk_bands.csv
  classification_reliability.csv
  confusion_matrices.csv
  error_slices.csv
  interval_calibration.csv
  paired_volume_slices.csv
  regression_coverage.csv
  summary.json
  summary.md

artifacts/predictions/confidence_error_v1/  # ignored, reproducible
  classification_confidence.parquet
  paired_volume_comparison.parquet
  regression_confidence.parquet
```

Key canonical artifacts:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Regression confidence Parquet | 44,636,053 | `73c410c1e1fc9a5313477a9a7104c9bf00abc998b7c50f7fbbe548111f396b06` |
| Classification confidence Parquet | 18,794,358 | `0d665712d7af3da719f7ed9804583dcf69b91e8e5b3941b9f06d899ee90e3693` |
| Paired volume Parquet | 18,114,789 | `8fc6403324e172f83da72382028ddc58a8a858c1949378f36dbbf1642bf739de` |
| Error slices CSV | 792,354 | `5fe1ae82559c019e751fe3807b870df5df17665eb725e939d8bd0b53df756262` |
| Interval calibration CSV | 1,811 | `cfa11e025a46e55868d2419187beeabc4ad4262c345662f8a62add838d983d73` |

## 5. Validation and Assurance Evidence

Executed with project-local CPython 3.11.9:

```text
.venv/Scripts/python.exe -m flowcast.cli analyze-confidence
.venv/Scripts/python.exe -m pytest -q tests/unit/test_confidence_analysis.py tests/data_contracts/test_confidence_analysis_contract.py
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m compileall -q src tests
.venv/Scripts/python.exe -m flowcast.cli analyze-confidence --help
.venv/Scripts/python.exe -m pip check
git diff --check
```

Verified results:

- Canonical Step 16 build: passed in 18.35 seconds.
- Focused Step 16 unit/full-artifact contracts: 10 passed in 19.04 seconds.
- Complete repository suite: 159 passed in 513.90 seconds with pytest exit
  code 0.
- The complete suite retrained bounded classical families in temporary roots,
  verified every prior pipeline layer, reran Step 16 determinism/tamper
  contracts, and executed the isolated EDA notebook without changing its
  canonical summary hash.
- A repeated canonical build reproduces every tracked committed metric/report
  byte exactly.
- Deliberate output-byte tampering is rejected before a confidence table loads.
- All three row-level Parquets reconcile to their frozen prediction sources.
- Every recurrent volume prediction maps to one exact classical row with the
  same actual value.
- All interval bounds are ordered and all classifier uncertainty values are
  finite.
- Dependency consistency, CLI help, source/test byte compilation, whitespace,
  and source-size assurance pass.
- Every source file remains below 400 physical lines.

## 6. Decisions and Constraints

- Confidence configuration is separate from frozen training and registry
  configuration. Reporting changes cannot invalidate model-selection hashes.
- Regression interval widths are target/horizon/model specific, validation
  fitted, and then immutable. There is no test-driven interval widening.
- Recurrent and classical volume use the same external interval method so their
  uncertainty is comparable.
- Risk-band labels are relative rankings around each frozen accident operating
  threshold; they are not calibrated claims of absolute real-world danger.
- Unsupported groups are retained instead of silently dropped or reported with
  unstable metrics.
- The current PyTorch environment remains `2.13.0+cu130` with a verified RTX
  5070 path, but CPU fallback is mandatory. Step 16 correctly stayed on CPU.

## 7. Risks and Unresolved Work

- Congestion Macro-F1 and accident ROC-AUC formal targets remain unmet.
- Low accident prevalence makes subgroup ranking estimates volatile even with
  minimum-positive rules.
- The recurrent model still trails classical at the 120-minute test horizon,
  especially in late-night slices.
- Inference/report services, Streamlit views, upload/retraining controls, and
  final clean reproduction remain.
- Generated model and row-level prediction artifacts are ignored by Git and
  must be rebuilt with documented CLI commands after a clean clone.

## 8. Next Gate

Proceed only to **Step 17 - Build the Inference and Reporting Services**. The
bounded action and evidence gate are maintained in `NEXT_STEP.md`.
