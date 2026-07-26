# FlowCast v1.0 Final Technical Report

## Executive Summary

FlowCast v1.0 is a reproducible Streamlit traffic-intelligence application for
the 25-segment Northline Corridor. It rebuilds immutable traffic, weather, and
calendar inputs into a leakage-safe, four-horizon forecasting system covering
traffic volume, average speed, congestion class, travel time, accident risk,
and prediction confidence.

The final acceptance run completed all 16 pipeline stages from the delivered
source files in 520.287 seconds on CPU. A permanent verifier reconciled every
stage record, source hash, selected model, frozen metric, prediction batch, and
report with a maximum numeric difference of
`1.0842021724855044e-17`, below the `1e-12` tolerance. The complete isolated
test suite passed 192 tests. The ten-route dashboard then passed navigation,
prediction, upload-validation, report-export, lineage, and retraining-safety
acceptance against the reproduced artifacts.

The volume target is met at all four horizons. Congestion classification,
accident ranking, and the requirement that the recurrent model beat the
classical volume model at every horizon are not fully met; they remain visible
limitations rather than being hidden by post-test tuning.

## Reproduction Contract

The canonical delivery path is:

```powershell
python -m flowcast.cli run-all `
  --output-root artifacts/reproductions/flowcast_v1_final_cpu `
  --recurrent-device cpu
python -m flowcast.cli verify-reproduction `
  --output-root artifacts/reproductions/flowcast_v1_final_cpu
```

`run-all` requires an empty child directory beneath `artifacts/reproductions`.
It redirects every writable raw copy, interim table, processed table,
quarantine record, model, metric, report, prediction, and log beneath that
root. `FlowCast-project_file/` remains the read-only source of truth.

The verified run used:

- CPython 3.11.9.
- NumPy 2.3.3, pandas 3.0.3, PyArrow 24.0.0.
- scikit-learn 1.9.0, XGBoost 3.2.0.
- PyTorch 2.13.0 CPU build.
- Streamlit 1.59.2.
- Run ID `flowcast_v1_final_cpu`.
- Manifest SHA-256
  `37203a16e6859dffeef9cad543edd32d62b9de80dd0b1c3e5d142c2cf2e26e7d`.

The three delivered CSV hashes were unchanged:

| Source | SHA-256 |
|---|---|
| `traffic_sensor_log.csv` | `8f793f3643c891d4fdda7b66c5c4792d24f4db3f26a07cccb8f1d613e254062a` |
| `weather_observations.csv` | `63f3dc54a491dfd5d4663d8bf0602779084c30a1396f0c7b4fd177e132bc8a31` |
| `calendar_events.csv` | `60d3de6b731486e02f6edaa3515af87c2472231a211b48d94d7a3cad38799b9c` |

## Data Quality and Preparation

The traffic source contains 178,468 rows. Validation identified 1,767 exact
and key duplicates, leaving 176,701 unique road/timestamp observations.
Cleaning reconstructed the complete 25-road, 151-day, half-hour grid of
181,200 rows by inserting 4,499 missing windows and retaining explicit
observation and repair lineage.

Key quality findings:

- 241 raw rows had physically invalid negative traffic volume.
- 26,883 raw congestion labels were blank.
- 1,669 raw rows were accident-positive, a 0.9352% positive rate.
- 60,347 vehicle-distribution rows required normalization.
- 31,123 congestion labels were derived from the documented volume/capacity
  rule, with no disagreement against existing valid labels.
- Inserted accident windows remain unknown rather than being relabelled as
  non-accidents.
- Weather covers all 10,872 expected station-hours. Causal, station-local
  forward fill repaired 167 temperature and 111 visibility values.
- Calendar covers all 151 dates, including six public holidays, six event
  days, and eleven roadwork days.
- Weather and calendar joins are validated many-to-one joins with 181,200
  matches, zero misses, and zero row multiplication.

Feature engineering preserves strict road-local timestamp order. Lagged and
rolling values are shifted before aggregation so current and future targets
cannot leak into explanatory inputs. The processed table retains 181,200
unique origins, 62 model-candidate features, and 20 target definitions: five
outputs at 30, 60, 90, and 120 minutes. Trailing and unobserved targets remain
null with explicit availability masks.

## Evaluation and Mathematics

Evaluation uses frozen chronological train, validation, and test periods with
five expanding, horizon-gapped cross-validation folds inside the training
period. Imputers, encoders, and scalers learn only from training data. Model
selection and calibration decisions are persisted before the test partition is
loaded once for final evaluation.

The required NumPy linear regression implements:

`prediction = Xw + b`

and minimizes mean squared error with an explicit gradient-descent loop:

`dw = (2/n) X^T (prediction - y)` and
`db = (2/n) sum(prediction - y)`.

Central finite-difference checks validate the analytical gradients. On the
real validation comparison, the scratch implementation converged after 4,153
updates with RMSE 86.93, compared with 80.87 for scikit-learn Linear
Regression on the identical rows.

Regression intervals use finite-sample split conformal absolute residuals
calibrated on validation only. For calibration size `n` and confidence
`1-alpha`, the selected residual is the
`ceil((n + 1) * (1-alpha))` order statistic, capped at `n`.
Classification confidence uses frozen probabilities, maximum probability,
normalized entropy, reliability bins, and validation-selected accident
thresholds.

## Classical Models

Linear Regression, Decision Tree, Random Forest, and XGBoost were compared for
each regression task. Random Forest won all twelve validation selections.

| Target | Horizon range | Test RMSE range | Test MAPE range |
|---|---:|---:|---:|
| Volume | 30-120 min | 62.0092-65.3058 | 10.218%-10.952% |
| Speed | 30-120 min | 3.7400-3.7940 | 9.029%-9.145% |
| Travel time | 30-120 min | 1.0822-1.1426 | 9.012%-9.753% |

Decision Tree, Random Forest, XGBoost, and scaled SVM were compared for
classification. Congestion uses Macro-F1 for selection; accident risk uses
ROC-AUC with PR-AUC and threshold evidence retained.

| Horizon | Congestion Macro-F1 | Accident ROC-AUC |
|---:|---:|---:|
| 30 min | 0.7540 | 0.6209 |
| 60 min | 0.7503 | 0.6237 |
| 90 min | 0.7493 | 0.5980 |
| 120 min | 0.7468 | 0.5894 |

The final classical registry contains twenty tamper-checked entries: twelve
regression and eight classification models, each linked to its model card,
feature schema, preprocessing version, split, metric record, and persisted
predictions.

## Recurrent Volume Model

The recurrent model is a from-scratch PyTorch LSTM with no pretrained weights.
The canonical candidate `lstm_s12_h32` consumes twelve road-local half-hour
steps and emits four volume forecasts in one pass. It was selected at epoch 8
using validation evidence, with early stopping, learning-rate control, dropout,
portable state-dictionary persistence, and explicit `map_location` loading.

| Horizon | Recurrent test RMSE | Classical RMSE on identical rows | Deep wins |
|---:|---:|---:|---|
| 30 min | 60.1443 | 63.2354 | Yes |
| 60 min | 60.8154 | 62.6833 | Yes |
| 90 min | 61.2014 | 65.0565 | Yes |
| 120 min | 61.8966 | 61.8495 | No |

The model beats the classical comparator at three of four horizons. At 120
minutes it trails by 0.0471 RMSE. The worst supported relative error slices
are late-night origins, especially hour 22 at 120 minutes. This limitation did
not trigger post-test architecture changes.

An optional CUDA diagnostic verified PyTorch 2.13.0+cu130 on the NVIDIA
GeForce RTX 5070 Laptop GPU. CUDA training produced a different
validation-selected candidate because floating-point training trajectories are
device-dependent. Those results were not frozen or published. The canonical
acceptance identity therefore uses explicit CPU recurrent training; CUDA
remains an optional acceleration experiment, not a hidden requirement.

## Confidence and Error Analysis

Across sixteen classical/recurrent regression groups, empirical 90% test
interval coverage ranges from 0.8924 to 0.9055. This is close to the nominal
level but is not a guarantee under future distribution shift.

Congestion expected calibration error ranges from 0.0029 to 0.0602. The
dominant congestion confusion is Free-flow predicted as Moderate at the
60-minute horizon. Accident prevalence is very low, so ROC-AUC is reported
with PR-AUC, validation-selected thresholds, positive-event support, and risk
bands. Slices below the configured support threshold remain present with blank
metrics rather than being silently omitted.

## Product and Runtime Acceptance

The Streamlit application exposes all nine required real-output views plus a
Data and training control page. All views are directly reachable from grouped
navigation and read persisted artifacts rather than retraining on rerun.

Acceptance against the clean CPU reproduction verified:

- all ten routes rendered without page exceptions or browser-console errors;
- a full 25-road, four-horizon request produced 100 prediction rows;
- cold prediction completed in 3.309 seconds, below the 30-second goal;
- upload validation recognized all 178,468 traffic rows, accounted for 176,701
  valid and 1,767 rejected duplicate rows, and staged only a versioned copy;
- CSV and self-contained HTML reports verified their source batch before
  export; the HTML export produced a browser download event;
- audit, artifact, model, metric, and forecast lineage were visible;
- an incorrect retraining confirmation was rejected, and active routing was
  unchanged;
- the original reference directory and immutable inputs were not edited.

## Acceptance Target Summary

| Target | Requirement | Result | Status |
|---|---:|---:|---|
| Volume accuracy | MAPE <= 12% at all horizons | 10.218%-10.952% | Met |
| Congestion classification | Macro-F1 >= 0.80 | 0.7468-0.7540 | Not met |
| Accident ranking | ROC-AUC >= 0.75 | 0.5894-0.6237 | Not met |
| Deep volume benchmark | Beat classical RMSE at all horizons | Wins 3 of 4 | Not met |
| Inference latency | <= 30 seconds | 3.309 seconds cold | Met |
| Interval behavior | Near 90% empirical coverage | 0.8924-0.9055 | Met as diagnostic |
| Reproduction | Clean CPU rebuild and metric reconciliation | 16 stages; verifier passed | Met |
| Assurance | Complete isolated suite | 192 passed | Met |
| Dashboard | Nine required views and support workflows | Ten routes passed | Met |

## Limitations and Recommendations

1. Do not deploy accident probabilities as an autonomous safety decision.
   Collect more positive-event examples, improve incident exposure features,
   and reassess PR-AUC and operating thresholds on a later chronological
   period.
2. Improve congestion separation around the Free-flow/Moderate boundary with
   better capacity/context signals before increasing model complexity.
3. Investigate late-night recurrent errors and consider horizon-specific loss
   weighting only through a new validation-frozen experiment.
4. Preserve CPU as the reproducibility identity. Treat CUDA training as a
   separately versioned experiment with its own frozen metrics and environment
   record.
5. Run the documented CPU acceptance path on macOS and Linux as the remaining
   independent portability check. The implementation is platform-neutral, but
   the final clean full training run was measured on Windows.

## Final Disposition

FlowCast v1.0 satisfies the reproducible data pipeline, modelling coverage,
confidence, inference, reporting, dashboard, lineage, safety-control, and
testing delivery gates. It is suitable for reviewer evaluation and
demonstration as an analytical decision-support system. It is not a
production traffic-signal controller, and the unmet classifier and
all-horizon deep-benchmark goals must remain visible in any release notes.
