# FlowCast Classical Classification

## Evaluation contract

- Version: `classical_classification_v1`; seed: `42`.
- Jobs: 8 (2 tasks x 4 horizons).
- Search: 8 candidates across 4 required families and 5 expanding folds.
- Congestion selection uses Macro-F1; accident selection uses ROC-AUC with PR-AUC visible.
- Calibration and accident thresholds were frozen from chronological validation evidence before one explicit test load.

## Frozen hold-out scoreboard

| Task | Horizon | Family | Calibration | Threshold | Validation primary | Test primary | Test secondary |
|---|---:|---|---|---:|---:|---:|---:|
| congestion | 30 | random_forest | not applied | - | 0.7506 | 0.7540 | 0.7659 |
| congestion | 60 | xgboost | applied | - | 0.7497 | 0.7503 | 0.7736 |
| congestion | 90 | xgboost | applied | - | 0.7491 | 0.7493 | 0.7762 |
| congestion | 120 | random_forest | applied | - | 0.7337 | 0.7468 | 0.7751 |
| accident | 30 | svm | applied | 0.0133 | 0.5763 | 0.6209 | 0.0209 |
| accident | 60 | svm | applied | 0.0190 | 0.5813 | 0.6237 | 0.0182 |
| accident | 90 | svm | applied | 0.0132 | 0.5603 | 0.5980 | 0.0161 |
| accident | 120 | svm | applied | 0.0102 | 0.5471 | 0.5894 | 0.0165 |

## Coverage and persistence

- Required family/job comparisons: 32.
- Successful CV fold evaluations: 320.
- Reloadable selected classifiers/model cards: 8.
- Persisted validation/test predictions: 428257 rows.

## Acceptance targets

- Congestion Macro-F1 target met at all horizons: `False`.
- Accident ROC-AUC target met at all horizons: `False`.

## Limitations

- CV search uses evenly spaced timestamps spanning each fold; final family fits use all eligible training rows.
- Accident positives are rare; ROC-AUC is paired with PR-AUC and validation-selected operating metrics.
- Probability calibration uses a chronological validation split and is applied only when required or when Brier score improves enough.
- Confidence displays and segmented error analysis remain Step 16 work.
