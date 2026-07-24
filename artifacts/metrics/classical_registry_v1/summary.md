# FlowCast Classical Model Registry

## Contract

- Version: `classical_registry_v1`.
- Coverage: 20 target/horizon entries across five required outputs and four forecast horizons.
- Prediction mapping: 1,078,957 persisted validation/test rows indexed in place; no data was copied.
- All selections came from frozen validation evidence. Test metrics are reported honestly and were not used to change a winner.

## Combined scoreboard

| Target | Horizon | Family | Primary metric | Validation | Test | Acceptance |
|---|---:|---|---|---:|---:|---|
| volume | 30 min | random_forest | rmse | 61.1926 | 63.4595 | met |
| volume | 60 min | random_forest | rmse | 61.1236 | 62.8626 | met |
| volume | 90 min | random_forest | rmse | 65.6565 | 65.3058 | met |
| volume | 120 min | random_forest | rmse | 60.5924 | 62.0092 | met |
| speed | 30 min | random_forest | rmse | 3.72298 | 3.73998 | not specified |
| speed | 60 min | random_forest | rmse | 3.75908 | 3.7683 | not specified |
| speed | 90 min | random_forest | rmse | 3.85441 | 3.79402 | not specified |
| speed | 120 min | random_forest | rmse | 3.84186 | 3.79233 | not specified |
| travel_time | 30 min | random_forest | rmse | 1.10048 | 1.14263 | not specified |
| travel_time | 60 min | random_forest | rmse | 1.08534 | 1.09486 | not specified |
| travel_time | 90 min | random_forest | rmse | 1.087 | 1.08217 | not specified |
| travel_time | 120 min | random_forest | rmse | 1.08225 | 1.10158 | not specified |
| congestion | 30 min | random_forest | macro_f1 | 0.750623 | 0.75397 | not met |
| congestion | 60 min | xgboost | macro_f1 | 0.749702 | 0.750282 | not met |
| congestion | 90 min | xgboost | macro_f1 | 0.749073 | 0.749251 | not met |
| congestion | 120 min | random_forest | macro_f1 | 0.733744 | 0.746787 | not met |
| accident | 30 min | svm | roc_auc | 0.5763 | 0.620908 | not met |
| accident | 60 min | svm | roc_auc | 0.581285 | 0.623673 | not met |
| accident | 90 min | svm | roc_auc | 0.560282 | 0.598019 | not met |
| accident | 120 min | svm | roc_auc | 0.547132 | 0.58941 | not met |

## Selection rationale

### volume_h1

Selected before test access because random_forest/forest_deep won the frozen validation comparison on RMSE (61.19257047) after time-ordered CV (mean 80.75278107, standard deviation 7.262159316). Validation fit/prediction time was 2.20099s/0.066941s. Interpretability context: tree-ensemble feature importance. Runtime and interpretability are governance context only; test metrics were not selection inputs.

### volume_h2

Selected before test access because random_forest/forest_deep won the frozen validation comparison on RMSE (61.12355559) after time-ordered CV (mean 89.49765824, standard deviation 12.76402858). Validation fit/prediction time was 2.16751s/0.068367s. Interpretability context: tree-ensemble feature importance. Runtime and interpretability are governance context only; test metrics were not selection inputs.

### volume_h3

Selected before test access because random_forest/forest_balanced won the frozen validation comparison on RMSE (65.65649039) after time-ordered CV (mean 96.20647212, standard deviation 14.75140431). Validation fit/prediction time was 1.36572s/0.069324s. Interpretability context: tree-ensemble feature importance. Runtime and interpretability are governance context only; test metrics were not selection inputs.

### volume_h4

Selected before test access because random_forest/forest_deep won the frozen validation comparison on RMSE (60.59237758) after time-ordered CV (mean 104.8584418, standard deviation 20.53671075). Validation fit/prediction time was 2.11979s/0.069171s. Interpretability context: tree-ensemble feature importance. Runtime and interpretability are governance context only; test metrics were not selection inputs.

### speed_h1

Selected before test access because random_forest/forest_deep won the frozen validation comparison on RMSE (3.722982197) after time-ordered CV (mean 4.687710051, standard deviation 0.4849137806). Validation fit/prediction time was 2.12881s/0.067845s. Interpretability context: tree-ensemble feature importance. Runtime and interpretability are governance context only; test metrics were not selection inputs.

### speed_h2

Selected before test access because random_forest/forest_deep won the frozen validation comparison on RMSE (3.759077774) after time-ordered CV (mean 5.062842974, standard deviation 0.5910077481). Validation fit/prediction time was 2.14368s/0.070241s. Interpretability context: tree-ensemble feature importance. Runtime and interpretability are governance context only; test metrics were not selection inputs.

### speed_h3

Selected before test access because random_forest/forest_balanced won the frozen validation comparison on RMSE (3.854410876) after time-ordered CV (mean 5.279576189, standard deviation 0.8290471817). Validation fit/prediction time was 1.37909s/0.068725s. Interpretability context: tree-ensemble feature importance. Runtime and interpretability are governance context only; test metrics were not selection inputs.

### speed_h4

Selected before test access because random_forest/forest_balanced won the frozen validation comparison on RMSE (3.841860581) after time-ordered CV (mean 5.287603673, standard deviation 0.8017266342). Validation fit/prediction time was 1.38377s/0.065733s. Interpretability context: tree-ensemble feature importance. Runtime and interpretability are governance context only; test metrics were not selection inputs.

### travel_time_h1

Selected before test access because random_forest/forest_deep won the frozen validation comparison on RMSE (1.100480954) after time-ordered CV (mean 1.60064824, standard deviation 0.3660809108). Validation fit/prediction time was 2.22781s/0.070133s. Interpretability context: tree-ensemble feature importance. Runtime and interpretability are governance context only; test metrics were not selection inputs.

### travel_time_h2

Selected before test access because random_forest/forest_deep won the frozen validation comparison on RMSE (1.085335142) after time-ordered CV (mean 1.640693946, standard deviation 0.3878922675). Validation fit/prediction time was 2.12565s/0.077605s. Interpretability context: tree-ensemble feature importance. Runtime and interpretability are governance context only; test metrics were not selection inputs.

### travel_time_h3

Selected before test access because random_forest/forest_deep won the frozen validation comparison on RMSE (1.087003399) after time-ordered CV (mean 1.653024838, standard deviation 0.4274760867). Validation fit/prediction time was 2.15045s/0.070239s. Interpretability context: tree-ensemble feature importance. Runtime and interpretability are governance context only; test metrics were not selection inputs.

### travel_time_h4

Selected before test access because random_forest/forest_deep won the frozen validation comparison on RMSE (1.082250276) after time-ordered CV (mean 1.676574606, standard deviation 0.4202603644). Validation fit/prediction time was 2.26347s/0.070092s. Interpretability context: tree-ensemble feature importance. Runtime and interpretability are governance context only; test metrics were not selection inputs.

### congestion_h1

Selected before test access because random_forest/forest_deep won the frozen validation comparison on MACRO_F1 (0.7506226006) after time-ordered CV (mean 0.6811728173, standard deviation 0.0134301453). Validation fit/prediction time was 4.08836s/0.194412s. Interpretability context: tree-ensemble feature importance. Runtime and interpretability are governance context only; test metrics were not selection inputs.

### congestion_h2

Selected before test access because xgboost/xgb_deep won the frozen validation comparison on MACRO_F1 (0.7497023949) after time-ordered CV (mean 0.6300991188, standard deviation 0.0308273092). Validation fit/prediction time was 1.19727s/0.102608s. Interpretability context: boosted-tree feature importance. Runtime and interpretability are governance context only; test metrics were not selection inputs.

### congestion_h3

Selected before test access because xgboost/xgb_deep won the frozen validation comparison on MACRO_F1 (0.7490727035) after time-ordered CV (mean 0.6174313648, standard deviation 0.0404146001). Validation fit/prediction time was 1.10427s/0.102677s. Interpretability context: boosted-tree feature importance. Runtime and interpretability are governance context only; test metrics were not selection inputs.

### congestion_h4

Selected before test access because random_forest/forest_balanced won the frozen validation comparison on MACRO_F1 (0.7337437301) after time-ordered CV (mean 0.6297798554, standard deviation 0.036865221). Validation fit/prediction time was 1.72244s/0.149351s. Interpretability context: tree-ensemble feature importance. Runtime and interpretability are governance context only; test metrics were not selection inputs.

### accident_h1

Selected before test access because svm/svm_regularized won the frozen validation comparison on ROC_AUC (0.5763001149) after time-ordered CV (mean 0.5326621042, standard deviation 0.0563276169). Validation fit/prediction time was 5.62036s/0.055687s. Interpretability context: linear decision coefficients. Runtime and interpretability are governance context only; test metrics were not selection inputs.

### accident_h2

Selected before test access because svm/svm_regularized won the frozen validation comparison on ROC_AUC (0.5812852433) after time-ordered CV (mean 0.5298677956, standard deviation 0.0277102461). Validation fit/prediction time was 7.0784s/0.05054s. Interpretability context: linear decision coefficients. Runtime and interpretability are governance context only; test metrics were not selection inputs.

### accident_h3

Selected before test access because svm/svm_default won the frozen validation comparison on ROC_AUC (0.5602821469) after time-ordered CV (mean 0.4833379964, standard deviation 0.0304646615). Validation fit/prediction time was 9.90982s/0.049856s. Interpretability context: linear decision coefficients. Runtime and interpretability are governance context only; test metrics were not selection inputs.

### accident_h4

Selected before test access because svm/svm_regularized won the frozen validation comparison on ROC_AUC (0.5471324772) after time-ordered CV (mean 0.5092951094, standard deviation 0.0517277048). Validation fit/prediction time was 4.72194s/0.051313s. Interpretability context: linear decision coefficients. Runtime and interpretability are governance context only; test metrics were not selection inputs.

## Acceptance summary

- volume: 4/4 horizons met `mape_percent <= 12.0`; observed test values remain frozen and visible.
- congestion: 0/4 horizons met `macro_f1 >= 0.8`; observed test values remain frozen and visible.
- accident: 0/4 horizons met `roc_auc >= 0.75`; observed test values remain frozen and visible.

## Governance

- Each entry records its model, model card, source predictions, selection manifest, preprocessing version, feature schema hash, processed data hash, seed, windows, metrics, and limitations.
- The verified loader checks the registry, both upstream summaries, every referenced artifact, and the independent registry configuration before returning a model.
- Runtime and interpretability are supplied as operating context; they do not override the validation-metric winner.
