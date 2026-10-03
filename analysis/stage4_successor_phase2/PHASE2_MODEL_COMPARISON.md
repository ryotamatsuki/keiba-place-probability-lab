# Stage 4 Successor Phase 2 — Model-family Comparison

Status: **PASS — preregistered outer 2023/2024 comparison completed**

## Decision

- winner under frozen v2 rule: **xgboost**
- point-Brier-best: **xgboost**
- incumbent retained: **False**
- reason: point-Brier-best selected: no incumbent was supplied or the incumbent fell outside the paired one-SE Brier retention band
- winner race-macro Brier: 0.151494

This is a preregistered replacement decision, not a claim of statistical proof.

## Outer metrics

| name                   |   race_macro_brier |   race_macro_log_loss |   runner_micro_brier |   runner_micro_log_loss |   calibration_intercept |   calibration_slope |   ece_10bin |   races |   rows |
|:-----------------------|-------------------:|----------------------:|---------------------:|------------------------:|------------------------:|--------------------:|------------:|--------:|-------:|
| xgboost                |           0.151494 |              0.46602  |             0.146529 |                0.455735 |               -0.029634 |             1.02083 |    0.009373 |     495 |   6313 |
| hist_gradient_boosting |           0.152193 |              0.468353 |             0.147132 |                0.457752 |               -0.023375 |             1.0203  |    0.007296 |     495 |   6313 |
| l2_logistic_c0.1       |           0.153096 |              0.470751 |             0.147285 |                0.458203 |               -0.054207 |             1.01762 |    0.014399 |     495 |   6313 |
| random_forest          |           0.155078 |              0.476412 |             0.149069 |                0.463475 |                0.243908 |             1.23474 |    0.018154 |     495 |   6313 |

## Frozen v2 selection table

| name                   |   race_macro_brier |   mean_brier_delta_vs_best |   se_brier_delta_vs_best | within_one_se_brier   |   mean_brier_delta_vs_incumbent |   se_brier_delta_vs_incumbent |
|:-----------------------|-------------------:|---------------------------:|-------------------------:|:----------------------|--------------------------------:|------------------------------:|
| xgboost                |           0.151494 |                   0        |                 0        | True                  |                       -0.001602 |                      0.00067  |
| hist_gradient_boosting |           0.152193 |                   0.000699 |                 0.000299 | False                 |                       -0.000903 |                      0.000768 |
| l2_logistic_c0.1       |           0.153096 |                   0.001602 |                 0.00067  | False                 |                        0        |                      0        |
| random_forest          |           0.155078 |                   0.003584 |                 0.000726 | False                 |                        0.001982 |                      0.000934 |

## Selected inner configurations

|   outer_year | family                 | config_id   |   complexity_rank | params_json                                                                                     |
|-------------:|:-----------------------|:------------|------------------:|:------------------------------------------------------------------------------------------------|
|         2023 | random_forest          | RF04        |                 4 | {"max_depth": 8, "max_features": "sqrt", "min_samples_leaf": 20}                                |
|         2023 | hist_gradient_boosting | HGB01       |                 1 | {"l2_regularization": 2.0, "learning_rate": 0.03, "max_leaf_nodes": 15, "min_samples_leaf": 50} |
|         2023 | xgboost                | XGB01       |                 1 | {"learning_rate": 0.03, "max_depth": 3, "min_child_weight": 20, "reg_lambda": 5.0}              |
|         2024 | random_forest          | RF04        |                 4 | {"max_depth": 8, "max_features": "sqrt", "min_samples_leaf": 20}                                |
|         2024 | hist_gradient_boosting | HGB01       |                 1 | {"l2_regularization": 2.0, "learning_rate": 0.03, "max_leaf_nodes": 15, "min_samples_leaf": 50} |
|         2024 | xgboost                | XGB01       |                 1 | {"learning_rate": 0.03, "max_depth": 3, "min_child_weight": 20, "reg_lambda": 5.0}              |

## Outer-year stability

|   outer_year | name                   |   race_macro_brier |   race_macro_log_loss |   races |   rows |
|-------------:|:-----------------------|-------------------:|----------------------:|--------:|-------:|
|         2023 | xgboost                |           0.150204 |              0.462082 |     250 |   3194 |
|         2023 | hist_gradient_boosting |           0.150434 |              0.463347 |     250 |   3194 |
|         2023 | random_forest          |           0.152856 |              0.470474 |     250 |   3194 |
|         2023 | l2_logistic_c0.1       |           0.153181 |              0.470891 |     250 |   3194 |
|         2024 | xgboost                |           0.15281  |              0.470038 |     245 |   3119 |
|         2024 | l2_logistic_c0.1       |           0.15301  |              0.470607 |     245 |   3119 |
|         2024 | hist_gradient_boosting |           0.153987 |              0.473461 |     245 |   3119 |
|         2024 | random_forest          |           0.157345 |              0.48247  |     245 |   3119 |

## Audit

- evaluation rows: 6,313
- evaluation races: 495
- evaluation-row SHA256: a72fa10cc9adcd1ceee4c5b5eacd0dfdcdfc33fc984bf0b7866ec70b617bfd85
- incumbent reproduction max absolute probability difference vs Phase 1: 5.000e-10
- 2025 rows loaded: **no**
- market fields loaded: **no**
- post-hoc calibration: **none**

Phase 3 must not start until this Phase-2 result is frozen.
