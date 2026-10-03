# Stage 4 Successor Phase 3A — Relative Ability

Status: **PASS — preregistered feature-block comparison completed**

## Decision

- incumbent: **xgboost_phase2_xgb01**
- challenger: **xgboost_xgb01_plus_relative_ability**
- winner under frozen v2 rule: **xgboost_xgb01_plus_relative_ability**
- challenger accepted: **True**
- reason: point-Brier-best selected: no incumbent was supplied or the incumbent fell outside the paired one-SE Brier retention band

This is a preregistered replacement decision, not a significance-test claim.

## Outer metrics

| name                                |   race_macro_brier |   race_macro_log_loss |   runner_micro_brier |   runner_micro_log_loss |   calibration_intercept |   calibration_slope |   ece_10bin |   races |   rows |
|:------------------------------------|-------------------:|----------------------:|---------------------:|------------------------:|------------------------:|--------------------:|------------:|--------:|-------:|
| xgboost_xgb01_plus_relative_ability |           0.148959 |              0.459405 |             0.144733 |                0.450894 |               -0.00825  |             1.02263 |    0.012196 |     495 |   6313 |
| xgboost_phase2_xgb01                |           0.151494 |              0.46602  |             0.146529 |                0.455735 |               -0.029634 |             1.02083 |    0.009373 |     495 |   6313 |

## Selection table

| name                                |   race_macro_brier |   mean_brier_delta_vs_best |   se_brier_delta_vs_best | within_one_se_brier   |
|:------------------------------------|-------------------:|---------------------------:|-------------------------:|:----------------------|
| xgboost_xgb01_plus_relative_ability |           0.148959 |                   0        |                 0        | True                  |
| xgboost_phase2_xgb01                |           0.151494 |                   0.002535 |                 0.000882 | False                 |

## Outer-year stability

|   outer_year | name                                |   race_macro_brier |   race_macro_log_loss |   races |   rows |
|-------------:|:------------------------------------|-------------------:|----------------------:|--------:|-------:|
|         2023 | xgboost_xgb01_plus_relative_ability |           0.149101 |              0.459014 |     250 |   3194 |
|         2023 | xgboost_phase2_xgb01                |           0.150204 |              0.462082 |     250 |   3194 |
|         2024 | xgboost_xgb01_plus_relative_ability |           0.148814 |              0.459805 |     245 |   3119 |
|         2024 | xgboost_phase2_xgb01                |           0.15281  |              0.470038 |     245 |   3119 |

## Relative-feature coverage

|   outer_year | feature                          |   relative_coverage |   eligible_coverage |   eligible_rows |
|-------------:|:---------------------------------|--------------------:|--------------------:|----------------:|
|         2023 | rel_career_top3_vs_others        |            1        |                   1 |            3194 |
|         2023 | rel_same_distance_top3_vs_others |            1        |                   1 |            3194 |
|         2023 | rel_same_course_top3_vs_others   |            1        |                   1 |            3194 |
|         2023 | rel_recent3_finish_vs_others     |            0.986663 |                   1 |            3194 |
|         2023 | rel_recent3_time_vs_others       |            0.986663 |                   1 |            3194 |
|         2024 | rel_career_top3_vs_others        |            1        |                   1 |            3119 |
|         2024 | rel_same_distance_top3_vs_others |            1        |                   1 |            3119 |
|         2024 | rel_same_course_top3_vs_others   |            1        |                   1 |            3119 |
|         2024 | rel_recent3_finish_vs_others     |            0.984183 |                   1 |            3119 |
|         2024 | rel_recent3_time_vs_others       |            0.984183 |                   1 |            3119 |

## Audit

- evaluation rows: 6,313
- evaluation races: 495
- evaluation-row SHA256: a72fa10cc9adcd1ceee4c5b5eacd0dfdcdfc33fc984bf0b7866ec70b617bfd85
- all relative features use full starter fields before eligible-row filtering
- 2025 rows loaded: **no**
- market fields loaded: **no**
- XGB01 hyperparameters retuned: **no**
- post-hoc calibration: **none**

The Phase-3A winner becomes the frozen incumbent for Phase 3B.
