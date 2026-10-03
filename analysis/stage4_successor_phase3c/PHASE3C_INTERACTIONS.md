# Stage 4 Successor Phase 3C — Explicit Interactions

Status: **PASS — preregistered feature-block comparison completed**

## Decision

- incumbent: **xgboost_xgb01_plus_relative_ability**
- challenger: **xgboost_xgb01_plus_relative_and_interactions**
- winner under frozen v2 rule: **xgboost_xgb01_plus_relative_ability**
- challenger accepted: **False**
- reason: incumbent retained: race-macro Brier is within one paired date-clustered SE of the point-Brier-best
- challenger - incumbent paired Brier delta: -0.000191
- date-clustered SE of paired Brier delta: 0.000216

This is a preregistered replacement decision, not a significance-test claim.

## Outer metrics

| name                                         |   race_macro_brier |   race_macro_log_loss |   runner_micro_brier |   runner_micro_log_loss |   calibration_intercept |   calibration_slope |   ece_10bin |   races |   rows |
|:---------------------------------------------|-------------------:|----------------------:|---------------------:|------------------------:|------------------------:|--------------------:|------------:|--------:|-------:|
| xgboost_xgb01_plus_relative_and_interactions |           0.148768 |              0.458978 |             0.14468  |                0.450816 |               -0.006047 |             1.02484 |    0.00879  |     495 |   6313 |
| xgboost_xgb01_plus_relative_ability          |           0.148959 |              0.459405 |             0.144733 |                0.450894 |               -0.00825  |             1.02263 |    0.012196 |     495 |   6313 |

## Selection table

| name                                         |   race_macro_brier |   mean_brier_delta_vs_best |   se_brier_delta_vs_best | within_one_se_brier   |
|:---------------------------------------------|-------------------:|---------------------------:|-------------------------:|:----------------------|
| xgboost_xgb01_plus_relative_and_interactions |           0.148768 |                   0        |                 0        | True                  |
| xgboost_xgb01_plus_relative_ability          |           0.148959 |                   0.000191 |                 0.000216 | True                  |

## Outer-year stability

|   outer_year | name                                         |   race_macro_brier |   race_macro_log_loss |   races |   rows |
|-------------:|:---------------------------------------------|-------------------:|----------------------:|--------:|-------:|
|         2023 | xgboost_xgb01_plus_relative_and_interactions |           0.148945 |              0.45875  |     250 |   3194 |
|         2023 | xgboost_xgb01_plus_relative_ability          |           0.149101 |              0.459014 |     250 |   3194 |
|         2024 | xgboost_xgb01_plus_relative_and_interactions |           0.148586 |              0.459211 |     245 |   3119 |
|         2024 | xgboost_xgb01_plus_relative_ability          |           0.148814 |              0.459805 |     245 |   3119 |

## Interaction-feature coverage

|   outer_year | feature                                    |   eligible_coverage |   eligible_rows |
|-------------:|:-------------------------------------------|--------------------:|----------------:|
|         2023 | interaction_draw_x_front_tendency          |                   1 |            3194 |
|         2023 | interaction_front_tendency_x_race_pressure |                   1 |            3194 |
|         2023 | interaction_rest_x_age                     |                   1 |            3194 |
|         2024 | interaction_draw_x_front_tendency          |                   1 |            3119 |
|         2024 | interaction_front_tendency_x_race_pressure |                   1 |            3119 |
|         2024 | interaction_rest_x_age                     |                   1 |            3119 |

## Audit

- evaluation rows: 6,313
- evaluation races: 495
- evaluation-row SHA256: a72fa10cc9adcd1ceee4c5b5eacd0dfdcdfc33fc984bf0b7866ec70b617bfd85
- recent-trend block carried forward: **no**
- 2025 rows loaded: **no**
- market fields loaded: **no**
- XGB01 hyperparameters retuned: **no**
- post-hoc calibration: **none**

Phase 3 development is complete after this comparison.
