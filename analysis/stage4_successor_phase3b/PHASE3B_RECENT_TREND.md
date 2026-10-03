# Stage 4 Successor Phase 3B — Recent Trend

Status: **PASS — preregistered feature-block comparison completed**

## Decision

- incumbent: **xgboost_xgb01_plus_relative_ability**
- challenger: **xgboost_xgb01_plus_relative_and_recent_trend**
- winner under frozen v2 rule: **xgboost_xgb01_plus_relative_ability**
- challenger accepted: **False**
- reason: incumbent retained: race-macro Brier is within one paired date-clustered SE of the point-Brier-best
- challenger - incumbent paired Brier delta: -0.000285
- date-clustered SE of paired Brier delta: 0.000322

This is a preregistered replacement decision, not a significance-test claim.

## Outer metrics

| name                                         |   race_macro_brier |   race_macro_log_loss |   runner_micro_brier |   runner_micro_log_loss |   calibration_intercept |   calibration_slope |   ece_10bin |   races |   rows |
|:---------------------------------------------|-------------------:|----------------------:|---------------------:|------------------------:|------------------------:|--------------------:|------------:|--------:|-------:|
| xgboost_xgb01_plus_relative_and_recent_trend |           0.148674 |              0.458824 |             0.14459  |                0.450611 |               -0.005799 |             1.02523 |    0.011276 |     495 |   6313 |
| xgboost_xgb01_plus_relative_ability          |           0.148959 |              0.459405 |             0.144733 |                0.450894 |               -0.00825  |             1.02263 |    0.012196 |     495 |   6313 |

## Selection table

| name                                         |   race_macro_brier |   mean_brier_delta_vs_best |   se_brier_delta_vs_best | within_one_se_brier   |
|:---------------------------------------------|-------------------:|---------------------------:|-------------------------:|:----------------------|
| xgboost_xgb01_plus_relative_and_recent_trend |           0.148674 |                   0        |                 0        | True                  |
| xgboost_xgb01_plus_relative_ability          |           0.148959 |                   0.000285 |                 0.000322 | True                  |

## Outer-year stability

|   outer_year | name                                         |   race_macro_brier |   race_macro_log_loss |   races |   rows |
|-------------:|:---------------------------------------------|-------------------:|----------------------:|--------:|-------:|
|         2023 | xgboost_xgb01_plus_relative_and_recent_trend |           0.148976 |              0.45847  |     250 |   3194 |
|         2023 | xgboost_xgb01_plus_relative_ability          |           0.149101 |              0.459014 |     250 |   3194 |
|         2024 | xgboost_xgb01_plus_relative_and_recent_trend |           0.148365 |              0.459185 |     245 |   3119 |
|         2024 | xgboost_xgb01_plus_relative_ability          |           0.148814 |              0.459805 |     245 |   3119 |

## Trend-feature coverage

|   outer_year | feature                            |   eligible_coverage |   eligible_rows |
|-------------:|:-----------------------------------|--------------------:|----------------:|
|         2023 | trend_recent3_finish_improvement   |                   1 |            3194 |
|         2023 | trend_recent3_time_improvement     |                   1 |            3194 |
|         2023 | trend_recent3_top3_count_change    |                   1 |            3194 |
|         2023 | trend_recent4_early_forward_change |                   1 |            3194 |
|         2024 | trend_recent3_finish_improvement   |                   1 |            3119 |
|         2024 | trend_recent3_time_improvement     |                   1 |            3119 |
|         2024 | trend_recent3_top3_count_change    |                   1 |            3119 |
|         2024 | trend_recent4_early_forward_change |                   1 |            3119 |

## Audit

- evaluation rows: 6,313
- evaluation races: 495
- evaluation-row SHA256: a72fa10cc9adcd1ceee4c5b5eacd0dfdcdfc33fc984bf0b7866ec70b617bfd85
- horse-history continuity includes warmup 2010-2015
- 2025 rows loaded: **no**
- market fields loaded: **no**
- XGB01 hyperparameters retuned: **no**
- post-hoc calibration: **none**

The Phase-3A incumbent remains frozen for Phase 3C.
