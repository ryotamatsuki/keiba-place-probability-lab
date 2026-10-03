# Scope Expansion Stage 1 — A/B/C Comparison

Status: **PASS — frozen training-population comparison completed**

## Question

Does broader turf training data improve prediction on the existing 1200m evaluation rows
without changing features or XGB01 hyperparameters?

## Decision

- incumbent: **A_current_1200**
- point-Brier-best: **B_turf_sprint_1000_1400**
- winner under frozen paired one-SE rule: **A_current_1200**
- incumbent retained: **True**
- reason: incumbent retained: race-macro Brier is within one paired date-clustered SE of the point-Brier-best

This is a model-selection rule, not a statistical-significance claim.

## Common 2023-2024 evaluation

- rows: **6,313**
- races: **495**
- fingerprint: a72fa10cc9adcd1ceee4c5b5eacd0dfdcdfc33fc984bf0b7866ec70b617bfd85
- Candidate A reproduces frozen Phase3A predictions within 9-decimal CSV rounding: **yes**
- max absolute A/reference difference: 4.99999999737e-10

## Overall metrics

| name                    |   race_macro_brier |   race_macro_log_loss |   runner_micro_brier |   runner_micro_log_loss |   calibration_intercept |   calibration_slope |   ece_10bin |   races |   rows |
|:------------------------|-------------------:|----------------------:|---------------------:|------------------------:|------------------------:|--------------------:|------------:|--------:|-------:|
| B_turf_sprint_1000_1400 |           0.148774 |              0.458557 |             0.144406 |                0.449709 |             -0.00829339 |            1.0196   |   0.0119423 |     495 |   6313 |
| A_current_1200          |           0.148959 |              0.459405 |             0.144733 |                0.450894 |             -0.00824957 |            1.02263  |   0.012196  |     495 |   6313 |
| C_turf_global_1000_2600 |           0.149394 |              0.460244 |             0.144563 |                0.450353 |             -0.0341282  |            0.984454 |   0.0122989 |     495 |   6313 |

## Selection table

| name                    |   race_macro_brier |   race_macro_log_loss |   mean_brier_delta_vs_best |   se_brier_delta_vs_best | within_one_se_brier   |
|:------------------------|-------------------:|----------------------:|---------------------------:|-------------------------:|:----------------------|
| B_turf_sprint_1000_1400 |           0.148774 |              0.458557 |                0           |              0           | True                  |
| A_current_1200          |           0.148959 |              0.459405 |                0.000184425 |              0.000324919 | True                  |
| C_turf_global_1000_2600 |           0.149394 |              0.460244 |                0.000619898 |              0.000511416 | False                 |

## Direct incumbent/challenger paired differences

| incumbent      | challenger              |   mean_incumbent_minus_challenger_brier |   se_incumbent_minus_challenger_brier |   mean_incumbent_minus_challenger_log_loss |   se_incumbent_minus_challenger_log_loss |
|:---------------|:------------------------|----------------------------------------:|--------------------------------------:|-------------------------------------------:|-----------------------------------------:|
| A_current_1200 | B_turf_sprint_1000_1400 |                             0.000184425 |                           0.000324919 |                                0.000848263 |                              0.000831554 |
| A_current_1200 | C_turf_global_1000_2600 |                            -0.000435473 |                           0.000557915 |                               -0.000838654 |                              0.00147167  |

## Outer-year metrics

|   outer_year | name                    |   race_macro_brier |   race_macro_log_loss |   calibration_intercept |   calibration_slope |   races |   rows |
|-------------:|:------------------------|-------------------:|----------------------:|------------------------:|--------------------:|--------:|-------:|
|         2023 | A_current_1200          |           0.149101 |              0.459014 |             -0.0493722  |            0.98532  |     250 |   3194 |
|         2023 | B_turf_sprint_1000_1400 |           0.149432 |              0.459408 |             -0.0498955  |            0.976245 |     250 |   3194 |
|         2023 | C_turf_global_1000_2600 |           0.149656 |              0.460098 |             -0.05796    |            0.956615 |     250 |   3194 |
|         2024 | B_turf_sprint_1000_1400 |           0.148103 |              0.457689 |              0.0357492  |            1.06597  |     245 |   3119 |
|         2024 | A_current_1200          |           0.148814 |              0.459805 |              0.0356636  |            1.06244  |     245 |   3119 |
|         2024 | C_turf_global_1000_2600 |           0.149127 |              0.460393 |             -0.00880473 |            1.01447  |     245 |   3119 |

## Training folds

| candidate               |   outer_year |   train_min_year |   train_max_year |   train_rows |   train_races |   train_context_rows |   prior_mean |   min_distance_m |   max_distance_m |
|:------------------------|-------------:|-----------------:|-----------------:|-------------:|--------------:|---------------------:|-------------:|-----------------:|-----------------:|
| A_current_1200          |         2023 |             2016 |             2022 |        23656 |          1822 |                27974 |     0.2035   |             1200 |             1200 |
| A_current_1200          |         2024 |             2016 |             2023 |        26850 |          2072 |                31723 |     0.204097 |             1200 |             1200 |
| B_turf_sprint_1000_1400 |         2023 |             2016 |             2022 |        35179 |          2800 |                42814 |     0.203815 |             1000 |             1400 |
| B_turf_sprint_1000_1400 |         2024 |             2016 |             2023 |        39994 |          3192 |                48660 |     0.204531 |             1000 |             1400 |
| C_turf_global_1000_2600 |         2023 |             2016 |             2022 |       104673 |          9469 |               132260 |     0.221662 |             1000 |             2600 |
| C_turf_global_1000_2600 |         2024 |             2016 |             2023 |       119366 |         10815 |               150840 |     0.221914 |             1000 |             2600 |

## Interpretation boundary

This experiment changes training-population width only. It does not validate B or C for
prediction on non-1200 races. Cross-distance deployment remains prohibited until the later
distance-wide evaluation stage.
