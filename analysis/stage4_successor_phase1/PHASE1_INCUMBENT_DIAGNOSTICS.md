# Stage 4 Successor Phase 1 — Incumbent Diagnostics

Status: **PASS — leakage-safe 2023/2024 incumbent OOF diagnosis completed**

No challenger model or feature block is selected in this phase.
2025 is not loaded.

## Outer walk-forward contract

- 2023 forecast: trained through 2022 only.
- 2024 forecast: refit through 2023 only.
- incumbent: L2 Logistic Regression, C=0.1.
- current Stage 4 feature engineering and prior strength 6.
- historical score: raw marginal P(top3); no sum-to-three adjustment on partial cohorts.

## Overall

| model               |   race_macro_brier |   race_macro_log_loss |   runner_micro_brier |   runner_micro_log_loss |   brier_skill_vs_field_size_baseline |   races |   rows |
|:--------------------|-------------------:|----------------------:|---------------------:|------------------------:|-------------------------------------:|--------:|-------:|
| l2_logistic_c0.1    |           0.153096 |              0.470751 |             0.147285 |                0.458203 |                             0.106786 |     495 |   6313 |
| field_size_baseline |           0.171399 |              0.525783 |             0.163678 |                0.508255 |                             0        |     495 |   6313 |

## By outer year

|   level |   race_macro_brier |   baseline_race_macro_brier |   brier_skill_vs_field_size_baseline |   observed_top3_rate |   mean_predicted |   calibration_intercept |   calibration_slope |   races |   rows |
|--------:|-------------------:|----------------------------:|-------------------------------------:|---------------------:|-----------------:|------------------------:|--------------------:|--------:|-------:|
|    2023 |           0.153181 |                    0.170511 |                             0.10164  |             0.208516 |         0.219645 |               -0.100483 |            0.975814 |     250 |   3194 |
|    2024 |           0.15301  |                    0.172305 |                             0.111982 |             0.20808  |         0.218946 |               -0.006316 |            1.06112  |     245 |   3119 |

## Lowest skill strata

Shown only for preregistered strata with at least 100 runner rows and 20 races.
These are diagnostic hypotheses, not evidence for changing a feature.

| dimension          | level   |   race_macro_brier |   baseline_race_macro_brier |   brier_skill_vs_field_size_baseline |   calibration_gap_observed_minus_predicted |   calibration_intercept |   calibration_slope |   races |   rows |
|:-------------------|:--------|-------------------:|----------------------------:|-------------------------------------:|-------------------------------------------:|------------------------:|--------------------:|--------:|-------:|
| race_class         | Class3  |           0.15817  |                    0.162695 |                             0.027816 |                                  -0.015317 |               -0.344999 |            0.78611  |      43 |    648 |
| career_starts_band | 11-20   |           0.16192  |                    0.170048 |                             0.0478   |                                  -0.00056  |               -0.15736  |            0.868783 |     328 |   1998 |
| racecourse         | 小倉      |           0.144915 |                    0.153917 |                             0.058481 |                                  -0.010928 |               -0.284507 |            0.831656 |     108 |   1474 |
| field_size_band    | 12-14   |           0.158843 |                    0.171921 |                             0.076074 |                                  -0.016311 |               -0.145918 |            0.955315 |      88 |    989 |
| race_class         | Class2  |           0.155901 |                    0.168948 |                             0.07722  |                                  -0.013715 |               -0.145276 |            0.947919 |      89 |   1262 |
| race_class         | Open    |           0.147917 |                    0.16038  |                             0.077708 |                                  -0.01706  |               -0.094315 |            1.02058  |      61 |    879 |
| career_starts_band | 21+     |           0.104491 |                    0.113336 |                             0.07804  |                                  -0.022774 |               -0.342979 |            0.915053 |     279 |   1117 |
| racecourse         | 中京      |           0.156506 |                    0.170298 |                             0.080983 |                                  -0.00908  |               -0.023804 |            1.0347   |      38 |    495 |
| rest_interval_band | 36-56   |           0.152516 |                    0.166857 |                             0.085947 |                                   0.00264  |                0.010961 |            0.994093 |     370 |   1073 |
| field_size_band    | 17+     |           0.136022 |                    0.149486 |                             0.09007  |                                  -0.006679 |               -0.056153 |            0.994765 |     122 |   1841 |
| rest_interval_band | 57-90   |           0.149389 |                    0.165543 |                             0.097579 |                                  -0.009638 |               -0.02554  |            1.03544  |     409 |   1263 |
| race_class         | Class1  |           0.152973 |                    0.169975 |                             0.100029 |                                  -0.010515 |               -0.097402 |            0.974774 |     158 |   2155 |

## Audit

- evaluation rows: 6,313
- evaluation races: 495
- common evaluation-row SHA256: a72fa10cc9adcd1ceee4c5b5eacd0dfdcdfc33fc984bf0b7866ec70b617bfd85
- 2025 rows loaded: **no**
- market fields loaded: **no**
- target-race outcome loaded: **no**

The next phase may use these diagnostics to motivate hypotheses, but its candidate registry
and tuning ranges must be frozen before Phase-2 model-family scores are generated.
