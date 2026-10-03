# Stage 4 Non-market P(top3) Baseline

Status: **PASS — historical fit / validation / held-out-2025 test completed**

## Timing and interpretation

This Stage 4 run was generated after the scheduled start time of the 2026-10-03 Kyoto 11R.
It is therefore a **blind retrospective reconstruction from the frozen pre-race matrix**, not a
claim that this Stage 4 probability table was locked before the start. No target outcome, final
odds, popularity, payout, or Stage 2 market probability is loaded by the Stage 4 code.

## Frozen model selection

- selected training cohort: `turf_1200`
- selected L2 logistic C: `0.1`
- validation set for every candidate: 2023-2024 turf 1200m (495 races / 6,313 rows)
- validation marginal Brier: `0.147302`
- validation marginal log loss: `0.458217`
- validation 10-bin ECE: `0.017653`
- empirical-Bayes prior: fitting-sample top3 prevalence, strength `6`
- missing values: fitting-sample numeric median + indicators; categorical `UNKNOWN`
- historical validation/test metric: raw marginal P(top3), because Phase-A filters runner rows
- target transform: common logit intercept with exact `sum P(top3)=3`
- target category aliases: `Kyoto -> 京都`, `Listed_open -> Open`

Candidate models were trained on different predeclared cohorts but evaluated on the same
target-like validation population. 2025 was not used for cohort, feature-block, or C selection.
Run #1 was discarded at QA because it imposed a three-slot race constraint on partial historical
race cohorts after the career-start eligibility filter. The correction is structural and does not
use 2025 performance to choose the model.

## Validation grid

| training_cohort       |    C |    brier |   log_loss |
|:----------------------|-----:|---------:|-----------:|
| turf_1200             |  0.1 | 0.147302 |   0.458217 |
| turf_1200             |  1   | 0.147326 |   0.458267 |
| turf_1200             | 10   | 0.147328 |   0.45827  |
| turf_sprint_1000_1400 |  0.1 | 0.147559 |   0.458873 |
| turf_sprint_1000_1400 |  1   | 0.147569 |   0.458885 |
| turf_sprint_1000_1400 | 10   | 0.147569 |   0.458885 |
| all_turf              |  0.1 | 0.147914 |   0.460014 |
| all_turf              |  1   | 0.147925 |   0.460052 |
| all_turf              | 10   | 0.147926 |   0.460054 |

## Held-out 2025 test

After selection, the chosen specification was refit on its 2016-2024 train+validation cohort
and evaluated once on the frozen 2025 turf-1200 test set.

- races: 248
- rows: 3,100
- marginal Brier: `0.150537`
- marginal log loss: `0.467970`
- 10-bin ECE: `0.016938`
- 2025 outcomes used in target fit: **no**

### 2025 reliability bins

|   bin |   rows |   mean_predicted |   observed_top3_rate |   calibration_gap |
|------:|-------:|-----------------:|---------------------:|------------------:|
|     0 |    310 |         0.050644 |             0.051613 |          0.000969 |
|     1 |    310 |         0.085094 |             0.070968 |         -0.014126 |
|     2 |    310 |         0.113249 |             0.151613 |          0.038364 |
|     3 |    310 |         0.141617 |             0.122581 |         -0.019036 |
|     4 |    310 |         0.175639 |             0.13871  |         -0.036929 |
|     5 |    310 |         0.214124 |             0.2      |         -0.014124 |
|     6 |    310 |         0.257693 |             0.26129  |          0.003597 |
|     7 |    310 |         0.30908  |             0.287097 |         -0.021983 |
|     8 |    310 |         0.378751 |             0.36129  |         -0.017461 |
|     9 |    310 |         0.513337 |             0.516129 |          0.002792 |

Race-sum QA is evaluated separately only on 2025 races where every starter survives the
pre-race eligibility filter:

- complete races: 135
- complete-race rows: 1,965
- adjusted Brier: `0.147279`
- adjusted log loss: `0.460704`
- mean race probability sum: `3.000000000000`
- max |race sum - 3|: `8.882e-16`

## Sensitivity — leave one feature block out

| excluded_block   |   validation_brier |   validation_log_loss |   delta_brier_vs_primary |
|:-----------------|-------------------:|----------------------:|-------------------------:|
| entry_condition  |           0.148052 |              0.459944 |                 0.00075  |
| history          |           0.149463 |              0.464489 |                 0.002161 |
| race_context     |           0.149635 |              0.464539 |                 0.002333 |
| recent_form      |           0.151369 |              0.471908 |                 0.004066 |

These ablations are diagnostic. They do not change the primary model after the 2025 test is viewed.

## 18-runner non-market output

Probability sum check: **3.000000000000**.

|   rank |   horse_no | horse_name   |   raw_logit |   raw_probability |   p_top3 |   sensitivity_p_min |   sensitivity_p_max |   sensitivity_rank_best |   sensitivity_rank_worst |
|-------:|-----------:|:-------------|------------:|------------------:|---------:|--------------------:|--------------------:|------------------------:|-------------------------:|
|      1 |         10 | ヒシアイラ        |   -0.505341 |          0.376286 | 0.370519 |            0.322034 |            0.393535 |                       1 |                        3 |
|      2 |         18 | ディアナザール      |   -0.584683 |          0.357856 | 0.352212 |            0.242722 |            0.366042 |                       2 |                        4 |
|      3 |         13 | クラスペディア      |   -0.889922 |          0.291126 | 0.286066 |            0.187019 |            0.334971 |                       2 |                        6 |
|      4 |          6 | リリージョワ       |   -1.0892   |          0.251769 | 0.247155 |            0.199535 |            0.258312 |                       4 |                        8 |
|      5 |          4 | メイショウヨゾラ     |   -1.1062   |          0.248581 | 0.244005 |            0.172972 |            0.29551  |                       3 |                        7 |
|      6 |          3 | タマモイカロス      |   -1.1253   |          0.24503  | 0.240499 |            0.195865 |            0.339681 |                       1 |                        8 |
|      7 |          8 | レッドエヴァンス     |   -1.21727  |          0.228417 | 0.224102 |            0.205199 |            0.269044 |                       3 |                        7 |
|      8 |         15 | タガノアラリア      |   -1.43765  |          0.19191  | 0.188117 |            0.11409  |            0.238208 |                       5 |                       11 |
|      9 |         17 | ヨシノイースター     |   -1.53384  |          0.177433 | 0.173864 |            0.137329 |            0.178783 |                       8 |                        9 |
|     10 |          1 | カルチャーデイ      |   -2.09935  |          0.10916  | 0.106786 |            0.074147 |            0.150222 |                      10 |                       16 |
|     11 |         14 | ヤマニンアルリフラ    |   -2.25276  |          0.095112 | 0.093012 |            0.081667 |            0.143859 |                       9 |                       15 |
|     12 |          2 | ショウナンアビアス    |   -2.33184  |          0.08852  | 0.086551 |            0.071106 |            0.107163 |                      12 |                       15 |
|     13 |         16 | フロムダスク       |   -2.34302  |          0.087622 | 0.085672 |            0.05685  |            0.138282 |                      11 |                       18 |
|     14 |          5 | フィオライア       |   -2.4763   |          0.077536 | 0.075792 |            0.067576 |            0.142064 |                      10 |                       16 |
|     15 |          7 | テーオーダヴィンチ    |   -2.47823  |          0.077399 | 0.075657 |            0.047632 |            0.11416  |                      10 |                       17 |
|     16 |         11 | オタルエバー       |   -2.68789  |          0.063692 | 0.062237 |            0.052763 |            0.087377 |                      14 |                       16 |
|     17 |          9 | タマモブラックタイ    |   -2.87332  |          0.053488 | 0.052254 |            0.045964 |            0.083807 |                      14 |                       17 |
|     18 |         12 | デイトナモード      |   -3.27738  |          0.036356 | 0.035502 |            0.032253 |            0.096548 |                      13 |                       18 |

`p_top3` is the Stage 4 non-market probability after race-level consistency adjustment.
It is not blended with Stage 2 market information; that remains Stage 5 work.

## Largest standardized model coefficients

Positive coefficients increase the raw logistic top-3 logit, holding other encoded variables fixed.
Negative coefficients decrease it. Coefficients are descriptive model parameters, not causal effects.

### Positive

- `categorical__race_class_Maiden`: +0.390643
- `numeric__turf_top3_shrunk`: +0.201483
- `categorical__racecourse_福島`: +0.184157
- `categorical__race_class_Class1`: +0.168081
- `numeric__log_turf_starts`: +0.154803
- `numeric__assigned_weight_kg`: +0.101725
- `numeric__recent3_graded_count`: +0.099461
- `numeric__career_top3_shrunk`: +0.094098
- `categorical__racecourse_小倉`: +0.080037
- `categorical__racecourse_新潟`: +0.064655
- `numeric__same_course_top3_shrunk`: +0.062900
- `numeric__recent3_open_plus_count`: +0.062470

### Negative

- `categorical__race_class_Open`: -0.583136
- `numeric__recent3_finish_pct_mean`: -0.456364
- `numeric__field_size`: -0.204942
- `numeric__recent3_relative_time_mean`: -0.196215
- `numeric__log_same_distance_starts`: -0.180856
- `numeric__log_career_starts`: -0.180141
- `numeric__recent4_early_pos_pct_mean`: -0.158556
- `numeric__recent3_top3_count`: -0.138552
- `numeric__log_days_since_prev`: -0.116736
- `numeric__front_forward_share`: -0.093725
- `categorical__sex_G`: -0.091451
- `categorical__racecourse_札幌`: -0.085667

## Reproducibility

Run `scripts/run_stage4_nonmarket.py` against the frozen Stage 3.6 Actions artifact and
`analysis/2026-10-03_kyoto11_opal/canonical_nonmarket_features_v1.csv`.
The workflow `.github/workflows/stage4-nonmarket.yml` pins the Stage 3.6 artifact run.
