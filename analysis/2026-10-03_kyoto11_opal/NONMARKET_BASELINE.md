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
| turf_1200             | 10   | 0.147328 |   0.458270 |
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

Race-sum QA is evaluated separately only on 2025 races where every starter survives the
pre-race eligibility filter:

- complete races: 135
- complete-race rows: 1,965
- adjusted Brier: `0.147279`
- adjusted log loss: `0.460704`
- mean race probability sum: `3.000000000000`
- max |race sum - 3|: `8.882e-16`

## Sensitivity — leave one feature block out

| excluded_block   | validation_brier | validation_log_loss | delta_brier_vs_primary |
|:-----------------|-----------------:|--------------------:|-----------------------:|
| entry_condition  | 0.148052 | 0.459944 | 0.000750 |
| history          | 0.149463 | 0.464489 | 0.002161 |
| race_context     | 0.149635 | 0.464539 | 0.002333 |
| recent_form      | 0.151369 | 0.471908 | 0.004066 |

These ablations are diagnostic. They do not change the primary model after the 2025 test is viewed.

## 18-runner non-market output

Probability sum check: **3.000000000000**.

| rank | horse_no | horse_name | raw_probability | p_top3 | sensitivity rank range |
|---:|---:|:---|---:|---:|:---|
| 1 | 10 | ヒシアイラ | 0.376286 | 0.370519 | 1–3 |
| 2 | 18 | ディアナザール | 0.357856 | 0.352212 | 2–4 |
| 3 | 13 | クラスペディア | 0.291126 | 0.286066 | 2–6 |
| 4 | 6 | リリージョワ | 0.251769 | 0.247155 | 4–8 |
| 5 | 4 | メイショウヨゾラ | 0.248581 | 0.244005 | 3–7 |
| 6 | 3 | タマモイカロス | 0.245030 | 0.240499 | 1–8 |
| 7 | 8 | レッドエヴァンス | 0.228417 | 0.224102 | 3–7 |
| 8 | 15 | タガノアラリア | 0.191910 | 0.188117 | 5–11 |
| 9 | 17 | ヨシノイースター | 0.177433 | 0.173864 | 8–9 |
| 10 | 1 | カルチャーデイ | 0.109160 | 0.106786 | 10–16 |
| 11 | 14 | ヤマニンアルリフラ | 0.095112 | 0.093012 | 9–15 |
| 12 | 2 | ショウナンアビアス | 0.088520 | 0.086551 | 12–15 |
| 13 | 16 | フロムダスク | 0.087622 | 0.085672 | 11–18 |
| 14 | 5 | フィオライア | 0.077536 | 0.075792 | 10–16 |
| 15 | 7 | テーオーダヴィンチ | 0.077399 | 0.075657 | 10–17 |
| 16 | 11 | オタルエバー | 0.063692 | 0.062237 | 14–16 |
| 17 | 9 | タマモブラックタイ | 0.053488 | 0.052254 | 14–17 |
| 18 | 12 | デイトナモード | 0.036356 | 0.035502 | 13–18 |

`p_top3` is the Stage 4 non-market probability after race-level consistency adjustment.
It is not blended with Stage 2 market information; that remains Stage 5 work.

## Reproducibility and provenance

Run `scripts/run_stage4_nonmarket.py` against the frozen Stage 3.6 Actions artifact and
`analysis/2026-10-03_kyoto11_opal/canonical_nonmarket_features_v1.csv`.
The workflow `.github/workflows/stage4-nonmarket.yml` pins the Stage 3.6 artifact run.

Final verified execution: Stage 4 workflow run #10, run id `37106873451`, head
`131240120f5304829b0cf7ae1d7b7bd0aee6d322`, SUCCESS.
Artifact: `stage4-nonmarket-baseline-results`, ID `11267173870`,
digest `sha256:72f395d8a5503cdc8a905053e331904133c33b6b3f9717f75f2ca795120cfcf7`.

The full artifact also contains validation/test calibration tables, validation grid,
sensitivity table, and machine-readable metrics.
