# Stage 5 — Calibration / Ensemble

Status: **PASS — historical market comparison, calibration, blend selection, and 2025 test completed**

## Timing

This execution occurred after the scheduled start of the 2026-10-03 Kyoto 11R.
It is a blind retrospective reconstruction from inputs frozen before the race, not a
pre-start prediction lock. No target outcome or later/final target odds are loaded.

## Historical market reconstruction

- exact keiba_results.csv SHA256: `fb9345273b21a7c23d41260dc77e45134a1bc2fe756899f3416b24ac0dd51de3`
- market input: historical win odds from the frozen Kaggle v1 source
- reciprocal odds normalized on every starter in each complete race
- top-3 marginal: same Harville / Plackett-Luce method as Stage 2
- historical market probabilities are constructed before Phase-A runner filtering

|   period |   eligible_races |   paired_market_races |   eligible_rows |   paired_rows |   race_coverage |   row_coverage |
|---------:|-----------------:|----------------------:|----------------:|--------------:|----------------:|---------------:|
|     2023 |              250 |                   250 |            3194 |          3194 |        1        |       1        |
|     2024 |              245 |                   245 |            3119 |          3119 |        1        |       1        |
|     2025 |              248 |                   245 |            3100 |          3070 |        0.987903 |       0.990323 |

## Calibration and selection protocol

- 2023: fit logit intercept/slope calibration separately for market and non-market
- 2024: choose convex blend weight on paired rows only
- candidate non-market weight: 0.00 to 1.00 by 0.05
- selection: Brier, then log loss, then smaller non-market weight
- after weight freeze: refit calibration parameters on 2023-2024
- 2025: one-time held-out Stage 5 test; no weight tuning

### 2023 calibration parameters used for 2024 selection

- market: intercept `-0.193566`, slope `0.794662`
- non-market: intercept `-0.100483`, slope `0.975814`

## 2024 selection result

- selected market weight: `0.95`
- selected non-market weight: `0.05`
- selected blend Brier: `0.133691`
- selected blend log loss: `0.419264`

| model                |    brier |   log_loss |   rows |   races |
|:---------------------|---------:|-----------:|-------:|--------:|
| selected_blend       | 0.133691 |   0.419264 |   3119 |     245 |
| market_calibrated    | 0.133732 |   0.419352 |   3119 |     245 |
| market_raw           | 0.134853 |   0.423193 |   3119 |     245 |
| nonmarket_calibrated | 0.145236 |   0.453649 |   3119 |     245 |
| nonmarket_raw        | 0.145339 |   0.454123 |   3119 |     245 |

The selected blend is therefore overwhelmingly market-led. The incremental gain over
calibrated market-only is small and should not be interpreted as evidence that the
non-market component is independently superior.

## 2025 held-out test

- paired races: 245
- paired rows: 3,070
- raw market Brier / log loss: `0.141103` / `0.439789`
- raw non-market Brier / log loss: `0.150388` / `0.467430`
- selected ensemble Brier / log loss: `0.139662` / `0.434544`

| model                |    brier |   log_loss |   rows |   races |
|:---------------------|---------:|-----------:|-------:|--------:|
| selected_blend       | 0.139662 |   0.434544 |   3070 |     245 |
| market_calibrated    | 0.139753 |   0.434701 |   3070 |     245 |
| market_raw           | 0.141103 |   0.439789 |   3070 |     245 |
| nonmarket_calibrated | 0.150325 |   0.467312 |   3070 |     245 |
| nonmarket_raw        | 0.150388 |   0.467430 |   3070 |     245 |

The direction of the small validation gain is preserved in the held-out 2025 test:
the 95/5 blend narrowly improves both Brier and log loss relative to calibrated
market-only. The effect size remains small.

## Target 18-runner Stage 5 output

Exact probability-sum check: **3.000000000000**.

|   rank |   horse_no | horse_name   |   p_market_stage2 |   p_nonmarket_stage4 |   p_ensemble |   uncertainty_low |   uncertainty_high |   component_abs_gap |
|-------:|-----------:|:-------------|------------------:|---------------------:|-------------:|------------------:|-------------------:|--------------------:|
|      1 |         10 | ヒシアイラ        |          0.383753 |             0.370519 |     0.342056 |          0.340497 |           0.373911 |            0.013234 |
|      2 |          6 | リリージョワ       |          0.349399 |             0.247155 |     0.311323 |          0.248073 |           0.314479 |            0.102244 |
|      3 |         18 | ディアナザール      |          0.309414 |             0.352212 |     0.287488 |          0.284162 |           0.355232 |            0.042798 |
|      4 |          3 | タマモイカロス      |          0.302453 |             0.240499 |     0.277096 |          0.241292 |           0.278869 |            0.061954 |
|      5 |          4 | メイショウヨゾラ     |          0.256106 |             0.244005 |     0.243440 |          0.243381 |           0.244864 |            0.012101 |
|      6 |          8 | レッドエヴァンス     |          0.239937 |             0.224102 |     0.230565 |          0.224593 |           0.230860 |            0.015835 |
|      7 |         16 | フロムダスク       |          0.197097 |             0.085672 |     0.192039 |          0.084453 |           0.197141 |            0.111425 |
|      8 |         13 | クラスペディア      |          0.147483 |             0.286066 |     0.162582 |          0.156615 |           0.287743 |            0.138583 |
|      9 |         17 | ヨシノイースター     |          0.146661 |             0.173864 |     0.156725 |          0.155925 |           0.173523 |            0.027203 |
|     10 |          1 | カルチャーデイ      |          0.143462 |             0.106786 |     0.151107 |          0.105679 |           0.153234 |            0.036676 |
|     11 |          5 | フィオライア       |          0.129350 |             0.075792 |     0.138254 |          0.074551 |           0.141222 |            0.053558 |
|     12 |         15 | タガノアラリア      |          0.101390 |             0.188117 |     0.119821 |          0.116619 |           0.187995 |            0.086727 |
|     13 |          9 | タマモブラックタイ    |          0.085251 |             0.052254 |     0.099562 |          0.051060 |           0.101792 |            0.032997 |
|     14 |         14 | ヤマニンアルリフラ    |          0.075660 |             0.093012 |     0.092674 |          0.091822 |           0.092702 |            0.017352 |
|     15 |          2 | ショウナンアビアス    |          0.047025 |             0.086551 |     0.064809 |          0.063858 |           0.085335 |            0.039526 |
|     16 |         12 | デイトナモード      |          0.038865 |             0.035502 |     0.054102 |          0.034455 |           0.054989 |            0.003363 |
|     17 |         11 | オタルエバー       |          0.024623 |             0.062237 |     0.039403 |          0.038413 |           0.061004 |            0.037614 |
|     18 |          7 | テーオーダヴィンチ    |          0.022071 |             0.075657 |     0.036954 |          0.035242 |           0.074416 |            0.053586 |

The uncertainty range is a model-sensitivity range across the selected blend, +/-0.10
non-market weight, and both component endpoints. It is not a confidence interval.

## Critical transport limitation

The historical Kaggle odds are effectively final historical win odds, but the target
market vector is the frozen **08:35** snapshot. Therefore historical test performance
measures blending against a mature market and is not perfectly time-matched to the target.
No later target odds are substituted to remove this mismatch.

## Reproducibility / provenance

- Stage 5 workflow run #2: `37108131492` — SUCCESS
- head SHA: `79a885bf46c1c398e6cabc60ddb22e4cbfc3847e`
- artifact: `stage5-calibration-ensemble-results`
- artifact ID: `11269180077`
- artifact digest: `sha256:ef7b6d93d20fa2b5762fec4329612b35f19ffe035d65230c832e98bea372ac44`
- `stage5_ensemble.csv` SHA256: `560387ab264462af0421109f1f045f9335de3c6e030c91fba8837b0becad365f`
- `stage5_metrics.json` SHA256: `6a82a0f83dada780384415ba2d8f27b24e6fa91e50f6a181dfb219a8795f801d`

## Stage boundary

Stage 5 compares/calibrates/ensembles the frozen Stage 2 and Stage 4 signals.
It does not create a pre-start lock retroactively. Stage 6 must preserve that distinction.
