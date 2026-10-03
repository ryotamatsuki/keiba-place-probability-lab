# Stage 5 v2 — Market / Non-market Ensemble

Status: **PASS — v2 historical-domain selection and known-outcome audit completed**

## Information boundary

- 2023 Stage-4 prediction: trained through 2022 only
- 2024 Stage-4 prediction: trained through 2023 only
- 2025 audit prediction: trained through 2024 only
- production model fitted through 2025 used to back-predict history: **no**
- 2025 is a known-outcome audit, not an untouched test

## 2024 market-only gate

- mandatory incumbent: market_raw
- gate winner: **market_calibrated**
- point-Brier-best: **market_calibrated**
- incumbent retained: **False**
- reason: point-Brier-best selected: no incumbent was supplied or the incumbent fell outside the paired one-SE Brier retention band

| name              |   race_macro_brier |   race_macro_log_loss |   runner_micro_brier |   runner_micro_log_loss |   races |   rows | point_brier_best   |   mean_brier_delta_vs_best |   se_brier_delta_vs_best |   mean_log_loss_delta_vs_best |   se_log_loss_delta_vs_best | within_one_se_brier   | within_one_se_log_loss   |   calibration_intercept |   calibration_slope |   ece_10bin |
|:------------------|-------------------:|----------------------:|---------------------:|------------------------:|--------:|-------:|:-------------------|---------------------------:|-------------------------:|------------------------------:|----------------------------:|:----------------------|:-------------------------|------------------------:|--------------------:|------------:|
| market_calibrated |           0.138085 |              0.428823 |             0.133732 |                0.419352 |     245 |   3119 | True               |                 0          |              0           |                    0          |                  0          | True                  | True                     |                0.014533 |            1.0088   |   0.01526   |
| market_raw        |           0.139486 |              0.432852 |             0.134853 |                0.423193 |     245 |   3119 | False              |                 0.00140105 |              0.000665296 |                    0.00402896 |                  0.00199408 | False                 | False                    |               -0.180565 |            0.801953 |   0.0302803 |

## 2024 blend selection

- market base: **market_calibrated**
- winner under v2 rule: **blend_w0.00**
- point-Brier-best: **blend_w0.05**
- market-only retained: **True**
- selected historical-domain non-market weight: **0.00**
- market-only race-macro Brier: 0.138085
- selected race-macro Brier: 0.138085

| candidate   |   nonmarket_weight |   race_macro_brier |   race_macro_log_loss |   mean_brier_delta_vs_best |   se_brier_delta_vs_best | within_one_se_of_best   |
|:------------|-------------------:|-------------------:|----------------------:|---------------------------:|-------------------------:|:------------------------|
| blend_w0.05 |               0.05 |           0.138036 |              0.428742 |                0           |              0           | True                    |
| blend_w0.10 |               0.1  |           0.13805  |              0.428876 |                1.35789e-05 |              0.000111888 | True                    |
| blend_w0.00 |               0    |           0.138085 |              0.428823 |                4.84784e-05 |              0.000111327 | True                    |
| blend_w0.15 |               0.15 |           0.138125 |              0.429207 |                8.9215e-05  |              0.000224394 | True                    |
| blend_w0.20 |               0.2  |           0.138263 |              0.42972  |                0.000226908 |              0.000337571 | True                    |
| blend_w0.25 |               0.25 |           0.138463 |              0.430407 |                0.000426659 |              0.000451475 | True                    |
| blend_w0.30 |               0.3  |           0.138725 |              0.43126  |                0.000688467 |              0.000566159 | False                   |
| blend_w0.35 |               0.35 |           0.139048 |              0.432273 |                0.00101233  |              0.000681675 | False                   |

## 2025 known-outcome audit

This section is descriptive only and cannot change the selected recipe.

| name                             |   race_macro_brier |   race_macro_log_loss |   runner_micro_brier |   runner_micro_log_loss |   races |   rows |   calibration_intercept |   calibration_slope |   ece_10bin |
|:---------------------------------|-------------------:|----------------------:|---------------------:|------------------------:|--------:|-------:|------------------------:|--------------------:|------------:|
| market_base_frozen_policy        |           0.142938 |              0.442153 |             0.139753 |                0.434701 |     245 |   3070 |              -0.0168408 |            0.970259 |   0.0118166 |
| selected_historical_domain_blend |           0.142938 |              0.442153 |             0.139753 |                0.434701 |     245 |   3070 |              -0.0168408 |            0.970259 |   0.0118166 |
| market_raw                       |           0.143998 |              0.446711 |             0.141103 |                0.439789 |     245 |   3070 |              -0.198087  |            0.774818 |   0.030682  |
| nonmarket_raw_stage4_v2          |           0.153039 |              0.470316 |             0.149349 |                0.462612 |     245 |   3070 |              -0.064792  |            0.965901 |   0.0198651 |
| nonmarket_calibrated             |           0.153069 |              0.470382 |             0.149351 |                0.462658 |     245 |   3070 |              -0.0571085 |            0.944479 |   0.0185595 |

## Frozen 2026 target rehearsal

Canonical live Stage-5 v2 remains **raw market-only** because the historical market data are
final odds while the target market is the 08:35 snapshot. The historical-domain blend is
reported only as a shadow forecast.

|   canonical_rank |   horse_no | horse_name   |   p_market_live_canonical |   p_nonmarket_stage4_v2 |   p_historical_domain_shadow_blend |
|-----------------:|-----------:|:-------------|--------------------------:|------------------------:|-----------------------------------:|
|                1 |         10 | ヒシアイラ        |                  0.383753 |               0.375584  |                          0.362344  |
|                2 |          6 | リリージョワ       |                  0.349399 |               0.271637  |                          0.335506  |
|                3 |         18 | ディアナザール      |                  0.309414 |               0.404602  |                          0.304062  |
|                4 |          3 | タマモイカロス      |                  0.302453 |               0.234899  |                          0.298552  |
|                5 |          4 | メイショウヨゾラ     |                  0.256106 |               0.281141  |                          0.261468  |
|                6 |          8 | レッドエヴァンス     |                  0.239937 |               0.298454  |                          0.248322  |
|                7 |         16 | フロムダスク       |                  0.197097 |               0.0939756 |                          0.212758  |
|                8 |         13 | クラスペディア      |                  0.147483 |               0.373589  |                          0.1697    |
|                9 |         17 | ヨシノイースター     |                  0.146661 |               0.134884  |                          0.168964  |
|               10 |          1 | カルチャーデイ      |                  0.143462 |               0.206193  |                          0.166092  |
|               11 |          5 | フィオライア       |                  0.12935  |               0.081674  |                          0.153255  |
|               12 |         15 | タガノアラリア      |                  0.10139  |               0.247546  |                          0.126865  |
|               13 |          9 | タマモブラックタイ    |                  0.085251 |               0.0370349 |                          0.110899  |
|               14 |         14 | ヤマニンアルリフラ    |                  0.07566  |               0.182752  |                          0.101087  |
|               15 |          2 | ショウナンアビアス    |                  0.047025 |               0.093228  |                          0.0698346 |
|               16 |         12 | デイトナモード      |                  0.038865 |               0.0450126 |                          0.0601888 |
|               17 |         11 | オタルエバー       |                  0.024623 |               0.0556867 |                          0.0421154 |
|               18 |          7 | テーオーダヴィンチ    |                  0.022071 |               0.061342  |                          0.0386514 |

- canonical raw-market sum: 3.000000000
- raw Stage-4 v2 sum: 3.479234352
- historical-domain shadow-blend sum: 3.230662931
- shadow sum-to-three adjustment: **none**

## Operational conclusion

The historical-final-odds result does not by itself authorize a live 08:35 blend. Live
adoption requires repeated pre-race locks of market-only and challenger forecasts under
time-matched snapshots, followed by the preregistered review rule.
