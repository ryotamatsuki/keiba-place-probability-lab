# Probability Improvement V1 — historical final-odds research

No 10-minute-before-off market-performance claim; no production promotion.

## A — common 2023/2024 development rows

| name          |   race_macro_brier |   race_macro_log_loss |   runner_micro_brier |   runner_micro_log_loss |   races |   rows | point_brier_best   |   mean_brier_delta_vs_best |   se_brier_delta_vs_best |   mean_log_loss_delta_vs_best |   se_log_loss_delta_vs_best | within_one_se_brier   | within_one_se_log_loss   |
|:--------------|-------------------:|----------------------:|---------------------:|------------------------:|--------:|-------:|:-------------------|---------------------------:|-------------------------:|------------------------------:|----------------------------:|:----------------------|:-------------------------|
| A1_calibrated |           0.137963 |              0.428777 |             0.134951 |                0.422833 |     495 |   6313 | True               |                0           |              0           |                   0           |                 0           | True                  | True                     |
| A2_corrected  |           0.137978 |              0.428862 |             0.13494  |                0.422821 |     495 |   6313 | False              |                1.44776e-05 |              0.000116354 |                   8.56602e-05 |                 0.000260616 | True                  | True                     |
| A0_raw        |           0.138939 |              0.43218  |             0.135953 |                0.426728 |     495 |   6313 | False              |                0.000975244 |              0.000477445 |                   0.00340341  |                 0.00143864  | False                 | False                    |

## B — 2024 diagnostic only, frozen earlier Stage4 V2 OOF

| name           |   race_macro_brier |   race_macro_log_loss |   runner_micro_brier |   runner_micro_log_loss |   races |   rows | point_brier_best   |   mean_brier_delta_vs_best |   se_brier_delta_vs_best |   mean_log_loss_delta_vs_best |   se_log_loss_delta_vs_best | within_one_se_brier   | within_one_se_log_loss   |
|:---------------|-------------------:|----------------------:|---------------------:|------------------------:|--------:|-------:|:-------------------|---------------------------:|-------------------------:|------------------------------:|----------------------------:|:----------------------|:-------------------------|
| B1_logit_stack |           0.137985 |              0.42859  |             0.133649 |                0.419198 |     245 |   3119 | True               |                0           |              0           |                   0           |                 0           | True                  | True                     |
| B0_blend005    |           0.138038 |              0.428755 |             0.133677 |                0.419209 |     245 |   3119 | False              |                5.32023e-05 |              0.000129319 |                   0.000165102 |                 0.000323905 | True                  | True                     |
| B0_blend010    |           0.138048 |              0.428877 |             0.133677 |                0.419275 |     245 |   3119 | False              |                6.31318e-05 |              8.75255e-05 |                   0.000287697 |                 0.000245696 | True                  | False                    |
| B0_market      |           0.13809  |              0.428853 |             0.133732 |                0.419359 |     245 |   3119 | False              |                0.000105185 |              0.000226174 |                   0.000263577 |                 0.000624426 | True                  | True                     |

**B2 and C have now been evaluated.** See B2_manifest.json, C_manifest.json and JOINT_manifest.json. B2 individually clears one SE but the global market-only incumbent remains retained. Stage4 V2 B result is NOT evidence for Scope V3. 2025 has not been reselected. Market-only is retained live.
