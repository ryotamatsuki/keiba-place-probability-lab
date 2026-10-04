# Scope V3 strict OOF SHAP audit

Importance uses XGBoost contribution values in log-odds space on the frozen 2023-2024 outer folds.
No 2025 outcomes or current-race outcomes are used. Routed predictions must reproduce the stored OOF output within 1e-7.

Prediction parity max absolute error: 9.89e-17.

## All routed turf 1000-2600m

### Source features

| name                           |   mean_abs_shap_logodds | share_of_total_abs   |
|:-------------------------------|------------------------:|:---------------------|
| rel_recent3_finish_vs_others   |                0.341301 | 16.6%                |
| rel_career_top3_vs_others      |                0.251924 | 12.3%                |
| field_size                     |                0.212693 | 10.3%                |
| rel_recent3_time_vs_others     |                0.182051 | 8.9%                 |
| recent3_graded_count           |                0.086344 | 4.2%                 |
| recent3_relative_time_mean     |                0.075801 | 3.7%                 |
| recent4_early_pos_pct_mean     |                0.0727   | 3.5%                 |
| turf_top3_shrunk               |                0.071813 | 3.5%                 |
| log_career_starts              |                0.067797 | 3.3%                 |
| log_days_since_prev            |                0.063117 | 3.1%                 |
| rel_same_course_top3_vs_others |                0.056514 | 2.7%                 |
| draw_pct                       |                0.052612 | 2.6%                 |
| recent3_finish_pct_mean        |                0.05217  | 2.5%                 |
| distance_change_from_prev_m    |                0.048626 | 2.4%                 |
| race_class                     |                0.046387 | 2.3%                 |

### Feature families

| name                     |   mean_abs_shap_logodds | share_of_total_abs   |
|:-------------------------|------------------------:|:---------------------|
| relative_ability         |                0.672071 | 49.4%                |
| race_context             |                0.228018 | 16.7%                |
| recent_form              |                0.169533 | 12.5%                |
| entry_condition          |                0.144892 | 10.6%                |
| history                  |                0.118711 | 8.7%                 |
| distance_regime          |                0.020128 | 1.5%                 |
| distance_near            |                0.005711 | 0.4%                 |
| distance_course_distance |                0.002387 | 0.2%                 |

## Turf 1000-1400m except 1200m

### Source features

| name                           |   mean_abs_shap_logodds | share_of_total_abs   |
|:-------------------------------|------------------------:|:---------------------|
| rel_recent3_finish_vs_others   |                0.37177  | 18.4%                |
| rel_career_top3_vs_others      |                0.24548  | 12.1%                |
| field_size                     |                0.229567 | 11.3%                |
| rel_recent3_time_vs_others     |                0.168325 | 8.3%                 |
| recent3_graded_count           |                0.084218 | 4.2%                 |
| recent3_relative_time_mean     |                0.07978  | 3.9%                 |
| log_career_starts              |                0.074295 | 3.7%                 |
| turf_top3_shrunk               |                0.067587 | 3.3%                 |
| recent4_early_pos_pct_mean     |                0.062052 | 3.1%                 |
| rel_same_course_top3_vs_others |                0.060964 | 3.0%                 |
| race_class                     |                0.052698 | 2.6%                 |
| log_days_since_prev            |                0.051969 | 2.6%                 |
| distance_change_from_prev_m    |                0.05134  | 2.5%                 |
| recent3_top3_count             |                0.04847  | 2.4%                 |
| draw_pct                       |                0.048408 | 2.4%                 |

### Feature families

| name             |   mean_abs_shap_logodds | share_of_total_abs   |
|:-----------------|------------------------:|:---------------------|
| relative_ability |                0.687425 | 51.3%                |
| race_context     |                0.248581 | 18.5%                |
| recent_form      |                0.149903 | 11.2%                |
| entry_condition  |                0.135308 | 10.1%                |
| history          |                0.119457 | 8.9%                 |

## Turf exactly 1200m

### Source features

| name                           |   mean_abs_shap_logodds | share_of_total_abs   |
|:-------------------------------|------------------------:|:---------------------|
| rel_recent3_finish_vs_others   |                0.281581 | 14.2%                |
| rel_recent3_time_vs_others     |                0.211778 | 10.7%                |
| rel_career_top3_vs_others      |                0.19789  | 10.0%                |
| field_size                     |                0.141027 | 7.1%                 |
| recent4_early_pos_pct_mean     |                0.122533 | 6.2%                 |
| turf_top3_shrunk               |                0.093836 | 4.7%                 |
| recent3_graded_count           |                0.085236 | 4.3%                 |
| log_days_since_prev            |                0.079026 | 4.0%                 |
| recent3_finish_pct_mean        |                0.069603 | 3.5%                 |
| draw_pct                       |                0.068486 | 3.4%                 |
| log_career_starts              |                0.054097 | 2.7%                 |
| recent3_relative_time_mean     |                0.05254  | 2.6%                 |
| rel_same_course_top3_vs_others |                0.051086 | 2.6%                 |
| distance_change_from_prev_m    |                0.044984 | 2.3%                 |
| assigned_weight_kg             |                0.040587 | 2.0%                 |

### Feature families

| name             |   mean_abs_shap_logodds | share_of_total_abs   |
|:-----------------|------------------------:|:---------------------|
| relative_ability |                0.587675 | 47.6%                |
| recent_form      |                0.182109 | 14.8%                |
| entry_condition  |                0.155934 | 12.6%                |
| race_context     |                0.153007 | 12.4%                |
| history          |                0.106637 | 8.6%                 |
| distance_regime  |                0.049278 | 4.0%                 |

## Turf 1401-2000m

### Source features

| name                           |   mean_abs_shap_logodds | share_of_total_abs   |
|:-------------------------------|------------------------:|:---------------------|
| rel_recent3_finish_vs_others   |                0.371692 | 18.5%                |
| rel_career_top3_vs_others      |                0.248518 | 12.4%                |
| field_size                     |                0.221944 | 11.0%                |
| rel_recent3_time_vs_others     |                0.160113 | 8.0%                 |
| recent3_graded_count           |                0.095212 | 4.7%                 |
| recent3_relative_time_mean     |                0.081583 | 4.1%                 |
| log_career_starts              |                0.071525 | 3.6%                 |
| turf_top3_shrunk               |                0.070451 | 3.5%                 |
| recent4_early_pos_pct_mean     |                0.059757 | 3.0%                 |
| rel_same_course_top3_vs_others |                0.056934 | 2.8%                 |
| race_class                     |                0.054498 | 2.7%                 |
| distance_change_from_prev_m    |                0.053236 | 2.6%                 |
| log_days_since_prev            |                0.052683 | 2.6%                 |
| recent3_top3_count             |                0.047943 | 2.4%                 |
| draw_pct                       |                0.04693  | 2.3%                 |

### Feature families

| name             |   mean_abs_shap_logodds | share_of_total_abs   |
|:-----------------|------------------------:|:---------------------|
| relative_ability |                0.686105 | 50.8%                |
| race_context     |                0.238463 | 17.6%                |
| recent_form      |                0.16678  | 12.3%                |
| entry_condition  |                0.141871 | 10.5%                |
| history          |                0.118219 | 8.7%                 |

## Turf 2001-2600m

### Source features

| name                           |   mean_abs_shap_logodds | share_of_total_abs   |
|:-------------------------------|------------------------:|:---------------------|
| rel_career_top3_vs_others      |                0.35758  | 15.0%                |
| rel_recent3_finish_vs_others   |                0.292535 | 12.3%                |
| field_size                     |                0.277812 | 11.7%                |
| rel_recent3_time_vs_others     |                0.232123 | 9.7%                 |
| log_days_since_prev            |                0.087813 | 3.7%                 |
| recent3_relative_time_mean     |                0.087139 | 3.7%                 |
| log_career_starts              |                0.06985  | 2.9%                 |
| rel_same_course_top3_vs_others |                0.06     | 2.5%                 |
| recent3_finish_pct_mean        |                0.056254 | 2.4%                 |
| age                            |                0.055663 | 2.3%                 |
| recent3_graded_count           |                0.054735 | 2.3%                 |
| log_turf_starts                |                0.053328 | 2.2%                 |
| draw_pct                       |                0.052947 | 2.2%                 |
| recent4_early_pos_pct_mean     |                0.052316 | 2.2%                 |
| recent3_top3_count             |                0.045974 | 1.9%                 |

### Feature families

| name                     |   mean_abs_shap_logodds | share_of_total_abs   |
|:-------------------------|------------------------:|:---------------------|
| relative_ability         |                0.73997  | 45.6%                |
| race_context             |                0.290813 | 17.9%                |
| recent_form              |                0.176025 | 10.9%                |
| entry_condition          |                0.146808 | 9.1%                 |
| history                  |                0.139492 | 8.6%                 |
| distance_regime          |                0.069133 | 4.3%                 |
| distance_near            |                0.042124 | 2.6%                 |
| distance_course_distance |                0.017607 | 1.1%                 |
