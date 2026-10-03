# Historical Panel QA Report

Status: PASS

All numerical values below are computed by Python from materialized 2010-2025 data.

## Source and date mapping

- source: Japan Horse Racing Data 2010-2025
- license: CC0: Public Domain
- lastUpdated: 2025-12-28T07:41:37.95Z
- version: 1
- mapped race IDs: 55,268 / 55,268 (100.000000%)
- date-map SHA256: f75a44b593f5f9fff3fd49e9d7c1083b30ec550024f29b3b19b61689518a3ce3

The source date column is never used. Every JRA race_id is mapped through the official
JRA annual-results PDF archive at year/course/meeting/meeting-day granularity.

## Final JRA flat historical panel

- races: 53,220
- runner rows: 753,387
- unique horses: 81,900
- actual date range: 2010-01-05 to 2025-12-28
- obstacle races excluded: 2,048
- malformed raw result rows: 217
- malformed JRA result rows: 0

## Annual JRA reconciliation

|   year |   official_race_days |   official_12_race_slots |   source_JRA_races |   unfilled_slots_vs_12_race_cards |   flat_races |   obstacle_races |   runner_rows |
|-------:|---------------------:|-------------------------:|-------------------:|----------------------------------:|-------------:|-----------------:|--------------:|
|   2010 |                  288 |                     3456 |               3454 |                                 2 |         3320 |              134 |         48141 |
|   2011 |                  288 |                     3456 |               3453 |                                 3 |         3331 |              122 |         47389 |
|   2012 |                  288 |                     3456 |               3454 |                                 2 |         3321 |              133 |         48097 |
|   2013 |                  288 |                     3456 |               3454 |                                 2 |         3321 |              133 |         48250 |
|   2014 |                  288 |                     3456 |               3451 |                                 5 |         3323 |              128 |         48522 |
|   2015 |                  288 |                     3456 |               3454 |                                 2 |         3321 |              133 |         48184 |
|   2016 |                  288 |                     3456 |               3454 |                                 2 |         3322 |              132 |         48275 |
|   2017 |                  288 |                     3456 |               3455 |                                 1 |         3328 |              127 |         47563 |
|   2018 |                  288 |                     3456 |               3454 |                                 2 |         3327 |              127 |         46853 |
|   2019 |                  288 |                     3456 |               3452 |                                 4 |         3323 |              129 |         45748 |
|   2020 |                  287 |                     3444 |               3456 |                               -12 |         3332 |              124 |         46578 |
|   2021 |                  288 |                     3456 |               3456 |                                 0 |         3329 |              127 |         46168 |
|   2022 |                  288 |                     3456 |               3456 |                                 0 |         3331 |              125 |         45649 |
|   2023 |                  288 |                     3456 |               3456 |                                 0 |         3329 |              127 |         46039 |
|   2024 |                  288 |                     3456 |               3454 |                                 2 |         3327 |              127 |         45611 |
|   2025 |                  288 |                     3456 |               3455 |                                 1 |         3335 |              120 |         46320 |

official_12_race_slots is a schedule-capacity diagnostic, not a forced race count.
Cancelled or short-card slots are never invented. Every realized source JRA race_id
must map to an official venue-day; details are in HISTORICAL_JRA_DAY_RECONCILIATION.csv.

## Temporal split

| split | races | runner rows | top3 prevalence |
|---|---:|---:|---:|
| warmup | 19,937 | 288,583 | 0.207459 |
| train | 23,292 | 326,834 | 0.213962 |
| validation | 6,656 | 91,650 | 0.218058 |
| test | 3,335 | 46,320 | 0.216213 |

## Leakage and structural QA

- chronology_violations: 0
- duplicate_race_horse_keys: 0
- duplicate_target_rows: 0
- market_leakage_columns: 0
- future_year_rows: 0
- same_race_predictor_contamination: 0
- same_race_pace_contamination: 0
- unresolved_date_mappings: 0
- malformed_JRA_rows: 0
- unclassified_JRA_races: 0

## Feature coverage and historical-v1.1 decision

| feature | known | total | coverage |
|---|---:|---:|---:|
| race_class | 752,334 | 753,387 | 99.860231% |
| course_layout | 96,144 | 753,387 | 12.761569% |
| handicap_indicator | 424,527 | 753,387 | 56.349127% |
| is_open_plus | 752,334 | 753,387 | 99.860231% |
| is_graded | 750,686 | 753,387 | 99.641486% |

course_layout and handicap_indicator are retained as audit columns but excluded from
the Stage 4 historical model allowlist because official-PDF extraction is incomplete
across the full period. No fake/default value is imputed. The separate 18-runner
current-race matrix remains unchanged.

## Missingness

| feature                            |   missing_count |   missing_rate |
|:-----------------------------------|----------------:|---------------:|
| draw_pct                           |               0 |     0          |
| sex                                |               0 |     0          |
| age                                |               0 |     0          |
| assigned_weight_kg                 |               0 |     0          |
| assigned_weight_delta_from_prev_kg |           81900 |     0.108709   |
| days_since_prev                    |           81900 |     0.108709   |
| distance_change_from_prev_m        |           81900 |     0.108709   |
| surface_changed_from_prev          |           81900 |     0.108709   |
| career_starts                      |               0 |     0          |
| career_top3                        |               0 |     0          |
| turf_starts                        |               0 |     0          |
| turf_top3                          |               0 |     0          |
| same_distance_starts               |               0 |     0          |
| same_distance_top3                 |               0 |     0          |
| same_course_starts                 |               0 |     0          |
| same_course_top3                   |               0 |     0          |
| recent3_finish_pct_mean            |           82118 |     0.108998   |
| recent3_top3_count                 |           81900 |     0.108709   |
| recent3_open_plus_count            |           81903 |     0.108713   |
| recent3_graded_count               |           82069 |     0.108933   |
| recent3_relative_time_mean         |           82118 |     0.108998   |
| recent4_early_pos_pct_mean         |           81979 |     0.108814   |
| front_forward_share                |           64154 |     0.0851541  |
| field_size                         |               0 |     0          |
| declared_field_size                |               0 |     0          |
| racecourse                         |               0 |     0          |
| surface                            |               0 |     0          |
| distance_m                         |               0 |     0          |
| turn_direction                     |               0 |     0          |
| race_class                         |            1053 |     0.00139769 |
| course_layout                      |          657243 |     0.872384   |
| handicap_indicator                 |          328860 |     0.436509   |

## Finish-status policy

- 取 / 除: non-starters; excluded from entrant/start history.
- 中 / 失: started; career start counted, top3_label=0, placing/time-derived values missing.
- numeric ranks including (降) / (再): numeric official placing prefix is used.

## Obstacle classification

Official PDF race-condition classification is used where machine-readable. Else source
race metadata is combined with obstacle-name markers and the source winner-last3F
encoding below 20. That fallback is only a race-type classifier, never a predictor,
and any official/fallback disagreement is a hard failure.

## Final artifact

- file: jra_flat_historical_panel_v1.parquet
- SHA256: cee9ae9a099f521f12b1bcdd371c25a3d7b9ba3555fbcf5a46f2c9098f59a33d
- bytes: 21,786,362
- Actions artifact directory: data/historical_processed/stage36_freeze_v1/

2025 is an untouched final historical test. No 2025 outcome is used for feature selection,
hyperparameter tuning, or cohort selection. No 2026 race outcome is present.

