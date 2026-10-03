# Stage 4 Successor Development — Phase 3A Preregistration

Status: **FROZEN BEFORE ANY PHASE-3A OUTER SCORE IS GENERATED**  
Base main: `c34a00f320e44efc6c85dc2e1c6d77b83ba49a4f`  
Phase-2 winner: `xgboost / XGB01`  
Phase-2 evaluation-row SHA256: `a72fa10cc9adcd1ceee4c5b5eacd0dfdcdfc33fc984bf0b7866ec70b617bfd85`

## Question

Does adding a small, pre-race **within-race relative ability block** improve the frozen Phase-2
XGBoost predictor enough to satisfy the same v2 paired one-SE replacement rule?

Only the feature information changes. Model family and hyperparameters remain fixed.

## Frozen incumbent

XGBoost XGB01 from Phase 2:

- `learning_rate=0.03`
- `max_depth=3`
- `min_child_weight=20`
- `reg_lambda=5.0`
- `n_estimators=500`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `reg_alpha=0`
- `objective=binary:logistic`
- `eval_metric=logloss`
- `tree_method=hist`
- seed `20261003`

No hyperparameter tuning is permitted in Phase 3A.

## Relative-ability block

Five numeric features are added. Each is a leave-one-out difference between the runner and the
other **actual starters in the same race**, using only pre-race variables.

Positive values always mean "stronger than the other starters" under the source feature.

1. `rel_career_top3_vs_others`  
   `career_top3_shrunk_i - mean(career_top3_shrunk_others)`

2. `rel_same_distance_top3_vs_others`  
   `same_distance_top3_shrunk_i - mean(same_distance_top3_shrunk_others)`

3. `rel_same_course_top3_vs_others`  
   `same_course_top3_shrunk_i - mean(same_course_top3_shrunk_others)`

4. `rel_recent3_finish_vs_others`  
   `mean(recent3_finish_pct_mean_others) - recent3_finish_pct_mean_i`  
   Lower source values are better, so the sign is reversed.

5. `rel_recent3_time_vs_others`  
   `mean(recent3_relative_time_mean_others) - recent3_relative_time_mean_i`  
   Lower source values are better, so the sign is reversed.

### Full-field context rule

The relative block **must not** be computed only from the Phase-A eligible runners. The frozen
`jra_flat_historical_panel_v1.parquet` is used to recover all starters in every selected race.
The eligible Phase-A row receives its relative feature after the all-starter context is computed.

Hard QA:

- each selected race must have exactly `field_size` context rows;
- no top3 label, finish position, market field, payout, or post-race variable may enter a relative feature;
- shrinkage prior is estimated from the training side only and then applied to both training and
  evaluation context;
- missing source feature values remain missing when a valid leave-one-out peer mean cannot be
  formed; downstream training-only imputation handles them.

## Outer evaluation

Exactly the same outer folds and rows as Phase 2:

- outer 2023: train 2016-2022, predict 2023;
- outer 2024: train 2016-2023, predict 2024.

The concatenated evaluation fingerprint must equal:

`a72fa10cc9adcd1ceee4c5b5eacd0dfdcdfc33fc984bf0b7866ec70b617bfd85`

2025 is prohibited.

## Selection

Incumbent probabilities are the frozen Phase-2 XGBoost outer predictions.

Challenger:
`xgboost_xgb01_plus_relative_ability`

Apply `docs/MODEL_SELECTION_PROTOCOL_V2.md` without modification:

- primary: race-macro Brier;
- paired by race;
- SE clustered by race date;
- incumbent retained if within one paired clustered SE of the point-Brier-best;
- log loss is mandatory secondary diagnostic / exact-Brier tie-breaker;
- calibration is diagnostic only.

This is a replacement rule, not a significance test.

## Required outputs

- `phase3a_outer_predictions.csv`
- `phase3a_outer_metrics.csv`
- `phase3a_outer_year_metrics.csv`
- `phase3a_selection_table.csv`
- `phase3a_relative_feature_coverage.csv`
- `phase3a_reliability.csv`
- `phase3a_manifest.json`
- `PHASE3A_RELATIVE_ABILITY.md`

If the challenger is adopted, it becomes the incumbent for Phase 3B. If not, Phase-2 XGB01
remains the incumbent for Phase 3B.
