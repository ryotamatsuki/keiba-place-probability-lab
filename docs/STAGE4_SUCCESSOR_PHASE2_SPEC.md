# Stage 4 Successor Development — Phase 2 Preregistration

Status: **FROZEN BEFORE ANY PHASE-2 CHALLENGER SCORE IS GENERATED**  
Base main: `7506aa0a63500fd9852d4605dd6cd100fbe13906`  
Phase-1 evaluation-row SHA256: `a72fa10cc9adcd1ceee4c5b5eacd0dfdcdfc33fc984bf0b7866ec70b617bfd85`  
Historical panel SHA256: `cee9ae9a099f521f12b1bcdd371c25a3d7b9ba3555fbcf5a46f2c9098f59a33d`

## Scientific question

With **exactly the same Stage-4 information set**, does a nonlinear model family satisfy the
pre-frozen v2 replacement rule against the incumbent L2 logistic model?

Phase 2 changes the model family only. It does not add a feature block, market information,
post-hoc calibration, race-sum adjustment, or 2025 development data.

## Common contract

- population: Stage-3.6 `turf_1200` eligible rows;
- features: current Stage-4 allowlist and feature engineering only;
- common empirical-Bayes history shrinkage: prior strength 6;
- target: binary `top3_label`;
- historical prediction: raw marginal `P(top3)`;
- post-hoc calibration: **none**;
- race-level sum-to-three adjustment: **none** on runner-filtered historical cohorts;
- primary outer metric: race-macro Brier;
- outer replacement rule: paired race-date-clustered 1-SE incumbent gate;
- secondary diagnostic / exact-Brier tie-breaker: race-macro log loss;
- random seed: `20261003`;
- 2025 rows: prohibited.

Tree models use the same engineered information but do not standardize numeric features.
Categorical variables are fitted with the same training-only `UNKNOWN` imputation and one-hot
encoding concept as the incumbent. One-hot output is dense for a common tree-model interface.
All preprocessing is fitted inside the relevant training fold.

## Outer evaluation

Two genuine expanding-window outer forecasts:

- outer 2023: train/refit using 2016-2022 only; predict 2023;
- outer 2024: train/refit using 2016-2023 only; predict 2024.

The outer evaluation rows must reproduce the Phase-1 fingerprint exactly.

## Inner tuning

Challenger tuning uses only years available before the outer year.

For outer 2023:
- inner validation 2021: train 2016-2020;
- inner validation 2022: train 2016-2021.

For outer 2024:
- inner validation 2021: train 2016-2020;
- inner validation 2022: train 2016-2021;
- inner validation 2023: train 2016-2022.

For each hyperparameter specification, concatenate its inner out-of-time predictions and calculate
race-macro Brier.

### Inner one-SE simplification rule

Within each challenger family:

1. identify the hyperparameter specification with minimum inner race-macro Brier;
2. compute paired per-race Brier differences against that best specification;
3. estimate SE clustered by race date;
4. retain all specifications within one paired SE of the best;
5. among those, choose the lowest preregistered `complexity_rank`;
6. then break an exact complexity tie by lower Brier, lower log loss, and stable config ID.

This is tuning regularization, not a significance test.

No random K-fold, random validation split, or model-internal early stopping is allowed.

## Candidate registry

### Incumbent — L2 Logistic Regression

Frozen, not retuned:

```text
C=0.1
penalty=L2
solver=lbfgs
numeric preprocessing=median imputation + missing indicator + StandardScaler
categorical preprocessing=UNKNOWN imputation + one-hot(min_frequency=5)
```

### Random Forest

Fixed:
- `n_estimators=500`
- `criterion=log_loss`
- `bootstrap=true`
- `class_weight=None`
- `n_jobs=2`

Six specifications, ordered from simpler/more regularized to more flexible:

| ID | complexity_rank | max_depth | min_samples_leaf | max_features |
|---|---:|---:|---:|---|
| RF01 | 1 | 5 | 50 | sqrt |
| RF02 | 2 | 5 | 20 | sqrt |
| RF03 | 3 | 8 | 50 | sqrt |
| RF04 | 4 | 8 | 20 | sqrt |
| RF05 | 5 | 8 | 20 | 0.7 |
| RF06 | 6 | None | 30 | 0.7 |

### HistGradientBoosting

Fixed:
- `loss=log_loss`
- `early_stopping=false`
- `max_iter=300`

| ID | complexity_rank | learning_rate | max_leaf_nodes | min_samples_leaf | l2_regularization |
|---|---:|---:|---:|---:|---:|
| HGB01 | 1 | 0.03 | 15 | 50 | 2.0 |
| HGB02 | 2 | 0.03 | 31 | 50 | 2.0 |
| HGB03 | 3 | 0.05 | 15 | 30 | 1.0 |
| HGB04 | 4 | 0.05 | 31 | 30 | 1.0 |
| HGB05 | 5 | 0.08 | 31 | 20 | 1.0 |
| HGB06 | 6 | 0.08 | 63 | 20 | 0.0 |

### XGBoost

Fixed:
- `objective=binary:logistic`
- `eval_metric=logloss`
- `tree_method=hist`
- `n_estimators=500`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `reg_alpha=0`
- `n_jobs=2`
- no early stopping

| ID | complexity_rank | learning_rate | max_depth | min_child_weight | reg_lambda |
|---|---:|---:|---:|---:|---:|
| XGB01 | 1 | 0.03 | 3 | 20 | 5.0 |
| XGB02 | 2 | 0.03 | 4 | 10 | 5.0 |
| XGB03 | 3 | 0.05 | 3 | 10 | 2.0 |
| XGB04 | 4 | 0.05 | 5 | 10 | 2.0 |
| XGB05 | 5 | 0.08 | 4 | 5 | 1.0 |
| XGB06 | 6 | 0.08 | 6 | 5 | 1.0 |

LightGBM is not part of Phase 2 and cannot be added after scores are viewed.

## Outer winner rule

Concatenate 2023 and 2024 outer predictions on the identical 6,313 Phase-1 rows. Then apply
`docs/MODEL_SELECTION_PROTOCOL_V2.md` with:

```text
incumbent = l2_logistic_c0.1
primary = race-macro Brier
replacement = incumbent outside 1 paired date-clustered SE of point-Brier-best
```

The phrase "wins statistically" is prohibited. The correct interpretation is "satisfies the
pre-frozen replacement rule."

## Required diagnostics

For every outer candidate report:

- race-macro Brier and log loss;
- runner-micro Brier and log loss;
- calibration intercept and slope;
- 10-bin ECE and reliability table;
- 2023 and 2024 scores separately;
- paired Brier delta and date-clustered SE versus the point-Brier-best;
- paired delta and date-clustered SE versus the incumbent.

## Required artifacts

- `phase2_candidate_registry.csv`;
- `phase2_inner_tuning.csv`;
- `phase2_selected_configs.csv`;
- `phase2_outer_predictions.csv`;
- `phase2_outer_metrics.csv`;
- `phase2_outer_year_metrics.csv`;
- `phase2_reliability.csv`;
- `phase2_selection_table.csv`;
- `phase2_manifest.json`;
- `PHASE2_MODEL_COMPARISON.md`.

Phase 3 must not begin until Phase 2 is frozen and merged.
