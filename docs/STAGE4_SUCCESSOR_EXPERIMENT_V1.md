# Stage 4 Successor Development Experiment v1

Status: **PRE-REGISTERED BEFORE EXECUTION**

This experiment develops a successor to the frozen Stage 4 non-market L2-logistic model. It does
not rewrite the historical Stage 4/5 audit and does not use 2025 performance to choose a model.

## Scientific question

Can a market-free `P(top3)` forecaster improve on the current Stage 4 incumbent under the frozen
Model Selection Protocol v2 when model family and a small set of feature-engineering blocks are
changed without using future information?

“Improve” means satisfying the repository's pre-registered selection rule. It does not mean that a
statistical superiority hypothesis has been proven.

## Fixed incumbent

- family: L2 Logistic Regression
- `C = 0.1`
- feature population: current canonical Stage 4 allowlist
- cohort: Phase-A turf 1200 m eligible runners
- empirical-Bayes prior strength: 6
- post-hoc calibration transform: **none**
- historical race-sum transform: **none** on runner-filtered partial race cohorts

The incumbent is refit from past data separately for each outer prediction year. A model trained
through 2024 is never used to predict 2023.

## Challenger families

The compact first comparison contains:

1. current L2 Logistic Regression;
2. Random Forest;
3. HistGradientBoosting;
4. XGBoost.

LightGBM is deferred. Adding another closely related boosting implementation at the first pass
would increase selection multiplicity without answering a distinct scientific question.

Class weights are not used because the task is probability estimation rather than threshold
classification. Internal random validation and early stopping are disabled. Iteration counts are
part of the frozen finite grids, so no candidate can inspect the outer year.

### Logistic
- `C=0.1` only.

### Random Forest
- 400 trees, max depth 8, min leaf 20, sqrt features;
- 400 trees, max depth 12, min leaf 10, 0.7 features;
- 500 trees, unlimited depth, min leaf 40, sqrt features.

All use `criterion=log_loss`, bootstrap, no class weighting, seed 20261003.

### HistGradientBoosting
- 250 iterations, lr 0.05, 15 leaves, min leaf 30, L2 1;
- 300 iterations, lr 0.04, 31 leaves, min leaf 30, L2 2;
- 300 iterations, lr 0.05, 31 leaves, min leaf 60, L2 5.

`early_stopping=False`; seed 20261003.

### XGBoost
- 400 trees, lr 0.03, depth 3, min child 20, lambda 5, alpha 0;
- 350 trees, lr 0.05, depth 4, min child 10, lambda 5, alpha 0;
- 500 trees, lr 0.03, depth 3, min child 10, lambda 10, alpha 0.2.

All use binary logistic loss, histogram trees, subsample 0.9, column subsample 0.9, seed 20261003.

## Model-specific preprocessing

All families receive the same information set, but preprocessing is estimator-appropriate.

- Logistic: median imputation + missingness indicators + standardization; one-hot categorical data.
- RF / HGB / XGBoost: median imputation + missingness indicators without numeric scaling; one-hot
  categorical data.
- unknown categories are ignored safely;
- market variables remain hard forbidden.

## Temporal nested validation

Only 2016-2024 may influence development selection.

### Outer evaluation
- outer 2023: train only on years before 2023;
- outer 2024: train only on years before 2024.

The final comparison concatenates the two sequentially generated outer prediction sets. Every
candidate is evaluated on identical runner rows.

### Inner tuning

For outer 2023:
- validate 2021 from training data ending 2020;
- validate 2022 from training data ending 2021.

For outer 2024:
- validate 2022 from training data ending 2021;
- validate 2023 from training data ending 2022.

Within a family, configuration 0 is the default incumbent and the paired date-clustered 1-SE rule
is used to avoid unnecessary hyperparameter churn. Family choice likewise uses Logistic as the
inner incumbent. No random K-fold split is permitted.

## Feature engineering blocks

Feature blocks use only fields already present in the frozen pre-race historical panel. They are
pre-registered before outer execution.

### Base
The current Stage 4 engineered features.

### Relative ability
For selected pre-race ability/form signals, add within-race centered values and percentile ranks.
Directions are normalized so that a larger transformed value means stronger evidence. Because the
Stage 3.6 Phase-A cohort removes runners with fewer than three prior starts, these are explicitly
**eligible-runner relative features**, not guaranteed full-field ranks.

### Recent deviation
The frozen panel contains recent rolling summaries but not the ordered last-three sequence, so a
true time slope cannot be reconstructed without rebuilding Stage 3.6. Therefore this block is
called `recent_deviation`, not a trend slope. It contains:

- recent-3 top3 rate;
- recent top3 rate minus career shrunk top3 rate;
- recent top3 rate minus turf shrunk top3 rate;
- recent open / graded share;
- absolute distance change;
- absolute assigned-weight change.

### Interactions
A small domain-driven set only:

- draw position × early-position tendency;
- early-position tendency × expected front-forward share;
- draw position × expected front-forward share;
- age × log days since previous start;
- age × assigned-weight change.

### All
Union of the three new blocks.

For each outer fold, model family/config is selected on inner data first. Feature variants are then
compared on the same inner-only predictions, with `base` as incumbent under the paired 1-SE gate.
The outer year remains untouched throughout both decisions.

## Calibration policy

This first successor experiment fixes **no post-hoc probability calibration transform**.
Calibration is measured, not fitted. This isolates model-family and feature-engineering effects and
keeps the current incumbent contract comparable.

Mandatory diagnostics on outer predictions:

- race-macro Brier;
- race-macro log loss;
- ECE (10 equal-frequency bins, descriptive only);
- calibration intercept;
- calibration slope;
- ROC-AUC (diagnostic only).

A separate calibration experiment may be pre-registered after this model/feature experiment and
must use its own temporally separated calibration data.

## Winner rule

The final outer candidate registry is:

- incumbent Logistic base;
- tuned RF base;
- tuned HGB base;
- tuned XGBoost base;
- the fully nested `adaptive_successor` whose family, configuration, and feature variant are
  chosen only from data prior to each outer year.

Primary winner metric: **race-macro Brier**.

The incumbent is retained if its mean paired Brier loss difference from the point-Brier-best is
within one race-date-clustered standard error. Otherwise the point-Brier-best becomes the
development winner. Log loss is a mandatory diagnostic / exact-Brier tie-breaker, not a second
optimization objective.

## Condition-level incumbent diagnostics

Using only the outer 2023-2024 predictions, report incumbent performance by:

- field size;
- race class;
- racecourse;
- layoff interval;
- prior career starts;
- year.

For each group report race-macro Brier, calibration gap, and Brier skill versus an exchangeable
`3 / field_size` forecast. This avoids interpreting raw Brier differences driven mainly by base
rates as model weakness.

## 2025 boundary

The 2025 outcomes and v1 metrics were already observed before this experiment. Therefore:

- 2025 is not evaluated, ranked, or used to choose family/config/features;
- the 2025 split must not be loaded until the development winner is frozen;
- after freeze, 2025 may be added to the production fit as legitimately historical data for a
  later target;
- no 2025 score may appear in the selection report.

The next genuinely untouched evidence remains repeated live races with Stage 6 locked before post
time and Stage 7 performed after the official result.

## 2026-10-03 target handling

After the development decision and production refit, the frozen 2026-10-03 feature matrix may be
scored to verify the successor pipeline. The race outcome must not be loaded. This is a
retrospective application only and cannot count as validation evidence.

Because all 18 target runners are present, a common logit intercept is applied so the final target
marginals satisfy `sum P(top3) = 3` exactly. Historical partial cohorts are never given this
constraint.

## Required artifacts

- outer 2023-2024 row-level predictions;
- inner hyperparameter tables;
- inner family-selection tables;
- inner feature-selection tables;
- final candidate score / 1-SE table;
- calibration / AUC diagnostics;
- subgroup diagnostics;
- development decision and production recipe JSON;
- retrospective target probabilities;
- human-readable development report.
