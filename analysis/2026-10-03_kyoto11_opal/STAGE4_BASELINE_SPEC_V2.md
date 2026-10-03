# Stage 4 Baseline Specification v2 — empirical historical model

Race: 2026-10-03 Kyoto 11R Opal Stakes  
Purpose: transparent **non-market** P(top3) baseline  
Status: frozen before Stage 4 historical fitting

## Why v2 exists

The v1 specification was written before Stage 3.6 had a populated historical panel and therefore used a hand-built structural score. Stage 3.6 is now COMPLETE / QA PASS with a 2010–2025 leakage-safe panel. Stage 4 therefore switches, before looking at the target-race outcome, to a simple fitted statistical baseline. v1 remains in the repository for audit history.

## Data and time split

Only the frozen Stage 3.6 artifact is used:

- 2016–2022: model-training period;
- 2023–2024: model/cohort/regularization selection period;
- 2025: untouched final historical test, evaluated once after selection;
- 2026 target race: prediction only; no race outcome, final odds, popularity or payout.

Phase-A eligibility remains turf, field size >= 8, and at least 3 prior career starts.

## Candidate training domains

Three predeclared Stage 3.6 cohorts are compared:

1. all turf;
2. turf 1000–1400m;
3. turf 1200m exactly.

For a fair comparison, **all candidate models are evaluated on the same 2023–2024 turf-1200 validation set**. Thus a wider training cohort does not get scored on an easier or different validation population.

## Model family

Use L2-regularized logistic regression. Candidate `C` values are `{0.1, 1.0, 10.0}`.

Selection criterion:

1. lowest race-consistent validation Brier score;
2. validation log loss as tie-breaker;
3. if still tied, smaller `C`.

No tree ensemble or opaque nonlinear model is eligible in Stage 4.

## Feature handling

Only Stage 3/3.6 market-free variables common to historical and target matrices may enter.

Historical top-3 counts are not converted to raw unsmoothed rates. For career, turf, same-distance and same-course histories, use one common empirical-Bayes transform:

`posterior = (top3 + p0 * 6) / (starts + 6)`

where `p0` is the top-3 prevalence in the fitting sample only. `log1p(starts)` is retained separately so the model can distinguish evidence strength. The prior strength 6 is fixed before validation.

Numeric missing values use the fitting-sample median plus missing indicators. Categorical missing values map to `UNKNOWN`; one-hot encoding uses `handle_unknown=ignore`. All numeric inputs are standardized using fitting-sample statistics.

`course_layout` and `handicap_indicator` remain excluded because Stage 3.6 found insufficient historical official coverage. `declared_field_size` and `turn_direction` are not used because they are not in the frozen target matrix.

## Probability consistency

The logistic model produces a raw logit and raw marginal probability. Within each race a single common intercept is then added to every horse's logit, solved by bisection so that:

`sum_i P(top3_i) = 3`

This preserves within-race ordering while enforcing the necessary three-slot marginal-probability identity.

## 2025 test and production fit

After cohort and `C` are frozen from 2023–2024 validation, the selected specification is refit on its 2016–2024 train+validation cohort and evaluated once on the 2025 turf-1200 test set. The same 2016–2024 refit is applied to the 18 target runners. 2025 outcomes are **not** used in the target-model fit.

## Sensitivity analysis

With the selected cohort and `C`, four leave-one-block-out models are evaluated on 2023–2024 validation:

- entry/current-condition block;
- historical suitability/shrinkage block;
- recent-form block;
- race-context block.

For the target race, report rank/probability movement across these ablations. This is diagnostic only and cannot change the selected primary model after the 2025 test is viewed.

## Forbidden information

Any column name containing `odds`, `market`, `popularity`, or `payout` causes a hard error. Stage 2 market probabilities are not loaded. The pre-analysis mention of horse 10 is not encoded as a variable, constraint, prior, weight, or selection criterion.
