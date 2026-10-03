# Stage 4 Baseline Specification v1 — frozen before score calculation

Race: 2026-10-03 Kyoto 11R Opal Stakes  
Purpose: transparent **non-market** structural baseline for P(top3)

## Important limitation

There is not yet a historical training panel large enough to estimate and out-of-sample calibrate coefficients.
Therefore Stage 4 will **not** fit a pseudo-ML model to 18 runners and will not call an arbitrary weighted score
a calibrated probability.

Instead it produces:

1. a transparent non-market evidence score;
2. a Plackett-Luce top-3 probability implied by that score;
3. explicit sensitivity analyses.

The resulting P(top3) is a **structural / uncalibrated baseline**. Calibration and market comparison belong to later stages.

## Frozen scoring blocks

Only variables with an interpretable monotone direction are used in the primary score.
Other materialized variables remain available for future historical estimation but are not assigned ad-hoc signs.

### Block A — recent performance

Equal weight within the block:

- lower `recent3_finish_pct_mean` is better;
- higher `recent3_top3_count` is better;
- lower `recent3_relative_time_mean` is better.

Each feature is converted to a within-race percentile score in [0,1], with 1 = strongest evidence.

### Block B — historical suitability / stability

Use the following numerator / denominator pairs:

- career_top3 / career_starts;
- turf_top3 / turf_starts;
- same_distance_top3 / same_distance_starts;
- same_course_top3 / same_course_starts.

To avoid 1/1 = 100% being treated as certainty, each rate uses the same pre-registered Beta prior:

```text
alpha = 1
beta = 5
prior mean = 1/6
prior strength = 6
posterior_mean = (top3 + 1) / (starts + 6)
```

The prior mean 1/6 equals the unconditional 3-of-18 top-three share in the target race.
This is a transparent shrinkage device, not a claim about a universal horse-racing base rate.

Each posterior mean is then converted to a within-race percentile score and equally averaged.

### Block C — recent class exposure

Equal weight:

- higher `recent3_open_plus_count` is stronger class-exposure evidence;
- higher `recent3_graded_count` is stronger class-exposure evidence.

These are proxies, not causal effects.

## Primary block aggregation

The three blocks receive equal weight:

```text
score = (recent_performance + suitability + class_exposure) / 3
```

No block weight is fitted or adjusted horse by horse.

## Variables deliberately not given a primary-score sign

The following are materialized but excluded from the primary hand-built score because a monotone direction is not
defensible without historical estimation:

- draw_pct;
- sex;
- age;
- assigned_weight_kg;
- days_since_prev;
- distance_change_from_prev_m;
- surface_changed_from_prev;
- recent4_early_pos_pct_mean;
- front_forward_share and other race-level constants.

`assigned_weight_delta_from_prev_kg` is evaluated only in a sensitivity variant, because lower weight is plausibly
helpful but handicap assignment also reflects latent ability and the net coefficient should ultimately be learned.

This exclusion is intentional: Stage 4 prefers omission to inventing unsupported coefficients.

## Probability transform

Let the 18 aggregate scores be standardized within the race:

```text
z_i = (score_i - mean(score)) / sd(score)
strength_i = exp(z_i / tau)
```

Primary `tau = 1.0`.

Normalize strengths to first-place shares and use the same exact Plackett-Luce / Harville top-3 enumeration as Stage 2.
Therefore:

```text
sum_i P_structural(top3_i) = 3
```

This gives a coherent ranking probability transform, but **not empirical calibration**.

## Sensitivity analyses

Pre-registered before output inspection:

1. temperature: tau in {0.75, 1.0, 1.50};
2. leave-one-block-out for each of A/B/C;
3. optional weight-change variant: add a fourth block in which lower assigned-weight change ranks better.

Report:
- rank under primary model;
- rank range across sensitivity runs;
- primary P(top3);
- P(top3) range across temperature settings;
- whether top-three membership is robust to leave-one-block-out.

## Forbidden inputs

Any column containing odds, market probability, popularity or payout is forbidden.
JBIS speed indices are not used in the primary score; they remain a separately labelled sensitivity source.
