# Stage 3.6 — Historical Training Panel Design

## Objective

Create one row per **horse × race**, with every predictor reconstructed from races strictly earlier
than that row's race date.

The target is:

```text
top3_label = 1 if official finish_position <= 3 else 0
```

For Phase A model fitting, races with fewer than eight starters are excluded so the research target
is aligned with the normal three-place context. The label remains a mathematical top-three label,
not a statement about official place-bet payout rules in small fields.

## Why a historical panel is required

A single 18-runner race cannot identify feature weights or probability calibration. Historical
rows allow us to estimate relationships and evaluate them on chronologically future races.

The key object is:

```text
race_date, race_id, horse_id
    -> information available before that race
    -> top3_label observed after the race
```

The outcome of the target row itself must never enter its predictors.

## Standardized event-row contract

The upstream adapter must produce the fields documented in
`docs/HISTORICAL_RAW_SCHEMA.csv`.

Minimum required concepts:

- race identity / actual date;
- horse identity;
- horse number and field size;
- sex / age / assigned weight;
- racecourse / surface / distance / course layout / class / handicap flag;
- official finishing position;
- race time in seconds;
- first usable corner position;
- open-plus and graded-race indicators.

The builder intentionally does not require target-race odds, popularity or payout.

## Leakage-safe feature construction

Rows are ordered chronologically. Horse-history features use `shift(1)` before rolling or
cumulative calculations.

Examples:

```text
career_top3(t)
 = sum(top3_label before t)

recent3_finish_pct_mean(t)
 = mean(normalized finishing position of the previous <=3 starts)

days_since_prev(t)
 = race_date(t) - previous race_date

recent3_relative_time_mean(t)
 = mean((horse_time - winner_time) / winner_time over previous <=3 starts)
```

The relative-time feature is calculated from each historical race and only becomes available to
later races for the same horse.

Running position is normalized within each historical field:

```text
early_position_pct = (position - 1) / (field_size - 1)
```

The race-level `front_forward_share` for race t is calculated from each entrant's **prior**
running-position history, never the target race's corner positions.

## Canonical training features

The historical builder materializes the Stage 3.5 canonical concepts:

- draw_pct
- sex / age
- assigned weight and change
- days since previous start
- distance / surface change
- career top-three counts with denominators
- turf / same-distance / same-course histories
- recent normalized finish
- recent top-three / class exposure
- recent race-relative time loss
- recent running position
- race-level front/forward share
- race context

The current v1 continues to exclude raw odds, popularity and target-race market information.

## Cold starts

Early career rows naturally lack previous-race features.

Stage 3.6 does not fill those values with fake zeros. It preserves missingness.

For the first Stage 4 experiment, the default cohort requires at least three previous valid starts.
A future model may explicitly support cold starts using separate missingness indicators / pedigree /
trainer / workout blocks.

## Temporal evaluation contract

No random train/test split.

The fixed first-pass split is:

| period | role |
|---|---|
| 2010-2015 | history warm-up |
| 2016-2022 | training |
| 2023-2024 | validation / feature-block and hyperparameter selection |
| 2025 | final untouched historical test |
| 2026+ | future/live application |

All runners from one race remain in the same split by construction.

## Cohort selection

The canonical raw panel is broad enough to support multiple predeclared cohorts.

Stage 4 should compare at least:

- all eligible JRA turf races;
- turf sprint races (1000-1400m);
- exact 1200m turf races.

The cohort is selected using validation-set probability metrics, **not** performance on the
2026-10-03 target race.

## Evaluation gate

Before Stage 4 model fitting, Stage 3.6 must report:

- number of races and runner rows by year;
- duplicate `race_id × horse_id` count;
- rows where previous date is not strictly earlier;
- missingness by canonical feature;
- top3 prevalence by split;
- distribution of field size;
- race-count reconciliation against an official or independently verified reference where feasible.

## Artifacts

Public repository:

- source decision and schema;
- leakage-safe panel builder;
- unit tests;
- panel QA script / report template;
- acquisition manifest.

Local / not committed by default:

- raw historical datasets;
- standardized row-level source table;
- full historical feature panel.
