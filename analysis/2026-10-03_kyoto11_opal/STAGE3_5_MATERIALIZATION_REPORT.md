# Stage 3.5 — Canonical Feature Materialization

Race: 2026-10-03 Kyoto 11R Opal Stakes  
Status: **COMPLETE / Stage 4 unblocked**

## Canonical matrix

The Stage 4 input matrix is now:

`canonical_nonmarket_features_v1.csv`

It contains 18 runners and zero market / odds / popularity columns.

The previous `nonmarket_features.csv` remains in the repository only as the original Stage 3 audit snapshot.

## Materialized v1 fields

All runner-level core fields were materialized:

- normalized draw position;
- sex and age;
- assigned weight and change from previous start;
- days since previous start;
- distance change and surface change;
- JRA-only career / turf / same-distance / same-course starts and top-three counts;
- field-size-normalized recent-three finish;
- recent-three top-three count;
- recent-three open-plus and graded exposure;
- recent-three relative time-loss measure;
- recent running-position measure.

Race-level context was also fixed:

- field size = 18;
- Kyoto;
- turf;
- 1200m;
- inner course;
- Listed/open;
- handicap race;
- front/forward diagnostic share = 6/18 = 0.333333.

## Definitions

### draw_pct

```text
(horse_no - 1) / (field_size - 1)
```

0 is the innermost horse number and 1 the outermost.

### recent3_finish_pct_mean

For each of the latest three valid starts:

```text
(finish_position - 1) / (field_size - 1)
```

then average. Lower is better.

An exclusion / non-start is skipped. For #13 クラスペディア, the June 13 exclusion is not treated as a finishing result.

### recent3_relative_time_mean

This project's v1 transparent relative-time feature is:

```text
reported_margin_seconds / estimated_winner_time_seconds
```

where

```text
estimated_winner_time_seconds
  = horse_time_seconds - reported_margin_seconds
```

and the metric is averaged over the latest three valid starts.

Lower is better; a win is 0.

This is a project-defined, rights-clean proxy inspired by race-relative timing principles. It is **not claimed to reproduce JRA-VAN's proprietary data-mining variable**. JRA-published times and margins are rounded, so very small margins can appear as 0.0.

### aggregate history

Career / turf / same-distance / same-course history is stored as numerator + denominator counts, not unsmoothed rates. These aggregate tables are JRA-only because the public SportsNavi tables used here state that they cover JRA-hosted races only.

Recent-form rows use the official JRA detailed card's most recent starts as displayed; consequently a recent NAR graded start may appear in recency features even though it is outside the JRA-only aggregate counts. This scope difference is explicit rather than hidden.

## Relative final-3F field

`recent3_relative_last3f_mean` was reviewed but **not materialized for canonical v1**.

Reason: the consolidated official race card provides each horse's own final-3F, but not a common race-level final-3F reference in a form that can be derived consistently for all 54 recent starts without opening and reconstructing every historical race result. Using raw final-3F seconds across different distances / courses would violate the normalization rule that motivated the feature.

Fallback:

- Stage 4 core proceeds without relative final-3F;
- JBIS speed-index features remain available only in the separately labelled sensitivity analysis;
- a later historical pipeline may add race-relative sectional metrics once reproducible source coverage exists.

This is an explicit missing-block decision, not silent imputation.

## Data-quality checks

Python checks before freeze:

- rows: 18;
- missing values in the canonical matrix: 0;
- forbidden market-column matches: 0;
- `draw_pct` range: [0, 1];
- `recent3_finish_pct_mean` range: [0.063889, 0.847059];
- count features are non-negative;
- `front_forward_share = 0.333333` for all runners.

## Leakage check

The JRA detailed race card currently also contains live odds. Those fields were not copied into this matrix.

No variable in `canonical_nonmarket_features_v1.csv` contains:

- odds;
- popularity;
- Stage 2 market probability;
- payout;
- post-race information from the target race.

## Stage 4 gate

PASS.

Stage 4 may now build the transparent non-market P(top3) baseline from this matrix only.
