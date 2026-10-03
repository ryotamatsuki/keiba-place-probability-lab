# Stage 4 Successor Development — Phase 3B Preregistration

Status: **FROZEN BEFORE ANY PHASE-3B OUTER SCORE IS GENERATED**  
Base main: `4d7de3ec9c4f8f405cbfe5a065beaa01a2aebfeb`  
Phase-3A incumbent: `XGB01 + full-field relative ability`  
Common evaluation-row SHA256: `a72fa10cc9adcd1ceee4c5b5eacd0dfdcdfc33fc984bf0b7866ec70b617bfd85`

## Question

Does a small, leakage-safe **recent-trend block** improve the frozen Phase-3A predictor enough
to satisfy the same v2 paired one-SE replacement rule?

Only four trend features are added. XGB01 hyperparameters, the Phase-3A relative-ability block,
population, preprocessing principles and evaluation protocol remain fixed.

## Recent-trend block

The Stage-3.6 panel already contains leakage-safe rolling pre-race summaries. For each horse,
sort all historical starts by `race_date, race_id`, including warmup history. Compare the
current pre-race rolling summary with the same summary stored at the horse's immediately previous
start.

This produces four features:

1. `trend_recent3_finish_improvement`  
   `previous_recent3_finish_pct_mean - current_recent3_finish_pct_mean`  
   Positive means the latest rolling window has improved because lower finish percentile is better.

2. `trend_recent3_time_improvement`  
   `previous_recent3_relative_time_mean - current_recent3_relative_time_mean`  
   Positive means the latest rolling window moved closer to winner time.

3. `trend_recent3_top3_count_change`  
   `current_recent3_top3_count - previous_recent3_top3_count`  
   Positive means more top-3 finishes in the latest 3-start window.

4. `trend_recent4_early_forward_change`  
   `previous_recent4_early_pos_pct_mean - current_recent4_early_pos_pct_mean`  
   Positive means the horse has recently been positioned more forward.

These are changes in already-frozen pre-race rolling statistics. No current-race outcome is used.

## History continuity

Trend construction uses the full frozen horse panel, not only turf-1200 eligible rows:

- warmup 2010-2015,
- train 2016-2022,
- validation 2023-2024.

This ensures the immediately previous start can be found across split/year boundaries and across
surface/distance changes.

2025 is not loaded.

## Missingness

A trend is missing if either the current rolling summary or the previous-start rolling summary is
missing. Missingness is not backfilled from future information. The existing training-only median
imputation with missing indicators is used downstream.

## Frozen model

The challenger keeps Phase-3A exactly intact:

- XGBoost XGB01 hyperparameters unchanged;
- five full-field relative-ability features unchanged;
- add only the four trend features above;
- no hyperparameter tuning;
- no post-hoc calibration.

## Outer evaluation

Exactly the same rows as Phase 1/2/3A:

- outer 2023: fit on eligible 2016-2022, predict 2023;
- outer 2024: fit on eligible 2016-2023, predict 2024.

The concatenated evaluation fingerprint must equal:

`a72fa10cc9adcd1ceee4c5b5eacd0dfdcdfc33fc984bf0b7866ec70b617bfd85`

The incumbent probabilities are the frozen Phase-3A outer predictions.

## Selection

Apply `docs/MODEL_SELECTION_PROTOCOL_V2.md` without modification.

- primary: race-macro Brier;
- paired race differences;
- SE clustered by race date;
- Phase-3A incumbent retained if within one paired clustered SE of the point-Brier-best;
- log loss is a mandatory secondary diagnostic / exact-Brier tie-breaker;
- calibration is diagnostic only.

This is a replacement rule, not a significance test.

## Required outputs

- `phase3b_outer_predictions.csv`
- `phase3b_outer_metrics.csv`
- `phase3b_outer_year_metrics.csv`
- `phase3b_selection_table.csv`
- `phase3b_trend_feature_coverage.csv`
- `phase3b_reliability.csv`
- `phase3b_manifest.json`
- `PHASE3B_RECENT_TREND.md`

If the challenger is adopted, it becomes the incumbent for Phase 3C. Otherwise Phase-3A remains
the incumbent for Phase 3C.
