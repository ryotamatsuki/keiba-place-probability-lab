# Stage 4 Successor Freeze v2

Status: **DEVELOPMENT FREEZE / QA PASS**  
Selection protocol: `docs/MODEL_SELECTION_PROTOCOL_V2.md`  
Evaluation population: turf 1200m Phase-A eligible runners  
Outer development years: 2023 and 2024  
Evaluation rows: 6,313  
Evaluation races: 495  
Evaluation-row SHA256: `a72fa10cc9adcd1ceee4c5b5eacd0dfdcdfc33fc984bf0b7866ec70b617bfd85`

## Frozen successor

The Stage-4 development winner is:

`XGBoost XGB01 + full-field relative-ability block`

### XGBoost

- objective: `binary:logistic`
- eval metric: `logloss`
- tree method: `hist`
- n_estimators: `500`
- learning_rate: `0.03`
- max_depth: `3`
- min_child_weight: `20`
- reg_lambda: `5.0`
- reg_alpha: `0`
- subsample: `0.8`
- colsample_bytree: `0.8`
- random seed: `20261003`

### Added relative-ability features

Computed leave-one-out against **all actual starters in the same race** before eligible-row
filtering:

1. `rel_career_top3_vs_others`
2. `rel_same_distance_top3_vs_others`
3. `rel_same_course_top3_vs_others`
4. `rel_recent3_finish_vs_others`
5. `rel_recent3_time_vs_others`

The empirical-Bayes shrinkage prior strength remains `6`, with the prior mean estimated from the
training side only.

## Development path

| Phase | Candidate | Race-macro Brier | Decision |
|---|---|---:|---|
| Phase 1 | L2 Logistic C=0.1 incumbent | 0.153096 | diagnostic baseline |
| Phase 2 | XGB01 | 0.151494 | adopted; Logistic outside 1-SE band |
| Phase 3A | XGB01 + relative ability | 0.148959 | adopted; XGB01 outside 1-SE band |
| Phase 3B | + recent-trend block | 0.148674 | rejected; incumbent within 1-SE band |
| Phase 3C | + explicit-interaction block | 0.148768 | rejected; incumbent within 1-SE band |

Phase 3B and 3C have slightly better point estimates, but neither clears the preregistered
complexity/replacement gate. They are not part of the frozen successor.

## Improvement versus original Stage 4 development incumbent

Original Phase-1 Logistic race-macro Brier:
`0.153095887`

Frozen successor race-macro Brier:
`0.148958593`

Absolute improvement:
`0.004137294`

Relative Brier reduction:
approximately `2.70%`.

This is a development-sample comparison under the frozen walk-forward protocol, not a future
performance guarantee.

## Information boundary

Development decisions used no 2025 rows and no target-race outcome. Market fields were excluded.

2025 must not be described as an untouched v2 test because its outcomes were known before this
development program. After this freeze, a production model for a later live race may refit the
**already-frozen specification** using legitimately historical data through the prediction date,
including 2025 where appropriate. Such refitting does not turn 2025 into a test set.

The true next confirmation is prospective:

1. materialize this frozen successor as the production Stage-4 predictor;
2. run the unchanged Stage-4/5/6 pipeline before scheduled post time;
3. commit the Stage-6 lock before the race;
4. evaluate only afterward;
5. accumulate multiple preregistered live races.

## Stage 5 consequence

A stronger non-market Stage-4 model does not imply that a market/non-market ensemble improves.
Stage 5 must be re-evaluated separately with market-only as the mandatory incumbent/reference
under the same v2 replacement philosophy.

Historical final odds are not time-matched to an early-morning live snapshot. Do not use the
historical-final-odds archive to claim that a blend has been optimized for morning odds.

## Frozen exclusions

Not included in the successor:

- Phase-3B recent-trend block;
- Phase-3C explicit interaction block;
- LightGBM;
- jockey/trainer rolling statistics;
- current-day weather/going/body-weight features;
- post-hoc calibration;
- market information.

These may only enter a future development version through a new preregistered experiment.
