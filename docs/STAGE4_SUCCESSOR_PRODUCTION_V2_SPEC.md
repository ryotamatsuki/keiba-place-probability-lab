# Stage 4 Successor Production Contract v2

Status: **FROZEN BEFORE PRODUCTION REFIT / TARGET SCORING**  
Base main: `a38c932e70c54b8591ac50efb3654fc8d4046eac`  
Development freeze: `docs/STAGE4_SUCCESSOR_FREEZE_V2.md`

## Purpose

Productionize the frozen Stage-4 successor without reopening development selection.

Frozen predictor:

`XGBoost XGB01 + full-field relative-ability block`

This production step may refit the already-frozen specification on all legitimately historical
data available before the prediction date, including 2025. That does not make 2025 an untouched
test and does not alter the development evidence.

## Production fit

For a post-2025 live prediction, fit on the frozen Phase-A turf-1200 eligible population through
the latest completed historical date available before the target race.

For the current reproducibility rehearsal:

- fit period: 2016-2025;
- eligible training population:
  - turf;
  - 1200m exactly;
  - actual field size >= 8;
  - runner career starts before the race >= 3;
- use frozen XGB01 hyperparameters;
- use empirical-Bayes prior strength 6;
- estimate the prior mean from the production fitting sample only;
- include the five frozen full-field leave-one-out relative-ability features;
- no Phase-3B trend block;
- no Phase-3C interaction block;
- no market information;
- no post-hoc calibration.

## Target eligibility

A production Stage-4 probability is emitted only for runners satisfying the frozen target scope:

- surface = turf;
- distance = 1200m;
- active field size >= 8;
- career starts before target race >= 3.

If a target race is outside this scope, the production runner must hard-fail rather than silently
extrapolate.

If only some active runners fail the runner-level eligibility rule, Stage 4 v2 may emit
probabilities only for the eligible subset, but:

- the relative-ability features are still computed against **all active starters**;
- no probability-sum-to-three adjustment is allowed on the eligible subset;
- downstream Stage 5 must know that the non-market component is partial-coverage.

## Scratch / withdrawal contract

The target snapshot distinguishes:

- `declared_field_size`: field size used when official horse numbers were assigned;
- `field_size`: active starters at the Stage-4/Stage-6 lock time.

Historical `draw_pct` is defined as:

`(horse_no - 1) / (declared_field_size - 1)`

Therefore, if a runner scratches before lock:

1. remove the scratched runner from the active full-field context;
2. reduce active `field_size`;
3. preserve official horse numbers;
4. preserve `declared_field_size`;
5. recompute/validate `draw_pct` from horse number and declared field size;
6. recompute all full-field leave-one-out relative features using active starters only.

A scratch after the Stage-6 lock does not mutate the locked prediction file. It is recorded as an
operational event for post-race evaluation.

If `declared_field_size` cannot be reconstructed unambiguously in a scratched field, hard-fail.

## Probability output

The production Stage-4 v2 output is the raw marginal `P(top3)` from the frozen predictor.

Do **not** force the Stage-4 production probabilities to sum to 3. The development comparison was
performed on raw marginal probabilities on the eligible population.

A sum-to-three transformation may be calculated only as a separately labeled diagnostic on a race
where every active starter is Stage-4 eligible. It is not the canonical v2 production probability
and must not be supplied to Stage 5 as though it were the evaluated predictor.

## Required QA

The production runner must verify:

- frozen model specification only;
- no market-like columns;
- no target outcome columns;
- no 2026 target outcome source;
- training dates strictly before target date;
- target race scope matches turf 1200m / field >= 8;
- target active-runner identities are unique;
- active row count equals `field_size`;
- horse numbers are within `declared_field_size`;
- `draw_pct` matches the frozen declared-field formula;
- relative features use all active starters;
- no sum-to-three adjustment is applied to canonical output.

## Reproducibility rehearsal

The frozen 2026-10-03 Kyoto 11R pre-race matrix is used only to prove that the new production
pipeline can train and score end to end.

This is retrospective and must not be described as a genuine pre-start v2 prediction. The target
outcome must not be loaded.

## Outputs

- `stage4_successor_v2.csv`
- `stage4_successor_v2_manifest.json`
- `STAGE4_SUCCESSOR_PRODUCTION_V2.md`
- serialized fitted pipeline as a workflow artifact (not required in git)

## Stage 5 boundary

Stage 5 v2 is a separate step.

Historical Stage-5 evaluation must use genuinely out-of-time Stage-4 v2 predictions for each
evaluation year. It must not score 2023-2024 using a model refit through 2025.

Market-only is the mandatory Stage-5 incumbent/reference. Historical final odds must not be
described as time-matched evidence for an early-morning live market snapshot.
