# Stage 4 Successor Development — Phase 3C Preregistration

Status: **FROZEN BEFORE ANY PHASE-3C OUTER SCORE IS GENERATED**  
Base main: `11bc3d8def9b6aab24056d69c378a78fac1c8ddc`  
Incumbent: Phase-3A `XGB01 + full-field relative ability`  
Phase-3B recent-trend block: rejected by the frozen one-SE gate  
Common evaluation-row SHA256: `a72fa10cc9adcd1ceee4c5b5eacd0dfdcdfc33fc984bf0b7866ec70b617bfd85`

## Question

Do three small, domain-motivated explicit interaction features improve the frozen Phase-3A
incumbent enough to satisfy the same v2 paired one-SE replacement rule?

XGBoost can already learn interactions. Therefore this phase tests whether making three specific
relationships explicit adds robust predictive information rather than assuming that it will.

## Interaction block

The challenger adds exactly three numeric features to the frozen Phase-3A feature set.

1. `interaction_draw_x_front_tendency`

   `draw_pct * (1 - recent4_early_pos_pct_mean)`

   Higher values represent a more outward draw combined with a stronger recent tendency to race
   forward.

2. `interaction_front_tendency_x_race_pressure`

   `(1 - recent4_early_pos_pct_mean) * front_forward_share`

   Higher values represent a forward-running horse in a race containing a larger share of runners
   with forward-running profiles.

3. `interaction_rest_x_age`

   `log_days_since_prev * age`

   This allows the effect of time since the previous start to differ with age.

No other interaction, ratio, polynomial term, feature selection, or hyperparameter search is
allowed in Phase 3C.

## Frozen incumbent components

The challenger preserves exactly:

- XGBoost XGB01 hyperparameters from Phase 2;
- five Phase-3A full-field relative-ability features;
- Stage-4 original feature set;
- preprocessing and missingness handling;
- no Phase-3B recent-trend features;
- no post-hoc calibration;
- no market data.

## Leakage / missingness

All interaction inputs are already pre-race features in the frozen historical panel or deterministic
functions of them.

- `draw_pct`: target-race entry information.
- `recent4_early_pos_pct_mean`: history strictly before the target race.
- `front_forward_share`: target-race aggregate of pre-race historical running-style summaries.
- `log_days_since_prev`: deterministic transform of pre-race days since prior start.
- `age`: target-race entry information.

If any source is missing, the interaction remains missing and is handled by the existing
training-only median imputation + missing indicator.

## Outer evaluation

Exactly the same rows and folds as Phase 1/2/3A/3B:

- outer 2023: train 2016-2022, predict 2023;
- outer 2024: train 2016-2023, predict 2024.

The concatenated evaluation fingerprint must equal:

`a72fa10cc9adcd1ceee4c5b5eacd0dfdcdfc33fc984bf0b7866ec70b617bfd85`

The incumbent probabilities are the frozen Phase-3A outer predictions.

2025 is prohibited.

## Selection

Apply `docs/MODEL_SELECTION_PROTOCOL_V2.md` without modification.

- primary: race-macro Brier;
- paired race differences;
- SE clustered by race date;
- Phase-3A incumbent retained if within one paired clustered SE of the point-Brier-best;
- log loss is mandatory secondary diagnostic / exact-Brier tie-breaker;
- calibration is diagnostic only.

This is a replacement rule, not a significance test.

## Required outputs

- `phase3c_outer_predictions.csv`
- `phase3c_outer_metrics.csv`
- `phase3c_outer_year_metrics.csv`
- `phase3c_selection_table.csv`
- `phase3c_interaction_feature_coverage.csv`
- `phase3c_reliability.csv`
- `phase3c_manifest.json`
- `PHASE3C_INTERACTIONS.md`

Phase 3 development is complete after this comparison. The winner is then frozen as the
Stage-4 successor specification for subsequent productionization and prospective live testing.
