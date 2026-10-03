# Stage 4 Successor Development — Phase 1 Preregistration

Status: **FROZEN BEFORE PHASE-1 DIAGNOSTIC EXECUTION**  
Base main: `46c7d46579677202eaaa1c1ac7016db140c7a456`  
Historical panel: Stage 3.6 freeze v1, SHA256 `cee9ae9a099f521f12b1bcdd371c25a3d7b9ba3555fbcf5a46f2c9098f59a33d`

## Purpose

Develop a successor to the frozen Stage 4 non-market model without using 2025 outcomes for model
selection and without changing multiple scientific components at once.

Development is split into independent phases.

### Phase 1 — incumbent diagnosis (this phase)

Reconstruct genuine out-of-time predictions for the frozen incumbent:

- incumbent: L2 Logistic Regression;
- `C=0.1`;
- current Stage 4 feature allowlist and engineering;
- current common empirical-Bayes shrinkage with prior strength 6;
- turf 1200m eligible cohort;
- raw marginal `P(top3)` only;
- no race-sum-to-three adjustment on runner-filtered historical cohorts;
- no post-hoc probability calibration.

Outer predictions:

- 2023: fit only on 2016-2022, predict 2023;
- 2024: refit only on 2016-2023, predict 2024.

The two outer prediction sets are concatenated and treated as the incumbent development OOF
forecast. No 2025 row is loaded by this Phase-1 runner.

### Phase 2 — model-family comparison (next phase, not executed here)

Feature information is held fixed. Predeclared model families:

1. incumbent L2 Logistic, `C=0.1`;
2. Random Forest;
3. HistGradientBoosting;
4. XGBoost.

LightGBM is deliberately deferred.

Hyperparameters for challengers will be tuned only inside the training side of each outer fold.
The inner validation years are expanding-window and chronological:

- outer 2023: inner validation 2021 and 2022;
- outer 2024: inner validation 2021, 2022 and 2023.

No random K-fold is allowed. No outer-year outcome may tune the model used to predict that year.

Phase 2 winner criterion is the already-frozen
`docs/MODEL_SELECTION_PROTOCOL_V2.md`: race-macro Brier plus the paired,
race-date-clustered 1-SE incumbent-retention gate.

### Phase 3 — feature-block experiments (later phase)

Only after Phase 2 is frozen, test feature blocks one at a time:

- within-race relative ability;
- recent-form change / trend;
- small predeclared interaction block.

Jockey/trainer rolling features require additional historical construction and are deferred until
the existing-panel blocks are exhausted.

## Phase-1 diagnostic questions

The incumbent OOF forecast is audited by:

- year;
- field-size band;
- race class;
- racecourse;
- rest-interval band;
- prior-career-start band.

For every stratum report:

- race-macro Brier;
- runner-micro Brier;
- no-information field-size baseline Brier using `3 / field_size`;
- Brier skill score versus that baseline;
- observed top-3 rate;
- mean predicted probability;
- calibration gap;
- calibration intercept and slope where estimable.

Subgroup scores are diagnostic. They do not select a new feature or model in Phase 1.

## Information boundaries

- 2025 is excluded from development and is not described as an untouched v2 test.
- The 2026-10-03 target outcome is never loaded.
- No market odds, popularity, payout or market-derived probability is loaded.
- The existing Stage 4/5 historical audit files are not rewritten.
- Phase 1 may identify hypotheses for Phase 3, but those hypotheses must be preregistered before
  their evaluation.

## Output

Phase 1 must produce:

- `incumbent_oof_2023_2024.csv`;
- `incumbent_overall_metrics.csv`;
- `incumbent_subgroup_diagnostics.csv`;
- `phase1_manifest.json`;
- `PHASE1_INCUMBENT_DIAGNOSTICS.md`.

