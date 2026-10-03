# Scope Expansion Stage 1 — Data Audit and Frozen Comparison Specification

Status: **FROZEN BEFORE A/B/C OUTER SCORES ARE GENERATED**  
Base main: `0f01ad47d40a3ee8b86b5aac7a91c2af319fb0b7`  
Historical panel: `jra_flat_historical_panel_v1.parquet`  
Panel SHA256: `cee9ae9a099f521f12b1bcdd371c25a3d7b9ba3555fbcf5a46f2c9098f59a33d`

## Question

Does broadening the **training population only**, while keeping the frozen Stage-4 v2 model
definition unchanged, improve prediction on the already validated turf-1200 evaluation rows?

This stage deliberately does **not** add new features, retune XGBoost, change eligibility, or
authorize deployment to non-1200 races.

## Scope audit

Audit all JRA flat turf races from 2010-2025 with distance 1000-2600m.

Primary model-development scope excludes straight-course races. Straight-course races are reported
separately and remain out of scope for the A/B/C comparison.

The existing eligibility rules remain frozen:

- surface = turf;
- actual field size >= 8;
- career starts strictly before the race >= 3;
- flat races only;
- race distance inside the candidate training scope.

The historical feature history for every runner remains built from the full chronological horse
history, including starts outside the candidate distance band and outside turf.

The full-field relative-ability context must use **all actual starters** in each selected race,
including runners that are not Stage-A eligible.

## Data-audit outputs

For the 1000-2600m turf population save:

- race count;
- eligible runner-row count;
- actual-starter context-row count;
- distinct race dates;
- counts by year × distance;
- counts by year × racecourse;
- counts by distance × racecourse;
- straight-course counts separately;
- feature missingness for the frozen Stage-4 v2 source fields;
- evaluation-row fingerprints.

Hard QA:

1. 2023-2024 turf-1200 eligible evaluation rows must equal **6,313**.
2. 2023-2024 turf-1200 evaluation races must equal **495**.
3. Their fingerprint must equal
   `a72fa10cc9adcd1ceee4c5b5eacd0dfdcdfc33fc984bf0b7866ec70b617bfd85`.
4. For every selected race, full-field context row count must equal actual `field_size`.
5. No market-like fields may enter the model inputs.
6. No 2025 row may enter the A/B/C development evaluation.

## Frozen candidates

All candidates use exactly the frozen Stage-4 v2 model definition:

- XGBoost XGB01;
- objective `binary:logistic`;
- eval metric `logloss`;
- tree method `hist`;
- 500 estimators;
- learning rate 0.03;
- max depth 3;
- min child weight 20;
- L2 5.0;
- L1 0;
- subsample 0.8;
- colsample_bytree 0.8;
- random seed 20261003;
- empirical-Bayes shrinkage strength 6;
- current base feature set;
- current five full-field relative-ability features;
- no Phase-3B trend block;
- no Phase-3C interaction block;
- no post-hoc calibration;
- no market information;
- no probability sum-to-three adjustment.

Only the **training population** differs.

### A — current successor

Training population:
- turf;
- exactly 1200m;
- non-straight;
- field size >= 8;
- prior starts >= 3.

### B — Turf Sprint

Training population:
- turf;
- 1000-1400m inclusive;
- non-straight;
- field size >= 8;
- prior starts >= 3.

### C — Turf Global

Training population:
- turf;
- 1000-2600m inclusive;
- non-straight;
- field size >= 8;
- prior starts >= 3.

## Common outer evaluation

All A/B/C candidates are evaluated on the **same turf-1200 rows**:

- outer 2023: train 2016-2022, predict 2023;
- outer 2024: train 2016-2023, predict 2024.

The evaluation set is the existing 6,313-row / 495-race turf-1200 development set.

For B and C, the training data include eligible races in their wider distance bands, but prediction
is still only on the common 1200m evaluation rows.

Each fold estimates:
- empirical-Bayes prior mean from that candidate's training side only;
- numeric imputation from that candidate's training side only;
- categorical encoding from that candidate's training side only.

Relative features for both training and evaluation are always computed from the complete actual
starter fields belonging to the selected races.

## Selection rule

Incumbent: **A — current successor**.

Primary metric:
- race-macro Brier.

Secondary:
- race-macro log loss;
- runner-micro Brier / log loss;
- calibration intercept / slope;
- 10-bin ECE / reliability.

Use the frozen paired race-date-clustered one-SE incumbent gate.

A is retained when its Brier is within one paired clustered SE of the point-Brier-best candidate.
A challenger may replace A only if A falls outside that band.

This is a model-selection regularizer, not a statistical-significance claim.

## Interpretation

This stage answers only:

> Does additional training data from nearby or broader turf distances improve 1200m prediction
> under the same frozen model and features?

It does **not** establish that B or C is valid for prediction at 1600m, 2000m, or 2400m.

If B/C fails, the next experiment may investigate distance-suitability features or segmentation.
If B/C succeeds, it becomes a candidate replacement for the 1200m successor only; cross-distance
deployment still requires Stage 5 of the scope-expansion plan.

## Required Stage-1 outputs

Audit:
- `scope_stage1_counts_year_distance.csv`
- `scope_stage1_counts_year_course.csv`
- `scope_stage1_counts_distance_course.csv`
- `scope_stage1_missingness.csv`
- `scope_stage1_context_qa.csv`
- `scope_stage1_audit_manifest.json`
- `SCOPE_EXPANSION_STAGE1_AUDIT.md`

Comparison:
- `scope_stage1_outer_predictions.csv`
- `scope_stage1_outer_metrics.csv`
- `scope_stage1_outer_year_metrics.csv`
- `scope_stage1_selection.csv`
- `scope_stage1_race_losses.csv`
- `scope_stage1_relative_coverage.csv`
- `scope_stage1_reliability.csv`
- `scope_stage1_manifest.json`
- `SCOPE_EXPANSION_STAGE1_REPORT.md`
