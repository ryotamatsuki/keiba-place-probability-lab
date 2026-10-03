# Stage 5 Calibration / Ensemble Specification v1

Date: 2026-10-03 JST  
Target: 2026-10-03 Kyoto 11R Opal Stakes  
Status: **FROZEN BEFORE STAGE 5 HISTORICAL EXECUTION**

## Purpose

Stage 5 asks whether the frozen Stage 2 win-market baseline and the Stage 4 non-market model
should be recalibrated and combined. No weight is chosen from the 18 target runners.

## Historical market reconstruction

Use the exact Stage 3.6 primary source:

- `noriyukifurufuru/japan-horse-racing-2010-2025`
- version 1
- `keiba_results.csv` SHA256
  `fb9345273b21a7c23d41260dc77e45134a1bc2fe756899f3416b24ac0dd51de3`

The source `odds` field is historical win odds. For each historical race:

1. exclude official non-starters (`取`, `除`);
2. require a valid win odds value for **every starter**;
3. normalize reciprocal win odds across the complete starter field;
4. derive marginal P(top3) with the same Harville / Plackett-Luce enumeration as Stage 2;
5. only then join to the Stage 4 Phase-A eligible rows.

A race with incomplete starter odds is excluded from paired Stage 5 evaluation. It is never
partially normalized.

## Time split inside the frozen Stage 4 validation period

Stage 4 has already selected the non-market model family/cohort/C without Stage 5 market input.

Stage 5 now uses:

- 2023: calibration-fit subset;
- 2024: blend-selection subset;
- 2025: held-out Stage 5 final test.

The Stage 4 validation model used for 2023/2024 predictions is trained on 2016-2022 only.
The Stage 4 production model used for 2025/target predictions is trained on 2016-2024 only.

## Calibration family

For market and non-market probabilities separately, fit:

`logit(P_cal) = a + b * logit(P_raw)`

This two-parameter logistic recalibration is the only Stage 5 calibration family. It is fitted
on 2023 paired rows for 2024 model selection.

After the blend weight is frozen, the same calibration family is refit on the combined
2023-2024 paired rows and then evaluated once on 2025.

## Ensemble selection

On 2024 paired rows, evaluate convex probability blends:

`P_blend = (1-w) * P_market_cal + w * P_nonmarket_cal`

for `w in {0.00, 0.05, ..., 1.00}`.

Selection order:

1. lowest Brier score;
2. lower log loss;
3. smaller non-market weight as deterministic final tie-break.

Because endpoints are included, Stage 5 is allowed to conclude that no blend is warranted.

## Target transform

Target inputs are frozen:

- market: Stage 2 08:35 Harville P(top3);
- non-market: Stage 4 raw marginal probability from the frozen non-market output.

Apply the 2023-2024 refit calibrators, the frozen 2024-selected blend weight, then one common
race-level logit intercept so the 18 final marginals satisfy exactly:

`sum P(top3) = 3`.

## Sensitivity / uncertainty range

This is a **model sensitivity range**, not a statistical confidence interval.

For each target runner, recompute the race-consistent ensemble at:

- selected `w`;
- `w - 0.10` clipped to [0,1];
- `w + 0.10` clipped to [0,1];
- market-only endpoint;
- non-market-only endpoint.

The minimum/maximum are reported as `uncertainty_low/high`.

## Critical transport limitation

Historical Kaggle odds are effectively final historical win odds, while the target market input is
the frozen 08:35 snapshot. The historical comparison therefore estimates the value of combining
non-market evidence with a mature/final market, not a perfectly time-matched 08:35 market.

This mismatch must be reported. Stage 5 must not fetch or substitute later/final target odds.

## Forbidden target information

No target outcome, final target odds, post-start market snapshot, payout or popularity is loaded.
Stage 5 remains a blind retrospective reconstruction from inputs frozen before the target start.
