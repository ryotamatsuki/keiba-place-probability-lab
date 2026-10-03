# Stage 5 v2 — Market / Non-market Ensemble Preregistration

Status: **FROZEN BEFORE STAGE-5 V2 SELECTION SCORES ARE GENERATED**  
Base main: `690a7c4643297a82ec651b302f3c0a2108609773`  
Stage-4 successor: `XGB01 + full-field relative ability`  
Selection protocol: `docs/MODEL_SELECTION_PROTOCOL_V2.md`

## Purpose

Re-evaluate Stage 5 using the frozen Stage-4 successor and the v2 probability-model selection
rule. The historical Stage-5 v1 95/5 result remains an immutable audit result and is not carried
forward as a default.

## Domains

Historical market source:
- complete-field final win odds from frozen Kaggle v1;
- normalized reciprocal win market;
- Harville/Plackett-Luce marginal P(top3);
- market P(top3) constructed before Stage-4 eligibility filtering.

Live target market:
- frozen 2026-10-03 08:35 market snapshot.

These are **not time-matched domains**. A historical-final-odds blend weight may be measured as a
historical-domain result, but it cannot become the canonical live-morning ensemble merely because
it wins historically.

Until time-matched snapshots are accumulated prospectively, the canonical live Stage-5 v2 output
remains **raw market-only**, with the non-market model and any historically selected blend stored
as shadow/challenger forecasts.

## Historical Stage-4 inputs

Every historical Stage-4 prediction used by Stage 5 must be genuinely out of time.

- 2023: frozen Phase-3A outer prediction trained only through 2022;
- 2024: frozen Phase-3A outer prediction trained only through 2023;
- 2025 audit: regenerate the frozen successor trained only through 2024, then predict 2025.

The 2016-2025 production refit must **never** be used to back-predict 2023-2025.

2025 outcomes were already known before this v2 cycle, so 2025 is not an untouched test and cannot
change any Stage-5 v2 selection decision.

## Step 1 — 2023 calibration fit

Fit the same transparent logit intercept+slope calibration family separately to:

- historical final-odds market marginal P(top3);
- frozen Stage-4 successor raw marginal P(top3).

Calibration is fitted on 2023 paired rows only.

No 2023 score is used as an out-of-sample winner criterion.

## Step 2 — 2024 market-only calibration gate

On 2024 paired rows compare:

- `market_raw` — mandatory incumbent;
- `market_calibrated` — 2023-fitted logit calibrator applied to 2024 market probabilities.

Use race-macro Brier and the paired race-date-clustered one-SE incumbent gate.

If calibrated market does not dislodge raw market, `market_raw` is the Stage-5 historical-domain
market-only base. If it does dislodge it, `market_calibrated` is the market-only base.

This prevents calibration from silently degrading the mandatory market reference.

## Step 3 — 2024 blend gate

The non-market component is the frozen Stage-4 successor raw probability after the 2023-fitted
non-market logit calibrator.

Candidate non-market weights are frozen at:

`0.00, 0.05, 0.10, ..., 1.00`

For every weight:

`p_blend = (1-w) * p_market_base + w * p_nonmarket_calibrated`

No race-sum-to-three transform is applied.

The mandatory incumbent is weight 0, i.e. the selected market-only base.

Apply the v2 paired one-SE gate across all weights. A non-zero weight is adopted in the
**historical final-odds domain** only if market-only lies outside one paired date-clustered SE of
the point-Brier-best candidate.

Tie-breakers:
- lower race-macro log loss;
- then lower non-market weight.

## Step 4 — 2025 known-outcome audit

After the 2024 decision is frozen:

- fit the frozen Stage-4 successor on 2016-2024 only;
- predict 2025 out of time;
- refit the already-frozen logit calibration family on 2023-2024 paired rows;
- apply the frozen market-calibration policy from Step 2;
- apply the frozen blend weight from Step 3;
- report 2025 race-macro Brier/log loss/calibration diagnostics.

This is **KNOWN-OUTCOME AUDIT ONLY**. It is not an untouched test and cannot alter the 2024
selection.

## Live 2026 target policy

For the frozen 2026-10-03 retrospective rehearsal:

- `p_market_live_canonical` = frozen 08:35 raw Stage-2 market probability;
- `p_nonmarket_stage4_v2` = frozen Stage-4 v2 production probability;
- `p_historical_domain_shadow_blend` may be reported using the historical-domain recipe;
- canonical Stage-5 live probability remains market-only because final-odds historical calibration
  and blend weights are not time-matched to 08:35.

No target probability is adjusted to sum to 3 except that the Stage-2 market construction itself
naturally sums to 3.

## Required diagnostics

For 2024 selection:
- race-macro Brier / log loss;
- runner-micro Brier / log loss;
- paired Brier deltas and date-clustered SE;
- calibration intercept / slope;
- 10-bin ECE;
- reliability tables;
- common paired-row fingerprint.

For 2025 audit:
- the same metrics, clearly labeled known-outcome/non-confirmatory.

## Required outputs

- `stage5_v2_candidate_registry.csv`
- `stage5_v2_calibration_2023.json`
- `stage5_v2_market_gate_2024.csv`
- `stage5_v2_blend_grid_2024.csv`
- `stage5_v2_selection_2024.csv`
- `stage5_v2_2025_audit.csv`
- `stage5_v2_reliability.csv`
- `stage5_v2_target_shadow.csv`
- `stage5_v2_manifest.json`
- `STAGE5_V2_REPORT.md`

## Interpretation boundary

A historical-final-odds blend clearing the gate does not authorize a live 08:35 blend. The true
live adoption decision requires repeated pre-race locks with time-matched market snapshots and a
predeclared review point.
