# B_raw-centred ensemble V1: frozen candidate registry before calculating new scores

Status: PRE-REGISTERED. Parent: PR #34 `research/input-learning-v1-20261031` at `87650e55e3eccf98887a3992a4ad9d99a07b7193`. This experiment is independent of the historical frozen selections and the production model.

## Objective and incumbent

Test whether ensembles with the actual previous strongest challenger **B_raw** (F3 information, depth 2 XGBoost, 400 trees selected strictly inside earlier years) improve race-macro Brier compared with B_raw on the identical horse/race/date sample. The incumbent for selection is **B_raw**, not the older X2 or E5.

In the historical 2024 development sample B_raw Brier was previously seen (0.135721276). The known outcome is NOT untouched or prospective. Interpret one-SE as the existing development criterion, not significance. Stage5 production market-only is unchanged. Historical odds are FINAL, not T−10 or fixed pre-post snapshots.

All parameters, candidate names, learning boundaries and data sources below are fixed in this file, which MUST be committed before computing any new ensemble score. No posthoc alpha/weight/mixture candidate changes.

## Original immutable inputs and integrity checks

Required all original out-of-time (OOF) predictions from PR #32/#34:
- `analysis/input_learning_v1/B_outer_oof_predictions.csv`: `p_M0`, `p_B_raw` and `p_B_scaled`, evaluated on 2023/2024. B_raw for 2023 used boosting rounds selected with inner 2021/2022 only; 2024 used inner 2021/2022/2023 only (both 400 trees). Its separate postfit alpha is NOT multiplied into these ensembles.
- `analysis/input_learning_v1/C_outer_oof_predictions.csv`: `p_LightGBM_C1`, `p_CatBoost_C1`, whose 2023 predictions used only 2018–2022 model fit; 2024 predictions used 2018–2023.
- `analysis/information_expansion_v1/outer_oof_predictions.csv`: original M0 and X2, full 2023+2024 key fingerprint `cb984b2553647d423e05742d0625988da3129dab3183f79b6fb84c74776a582f`.
- `analysis/model_ensemble_v1/2024_predictions.csv`: immutable M0/X2/E5 2024 comparator.
- `analysis/input_learning_v1/C2_2024_oof_predictions.csv`: LightGBM_scaled and CatBoost_scaled, each separately scaled using only 2023 OOF.
- `analysis/input_learning_v1/D2_2024_oof_predictions.csv`: previous X2-anchored regularized ensemble D3 as a fixed historical challenger.

Treat `race_date, race_id, horse_id, horse_no` as exact canonical composite key; every source must have identical keys on both years or its stated 2024 subset, binary target agreement and same calibrated market probability to `atol=1e-10`, no duplicate rows or invalid candidate probabilities. The 2024 subset has exactly 2,844 races/31,225 horse rows; the joint 2023–24 outer OOF has exactly 5,731 races/63,276 rows. Record SHA256 per source and sorted-row fingerprints. Original C1 and B_raw are out-of-time but still HISTORICAL development, not a new unseen test.

## Preregistered candidate predictions (on common 2024 rows)

All use existing MARKET M0 calibrated Harville as the logit baseline. Define `f_B=logit(B_raw)-logit(M0)`, `f_L=logit(LightGBM)-logit(M0)`, `f_C=logit(CatBoost)-logit(M0)`. The fixed weighted correction model is

`p(w)=sigmoid(logit(M0)+w_B*f_B+w_L*f_L+w_C*f_C)`.

Fixed **reference** candidates (not selection-tuned in this study):
- `M0` market-only; `X2`; `E5`; `B_raw` incumbent; `LGB_raw`; `CAT_raw`; `LGB_scaled` and `CAT_scaled` from prior C2 (2023-only alpha); `D3_old` from prior X2-anchored experiment.

Fixed **new** candidate registry:
- `BL_prob_75_25`: simple *probability* mix 75% B_raw + 25% LGB_raw.
- `BL_logit_75_25`: logit correction weights [0.75, 0.25, 0].
- `BL_logit_50_50`: logit correction weights [0.50, 0.50, 0].
- `BC_logit_75_25`: logit correction weights [0.75, 0, 0.25].
- `BC_logit_50_50`: logit correction weights [0.50, 0, 0.50].
- `BLC_logit_50_25_25`: logit correction weights [0.50, 0.25, 0.25].
- `BLC_logit_equal`: logit correction weights [1/3, 1/3, 1/3].
- `BL_scalL_logit_75_25`: fixed logit mix of B_raw 75% and 2023-scaled LGB 25%, no new metatraining.
- `BLC_ridge_2023`: learned nonnegative correction weights [w_B,w_L,w_C], fit **only** to 2023 previously out-of-time OOF; initial [1,0,0], bounds [0,1.5] each and total sum <= 2.0, deterministic SciPy SLSQP (maxiter 300, ftol 1e-12), 2023 *race-macro* Brier plus `0.0005*sum((w-[1,0,0])**2)` (same lambda as earlier D3). The market-only weights [0,0,0] and B_raw-only [1,0,0] are feasible. Do not reuse older D3 weights.
- `BL_ridge_2023`: as above but freeze w_C=0, same loss/penalty/constraints. At most two fitted parameters.

No selected candidate may be auto-used as a learner's source on its own training year. In particular, the 2023 meta learner may use 2023 OOF B_raw, LGB and CAT raw; it must NOT consume 2023 *in-sample* predictions from an alpha calibrator fitted on 2023. Calibrated C2 forecasts are used only in the fixed `BL_scalL_logit_75_25` 2024 candidate.

## Scoring, predeclared incumbent selection, diagnostics

Primary race-macro Brier, secondary race-macro log loss, using original `candidate_score` and `paired_delta`. Report paired differences `B_raw Brier - candidate Brier` and SE of race losses clustered by **same official racing date**. Also compare each new candidate with previous E5 and best simple/learned baseline (not choosing by ROI/hit rate). The *full predeclared candidate registry* enters `select_with_incumbent_one_se` with incumbent **B_raw**; candidate replaces it only if best point-Brier exceeds paired one-SE. Report individual pairwise one-SE decisions separately. Keep negative results.

Save per-horse 2024 common-cohort OOF predictions, candidate Brier/logloss, date-clustered paired SE, 2023 learned coefficients, model/residual correlations, per-race loss and paired deltas, calibration intercept/slope, reliability bins, surface/distance/class breakdown, source SHA256 and runner fingerprint, runtime dependency versions, seed, explicit candidate registry/hash, full run manifest, self-contained Python CLI and regression tests. Do not store raw odds/history files in git. Stop on any label/row/probability/parity error.

2025 and 2026 have known outcomes and are not new tests. Do not change the production 10–9 minute lock or forward protocol (1,000 complete races, 60 dates, 90 days, model frozen and no interim promotion). No claim of market outperformance at T−10 without genuinely timestamped same-clock odds.
