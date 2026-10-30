# Stage D2 — new learner ensemble, frozen before C1 accuracy scores

This registry is frozen while CatBoost/LightGBM C1 is still running. C1 first requires passing its tested market-offset API and identical F3 previous-only feature rows. Failing or missing C1 predictions => D2 DATA_INSUFFICIENT, no fabricated zero predictions.

## Source and train/eval chronology

- PR32 original `analysis/information_expansion_v1/outer_oof_predictions.csv`: 2023/24 strict OOF calibrated market `M0` and XGBoost `X2`. PR33 original 2024 `E5` scalar after prior 2023 OOF.
- C1 original `analysis/input_learning_v1/C_outer_oof_predictions.csv`: 2023/24 strictly earlier-year-trained CatBoost/LightGBM market offset, identical race/horse keys and 2023/24 targets. Exact original 2023/24 OOF sorted row fingerprint `cb984b2553647d423e05742d0625988da3129dab3183f79b6fb84c74776a582f`, and original 2024 `E5` row equality required.
- Fit D3 weights using **2023 only** (2023 labels are NOT included in fit of the 2023 C1 component models, which used 2018–22). Predict 2024; never fit weights or estimate subgroups from 2024 labels. 2024 has been observed in prior PRs (development, NOT unseen validation).
- Baselines M0, X2, original E5, individual CatBoost and LightGBM must remain.

## Exactly four frozen combination candidates

Represent each learner's market logit correction as `f_k = logit(p_k)-logit(M0)`. Predictions are `sigmoid(logit(M0)+sum_k w_k*f_k)` with no probability clipping beyond logit numerical domain guard.

- D0: fixed 50/50 residual/logit blend X2+LightGBM.
- D1: fixed 50/50 residual/logit blend X2+CatBoost.
- D2: equal 1/3 residual blend X2+LightGBM+CatBoost.
- D3: **low-dimensional nonnegative regularized residual stacking**; fit three weights `w=[w_X2,w_LGB,w_CAT]` on 2023 race-macro Brier plus `0.0005 * ((w_X2-1)^2+w_LGB^2+w_CAT^2)`; bounds each weight [0,1.5], total weight ≤2.0, deterministic SciPy SLSQP initial (1,0,0), maxiter 300. This contains zero weights market-only and X2-only, and explicitly shrinks toward original X2.
- No post-hoc choosing the best correlating model, loss-based race-specific gating or adjustment of lambda to 2024 results. Report correction correlation, market-to-model residual magnitude, **paired race loss differences**, joint fitted 2023 weights, calibration/reliability/2024 segment audit.
- Race-macro Brier primary, log loss secondary, race-date paired one-SE gate vs E5 and X2 on exactly 2024 2844 races / 31225 eligible horses. No ROI, accuracy or G1-based selection.
- This is historical **FINAL** win-market modeling; no claim of 10 minutes pre-post improvement. Production Stage5 market only unchanged; forward locked protocol 1000 complete races/60 dates/90 days.
