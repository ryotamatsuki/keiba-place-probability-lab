# Stage D1: conditional market-offset scaling — preregistration BEFORE scores

Stage D1 is separate from new XGB F0–F3/learning curves and learner families. It uses original PR32 2023/24 chronological X2 OOF as frozen (no model training on any target fold) and original PR33 2024 E5. Its goal is to test whether the 2023-fitted **condition-dependent correction amplitude** can improve 2024 race-macro Brier relative to already known E5. The 2024 results have been explored many times and are **not a pristine unseen test**.

## Score boundaries and identically paired rows

- Only fit 2023 dated rows (2,887 races, 32,051 horse rows), eval 2024 (2,844 races, 31,225). 2024 label not used in design, optimization, stopping or feature engineering. 2025/26 known excluded. Strict SHA sorted 2023+2024 cohort `cb984b2553647d423e05742d0625988da3129dab3183f79b6fb84c74776a582f`. Require perfect same-row check with PR33 E5.
- Fixed M0 market-only, X2 unscaled, original PR33 E5 2023-OOF scalar, plus **two preregistered candidates** D1 and D2. Do not add more conditional candidates after inspecting scores.
- `f=logit(X2)-logit(M0)`. `P=sigmoid(logit(M0)+clip(alpha(x),0,1.5)*f)`, so alpha=0 exactly M0, alpha=1 exactly X2; clipping is on alpha, NOT additive probability clipping. Never insert a final payout, current race result or a future market update as a predictor.
- Common coefficient `a0` and deviations on prespecified input only. D1: `alpha=a0 + a1*(1[surface=turf]-0.5)`. D2: D1 plus `a2*(distance_m-1600)/1000 + a3*(M0_p-0.3)`. All these have the same target-time information as M0. No nonlinear search or learned segment-specific trees.
- Frozen `a0 in [0,1.5]`, `a1,a2,a3 in [-0.5,0.5]`. Optimize **2023 race-macro Brier** plus `0.0001*(a0-1)^2 + 0.001*(a1^2+a2^2+a3^2)` with deterministic SciPy L-BFGS-B, zero-start deviations, `a0=1`. No early stopping using 2024 outcomes. If optimization fails, mark untested; no data-dependent candidate changes.
- Compare on identical 2024 runners M0 X2 E5 D1 D2, using original repository one-SE selection with E5 incumbent and date-clustered paired Brier differences; log loss, calibration intercept/slope, reliability, surface/distance/class, per-race loss table. Selection pass is not statistical significance.
- This is historical **final** win-odds domain; actual T−10 deployment unknown, no commercial license concerns because no new model weights. No production adoption.
- If no improvement on 2024, retain E5 as research leader, M0 unchanged live; 1000 races/60 dates/90 days forward locked T−10 protocol unchanged.
