# C2 model-specific calibration of new learner offsets — frozen before C1 scores

The C1 LightGBM/CatBoost training and prediction registry was frozen before any model scores. Before C1 final scores are inspected, register this explicit calibration experiment on their *original* strictly chronological 2023/2024 OOF.

- Input: PR32 original M0/X2 2023/24 OOF and C1 LightGBM/CatBoost 2023/24 strictly out-of-time predictions, **same exact 5731 races and 63276 eligible runners**. Distinct 2024 reference: PR33 original E5 fitted only on 2023. Strict SHA/fingerprint and 2024 runner/label equality.
- Fit **separate scalar alpha per new learner** from 2023 OOF **only**; no train labels from the same model being scored as OOF, no 2024 label for fitting. Input: `p_alpha=sigmoid(logit(M0)+alpha*(logit(C1_model)-logit(M0)))`. Fixed alpha range `[0,1.5]`, penalty `0.0001*(alpha-1)^2`, deterministic bounded one-dimensional SciPy optimization including both endpoints. alpha0 exactly M0 and alpha1 exactly original model.
- Fixed candidates: market M0, original X2, original E5, raw LightGBM, raw CatBoost, scalar-calibrated LightGBM, scalar-calibrated CatBoost. No extra calibration attempts after 2024 scores, no changes to C1 tree params or pre-race feature lists.
- Compare exact common 2024 race-macro Brier and log loss with same-date paired one-SE gate **against E5 and the same raw model**; save each alpha, row/hash, score, matched predictions and per-race loss contrasts; model family/candidate where alpha hits bound is diagnostic.
- No 2025/26 unseen test, no real T−10 intraday market, no production update. Staged D2 heterogeneous ensemble weights are registered separately in `docs/DIVERSE_ENSEMBLE_D2_SPEC.md`, must not be retroactively changed based on C2 outcomes.
