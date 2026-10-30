# Model-family evaluation V1 — verified constraints and next executable experiments

This is an operational protocol and **not a claim that CatBoost, LightGBM, RealMLP, TabICLv2 or TabPFN-3.5 have improved the horse racing model**. Execute only after the input F0–F3 ablation and inner-year XGBoost learning curve has passed regression and generated matched 2023/2024 row fingerprints.

## Why the sequence matters

In PR32 X2, the actual 42 numerical inputs + three categories (sex/racecourse/race_class) did not explicitly expose market confidence, surface or the historical Scope V3 five full-field relative-ability variables as trees' split features. Inputs F1–F3 add exactly these, keeping XGB architecture and sample identical. Depth2/120-tree and logloss objective are frozen to identify the value of information rather than conflating it with model capacity. The nested curves then select boosting rounds by race-macro Brier. Earlier model E5 alpha=1.478253352 is *not automatically copied* to a newly fitted model.

## Fixed-order family experiment proposal — separate new preregistration required

Do not add any candidate to the current F0–F3 selection slate after inspecting its scores. The following is a **future experiment design**, to freeze with exact package/checkpoint versions, seeds, 3-way inner/outer chronological year split, hardware limit and candidate subset prior to training:

1. **LightGBM (CPU)**. Identical F3 pre-race numeric and 4 categorical predictors as F3, train full earlier folds with equal-race weights, 120 boosting steps/31 leaves at fixed strong regularization or its prior-frozen variant. Market offset `lgb.Dataset(init_score=market_logit)`; during inference restore held-aside market logit on the raw predicted correction. Verify raw predictor+margin with a synthetic exactly-zero correction/identity regression. Source: https://lightgbm.readthedocs.io/en/latest/Parameters.html . Ensure LightGBM training's own initial score and model intercept are not counted twice.
2. **CatBoost (CPU)**. Same information, not additional unverified current-jockey IDs. One native categorical encoder; `catboost.Pool(baseline=market_logit)`, `has_time=True` only after sorting by date, with horse-same-day leakage guards. Establish that `RawFormulaVal` on prediction `Pool(baseline=...)` includes the supplied offset (some APIs differ), with no invisible double addition. Source: https://catboost.ai/docs/en/concepts/python-reference_pool .
3. **RealMLP (CPU / GPU optional)**. `pytabkit.RealMLP_TD_Classifier` scikit-style interface; use same F3 information with market logit as ordinary input (a direct-calibrated probability estimator unless a true logit-offset implementation is verified). If whole 200k+ meta-history exceeds an agreed cap, conduct clearly labelled *small-sample research*, with identical train and evaluation cohort for all compared learners. Do not mislabel an undertrained CPU MLP loss as global superiority. Sources: https://github.com/dholzmueller/pytabkit and NeurIPS 2024 Better by Default.
4. **TabICLv2 (GPU / small benchmark)**. The official open TabICLv2 implementation https://github.com/soda-inria/tabicl is BSD-3-Clause, weights open, synthetic prior, recommends GPU for large tables; cap a 20,000–50,000 earlier-year training context and 2024 batch size before launch. Compare a matched-size XGB baseline. Source: https://proceedings.mlr.press/v306/qu26a.html .
5. **TabPFN-3.5 (GPU research only)**. Model card and license https://huggingface.co/Prior-Labs/tabpfn_3_5 and https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE . Weights' `tabpfn-3-5-license-v1.0` allow non-commercial, non-production research/benchmarking but **prohibit production, commercial/revenue-generating uses and use of outputs for internal commercial decisions**. Do not integrate weights or outputs into the live horse-betting app without an appropriate license. No credentials, GPU job or research-only model output in current PR.
6. **EBM/GAM**: bounded-interaction market residual correction, with market-only as a zero-offset candidate. Explainable models are interpretable but require the same train-only preprocessing and validation.

TabArena 2025 (https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html) notes model-family ensembles can outperform singular learners with careful validation, and warns of overfitting the validation set. Model comparison should focus on `f_k=logit(P_k)-logit(P_market)` and date-paired loss differences, not naive final-probability correlation.

## Stage D: correction diversity, calibration, ensemble

For each candidate family generate **out-of-time and out-of-fit** predictions for inner years 2021/22/23; fit *each model's own* `alpha` and calibration using only preceding folds. Preserve original M0, X2 and PR33 E5. Try simple probabilities mean and nonnegative L2-penalized corrections with market-only `f=0` in the candidate set. Keep same 2024 runner/race fields; independent 2023 meta-training cannot see 2023's own validation labels if targeting 2023. Subgroup leakage/no dynamic model choice.

## Stage E: low-dimensional conditional correction / subset probabilities

`P=sigmoid(logit(P_M)+alpha(surface,distance,market_p)*f)`, common coefficient + small regularized surface/distance/confidence offsets. The condition-dependent coefficient must be learned exclusively on earlier-year OOF, with shape, bounds and shrinkage penalties frozen before evaluating outer-year outcomes. Do not claim that PR33's high E5 alpha proves tree-depth inadequacy or universal under-confidence.

For a precisely-three-horses top3 subset distribution, a conditional Bernoulli fixed-cardinality family provides coherent `sum_i P_i(top3)=3` on **all actual starters**. Requires full-field forecasts, nonstarter/cancellation/tie policy and a separate preregistration. Properly evaluate all-starter Brier vs eligible-only Brier and avoid claiming a projection's all-field theorem for restricted subsets.

## Source/time/selection guardrails

F3 final-odds-market popularity is **not** pre-lock observable. To measure live T−10 improvement, collect actual timestamped T−10 odds + active roster and train/evaluate at that same information boundary. Preserve 1,000 complete races/60 dates/90 calendar days frozen forward evaluation; no interim model selection or G1 anecdote. Older 2023/24 are development samples; 2025/26 observed data cannot be called a pristine unseen test.
