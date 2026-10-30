# Model Diversity / Ensemble V1 — frozen before re-scoring

Research-only stacked on PR #32 `1b276b695db600753007d565ea52d670cd2d2917`; user question: can we improve further with alternative learners and ensemble design? **This first experiment tests actual ensemble potential using existing chronological OOF predictions; it does NOT claim to have trained CatBoost/LightGBM.** The 2023/24 results have already been inspected in prior research, and **2024 is not a newly unseen test**. Do not alter production model or model-selection policy.

## Canonical data & time
Source `analysis/information_expansion_v1/outer_oof_predictions.csv` frozen in PR32. 2023 = earlier-history-trained OOF, usable to set one scalar hyperparameter for predicting 2024. 2024 = development audit of the ensemble. Never use 2024 labels to fit weights or to design post-hoc new candidates. 2025/26 excluded. Exactly the same 2024 races, horses, final-win-odds timestamps and outcomes for every candidate. Require a sorted-row SHA256 and source-file hash. Race-macro Brier primary, log loss secondary, paired date-clustered one-SE relative to market M0 AND standalone X2.

## Frozen candidates — only these seven
- M0: untouched calibrated final-odds-only probabilities.
- X2: untouched strongest prior XGBoost market offset.
- E1: arithmetic probability pool 0.5 X0 + 0.5 X2.
- E2: logit pool sigmoid(0.5 logit X0 + 0.5 logit X2).
- E3: arithmetic probability pool 0.5 X1 + 0.5 X2.
- E4: market hedge 0.1 M0 + 0.9 X2 (probability mixture).
- E5: shrinkage of log-odds correction toward market: sigmoid(logit M0 + alpha*(logit X2 - logit M0)); fit one alpha using ONLY 2023 OOF (range [0,1.5], optimization 2023 race-macro Brier + penalty 0.0001*(alpha-1)^2); freeze this learned alpha BEFORE computing 2024 outcomes. alpha=0 is M0 and alpha=1 is X2.

All candidates remain proper [0,1] probabilities, but may not sum to 3 over full race when calibrated. Preserve eligibility, no post-hoc per-race candidate selection. Report whether the best combination improves X2 by more than paired one-SE, *independently* of whether it beats M0. No outcome-conditioned gating. Separate calibration/reliability diagnostics on the common 2024 cohort and race-level losses. CV/fit chronology must be verified. No modifications to the 2023/24 source OOF file.

## Next model families (RESEARCH ROADMAP ONLY, not part of E1–E5 experiment)
Follow-up pre-registration required for: CatBoost with raw categorical and chronological leakage-safe ordered boosting, LightGBM low-leaf offset, shallow XGB variance-diverse seeds/tree count; proper race-aware listwise/PL objective, and strictly OOF learned mixing/calibration. All train with market at the SAME timing domain, compare in 2023/24 already-explored development then lock a truly forward run. Do not market historical 2024 as an unseen validation.
