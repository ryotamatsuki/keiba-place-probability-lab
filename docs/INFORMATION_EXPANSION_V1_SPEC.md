# Information Expansion V1 — preregistered protocol (before score inspection)

Status: **FROZEN**; this branch is stacked on PR #31 head `16da80c0ede5ece7e79b19baefe598e0cefe5fa7`. Main remains at `a82cb4788190d9d82f9684cedc228a60d1077311`. Preserve all production bundles, postrace data, lock files, prior V1 results and final-odds caveats.

## Why another experiment

Prior PR #31 compared calibrations, training windows, stacking, and small residual regressions mainly on **turf 1200**. This V1 tests *additional pre-race information*, *latent opponent-adjusted performance*, and *model nonlinearity* separately on **all supported flat non-straight JRA turf/dirt, 1000–2600m**.

## Provenance and pre-off information availability

- Exact Kaggle `noriyukifurufuru/japan-horse-racing-2010-2025/versions/1`: `keiba_results.csv` SHA256 `fb9345273b21a7c23d41260dc77e45134a1bc2fe756899f3416b24ac0dd51de3`; `keiba_races.csv`, metadata/license source version checked. JRA only, official reconstructed `race_date` from frozen Stage 3.6 full panel `jra_flat_{warmup,train,validation,test}_v1.parquet` obtained from GitHub Actions run `37104212189`.
- Raw columns: `last_3f, passing, jockey_name, trainer, horse_weight` from result rows; `track_condition` from race rows. These are **postrace observed historical facts only**. For target race, absolutely no current last_3f, passing, finish, horse_weight (including increase/decrease), condition, or result-reported rider/trainer is admitted. Historical rider/trainer features are explicitly labelled **last observed horse connection**, not confirmed target-race lineup. Current assigned runner weight and race declaration/condition fields already allowed by frozen panel remain unchanged. All historical aggregates have `prior.race_date < target.race_date` (same-day outcomes also barred).
- Parse invalid fields to missing; no silent source-date substitution or fabricated times. Record JRA row coverage and invalid examples by year/surface. The first experiment never adds contemporaneous observed weight or going without trusted pre-lock timestamps. Subsequent live transport requires fresh official historical results and pre-race current jockey confirmation.
- All starters participate in race-relative `last_3f` and opponent ratings. Eligible evaluation horses must have >=3 prior starts, actual race field >=8, non-straight course; require complete final win market for the *entire actual starter field*. Stage4 Scope V3 1000–2600 target; do not convert unavailable outsiders into zero forecasts. Drop incomplete-market races for every candidate, not selective runner rows. Record exclusions and row fingerprints.

## Concrete fixed feature blocks

- `F_existing`: market-free frozen standardized panel predictors `feature_columns()` (date/IDs/current finish/win odds forbidden), plus existing full-field relative ability if available without leakage.
- `F_history`: past three starts' full-field relative last-3F percentile (smaller last-3F is better), past corner first-to-last gain/pace position and stability, prior known horse mass and mass change, past wet-ground top3 count/rate and wet-start count, last-observed jockey/trainer and their historical pre-race record (not current rider assignment). Missingness retained as NaN.
- `F_strength`: horse's *pre-race* Elo relative to opponent mean, and historical strength-adjusted last performance. Elo starts at 1500, Bradley–Terry comparison logistic base 10, rating scale 400 and per-race K=12. Rating updates only after finishing a whole date; nonfinishers get no update. Opponent ratings used for that day's feature snapshot are prior-day ratings. No market odds ever enter Elo.
- Never use same-day or later labels in any historical feature, Elo update or current-race predictor. Strict mutation/same-day tests.

## Frozen candidate registry and model architecture

All candidates are a **logit offset to the OOF market-only calibrated Harville marginal P(top3)**:
`P = sigmoid(logit(P_market_calibrated) + f(F))`. XGBoost binomial classifier uses calibrated-market logits as explicit `base_margin`, not prediction clipping or linear residual addition. All features exclude target current outcomes.
- M0: historical final-win-odds calibrated market only, zero correction (mandatory incumbent).
- L0: L2-penalized linear offset with `F_existing`, penalty 1.0.
- X0: XGBoost nonlinear offset with `F_existing`.
- X1: **same X0 algorithm and parameters** with `F_existing + F_history` (isolates incremental raw data).
- X2: **same X0 algorithm and parameters** with `F_existing + F_history + F_strength` (isolates opponent rating).
- No target-year tuning, no extra grid, no post-results fallback. X0/X1/X2 all use `n_estimators=120,max_depth=2,learning_rate=0.03,min_child_weight=60,subsample=0.8,colsample_bytree=0.8,reg_lambda=20,reg_alpha=2,tree_method=hist,n_jobs=2,random_state=20261031`. Missing features are permitted via XGBoost's native missing routing, with missing counts audited.
- The linear L0 uses existing PR31 `fit_market_offset` with prior-only training dates and fixed 1.0 L2 penalty, median imputed missing values **on training only** with missing-indicator flags. No fitted model receives its own target row's label indirectly via feature construction.
- Fixed source date windows: years 2016–2022 training for outer 2023; 2016–2023 for outer 2024. For correction training, use **chronological out-of-time calibrated market inputs** for 2018–2022 or 2018–2023 respectively: year t receives a calibrator fitted on 2016..(t−1) only. Market-only outer 2023 fits 2016–2022, outer 2024 fits 2016–2023.
- To avoid calibration-training label reuse, a meta training row from year t uses only market calibration data ending t−1; 2023/24 evaluation labels never enter their own model, calibrator or preprocessing. Historical final market odds are used **as market information at an explicitly hypothetical final-odds time**, never equated to T−10.
- Causal forward validation 2023 and 2024; **2025 and 2026 are known-outcome audits, not unseen tests**.

## Comparisons and interpretability

- All candidates on exactly the same race/horse rows; guard equality with sorted identity and SHA256 fingerprint. Primary race-macro Brier, secondary log loss, exact existing `model_selection.select_with_incumbent_one_se` and `paired_delta` clustered by meeting date. No accuracy, ROI, G1 anecdote, or per-race winner selection.
- Report 2023, 2024 and combined development performance; surface, distance bands [1000,1400], [1401,2000], [2001,2600], race class, including G1 when race_class reliably identifies G1; don't invent labels where class not granular enough. Calibration intercept/slope and reliability tables. Candidate factor comparisons X1−X0 (new information), X2−X1 (Elo), X0−L0 (nonlinearity) are diagnostic.
- Save exclusions/missingness, raw source hashes, artifact checksums, exact dependency versions, frozen settings, seed, OOF files, per-race losses/deltas and explicit limitations. No raw/redistribution-prohibited data git.
- If data unavailable, fail closed and preserve blockers. Any development positive result is **not** live promotion: forward threshold 1000 races / 60 meeting dates / 90 days; immutable model, T−10..−9 lock, no intermediate label-driven adoption changes.

## Blocking issues

JRA-VAN licensed intraday market is not present. Consequently all comparisons in this experiment are **final-odds-domain**; no claims of genuine T−10 market excess skill. If the sourced raw data cannot be aligned unambiguously with official dates/starters, record DATA_INSUFFICIENT instead of scoring invalid joins.
