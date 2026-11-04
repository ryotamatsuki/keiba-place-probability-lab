# Probability Improvement V1 — preregistered protocol

Status: **FROZEN BEFORE ANY NEW CANDIDATE SCORES**. Research only; no automatic production promotion.
Base main SHA: `a82cb4788190d9d82f9684cedc228a60d1077311`.
Frozen predecessors: `MODEL_SELECTION_PROTOCOL_V2.md`, `STAGE4_SCOPE_V3_FREEZE.md`, `STAGE5_V2_FREEZE.md`, `SCOPE_PROSPECTIVE_V3_SPEC.md`. Their outcomes, manifests, model bundle and locks remain immutable.

## Data and domain gate

- Historical 2010–2025 bootstrap: Kaggle `noriyukifurufuru/japan-horse-racing-2010-2025/versions/1`, `keiba_results.csv` expected SHA256 `fb9345273b21a7c23d41260dc77e45134a1bc2fe756899f3416b24ac0dd51de3`. Its odds are **final win odds**, NOT a 10-minute or morning snapshot. Download at execution; never commit raw rows.
- Audited historical processed panel: GitHub Actions run `37104212189`, artifact `stage36-historical-training-panel-v1`, with `2016–2022` training, `2023–2024` development and `2025` known-outcome audit. Source/row SHA checks are mandatory.
- Existing Stage4 outer OOF 2023/2024: `analysis/stage4_successor_phase3a/phase3a_outer_predictions.csv`. Scope V3 full-distance historical OOF exists, but for strictly comparable market-and-nonmarket experiments v1 primary cohort is **turf exactly 1200m** satisfying existing >=8 actual starters, >=3 strictly prior starts, non-straight course (if recorded). All active starters, including ineligible horses, participate in market normalization and race-relative feature construction. Evaluated rows are the intersection of full-market complete races and the frozen Stage4 eligible OOF, keyed by `race_date,race_id,horse_id,horse_no`. Exclude entire incomplete-odds/unknown-outcome/invalid field races, no cherry-picking; persist exclusions and sorted-row fingerprint before scoring.
- History horizon remains 2010–2025; historical nonmarket training rows begin 2016. 2025/2026 are **observed audits**, never v1 model-selection test.
- Market comparison results are expressly **historical final-odds-domain research**, not a claim about a 10-minute-before-off model. There are no authenticated historical 10-minute snapshots in the repository.
- Development external folds: train through 2022 -> evaluate 2023; train through 2023 -> evaluate 2024. Fit all calibrators, engineered encoders, stackers and hyperparameters on strictly earlier data. The B/C comparison on the paired primary 2024 cohort is only admissible where 2023 strictly OOF meta-training rows exist; if required preceding OOF is unavailable, B/C are labelled blocked, never trained on 2024 labels.

## Fixed candidate registry and comparison family

**A — full-field odds**, on identical eligible 2023–2024 evaluation keys:
- A0: existing normalized inverse final win odds and sequential Harville top3, no calibration.
- A1: existing logit-intercept/slope calibration, fitted only on dates prior to each outer evaluation year.
- A2: sequential first-place q, second-place q^gamma normalized among survivors, third-place q^delta normalized among survivors. Fixed grid gamma,delta in `{0.75,1.00,1.25}` (9 combinations). Freeze selection by inner chronological validation years 2020–2022; each inner fold fits on prior years (>=2016); choose lowest inner **race-macro Brier** then smaller distance to (1,1) then lexical tie break. Refit outer-training logit intercept/slope only after grid chosen. A2 must beat **A1** separately, plus A0 anchor; A2 gamma=delta=1 must reproduce A0 full-field within 1e-9 and sum(top3)=3 exactly within numerical tolerance. Do not transfer foreign-study exponents.

**B — market + strictly out-of-time nonmarket**, primarily 2024:
- B0_market: outer-frozen **A1** (not selected using 2024 labels), and B0_blend weights `{0.00,0.05,0.10}` with Stage4 OOF. Compare every candidate to market-only.
- B1: LogisticRegression `C=0.1`, L2, input `logit(p_market),logit(p_nonmarket)`; calibrated 2023 market and nonmarket inputs must be out-of-time; meta fit 2023 labels, predict 2024. All pre-2023 Stage4 predictions, if used, must be OOF.
- B2: constrained logit offset `logit(p_market) + f(X)`; f is linear in standardized prior-only `rel_recent3_finish_vs_others, rel_career_top3_vs_others, rel_recent3_time_vs_others, draw_pct, front_forward_share`, L2 coefficients penalized at lambda=1.0, intercept also penalized, L-BFGS fixed random seed `20261031`. The 2023 meta fit may use only 2023 OOF/nonmarket-pre-race features; never 2024 outcomes. f=0 is exactly B0_market. Missing features cause explicit data-block status, never zero-impute invisibly.
- B0–B2 must use **identical 2024 race/horse rows**, and market only is incumbent. No market model is selected by observing its 2024 evaluation labels.

**C — nonmarket identical Scope V3 1200 features, XGB01 and frozen parameters**:
- C0: all eligible 2016..fold-end training targets, retain pre-2016 contextual warmup; C1: last 5 calendar years of *training targets*; C2: all targets weighted `0.5 ** (age_in_years/3)`, 365.25-day years. Feature history must never be truncated to 5 years.
- Evaluate on 2023/2024 strictly forward folds, same rows as A/B wherever data permit. Use same original model specification and no retuning; C0 reference comparison to frozen Stage4 predictions.
- Fixed combinations in 2024: A1+C0 weight 0.05; A1+C1 weight 0.05; A1+C2 weight 0.05; A2+C0 weight 0.05. No other cross-search; label as unavailable when required OOF history is absent.

**S — auxiliary sum-to-three projection**, not production default:
- For complete active-starter fields of >=4 with exactly 3 top3 labels and no ties/cancellations, project `p` to `{0<=x_i<=1, sum x_i=3}` by Euclidean projection `x_i=clip(p_i-lambda,0,1)`, lambda bisection. Compare original and projected full-field Brier. Complete-field vector distance to valid binary outcome cannot increase; **this guarantee does not hold for an eligible-runner subset**. Do not alter incumbent eligibility or market-only supplementation. Already-valid Harville sum 3 should be fixed within tolerance.

## Statistics / lock

- Existing `model_selection.candidate_score`, `paired_delta`, `select_with_incumbent_one_se`; **race-macro Brier** primary, log loss secondary. Compare matched keys and identical snapshots, not separately merged incomplete populations. Date-clustered paired CR1 1-SE gate; market-only retained unless improvement > paired SE. No p-value, ROI, accuracy or single G1 determines winner.
- Diagnostic only: calibration intercept/slope, reliability, distance band, surface, class, year, source coverage. If actual subgroup is one constant level, report that honestly (e.g., turf 1200 only). Persist exclusions, row fingerprint, source data SHA256, code SHA, package versions, seeds, OOF prediction files and race-level paired deltas.
- **No interim-selection**. The real forward gate remains 1,000 complete races, >=60 dates and >=90 days, immutable model during collection, pre-off main-publication lock between -10 and -9 minutes, no intermediate score-based promotion.

## Future 60/30/10-minute capture

- Snapshot targets are `scheduled_off - 60min`, `-30min`, `-10min`. Select most recent **provider publication <= target** and **local captured_at <= target**. Freshness threshold: <=10 minutes for 60/30, <=5 minutes for 10. Never choose a future or final-odds value. Stale/partial field -> missing entire race checkpoint, no look-ahead.
- Schema: race_id,horse_id,bet_type,odds,place_odds_low,place_odds_high,provider_updated_at,captured_at,scheduled_off_at,source,snapshot_id. All timestamps timezone-aware, preserve raw odds source semantics. Place odds interval is not P(top3); no reciprocal-place-odds probability.
- JRA-VAN JV-Link requires licensed authorized environment; no credentials or paid history available in this research environment. Adapter accepts a licensed user export, not an unauthorized scraper. No invented captures.

## Blocking and reporting policy

- Fail closed if historical source, shared row fingerprints, required pre-2023 OOF, or source chronology cannot be verified; label **DATA_INSUFFICIENT**, not an invented score.
- Stage 5 production stays canonical live market-only; research scores are never automatically promoted. Publish blocked/negative outcomes along with executable rerun commands and hashes.
