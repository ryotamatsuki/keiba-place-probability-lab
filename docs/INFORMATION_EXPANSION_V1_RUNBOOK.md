# Information Expansion V1 — reproducibility and validity runbook

This experiment is stacked on PR #31, isolated from the production prediction lock. Source of truth is the prior-to-score-committed `docs/INFORMATION_EXPANSION_V1_SPEC.md` (commit `1bab50544f9317f78e40ea1eab754450f8082f16`).

## What is being answered

**Incremental pre-existing information** vs the same historical FINAL-win-odds market:

1. `M0_market` — market-only, chronological logit recalibration.
2. `L0_linear_old` — penalized linear market-logit offset using the old numeric+categorical predictor information.
3. `X0_nonlinear_old` — fixed shallow-XGBoost market-logit offset using precisely the same old information as `L0`.
4. `X1_nonlinear_history` — unchanged XGBoost with lagged 3F, corner, going, body mass, previous rider/trainer history.
5. `X2_nonlinear_strength` — unchanged XGBoost with all previous history and prior-day opponent Elo.

All candidates use identical market-available races and eligible runners, never a candidate-specific cohort. The extra information may already be priced by the market; a negative outcome is valid. These are not T−10 comparisons.

## Exact source and execution

Required frozen Stage3.6 artifact `stage36-historical-training-panel-v1` from workflow run `37104212189`, unpacks to:

```
/tmp/stage36/data/historical_processed/stage36_freeze_v1/
  jra_flat_warmup_v1.parquet
  jra_flat_train_v1.parquet
  jra_flat_validation_v1.parquet
```

Required original Kaggle 2010–2025 version-one files `keiba_results.csv` and `keiba_races.csv`. The historical results source SHA256 is required to be exactly `fb9345273b21a7c23d41260dc77e45134a1bc2fe756899f3416b24ac0dd51de3`. This data and current PR outputs are *not* a current race-day provider snapshot. The frozen source real calendar date—not Kaggle's pseudo-date—is authoritative.

```bash
python -m pip install -e '.[dev,historical,modeldev]' -c requirements-scope.txt
ruff check .
pytest -q

python -u scripts/run_information_expansion_v1.py \
  --historical-dir /tmp/stage36/data/historical_processed/stage36_freeze_v1 \
  --results-csv /tmp/kaggle_v1/keiba_results.csv \
  --races-csv /tmp/kaggle_v1/keiba_races.csv \
  --output-dir analysis/information_expansion_v1
```

If original archive extracts to a nested directory, supply the actual verified absolute CSV paths. The runnable GitHub Actions workflow is `.github/workflows/information-expansion-v1.yml`, which resolves exact paths with `find`.

## Interpretation and leakage boundaries

- The first 2010–2015 observations are history warmup; training for calibrated market forecasts begins with pre-2018 fit, meta-training 2018–2022 for 2023 and 2018–2023 for 2024. Source rows after 2024 are not needed and are never a new test.
- The raw source's race-level `last_3f`, `passing`, `horse_weight`, and condition are known **only after that historical race**. The model never exposes those variables from the target race; it sees lagged previous-date aggregates. Avoid target-race rider/trainer from results: only `last observed` connection metadata is in the model.
- Elo starts at 1500 and compares historical finish ranks to prior-day opponents, applying all same-date score updates only once the day closes. Source ranks are checked against the frozen target label and conflicts are suppressed from Elo updates. All target-day result fields are absent from the final predictor allowlist.
- Horse histories across all past distances/surfaces remain intact. Target evaluation requires actual field >=8, pre-race career starts >=3, turf/dirt 1000–2600m and nonstraight course. Relative, Elo, last3f are calculated with *all* past actual starters, not just eligible target runners.
- Because XGBoost `base_margin=logit(P_market)`, the zero correction analytically equals the baseline; calibration and offset training never receive same-year labels. X0 vs L0 compares model families on the same variables (although native missing-value treatment differs); X1 vs X0 identifies new observed information, X2 vs X1 identifies Elo.
- All Brier comparisons are race-macro, pairwise clustered by the **same race date**. The incumbent market remains unless best candidate improvement exceeds the paired one-SE. Avoid interpreting this as a frequentist significance test. G1-specific claims require a reliable current-race G1 indicator; class labels are only reported at their actual verified resolution.
- 2025 results have been seen in earlier work. A favorable historical development result **cannot** authorize production promotion: existing immutable T−10..−9 model/lock and >=1,000 races, >=60 dates and >=90-day forward period are unchanged.

## Audit outputs

When the eligible source is available, `raw_source_coverage.json` and `market_coverage.json` are written *before* fitting; after successful fit: `manifest.json`, `outer_oof_predictions.csv`, `joint_scores.csv`, `paired_vs_market.csv`, `race_differences.csv`, `calibration.csv`, `subgroups.csv` and `reliability.csv`.

The main manifest records raw/panel SHA256, code hash, library versions, seed, target row fingerprint, evaluation period and caveats. Source rows and actual credential-bearing files are not checked in. If a job fails, check the GitHub Actions traceback; a partial coverage audit cannot be presented as a completed accuracy experiment.

## Next independent evidence

A frozen market-relative model must be rebuilt using truly pre-off market snapshots from the same target timestamp before claims of live improvement. Historic final odds are useful for identifying candidate features, not for 10-minute-before-post adoption.
