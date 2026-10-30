# Frozen 1,600-round model-family V1 — run/recovery guide

Parent commit `00fd15638548285a8c8e4e27f4aee0860cb24750`, stacked Draft PR #35 → #34 → #33 → #32 → #31. The preregistration is `docs/ROUND_SWEEP_1600_V1_SPEC.md`, committed before any candidate scoring. Do not modify the saved PR #34 model predictions, original model_selection v2 gate, or live Stage5 market-only system.

## GitHub Actions

`.github/workflows/fair-1600-rounds-v1.yml` launches three independent Linux CPU jobs (`xgb`, `lgb`, `cat`), each capped at 100 wall-clock minutes for actual experiment and 110 minutes total for the job, max two learner CPU threads (total run ≤3 family×100 family-minutes). Original successful Stage3.6 artifact from run 37104212189 and Kaggle V1 exact odds are downloaded on the runner; do not require persistent licensed originals or secret credentials in git.

Each job writes `analysis/round_sweep_1600_v1/<family>/` and **commits after EVERY completed stage**:
- `source_manifest.json`: original source hashes, frozen F3 feature names and **stable source signature**; raw original full panel stays temporary in `/tmp/fair_rounds_f3_history.parquet`.
- `inner_2021.csv`, `inner_2021_metrics.csv`, `inner_2021_manifest.json`; likewise 2022 and 2023; six prediction prefixes 60/120/240/400/800/1600 and Brier/log loss.
- `outer_2023.csv`, `outer_2023_selection_curve.csv`, `outer_2023_manifest.json`; likewise 2024. Contains chosen-round raw/scaled true OOF, old 400/120 reproduction check, prior-year alpha and timings.

An interrupted job keeps **fully committed previous stage checkpoints**; partially fitted trees that were not output are discarded. After a failure, check Actions logs and the source signature and resume from the first missing stage, not by inventing scores. Full automatic script reproduces the same prior-only source panel; the CLI for local processing is below. Note that new GitHub runs may be required to process the remaining epochs; refitting from the last complete checkpoint is the intended recovery, not resuming Python's in-memory booster.

### Reproduction commands (Python)

```bash
python -m pip install -e '.[dev,historical,modeldev]' -c requirements-scope.txt
python -m pip install 'lightgbm==4.7.0' 'catboost==1.2.10'
ruff check .
pytest -q
```

Use a legitimate local copy of the exact three original Stage3.6 parquet partitions and hashed Kaggle V1 results/race CSVs; **never commit those files**.

```bash
FAMILY=xgb  # also lgb / cat
REF=analysis/information_expansion_v1/outer_oof_predictions.csv
B=analysis/input_learning_v1/B_outer_oof_predictions.csv
C=analysis/input_learning_v1/C_outer_oof_predictions.csv
OUT=analysis/round_sweep_1600_v1/$FAMILY
PANEL=/tmp/round_f3.parquet
for_args=(--family "$FAMILY" --panel "$PANEL" --out "$OUT" \
 --reference-oof "$REF" --braw-oof "$B" --c1-oof "$C")
python scripts/run_round_sweep_1600_v1.py "${for_args[@]}" \
 --stage prepare \
 --historical-dir /path/to/stage36_freeze_v1 \
 --results-csv /path/to/keiba_results.csv \
 --races-csv /path/to/keiba_races.csv
for year in 2021 2022 2023; do
  python scripts/run_round_sweep_1600_v1.py "${for_args[@]}" \
    --stage inner --year "$year"
done
for year in 2023 2024; do
  python scripts/run_round_sweep_1600_v1.py "${for_args[@]}" \
    --stage outer --year "$year"
done
```

Repeat all three families. Then run aggregate, which **never trains** any full model:

```bash
python scripts/finish_round_sweep_1600_v1.py \
 --folder analysis/round_sweep_1600_v1 \
 --b-oof analysis/input_learning_v1/B_outer_oof_predictions.csv \
 --c1-oof analysis/input_learning_v1/C_outer_oof_predictions.csv \
 --x2-oof analysis/information_expansion_v1/outer_oof_predictions.csv \
 --e5-2024 analysis/model_ensemble_v1/2024_predictions.csv \
 --out analysis/round_sweep_1600_v1/FINAL
```

The aggregate writes `manifest.json` with A/B/C/D decision, original model reproductions, same-row/source parity and candidate registry; `inner_all_family_curves.csv`, `outer_selected_rounds_and_alpha.csv`, all 2024 paired forecasts, scores, date-SE, calibration, reliability, per-race loss, surface/distance/class diagnostics and (if 2023-only preregistered diversity gate passes) three fixed ensemble candidates.

## Stop conditions / interpretation

- Missing 2023/24 orig reference or one historical full-starter row; mismatch in any current-market probability or target; original 400/120 reproducibility beyond `atol=1e-8`: **stop before any scoring/adoption**.
- Only completed family checkpoints but no all-learner comparison within the registered budget: **category D / not model ceiling**, identify stage/timeout and exact local restart.
- If selected iterations =1600: registered search reached upper bound and optimum beyond is unknown. **Do not extend grid** without a new preregistration.
- 2024 is already seen development, so the paired one-SE is a complexity/selection gate and not evidence of an independent prospective probability gain.
- The source uses historical **FINAL odds**, not locked T−10 observations. Live adoption requires model lock, 1,000 complete races / 60 distinct dates / 90 days with same-time observation, no midstream threshold/parameter changes.

## Artifact organization and rights

The research branch contains only licensed-safe derived predicted probabilities and metrics, source identities/hashes, scripts, environment and independent tests. Never commit original `keiba_results.csv`, `keiba_races.csv`, historical runner parquet, model credentials or anything from private JRA-VAN.


## Reproduction incident (2026-10-31, before evaluating new 2024 scores)

The first archived B_raw 400-tree check on outer 2023 failed at maximum probability difference `2.9796600298226394e-08`, at float32 probability rounding granularity; no new outer score was accepted. Research-spec erratum records a per-row XGBoost tolerance of at most two float32 ULP plus a strict absolute 1e-7 cap; other learners continue to require 1e-8. Original train years, data keys, features, parameters, number-of-trees candidates, decision gate and model fit procedure are identical. The first LightGBM outer 2024 source-row comparison also failed solely because pandas preserved earlier year-filtered index labels; both frames are now explicitly sorted and reset before comparing the composite keys. This correction does not alter race/horse inclusion or target labels. Each complete prior inner fold has a saved hash/provenance manifest and can be reused instead of wasting CPU on identical reruns.
