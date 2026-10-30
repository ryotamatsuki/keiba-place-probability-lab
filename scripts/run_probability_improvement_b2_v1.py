"""B2: market logit-offset model, fitting only pre-2024 (2023) OOF labels.

Uses the same 2024 rows and 2024 market probabilities as experiment A.
The full race-relative context is computed for *all starters*, not eligible-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from keiba_place_lab.model_selection import paired_delta, select_with_incumbent_one_se
from keiba_place_lab.probability_improvement_v1 import fit_market_offset, predict_market_offset
from keiba_place_lab.stage4_successor_phase3a import add_relative_ability_features

FEATURES = (
    "rel_recent3_finish_vs_others",
    "rel_career_top3_vs_others",
    "rel_recent3_time_vs_others",
    "draw_pct",
    "front_forward_share",
)
KEYS = ["race_date", "race_id", "horse_id", "horse_no"]


def features_for_year(historical: Path, evaluation: pd.DataFrame, year: int
                      ) -> pd.DataFrame:
    cohort = pd.read_parquet(historical/"phase_a_turf_1200_train_v1.parquet")
    more = pd.read_parquet(historical/"phase_a_turf_1200_validation_v1.parquet")
    all_cohort = pd.concat([cohort, more], ignore_index=True)
    all_cohort["race_date"] = pd.to_datetime(all_cohort.race_date)
    old = all_cohort.loc[all_cohort.race_date.dt.year.lt(year) &
                         all_cohort.race_date.dt.year.ge(2016)]
    if old.empty or old.race_date.max() >= pd.Timestamp(f"{year}-01-01"):
        raise ValueError("B2 prior training missing or crosses target date")
    context = pd.read_parquet(historical/"jra_flat_validation_v1.parquet")
    context["race_date"] = pd.to_datetime(context.race_date)
    context = context.loc[context.race_date.dt.year.eq(year)]
    for frame in (all_cohort, context, evaluation):
        frame["race_id"] = frame.race_id.astype(str)
        frame["horse_id"] = frame.horse_id.astype(str)
    context = context.loc[context.race_id.isin(evaluation.race_id)]
    part = all_cohort.loc[all_cohort.race_date.dt.year.eq(year) &
                          all_cohort.race_id.isin(evaluation.race_id)].copy()
    part["race_date"] = pd.to_datetime(part.race_date)
    prior = float(old.top3_label.mean())
    engineered, _coverage = add_relative_ability_features(
        part, context, prior_mean=prior)
    if len(engineered) != len(part):
        raise ValueError("Full-field feature engineering changed sample")
    engineered["race_id"] = engineered.race_id.astype(str)
    engineered["horse_id"] = engineered.horse_id.astype(str)
    for name in ("race_date",):
        engineered[name] = pd.to_datetime(engineered[name])
    merged = evaluation[KEYS].merge(engineered[KEYS+list(FEATURES)], on=KEYS,
                                    validate="one_to_one", how="left", indicator=True)
    if len(merged) != len(evaluation) or not merged["_merge"].eq("both").all():
        raise ValueError("B2 feature coverage differs from frozen A evaluation keys")
    return merged.drop(columns=["_merge"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--historical-dir", type=Path, required=True)
    p.add_argument("--a-oof", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.a_oof)
    frame.race_date = pd.to_datetime(frame.race_date)
    frame["race_id"] = frame.race_id.astype(str)
    frame["horse_id"] = frame.horse_id.astype(str)
    frame["horse_no"] = frame.horse_no.astype(int)
    if set(frame.race_date.dt.year) != {2023, 2024}:
        raise ValueError("B2 expects 2023 OOF for fit, 2024 for evaluation")
    blocks = {}
    for year in (2023, 2024):
        subset = frame.loc[frame.race_date.dt.year.eq(year)].copy()
        blocks[year] = features_for_year(args.historical_dir, subset, year)
    Xtrain = blocks[2023][list(FEATURES)].to_numpy(float)
    Xtest = blocks[2024][list(FEATURES)].to_numpy(float)
    if not np.isfinite(Xtrain).all() or not np.isfinite(Xtest).all():
        absent = {year: blocks[year][list(FEATURES)].isna().sum().to_dict()
                  for year in (2023, 2024)}
        (out/"B2_BLOCKED.json").write_text(json.dumps({
            "status":"DATA_INSUFFICIENT","reason":"Missing complete prior-only features",
            "feature_missing_counts": absent}, indent=2))
        print("B2 DATA_INSUFFICIENT: NaN features, no fabricated fill.")
        return
    train = frame.loc[frame.race_date.dt.year.eq(2023)]
    test = frame.loc[frame.race_date.dt.year.eq(2024)].copy()
    fit = fit_market_offset(train.p_A1_calibrated.to_numpy(float), Xtrain,
                            train.top3_label.to_numpy(int), train.race_date.to_numpy(),
                            max_training_date="2024-01-01", penalty=1.0)
    base = test.p_A1_calibrated.to_numpy(float)
    challenger = predict_market_offset(base, Xtest, fit)
    decision, scores = select_with_incumbent_one_se(test, {
        "B0_market": base, "B2_offset": challenger}, incumbent="B0_market")
    scores.to_csv(out/"B2_scores.csv", index=False)
    test["p_B0_market"] = base
    test["p_B2_offset"] = challenger
    test.to_csv(out/"B2_oof_predictions.csv", index=False)
    diff = paired_delta(test, base, challenger, candidate_name="B0_market",
                        reference_name="B2_offset")
    manifest = {
        "status": "RESEARCH_ONLY_FINAL_ODDS",
        "training_max_date": fit.fitted_max_date,
        "fitted_coefficients": fit.coefficients.tolist(),
        "coefficients_zero_recover_baseline": True,
        "selection": {"winner": decision.winner,
                      "incumbent_retained": decision.incumbent_retained},
        "paired_market_minus_b2": {"brier": diff.mean_brier_delta,
                                   "date_clustered_se": diff.se_brier_delta},
        "rows": len(test), "races": int(test.race_id.nunique()),
        "source_a_oof_sha256": hashlib.sha256(args.a_oof.read_bytes()).hexdigest(),
    }
    (out/"B2_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
