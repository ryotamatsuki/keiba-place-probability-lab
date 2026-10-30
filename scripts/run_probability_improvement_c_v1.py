"""Strict-chronology Scope V3 training-window sensitivity (C0/C1/C2).

Input MUST be a full standardized 2010+ historical panel including the pre-2016
feature warmup; never rebuild from eligible-only rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from keiba_place_lab.model_selection import (
    candidate_score, paired_delta, select_with_incumbent_one_se,
)
from keiba_place_lab.scope_expansion import full_context_for_races, straight_course_mask
from keiba_place_lab.scope_features import (
    augment_distance_history, block_columns, predict_scope_candidate, scope_features,
)
from keiba_place_lab.stage4_successor_phase3a import make_xgb01_relative_pipeline
from keiba_place_lab.historical_panel import select_phase_a_cohort

CANDIDATES = ("C0", "C1", "C2")
CONTEXT_BLOCKS = ("regime",)


def training_weights(train: pd.DataFrame, end_date: pd.Timestamp,
                     candidate: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Filter only target rows, not prior historical context."""
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown frozen candidate {candidate}")
    dates = pd.to_datetime(train.race_date)
    if (dates > end_date).any():
        raise ValueError("Training includes future relative to fold end")
    if candidate == "C1":
        start = end_date - pd.DateOffset(years=5) + pd.Timedelta(days=1)
        selected = train.loc[dates >= start].copy()
        return np.ones(len(selected)), selected
    if candidate == "C2":
        age_years = (end_date - dates).dt.days.to_numpy(float) / 365.25
        return 0.5 ** (age_years/3.0), train
    return np.ones(len(train)), train


def fit_scope_weighted(train: pd.DataFrame, context: pd.DataFrame,
                       weights: np.ndarray):
    if train.empty or len(weights) != len(train):
        raise ValueError("Invalid weighted cohort")
    if not set(train.race_id).issubset(set(context.race_id)):
        raise ValueError("Missing full-field training context")
    prior = float(np.average(train.top3_label.to_numpy(float), weights=weights))
    engineered = scope_features(train, context, prior_mean=prior,
                                blocks=CONTEXT_BLOCKS, surface="turf")
    model = make_xgb01_relative_pipeline()
    transformers = model.named_steps["preprocess"].transformers
    name, pipe, numeric = transformers[0]
    numeric = [*numeric, *(col for b in CONTEXT_BLOCKS for col in block_columns(b))]
    model.set_params(preprocess=ColumnTransformer(
        [(name, pipe, numeric), transformers[1]], remainder="drop", sparse_threshold=0.0
    ))
    model.fit(engineered, train.top3_label.astype(int),
              model__sample_weight=np.asarray(weights, dtype=float))
    return model, prior


def cohort(panel: pd.DataFrame) -> pd.DataFrame:
    eligible = select_phase_a_cohort(panel, surface="turf", min_field_size=8,
                                     min_prior_starts=3, min_distance_m=1200,
                                     max_distance_m=1200)
    eligible = eligible.loc[~straight_course_mask(eligible)]
    return eligible.sort_values(["race_date", "race_id", "horse_no"]).reset_index(drop=True)


def run(full: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    full = full.copy()
    full["race_date"] = pd.to_datetime(full.race_date)
    for key in ("race_id", "horse_id"):
        full[key] = full[key].astype(str)
    if full.race_date.min() > pd.Timestamp("2010-01-01") + pd.Timedelta(days=31):
        raise ValueError("Full pre-2016 history/warmup missing")
    if full.race_date.max() < pd.Timestamp("2024-12-20"):
        raise ValueError("Historical panel is not complete through 2024")
    if full.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Duplicate full-field keys")
    full = augment_distance_history(full)
    rows = cohort(full)
    selected = rows.loc[rows.race_date.dt.year.isin([2023, 2024])].copy()
    keys = ["race_date", "race_id", "horse_id", "horse_no"]
    reference = reference.copy()
    reference["race_date"] = pd.to_datetime(reference.race_date)
    for key in ("race_id", "horse_id"):
        reference[key] = reference[key].astype(str)
    if (not selected[keys].sort_values(keys).reset_index(drop=True).equals(
        reference[keys].sort_values(keys).reset_index(drop=True)
    )):
        raise ValueError("C cohort and frozen A common rows differ")
    pred = {name: np.full(len(selected), np.nan) for name in CANDIDATES}
    fitted_meta = []
    for year in (2023, 2024):
        end = pd.Timestamp(f"{year-1}-12-31")
        train_all = rows.loc[rows.race_date.dt.year.between(2016, year-1)]
        is_test = selected.race_date.dt.year.eq(year).to_numpy()
        test = selected.loc[is_test]
        all_train_context = full_context_for_races(full, set(train_all.race_id))
        test_context = full_context_for_races(full, set(test.race_id))
        for name in CANDIDATES:
            weights, train = training_weights(train_all, end, name)
            # Prior-feature lookup does NOT restrict to C1's five years.
            model, prior = fit_scope_weighted(
                train, full_context_for_races(all_train_context, set(train.race_id)),
                weights)
            pred[name][is_test] = predict_scope_candidate(
                model, test, test_context, prior_mean=prior,
                blocks=CONTEXT_BLOCKS, surface="turf")
            fitted_meta.append({"year": year, "candidate": name,
                                "training_rows": len(train),
                                "training_min_date": str(train.race_date.min().date()),
                                "training_max_date": str(train.race_date.max().date()),
                                "prior": prior, "weight_min": float(weights.min()),
                                "weight_max": float(weights.max())})
    if any(not np.isfinite(value).all() for value in pred.values()):
        raise ValueError("C candidate OOF missing")
    result = selected[keys+["top3_label", "surface", "distance_m", "race_class"]].copy()
    for name, values in pred.items():
        result[f"p_{name}"] = values
    return result, {"training": fitted_meta, "predictions": pred}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full-historical-panel", type=Path, required=True)
    p.add_argument("--a-oof", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if not args.full_historical_panel.exists():
        raise FileNotFoundError("No full 2010+ historical panel: C cannot be scored")
    panel = pd.read_parquet(args.full_historical_panel)
    reference = pd.read_csv(args.a_oof)
    result, metadata = run(panel, reference)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out/"C_oof_predictions.csv", index=False)
    base = result.copy()
    base["race_date"] = pd.to_datetime(base.race_date)
    pmap = {name: result[f"p_{name}"].to_numpy(float) for name in CANDIDATES}
    decision, scores = select_with_incumbent_one_se(base, pmap, incumbent="C0")
    scores.to_csv(out/"C_scores.csv", index=False)
    # C comparisons are not automatically market improvements.
    summary = {"status": "RESEARCH_ONLY", "winner": decision.winner,
               "incumbent_retained": decision.incumbent_retained,
               "training": metadata["training"],
               "input_sha256": hashlib.sha256(
                   args.full_historical_panel.read_bytes()).hexdigest()}
    (out/"C_manifest.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main()
