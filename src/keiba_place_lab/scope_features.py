"""Strictly prior-date distance histories and surface-specific scope candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from .nonmarket import validate_training_frame
from .stage4_successor_phase3a import (
    _leave_one_out_mean,
    add_relative_ability_features,
    make_xgb01_relative_pipeline,
)

BLOCKS = ("near", "regime", "course_distance")
BANDS = ((1000, 1400), (1401, 2000), (2001, 2600))


def distance_regime(distance: np.ndarray) -> np.ndarray:
    return np.digitize(distance, [1400.0, 2000.0, 2600.0], right=True)


def augment_distance_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute histories before cohort filtering, preserving caller row order.

    A strict date comparison excludes the current race and every later result,
    including same-day results. Unknown keys/results differ from known zero starts.
    Labels may be absent only in target rows (live callers must validate history).
    """
    out = panel.reset_index(drop=True).copy()
    required = {"horse_id", "race_date", "surface", "distance_m", "racecourse"}
    if required.difference(out.columns) or out["horse_id"].isna().any():
        raise ValueError("Missing distance-history keys")
    dates = pd.to_datetime(out["race_date"], errors="raise").to_numpy()
    if pd.isna(dates).any():
        raise ValueError("Missing history dates")
    labels = pd.to_numeric(out.get("top3_label", pd.Series(np.nan, index=out.index)))
    if not labels.dropna().isin([0, 1]).all():
        raise ValueError("History labels must be binary")
    y = labels.to_numpy(dtype=float)
    distance = pd.to_numeric(out["distance_m"], errors="coerce").to_numpy(float)
    surface = out["surface"].fillna("").astype(str).to_numpy()
    course = out["racecourse"].fillna("").astype(str).to_numpy()
    values = {k: np.full((len(out), 2), np.nan) for k in (*BLOCKS, "same_surface")}
    for positions in out.groupby("horse_id", sort=False).indices.values():
        d, s, c, t, target = (
            distance[positions], surface[positions], course[positions],
            dates[positions], y[positions],
        )
        prior = t[:, None] > t[None, :]
        same_surface = (s[:, None] == s[None, :]) & prior
        same_distance = d[:, None] == d[None, :]
        r = distance_regime(d)
        rules = {
            "same_surface": same_surface,
            "near": same_surface & (np.abs(d[:, None] - d[None, :]) <= 200),
            "regime": same_surface & (r[:, None] == r[None, :]),
            "course_distance": same_surface & same_distance & (c[:, None] == c[None, :]),
        }
        for key, mask in rules.items():
            valid = s != ""
            if key != "same_surface":
                valid &= np.isfinite(d)
            if key == "course_distance":
                valid &= c != ""
            unknown = (mask & ~np.isfinite(target)[None, :]).any(axis=1)
            counts = mask.sum(axis=1).astype(float)
            successes = mask @ np.nan_to_num(target)
            counts[~valid | unknown] = np.nan
            successes[~valid | unknown] = np.nan
            values[key][positions, 0] = counts
            values[key][positions, 1] = successes
    for key, pair in values.items():
        out[f"{key}_starts"] = pair[:, 0]
        out[f"{key}_top3"] = pair[:, 1]
    return out


def block_columns(block: str) -> list[str]:
    if block not in BLOCKS:
        raise ValueError(f"Unknown feature block {block}")
    return [f"{block}_{suffix}" for suffix in (
        "top3_shrunk", "log_starts", "zero_experience", "history_missing", "relative",
    )]


def surface_base(frame: pd.DataFrame, surface: str) -> pd.DataFrame:
    out = frame.copy()
    if not out["surface"].eq(surface).all():
        raise ValueError("Mixed surface in candidate/context")
    if surface == "dirt":
        out["turf_starts"] = out["same_surface_starts"]
        out["turf_top3"] = out["same_surface_top3"]
    elif surface != "turf":
        raise ValueError("Unsupported surface")
    return out


def scope_features(frame, context, *, prior_mean, blocks=(), surface="turf"):
    engineered, _ = add_relative_ability_features(
        surface_base(frame, surface), surface_base(context, surface), prior_mean=prior_mean,
    )
    keys = context[["race_id", "horse_id"]].astype("string").copy()
    for block in blocks:
        starts = pd.to_numeric(context[f"{block}_starts"], errors="coerce")
        top3 = pd.to_numeric(context[f"{block}_top3"], errors="coerce")
        invalid = (starts < 0) | (top3 < 0) | (top3 > starts)
        if invalid.any():
            raise ValueError("Invalid scope-history counts")
        rate = (top3 + 6 * prior_mean) / (starts + 6)
        work = context[["race_id"]].copy()
        work["rate"] = rate
        keys[f"{block}_top3_shrunk"] = rate.to_numpy()
        keys[f"{block}_log_starts"] = np.log1p(starts).to_numpy()
        keys[f"{block}_zero_experience"] = starts.eq(0).astype(float).to_numpy()
        keys[f"{block}_history_missing"] = starts.isna().astype(float).to_numpy()
        keys[f"{block}_relative"] = (rate - _leave_one_out_mean(work, "rate")).to_numpy()
    return engineered.merge(keys, on=["race_id", "horse_id"], validate="one_to_one")


def fit_scope_candidate(train, context, *, blocks=(), surface="turf"):
    validate_training_frame(train)
    prior = float(train["top3_label"].mean())
    engineered = scope_features(train, context, prior_mean=prior, blocks=blocks, surface=surface)
    model = make_xgb01_relative_pipeline()
    transformers = model.named_steps["preprocess"].transformers
    name, pipe, numeric = transformers[0]
    numeric = [*numeric, *(col for b in blocks for col in block_columns(b))]
    model.set_params(preprocess=ColumnTransformer(
        [(name, pipe, numeric), transformers[1]], remainder="drop", sparse_threshold=0.0,
    ))
    model.fit(engineered, train["top3_label"].astype(int))
    return model, prior


def predict_scope_candidate(model, frame, context, *, prior_mean, blocks=(), surface="turf"):
    engineered = scope_features(frame, context, prior_mean=prior_mean, blocks=blocks, surface=surface)
    p = np.asarray(model.predict_proba(engineered)[:, 1], dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid scope probability")
    return p
