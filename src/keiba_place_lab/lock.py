"""Stage 6 prediction-lock validation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_LOCK_COLUMNS = [
    "horse_no",
    "horse_name",
    "as_of_time",
    "p_market",
    "p_model_raw",
    "p_model_calibrated",
    "p_ensemble",
    "uncertainty_low",
    "uncertainty_high",
    "rank",
    "model_version",
    "locked_commit_sha",
]


def validate_lock_frame(frame: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_LOCK_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing lock columns: {missing}")
    if len(frame) != 18 or frame["horse_no"].nunique() != 18:
        raise ValueError("Stage 6 lock must contain exactly 18 unique runners")
    if frame["rank"].nunique() != 18 or set(frame["rank"]) != set(range(1, 19)):
        raise ValueError("Stage 6 ranks must be exactly 1..18")
    probability_cols = [
        "p_market",
        "p_model_raw",
        "p_model_calibrated",
        "p_ensemble",
        "uncertainty_low",
        "uncertainty_high",
    ]
    for col in probability_cols:
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.isna().any() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Invalid probabilities in {col}")
    if not np.isclose(frame["p_ensemble"].sum(), 3.0, atol=1e-9):
        raise ValueError("Stage 6 ensemble probabilities must sum to 3")
    if (frame["uncertainty_low"] > frame["p_ensemble"]).any():
        raise ValueError("uncertainty_low exceeds ensemble probability")
    if (frame["uncertainty_high"] < frame["p_ensemble"]).any():
        raise ValueError("uncertainty_high below ensemble probability")
    if frame["as_of_time"].nunique() != 1:
        raise ValueError("All runners must share one market as-of timestamp")
    if frame["model_version"].nunique() != 1:
        raise ValueError("All runners must share one model version")
    if frame["locked_commit_sha"].nunique() != 1:
        raise ValueError("All runners must share one locked source commit")
