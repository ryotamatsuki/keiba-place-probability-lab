"""Transparent, market-free Stage 4 P(top3) baseline helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_validation import forbidden_columns

HISTORY_PAIRS = {
    "career": ("career_top3", "career_starts"),
    "turf": ("turf_top3", "turf_starts"),
    "same_distance": ("same_distance_top3", "same_distance_starts"),
    "same_course": ("same_course_top3", "same_course_starts"),
}

NUMERIC_BLOCKS = {
    "entry_condition": [
        "draw_pct",
        "age",
        "assigned_weight_kg",
        "assigned_weight_delta_from_prev_kg",
        "log_days_since_prev",
        "distance_change_from_prev_m",
        "surface_changed_from_prev",
    ],
    "history": [
        "career_top3_shrunk",
        "log_career_starts",
        "turf_top3_shrunk",
        "log_turf_starts",
        "same_distance_top3_shrunk",
        "log_same_distance_starts",
        "same_course_top3_shrunk",
        "log_same_course_starts",
    ],
    "recent_form": [
        "recent3_finish_pct_mean",
        "recent3_top3_count",
        "recent3_open_plus_count",
        "recent3_graded_count",
        "recent3_relative_time_mean",
        "recent4_early_pos_pct_mean",
    ],
    "race_context": [
        "front_forward_share",
        "field_size",
        "distance_m",
    ],
}

CATEGORICAL_BLOCKS = {
    "entry_condition": ["sex"],
    "race_context": ["racecourse", "race_class"],
}

ALL_BLOCKS = tuple(NUMERIC_BLOCKS)

RACECOURSE_ALIASES = {
    "Sapporo": "札幌",
    "Hakodate": "函館",
    "Fukushima": "福島",
    "Niigata": "新潟",
    "Tokyo": "東京",
    "Nakayama": "中山",
    "Chukyo": "中京",
    "Kyoto": "京都",
    "Hanshin": "阪神",
    "Kokura": "小倉",
}

RACE_CLASS_ALIASES = {
    "Listed_open": "Open",
    "Listed": "Open",
    "Open": "Open",
    "G1": "Open",
    "G2": "Open",
    "G3": "Open",
}

REQUIRED_RAW_COLUMNS = {
    "top3_label",
    "race_id",
    *(column for pair in HISTORY_PAIRS.values() for column in pair),
    "days_since_prev",
    *(
        column
        for columns in NUMERIC_BLOCKS.values()
        for column in columns
        if not column.startswith("log_")
    ),
    *(column for columns in CATEGORICAL_BLOCKS.values() for column in columns),
}
REQUIRED_RAW_COLUMNS.discard("log_days_since_prev")
for prefix in HISTORY_PAIRS:
    REQUIRED_RAW_COLUMNS.discard(f"{prefix}_top3_shrunk")
    REQUIRED_RAW_COLUMNS.discard(f"log_{prefix}_starts")


@dataclass(frozen=True)
class BinaryEvaluation:
    brier: float
    log_loss: float
    rows: int
    races: int


@dataclass(frozen=True)
class Evaluation:
    brier: float
    log_loss: float
    mean_race_sum: float
    max_abs_race_sum_error: float
    rows: int
    races: int


def validate_market_free(columns: Iterable[str]) -> None:
    """Raise if a Stage 4 input contains market-like columns."""
    bad = forbidden_columns(columns)
    if bad:
        raise ValueError(f"Forbidden market columns in Stage 4 input: {bad}")


def canonicalize_target_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Map the frozen current-race labels onto historical category labels."""
    validate_market_free(frame.columns)
    out = frame.copy()
    if "racecourse" in out.columns:
        out["racecourse"] = out["racecourse"].replace(RACECOURSE_ALIASES)
    if "race_class" in out.columns:
        out["race_class"] = out["race_class"].replace(RACE_CLASS_ALIASES)
    return out


def complete_race_subset(frame: pd.DataFrame) -> pd.DataFrame:
    """Return races where the filtered cohort still contains every starter."""
    required = {"race_id", "field_size"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing complete-race columns: {sorted(missing)}")
    field_unique = frame.groupby("race_id")["field_size"].nunique()
    if field_unique.gt(1).any():
        raise ValueError("field_size is not constant within race")
    counts = frame.groupby("race_id").size()
    expected = frame.groupby("race_id")["field_size"].first().astype(int)
    complete_ids = counts.index[counts.eq(expected)]
    return frame.loc[frame["race_id"].isin(complete_ids)].copy()


def validate_training_frame(frame: pd.DataFrame) -> None:
    validate_market_free(frame.columns)
    missing = sorted(REQUIRED_RAW_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing Stage 4 training columns: {missing}")
    if frame["top3_label"].isna().any():
        raise ValueError("top3_label contains missing values")


def engineer_features(
    frame: pd.DataFrame,
    *,
    prior_mean: float,
    prior_strength: float = 6.0,
) -> pd.DataFrame:
    """Build transparent derived features using one common shrinkage rule."""
    if not 0.0 < prior_mean < 1.0:
        raise ValueError("prior_mean must be in (0, 1)")
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")

    validate_market_free(frame.columns)
    out = frame.copy()
    out["log_days_since_prev"] = np.log1p(
        pd.to_numeric(out["days_since_prev"], errors="coerce")
    )

    alpha = prior_mean * prior_strength
    beta = (1.0 - prior_mean) * prior_strength
    for prefix, (success_col, starts_col) in HISTORY_PAIRS.items():
        successes = pd.to_numeric(out[success_col], errors="coerce")
        starts = pd.to_numeric(out[starts_col], errors="coerce")
        invalid = (successes < 0) | (starts < 0) | (successes > starts)
        if invalid.fillna(False).any():
            raise ValueError(f"Invalid top3/start counts in {prefix}")
        out[f"{prefix}_top3_shrunk"] = (successes + alpha) / (
            starts + alpha + beta
        )
        out[f"log_{prefix}_starts"] = np.log1p(starts)
    return out


def feature_columns(
    excluded_blocks: Iterable[str] = (),
) -> tuple[list[str], list[str]]:
    excluded = set(excluded_blocks)
    unknown = excluded.difference(ALL_BLOCKS)
    if unknown:
        raise ValueError(f"Unknown feature blocks: {sorted(unknown)}")
    numeric = [
        col
        for block, cols in NUMERIC_BLOCKS.items()
        if block not in excluded
        for col in cols
    ]
    categorical = [
        col
        for block, cols in CATEGORICAL_BLOCKS.items()
        if block not in excluded
        for col in cols
    ]
    return numeric, categorical


def make_logistic_model(
    *, c_value: float, excluded_blocks: Iterable[str] = ()
) -> Pipeline:
    """Construct the transparent Stage 4 regularized logistic model."""
    if c_value <= 0:
        raise ValueError("c_value must be positive")
    numeric, categorical = feature_columns(excluded_blocks)
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    C=float(c_value),
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=20261003,
                ),
            ),
        ]
    )


def fit_model(
    train: pd.DataFrame,
    *,
    c_value: float,
    prior_strength: float = 6.0,
    excluded_blocks: Iterable[str] = (),
) -> tuple[Pipeline, float]:
    validate_training_frame(train)
    prior_mean = float(train["top3_label"].mean())
    engineered = engineer_features(
        train, prior_mean=prior_mean, prior_strength=prior_strength
    )
    model = make_logistic_model(
        c_value=c_value, excluded_blocks=excluded_blocks
    )
    model.fit(engineered, train["top3_label"].astype(int))
    return model, prior_mean


def predict_raw_probability(
    model: Pipeline,
    frame: pd.DataFrame,
    *,
    prior_mean: float,
    prior_strength: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    engineered = engineer_features(
        frame, prior_mean=prior_mean, prior_strength=prior_strength
    )
    raw_logit = np.asarray(model.decision_function(engineered), dtype=float)
    raw_probability = np.asarray(
        model.predict_proba(engineered)[:, 1], dtype=float
    )
    return raw_logit, raw_probability


def _adjust_one_race(
    probabilities: np.ndarray, slots: float
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("probabilities must be a non-empty 1D array")
    if not 0 < slots < probabilities.size:
        raise ValueError("slots must be between 0 and the race field size")
    p = np.clip(probabilities, 1e-9, 1 - 1e-9)
    logits = np.log(p / (1.0 - p))
    low, high = -40.0, 40.0
    for _ in range(100):
        mid = (low + high) / 2.0
        adjusted = 1.0 / (1.0 + np.exp(-(logits + mid)))
        if adjusted.sum() < slots:
            low = mid
        else:
            high = mid
    delta = (low + high) / 2.0
    return 1.0 / (1.0 + np.exp(-(logits + delta)))


def enforce_race_top3_sum(
    frame: pd.DataFrame,
    raw_probability: np.ndarray,
    *,
    race_col: str = "race_id",
    slots: float = 3.0,
) -> np.ndarray:
    """Shift logits by a common race intercept so each race sums exactly to three."""
    if race_col not in frame.columns:
        raise ValueError(f"Missing race column: {race_col}")
    if len(frame) != len(raw_probability):
        raise ValueError("frame and raw_probability length mismatch")
    adjusted = np.empty(len(frame), dtype=float)
    positions = pd.Series(np.arange(len(frame)), index=frame.index)
    for _, group in frame.groupby(race_col, sort=False):
        pos = positions.loc[group.index].to_numpy()
        race_slots = min(float(slots), float(len(pos)) - 1e-9)
        adjusted[pos] = _adjust_one_race(
            np.asarray(raw_probability)[pos], race_slots
        )
    return adjusted


def calibration_table(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    """Return equal-frequency reliability bins for marginal P(top3)."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if len(frame) != len(probability):
        raise ValueError("frame and probability length mismatch")
    work = pd.DataFrame(
        {
            "y": frame["top3_label"].astype(int).to_numpy(),
            "p": np.asarray(probability, dtype=float),
        }
    )
    if work["p"].isna().any():
        raise ValueError("probability contains missing values")
    q = min(bins, len(work))
    work["bin"] = pd.qcut(
        work["p"].rank(method="first"),
        q=q,
        labels=False,
        duplicates="drop",
    )
    out = (
        work.groupby("bin", observed=True)
        .agg(
            rows=("y", "size"),
            mean_predicted=("p", "mean"),
            observed_top3_rate=("y", "mean"),
            min_predicted=("p", "min"),
            max_predicted=("p", "max"),
        )
        .reset_index()
    )
    out["calibration_gap"] = out["observed_top3_rate"] - out["mean_predicted"]
    return out


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Weighted absolute reliability gap."""
    if table.empty:
        raise ValueError("calibration table is empty")
    weights = table["rows"] / table["rows"].sum()
    return float((weights * table["calibration_gap"].abs()).sum())


def evaluate_binary(
    frame: pd.DataFrame, probability: np.ndarray
) -> BinaryEvaluation:
    """Evaluate marginal P(top3) without requiring a complete race cohort."""
    y = frame["top3_label"].astype(int).to_numpy()
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    return BinaryEvaluation(
        brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        rows=len(frame),
        races=int(frame["race_id"].nunique()),
    )


def evaluate(
    frame: pd.DataFrame, adjusted_probability: np.ndarray
) -> Evaluation:
    y = frame["top3_label"].astype(int).to_numpy()
    p = np.clip(
        np.asarray(adjusted_probability, dtype=float), 1e-12, 1 - 1e-12
    )
    sums = pd.Series(p, index=frame.index).groupby(frame["race_id"]).sum()
    return Evaluation(
        brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        mean_race_sum=float(sums.mean()),
        max_abs_race_sum_error=float(
            np.max(np.abs(sums.to_numpy() - 3.0))
        ),
        rows=len(frame),
        races=int(frame["race_id"].nunique()),
    )
