"""Phase 3B recent-trend block on top of the frozen Phase-3A model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .nonmarket import feature_columns, validate_training_frame
from .stage4_successor_phase2 import RANDOM_SEED
from .stage4_successor_phase3a import (
    RELATIVE_FEATURES,
    add_relative_ability_features,
)

TREND_FEATURES = (
    "trend_recent3_finish_improvement",
    "trend_recent3_time_improvement",
    "trend_recent3_top3_count_change",
    "trend_recent4_early_forward_change",
)

TREND_SOURCES = (
    "recent3_finish_pct_mean",
    "recent3_relative_time_mean",
    "recent3_top3_count",
    "recent4_early_pos_pct_mean",
)


def build_recent_trend_table(history: pd.DataFrame) -> pd.DataFrame:
    """Build pre-race trend features from each horse's immediately prior panel row."""
    required = {
        "race_date",
        "race_id",
        "horse_id",
        *TREND_SOURCES,
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Missing recent-trend history columns: {sorted(missing)}")

    x = history[list(required)].copy()
    x["race_date"] = pd.to_datetime(x["race_date"], errors="raise")
    x["race_id"] = x["race_id"].astype("string")
    x["horse_id"] = x["horse_id"].astype("string")
    if x.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Duplicate race_id x horse_id in trend history")

    x = x.sort_values(
        ["horse_id", "race_date", "race_id"],
        kind="stable",
    ).reset_index(drop=True)
    horse = x.groupby("horse_id", sort=False)
    previous_date = horse["race_date"].shift(1)
    bad = previous_date.notna() & previous_date.ge(x["race_date"])
    if bad.any():
        raise ValueError("Recent-trend history is not strictly chronological")

    previous_finish = horse["recent3_finish_pct_mean"].shift(1)
    previous_time = horse["recent3_relative_time_mean"].shift(1)
    previous_top3 = horse["recent3_top3_count"].shift(1)
    previous_early = horse["recent4_early_pos_pct_mean"].shift(1)

    out = x[["race_id", "horse_id"]].copy()
    out["trend_recent3_finish_improvement"] = (
        previous_finish - pd.to_numeric(x["recent3_finish_pct_mean"], errors="coerce")
    )
    out["trend_recent3_time_improvement"] = (
        previous_time
        - pd.to_numeric(x["recent3_relative_time_mean"], errors="coerce")
    )
    out["trend_recent3_top3_count_change"] = (
        pd.to_numeric(x["recent3_top3_count"], errors="coerce") - previous_top3
    )
    out["trend_recent4_early_forward_change"] = (
        previous_early
        - pd.to_numeric(x["recent4_early_pos_pct_mean"], errors="coerce")
    )
    return out


def add_phase3b_features(
    eligible: pd.DataFrame,
    full_field_context: pd.DataFrame,
    horse_history: pd.DataFrame,
    *,
    prior_mean: float,
    prior_strength: float = 6.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add frozen Phase-3A relative features plus the Phase-3B trend block."""
    base, relative_coverage = add_relative_ability_features(
        eligible,
        full_field_context,
        prior_mean=prior_mean,
        prior_strength=prior_strength,
    )

    trend = build_recent_trend_table(horse_history)
    out = base.merge(
        trend,
        on=["race_id", "horse_id"],
        how="left",
        validate="one_to_one",
    )
    if len(out) != len(eligible):
        raise ValueError("Trend-feature merge changed eligible row count")

    coverage_rows = []
    for feature in TREND_FEATURES:
        coverage_rows.append(
            {
                "feature": feature,
                "eligible_rows": len(out),
                "eligible_nonmissing": int(out[feature].notna().sum()),
                "eligible_coverage": float(out[feature].notna().mean()),
            }
        )
    trend_coverage = pd.DataFrame(coverage_rows)
    relative_coverage = relative_coverage.copy()
    relative_coverage["block"] = "relative_ability"
    trend_coverage["block"] = "recent_trend"
    coverage = pd.concat(
        [relative_coverage, trend_coverage],
        ignore_index=True,
        sort=False,
    )
    return out, coverage


def make_xgb01_phase3b_pipeline() -> Pipeline:
    """Frozen XGB01 with Phase-3A relative and Phase-3B trend numeric features."""
    from xgboost import XGBClassifier

    numeric, categorical = feature_columns()
    numeric = [*numeric, *RELATIVE_FEATURES, *TREND_FEATURES]
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                    sparse_output=False,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        n_jobs=2,
        random_state=RANDOM_SEED,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=20,
        reg_lambda=5.0,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def fit_phase3b_candidate(
    train: pd.DataFrame,
    full_field_context: pd.DataFrame,
    horse_history: pd.DataFrame,
    *,
    prior_strength: float = 6.0,
) -> tuple[Pipeline, float, pd.DataFrame]:
    validate_training_frame(train)
    prior_mean = float(train["top3_label"].mean())
    engineered, coverage = add_phase3b_features(
        train,
        full_field_context,
        horse_history,
        prior_mean=prior_mean,
        prior_strength=prior_strength,
    )
    model = make_xgb01_phase3b_pipeline()
    model.fit(engineered, train["top3_label"].astype(int))
    return model, prior_mean, coverage


def predict_phase3b_candidate(
    model: Pipeline,
    frame: pd.DataFrame,
    full_field_context: pd.DataFrame,
    horse_history: pd.DataFrame,
    *,
    prior_mean: float,
    prior_strength: float = 6.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    engineered, coverage = add_phase3b_features(
        frame,
        full_field_context,
        horse_history,
        prior_mean=prior_mean,
        prior_strength=prior_strength,
    )
    probability = np.asarray(model.predict_proba(engineered)[:, 1], dtype=float)
    if not np.isfinite(probability).all():
        raise ValueError("Phase-3B probability contains non-finite values")
    return np.clip(probability, 0.0, 1.0), coverage
