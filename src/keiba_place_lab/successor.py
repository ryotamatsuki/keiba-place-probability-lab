"""Stage 4 successor-model development helpers.

The experiment is deliberately market-free. Candidate tuning uses only dates strictly
before each outer evaluation year; 2025 is never read by the selection routine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .nonmarket import (
    CATEGORICAL_BLOCKS,
    NUMERIC_BLOCKS,
    engineer_features,
    validate_training_frame,
)

RANDOM_STATE = 20261003
PRIOR_STRENGTH = 6.0

BASE_NUMERIC = tuple(col for cols in NUMERIC_BLOCKS.values() for col in cols)
BASE_CATEGORICAL = tuple(col for cols in CATEGORICAL_BLOCKS.values() for col in cols)

RELATIVE_SIGNALS = {
    "career_top3_shrunk": 1.0,
    "turf_top3_shrunk": 1.0,
    "same_distance_top3_shrunk": 1.0,
    "same_course_top3_shrunk": 1.0,
    "recent3_top3_count": 1.0,
    "recent3_finish_pct_mean": -1.0,
    "recent3_relative_time_mean": -1.0,
}

FEATURE_VARIANTS = ("base", "relative", "recent_deviation", "interactions", "all")

MODEL_GRIDS: dict[str, tuple[dict[str, Any], ...]] = {
    "logistic": ({"C": 0.1},),
    "random_forest": (
        {
            "n_estimators": 400,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 400,
            "max_depth": 12,
            "min_samples_leaf": 10,
            "max_features": 0.7,
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 40,
            "max_features": "sqrt",
        },
    ),
    "hist_gradient_boosting": (
        {
            "max_iter": 250,
            "learning_rate": 0.05,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 30,
            "l2_regularization": 1.0,
        },
        {
            "max_iter": 300,
            "learning_rate": 0.04,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "l2_regularization": 2.0,
        },
        {
            "max_iter": 300,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 60,
            "l2_regularization": 5.0,
        },
    ),
    "xgboost": (
        {
            "n_estimators": 400,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 20.0,
            "reg_lambda": 5.0,
            "reg_alpha": 0.0,
        },
        {
            "n_estimators": 350,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_child_weight": 10.0,
            "reg_lambda": 5.0,
            "reg_alpha": 0.0,
        },
        {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 10.0,
            "reg_lambda": 10.0,
            "reg_alpha": 0.2,
        },
    ),
}


@dataclass(frozen=True)
class FittedForecaster:
    model: Pipeline
    prior_mean: float
    family: str
    config: dict[str, Any]
    feature_variant: str


def config_id(family: str, index: int) -> str:
    return f"{family}__{index:02d}"


def _race_relative_feature(
    frame: pd.DataFrame,
    column: str,
    direction: float,
) -> tuple[pd.Series, pd.Series]:
    oriented = pd.to_numeric(frame[column], errors="coerce") * direction
    race_mean = oriented.groupby(frame["race_id"], sort=False).transform("mean")
    centered = oriented - race_mean
    pct = oriented.groupby(frame["race_id"], sort=False).rank(
        method="average", pct=True, na_option="keep"
    )
    return centered, pct


def add_feature_variant(engineered: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Add only leakage-safe transforms of already pre-race features."""
    if variant not in FEATURE_VARIANTS:
        raise ValueError(f"Unknown feature variant: {variant}")
    out = engineered.copy()

    if variant in {"relative", "all"}:
        for column, direction in RELATIVE_SIGNALS.items():
            centered, pct = _race_relative_feature(out, column, direction)
            out[f"rel__{column}__centered"] = centered
            out[f"rel__{column}__pct"] = pct

    if variant in {"recent_deviation", "all"}:
        out["recent3_top3_rate"] = pd.to_numeric(
            out["recent3_top3_count"], errors="coerce"
        ) / 3.0
        out["recent_vs_career_top3"] = (
            out["recent3_top3_rate"] - out["career_top3_shrunk"]
        )
        out["recent_vs_turf_top3"] = (
            out["recent3_top3_rate"] - out["turf_top3_shrunk"]
        )
        out["recent_open_share"] = pd.to_numeric(
            out["recent3_open_plus_count"], errors="coerce"
        ) / 3.0
        out["recent_graded_share"] = pd.to_numeric(
            out["recent3_graded_count"], errors="coerce"
        ) / 3.0
        out["abs_distance_change_m"] = pd.to_numeric(
            out["distance_change_from_prev_m"], errors="coerce"
        ).abs()
        out["abs_weight_change_kg"] = pd.to_numeric(
            out["assigned_weight_delta_from_prev_kg"], errors="coerce"
        ).abs()

    if variant in {"interactions", "all"}:
        out["interaction_draw_early"] = out["draw_pct"] * out[
            "recent4_early_pos_pct_mean"
        ]
        out["interaction_early_front_share"] = out[
            "recent4_early_pos_pct_mean"
        ] * out["front_forward_share"]
        out["interaction_draw_front_share"] = out["draw_pct"] * out[
            "front_forward_share"
        ]
        out["interaction_age_log_days"] = out["age"] * out["log_days_since_prev"]
        out["interaction_age_weight_delta"] = out["age"] * out[
            "assigned_weight_delta_from_prev_kg"
        ]

    return out


def numeric_columns_for_variant(variant: str) -> list[str]:
    columns = list(BASE_NUMERIC)
    if variant in {"relative", "all"}:
        for column in RELATIVE_SIGNALS:
            columns.extend(
                [f"rel__{column}__centered", f"rel__{column}__pct"]
            )
    if variant in {"recent_deviation", "all"}:
        columns.extend(
            [
                "recent3_top3_rate",
                "recent_vs_career_top3",
                "recent_vs_turf_top3",
                "recent_open_share",
                "recent_graded_share",
                "abs_distance_change_m",
                "abs_weight_change_kg",
            ]
        )
    if variant in {"interactions", "all"}:
        columns.extend(
            [
                "interaction_draw_early",
                "interaction_early_front_share",
                "interaction_draw_front_share",
                "interaction_age_log_days",
                "interaction_age_weight_delta",
            ]
        )
    return columns


def _preprocessor(*, numeric: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
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
    return ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, list(BASE_CATEGORICAL)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def make_model(family: str, config: dict[str, Any], feature_variant: str) -> Pipeline:
    numeric = numeric_columns_for_variant(feature_variant)
    if family == "logistic":
        pre = _preprocessor(numeric=numeric, scale_numeric=True)
        estimator = LogisticRegression(
            C=float(config["C"]),
            penalty="l2",
            solver="lbfgs",
            max_iter=3000,
            random_state=RANDOM_STATE,
        )
    elif family == "random_forest":
        pre = _preprocessor(numeric=numeric, scale_numeric=False)
        estimator = RandomForestClassifier(
            **config,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            criterion="log_loss",
            bootstrap=True,
        )
    elif family == "hist_gradient_boosting":
        pre = _preprocessor(numeric=numeric, scale_numeric=False)
        estimator = HistGradientBoostingClassifier(
            **config,
            loss="log_loss",
            early_stopping=False,
            random_state=RANDOM_STATE,
        )
    elif family == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - optional dependency gate
            raise RuntimeError("xgboost is required for the successor experiment") from exc
        pre = _preprocessor(numeric=numeric, scale_numeric=False)
        estimator = XGBClassifier(
            **config,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=2,
            verbosity=0,
        )
    else:
        raise ValueError(f"Unknown family: {family}")
    return Pipeline([("preprocess", pre), ("model", estimator)])


def prepare_features(
    frame: pd.DataFrame,
    *,
    prior_mean: float,
    feature_variant: str,
) -> pd.DataFrame:
    engineered = engineer_features(
        frame,
        prior_mean=prior_mean,
        prior_strength=PRIOR_STRENGTH,
    )
    return add_feature_variant(engineered, feature_variant)


def fit_forecaster(
    train: pd.DataFrame,
    *,
    family: str,
    config: dict[str, Any],
    feature_variant: str = "base",
) -> FittedForecaster:
    validate_training_frame(train)
    prior_mean = float(train["top3_label"].mean())
    x = prepare_features(
        train,
        prior_mean=prior_mean,
        feature_variant=feature_variant,
    )
    model = make_model(family, config, feature_variant)
    model.fit(x, train["top3_label"].astype(int))
    return FittedForecaster(
        model=model,
        prior_mean=prior_mean,
        family=family,
        config=dict(config),
        feature_variant=feature_variant,
    )


def predict_probability(fitted: FittedForecaster, frame: pd.DataFrame) -> np.ndarray:
    x = prepare_features(
        frame,
        prior_mean=fitted.prior_mean,
        feature_variant=fitted.feature_variant,
    )
    p = np.asarray(fitted.model.predict_proba(x)[:, 1], dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("model produced invalid probabilities")
    return p
