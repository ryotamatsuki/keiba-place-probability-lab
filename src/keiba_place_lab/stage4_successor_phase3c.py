"""Phase 3C explicit interaction block on top of the frozen Phase-3A incumbent."""

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

INTERACTION_FEATURES = (
    "interaction_draw_x_front_tendency",
    "interaction_front_tendency_x_race_pressure",
    "interaction_rest_x_age",
)


def add_phase3c_features(
    eligible: pd.DataFrame,
    full_field_context: pd.DataFrame,
    *,
    prior_mean: float,
    prior_strength: float = 6.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add frozen Phase-3A relative features plus three explicit interactions."""
    out, relative_coverage = add_relative_ability_features(
        eligible,
        full_field_context,
        prior_mean=prior_mean,
        prior_strength=prior_strength,
    )

    draw = pd.to_numeric(out["draw_pct"], errors="coerce")
    early = pd.to_numeric(out["recent4_early_pos_pct_mean"], errors="coerce")
    pressure = pd.to_numeric(out["front_forward_share"], errors="coerce")
    rest = pd.to_numeric(out["log_days_since_prev"], errors="coerce")
    age = pd.to_numeric(out["age"], errors="coerce")
    front_tendency = 1.0 - early

    out["interaction_draw_x_front_tendency"] = draw * front_tendency
    out["interaction_front_tendency_x_race_pressure"] = front_tendency * pressure
    out["interaction_rest_x_age"] = rest * age

    rows = []
    for feature in INTERACTION_FEATURES:
        rows.append(
            {
                "block": "interaction",
                "feature": feature,
                "eligible_rows": len(out),
                "eligible_nonmissing": int(out[feature].notna().sum()),
                "eligible_coverage": float(out[feature].notna().mean()),
            }
        )
    coverage = pd.concat(
        [
            relative_coverage.assign(block="relative_ability"),
            pd.DataFrame(rows),
        ],
        ignore_index=True,
        sort=False,
    )
    return out, coverage


def make_xgb01_phase3c_pipeline() -> Pipeline:
    """Frozen XGB01 with Phase-3A relative features and Phase-3C interactions."""
    from xgboost import XGBClassifier

    numeric, categorical = feature_columns()
    numeric = [*numeric, *RELATIVE_FEATURES, *INTERACTION_FEATURES]
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


def fit_phase3c_candidate(
    train: pd.DataFrame,
    full_field_context: pd.DataFrame,
    *,
    prior_strength: float = 6.0,
) -> tuple[Pipeline, float, pd.DataFrame]:
    validate_training_frame(train)
    prior_mean = float(train["top3_label"].mean())
    engineered, coverage = add_phase3c_features(
        train,
        full_field_context,
        prior_mean=prior_mean,
        prior_strength=prior_strength,
    )
    model = make_xgb01_phase3c_pipeline()
    model.fit(engineered, train["top3_label"].astype(int))
    return model, prior_mean, coverage


def predict_phase3c_candidate(
    model: Pipeline,
    frame: pd.DataFrame,
    full_field_context: pd.DataFrame,
    *,
    prior_mean: float,
    prior_strength: float = 6.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    engineered, coverage = add_phase3c_features(
        frame,
        full_field_context,
        prior_mean=prior_mean,
        prior_strength=prior_strength,
    )
    probability = np.asarray(model.predict_proba(engineered)[:, 1], dtype=float)
    if not np.isfinite(probability).all():
        raise ValueError("Phase-3C probability contains non-finite values")
    return np.clip(probability, 0.0, 1.0), coverage
