"""Phase-2 model-family comparison for Stage 4 successor development."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .model_selection import candidate_score, paired_delta
from .nonmarket import (
    engineer_features,
    feature_columns,
    fit_model,
    predict_raw_probability,
    validate_training_frame,
)

RANDOM_SEED = 20261003
PRIOR_STRENGTH = 6.0
INCUMBENT_NAME = "l2_logistic_c0.1"


@dataclass(frozen=True)
class CandidateConfig:
    family: str
    config_id: str
    complexity_rank: int
    params: dict[str, Any]


RF_CONFIGS = (
    CandidateConfig("random_forest", "RF01", 1, {"max_depth": 5, "min_samples_leaf": 50, "max_features": "sqrt"}),
    CandidateConfig("random_forest", "RF02", 2, {"max_depth": 5, "min_samples_leaf": 20, "max_features": "sqrt"}),
    CandidateConfig("random_forest", "RF03", 3, {"max_depth": 8, "min_samples_leaf": 50, "max_features": "sqrt"}),
    CandidateConfig("random_forest", "RF04", 4, {"max_depth": 8, "min_samples_leaf": 20, "max_features": "sqrt"}),
    CandidateConfig("random_forest", "RF05", 5, {"max_depth": 8, "min_samples_leaf": 20, "max_features": 0.7}),
    CandidateConfig("random_forest", "RF06", 6, {"max_depth": None, "min_samples_leaf": 30, "max_features": 0.7}),
)

HGB_CONFIGS = (
    CandidateConfig("hist_gradient_boosting", "HGB01", 1, {"learning_rate": 0.03, "max_leaf_nodes": 15, "min_samples_leaf": 50, "l2_regularization": 2.0}),
    CandidateConfig("hist_gradient_boosting", "HGB02", 2, {"learning_rate": 0.03, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 2.0}),
    CandidateConfig("hist_gradient_boosting", "HGB03", 3, {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 30, "l2_regularization": 1.0}),
    CandidateConfig("hist_gradient_boosting", "HGB04", 4, {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 30, "l2_regularization": 1.0}),
    CandidateConfig("hist_gradient_boosting", "HGB05", 5, {"learning_rate": 0.08, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 1.0}),
    CandidateConfig("hist_gradient_boosting", "HGB06", 6, {"learning_rate": 0.08, "max_leaf_nodes": 63, "min_samples_leaf": 20, "l2_regularization": 0.0}),
)

XGB_CONFIGS = (
    CandidateConfig("xgboost", "XGB01", 1, {"learning_rate": 0.03, "max_depth": 3, "min_child_weight": 20, "reg_lambda": 5.0}),
    CandidateConfig("xgboost", "XGB02", 2, {"learning_rate": 0.03, "max_depth": 4, "min_child_weight": 10, "reg_lambda": 5.0}),
    CandidateConfig("xgboost", "XGB03", 3, {"learning_rate": 0.05, "max_depth": 3, "min_child_weight": 10, "reg_lambda": 2.0}),
    CandidateConfig("xgboost", "XGB04", 4, {"learning_rate": 0.05, "max_depth": 5, "min_child_weight": 10, "reg_lambda": 2.0}),
    CandidateConfig("xgboost", "XGB05", 5, {"learning_rate": 0.08, "max_depth": 4, "min_child_weight": 5, "reg_lambda": 1.0}),
    CandidateConfig("xgboost", "XGB06", 6, {"learning_rate": 0.08, "max_depth": 6, "min_child_weight": 5, "reg_lambda": 1.0}),
)

FAMILY_CONFIGS = {
    "random_forest": RF_CONFIGS,
    "hist_gradient_boosting": HGB_CONFIGS,
    "xgboost": XGB_CONFIGS,
}


def registry_frame() -> pd.DataFrame:
    rows = []
    for family, configs in FAMILY_CONFIGS.items():
        for cfg in configs:
            rows.append(
                {
                    "family": family,
                    "config_id": cfg.config_id,
                    "complexity_rank": cfg.complexity_rank,
                    **cfg.params,
                }
            )
    return pd.DataFrame(rows)


def make_tree_preprocessor() -> ColumnTransformer:
    numeric, categorical = feature_columns()
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
    return ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def make_challenger_pipeline(config: CandidateConfig) -> Pipeline:
    if config.family == "random_forest":
        model = RandomForestClassifier(
            n_estimators=500,
            criterion="log_loss",
            bootstrap=True,
            class_weight=None,
            n_jobs=2,
            random_state=RANDOM_SEED,
            **config.params,
        )
    elif config.family == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=300,
            early_stopping=False,
            random_state=RANDOM_SEED,
            **config.params,
        )
    elif config.family == "xgboost":
        from xgboost import XGBClassifier

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
            **config.params,
        )
    else:
        raise ValueError(f"Unknown family: {config.family}")
    return Pipeline([("preprocess", make_tree_preprocessor()), ("model", model)])


def fit_challenger(
    train: pd.DataFrame,
    config: CandidateConfig,
) -> tuple[Pipeline, float]:
    validate_training_frame(train)
    prior_mean = float(train["top3_label"].mean())
    engineered = engineer_features(
        train,
        prior_mean=prior_mean,
        prior_strength=PRIOR_STRENGTH,
    )
    model = make_challenger_pipeline(config)
    model.fit(engineered, train["top3_label"].astype(int))
    return model, prior_mean


def predict_challenger(
    model: Pipeline,
    frame: pd.DataFrame,
    *,
    prior_mean: float,
) -> np.ndarray:
    engineered = engineer_features(
        frame,
        prior_mean=prior_mean,
        prior_strength=PRIOR_STRENGTH,
    )
    probability = np.asarray(model.predict_proba(engineered)[:, 1], dtype=float)
    if not np.isfinite(probability).all():
        raise ValueError("challenger probability contains non-finite values")
    return np.clip(probability, 0.0, 1.0)


def fit_predict_incumbent(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> np.ndarray:
    model, prior_mean = fit_model(
        train,
        c_value=0.1,
        prior_strength=PRIOR_STRENGTH,
    )
    _, probability = predict_raw_probability(
        model,
        evaluation,
        prior_mean=prior_mean,
        prior_strength=PRIOR_STRENGTH,
    )
    return probability


def select_inner_config(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    configs: tuple[CandidateConfig, ...],
) -> tuple[CandidateConfig, pd.DataFrame]:
    scores = []
    config_map = {cfg.config_id: cfg for cfg in configs}
    for config_id, probability in predictions.items():
        score = candidate_score(frame, probability, name=config_id)
        cfg = config_map[config_id]
        scores.append(
            {
                **score.__dict__,
                "config_id": config_id,
                "complexity_rank": cfg.complexity_rank,
            }
        )
    table = pd.DataFrame(scores).sort_values(
        ["race_macro_brier", "race_macro_log_loss", "config_id"],
        kind="stable",
    )
    best_id = str(table.iloc[0]["config_id"])
    diagnostics = []
    for config_id, probability in predictions.items():
        delta = paired_delta(
            frame,
            probability,
            predictions[best_id],
            candidate_name=config_id,
            reference_name=best_id,
        )
        diagnostics.append(
            {
                "config_id": config_id,
                "mean_brier_delta_vs_inner_best": delta.mean_brier_delta,
                "se_brier_delta_vs_inner_best": delta.se_brier_delta,
                "within_one_se_of_inner_best": (
                    delta.mean_brier_delta <= delta.se_brier_delta + 1e-15
                ),
            }
        )
    table = table.merge(pd.DataFrame(diagnostics), on="config_id", how="left")
    eligible = table.loc[table["within_one_se_of_inner_best"]].sort_values(
        [
            "complexity_rank",
            "race_macro_brier",
            "race_macro_log_loss",
            "config_id",
        ],
        kind="stable",
    )
    selected_id = str(eligible.iloc[0]["config_id"])
    table["inner_point_best"] = table["config_id"].eq(best_id)
    table["inner_selected"] = table["config_id"].eq(selected_id)
    return config_map[selected_id], table.reset_index(drop=True)
