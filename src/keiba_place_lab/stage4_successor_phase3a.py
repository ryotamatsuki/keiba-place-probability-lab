"""Phase 3A relative-ability feature block for the frozen XGB01 model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .nonmarket import engineer_features, feature_columns, validate_training_frame
from .stage4_successor_phase2 import RANDOM_SEED

RELATIVE_FEATURES = (
    "rel_career_top3_vs_others",
    "rel_same_distance_top3_vs_others",
    "rel_same_course_top3_vs_others",
    "rel_recent3_finish_vs_others",
    "rel_recent3_time_vs_others",
)

SOURCE_RULES = (
    ("career_top3_shrunk", "rel_career_top3_vs_others", 1.0),
    ("same_distance_top3_shrunk", "rel_same_distance_top3_vs_others", 1.0),
    ("same_course_top3_shrunk", "rel_same_course_top3_vs_others", 1.0),
    ("recent3_finish_pct_mean", "rel_recent3_finish_vs_others", -1.0),
    ("recent3_relative_time_mean", "rel_recent3_time_vs_others", -1.0),
)


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["race_id"] = out["race_id"].astype("string")
    out["horse_id"] = out["horse_id"].astype("string")
    return out


def assert_full_field_context(context: pd.DataFrame) -> None:
    """Require one context row per actual starter for every race."""
    required = {"race_id", "horse_id", "field_size"}
    missing = required.difference(context.columns)
    if missing:
        raise ValueError(f"Missing full-field context columns: {sorted(missing)}")
    if context.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Duplicate race_id x horse_id rows in context")

    unique_sizes = context.groupby("race_id")["field_size"].nunique()
    if unique_sizes.gt(1).any():
        raise ValueError("field_size is not constant within context race")
    observed = context.groupby("race_id").size()
    expected = context.groupby("race_id")["field_size"].first().astype(int)
    bad = observed.ne(expected)
    if bad.any():
        examples = pd.DataFrame(
            {"observed": observed[bad], "expected": expected[bad]}
        ).head(20)
        raise ValueError(
            "Relative-ability context is not a complete starter field: "
            + examples.to_dict(orient="index").__repr__()
        )


def _leave_one_out_mean(
    frame: pd.DataFrame,
    source_col: str,
) -> pd.Series:
    values = pd.to_numeric(frame[source_col], errors="coerce")
    valid = values.notna().astype(int)
    sums = values.fillna(0.0).groupby(frame["race_id"]).transform("sum")
    counts = valid.groupby(frame["race_id"]).transform("sum")
    peer_sum = sums - values.fillna(0.0)
    peer_count = counts - valid
    mean = peer_sum / peer_count.where(peer_count.gt(0))
    return mean.where(values.notna())


def add_relative_ability_features(
    eligible: pd.DataFrame,
    full_context: pd.DataFrame,
    *,
    prior_mean: float,
    prior_strength: float = 6.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Engineer base features and attach full-field leave-one-out relative features."""
    eligible_norm = _normalize_keys(eligible)
    context_norm = _normalize_keys(full_context)
    assert_full_field_context(context_norm)

    eligible_races = set(eligible_norm["race_id"].dropna())
    context_races = set(context_norm["race_id"].dropna())
    missing_races = sorted(eligible_races.difference(context_races))
    if missing_races:
        raise ValueError(f"Missing full-field context races: {missing_races[:20]}")

    extra_context = context_norm.loc[
        context_norm["race_id"].isin(eligible_races)
    ].copy()
    assert_full_field_context(extra_context)

    engineered_context = engineer_features(
        extra_context,
        prior_mean=prior_mean,
        prior_strength=prior_strength,
    )

    relative = engineered_context[["race_id", "horse_id"]].copy()
    coverage_rows: list[dict[str, object]] = []
    for source_col, output_col, direction in SOURCE_RULES:
        own = pd.to_numeric(engineered_context[source_col], errors="coerce")
        peer_mean = _leave_one_out_mean(engineered_context, source_col)
        if direction > 0:
            relative[output_col] = own - peer_mean
        else:
            relative[output_col] = peer_mean - own
        coverage_rows.append(
            {
                "feature": output_col,
                "source_feature": source_col,
                "context_rows": len(engineered_context),
                "context_nonmissing": int(own.notna().sum()),
                "relative_nonmissing": int(relative[output_col].notna().sum()),
                "relative_coverage": float(relative[output_col].notna().mean()),
            }
        )

    engineered_eligible = engineer_features(
        eligible_norm,
        prior_mean=prior_mean,
        prior_strength=prior_strength,
    )
    out = engineered_eligible.merge(
        relative,
        on=["race_id", "horse_id"],
        how="left",
        validate="one_to_one",
    )
    if len(out) != len(eligible_norm):
        raise ValueError("Relative-feature merge changed eligible row count")

    coverage = pd.DataFrame(coverage_rows)
    eligible_nonmissing = {
        feature: int(out[feature].notna().sum()) for feature in RELATIVE_FEATURES
    }
    coverage["eligible_rows"] = len(out)
    coverage["eligible_nonmissing"] = coverage["feature"].map(eligible_nonmissing)
    coverage["eligible_coverage"] = (
        coverage["eligible_nonmissing"] / coverage["eligible_rows"]
    )
    return out, coverage


def make_xgb01_relative_pipeline() -> Pipeline:
    """Build frozen XGB01 with the Phase-3A numeric block appended."""
    from xgboost import XGBClassifier

    numeric, categorical = feature_columns()
    numeric = [*numeric, *RELATIVE_FEATURES]
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


def fit_relative_candidate(
    train: pd.DataFrame,
    full_context: pd.DataFrame,
    *,
    prior_strength: float = 6.0,
) -> tuple[Pipeline, float, pd.DataFrame]:
    validate_training_frame(train)
    prior_mean = float(train["top3_label"].mean())
    engineered, coverage = add_relative_ability_features(
        train,
        full_context,
        prior_mean=prior_mean,
        prior_strength=prior_strength,
    )
    model = make_xgb01_relative_pipeline()
    model.fit(engineered, train["top3_label"].astype(int))
    return model, prior_mean, coverage


def predict_relative_candidate(
    model: Pipeline,
    frame: pd.DataFrame,
    full_context: pd.DataFrame,
    *,
    prior_mean: float,
    prior_strength: float = 6.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    engineered, coverage = add_relative_ability_features(
        frame,
        full_context,
        prior_mean=prior_mean,
        prior_strength=prior_strength,
    )
    probability = np.asarray(model.predict_proba(engineered)[:, 1], dtype=float)
    if not np.isfinite(probability).all():
        raise ValueError("Relative candidate probability contains non-finite values")
    return np.clip(probability, 0.0, 1.0), coverage
