import numpy as np
import pandas as pd

from keiba_place_lab.successor import (
    FEATURE_VARIANTS,
    MODEL_GRIDS,
    add_feature_variant,
    numeric_columns_for_variant,
)


def _engineered_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R2", "R2"],
            "career_top3_shrunk": [0.2, 0.4, 0.3, 0.3],
            "turf_top3_shrunk": [0.2, 0.5, 0.2, 0.4],
            "same_distance_top3_shrunk": [0.1, 0.6, 0.2, 0.5],
            "same_course_top3_shrunk": [0.1, 0.4, 0.3, 0.4],
            "recent3_top3_count": [0.0, 2.0, 1.0, 3.0],
            "recent3_finish_pct_mean": [0.8, 0.2, 0.6, 0.1],
            "recent3_relative_time_mean": [0.02, 0.00, 0.01, -0.01],
            "recent3_open_plus_count": [0.0, 2.0, 1.0, 3.0],
            "recent3_graded_count": [0.0, 1.0, 0.0, 2.0],
            "distance_change_from_prev_m": [-200.0, 0.0, 400.0, -400.0],
            "assigned_weight_delta_from_prev_kg": [-1.0, 1.0, 2.0, -2.0],
            "draw_pct": [0.1, 0.8, 0.2, 0.7],
            "recent4_early_pos_pct_mean": [0.2, 0.8, 0.4, 0.6],
            "front_forward_share": [0.5, 0.5, 0.25, 0.25],
            "age": [3.0, 5.0, 4.0, 6.0],
            "log_days_since_prev": np.log1p([21.0, 42.0, 28.0, 90.0]),
        }
    )


def test_feature_variants_are_frozen():
    assert FEATURE_VARIANTS == (
        "base",
        "relative",
        "recent_deviation",
        "interactions",
        "all",
    )
    assert set(MODEL_GRIDS) == {
        "logistic",
        "random_forest",
        "hist_gradient_boosting",
        "xgboost",
    }
    assert len(MODEL_GRIDS["logistic"]) == 1
    assert len(MODEL_GRIDS["random_forest"]) == 3
    assert len(MODEL_GRIDS["hist_gradient_boosting"]) == 3
    assert len(MODEL_GRIDS["xgboost"]) == 3


def test_relative_features_use_only_same_race_values():
    frame = _engineered_frame()
    out = add_feature_variant(frame, "relative")
    col = "rel__career_top3_shrunk__centered"
    assert out.loc[frame["race_id"].eq("R1"), col].mean() == 0.0
    assert out.loc[frame["race_id"].eq("R2"), col].mean() == 0.0
    assert (
        out.loc[1, "rel__career_top3_shrunk__pct"]
        > out.loc[0, "rel__career_top3_shrunk__pct"]
    )


def test_lower_is_better_relative_signal_is_oriented_upward():
    frame = _engineered_frame()
    out = add_feature_variant(frame, "relative")
    col = "rel__recent3_finish_pct_mean__pct"
    assert out.loc[1, col] > out.loc[0, col]


def test_recent_deviation_and_interactions_do_not_use_target_columns():
    frame = _engineered_frame()
    out = add_feature_variant(frame, "all")
    assert "top3_label" not in out.columns
    assert out.loc[1, "recent3_top3_rate"] == 2.0 / 3.0
    assert out.loc[0, "abs_distance_change_m"] == 200.0
    assert out.loc[0, "interaction_draw_early"] == 0.02


def test_numeric_column_registry_expands_monotonically():
    base = set(numeric_columns_for_variant("base"))
    all_cols = set(numeric_columns_for_variant("all"))
    assert base < all_cols
