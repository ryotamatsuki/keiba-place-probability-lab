import numpy as np
import pandas as pd
import pytest

from keiba_place_lab.nonmarket import (
    enforce_race_top3_sum,
    engineer_features,
    feature_columns,
    validate_market_free,
)


def test_market_columns_are_rejected():
    with pytest.raises(ValueError, match="Forbidden market columns"):
        validate_market_free(["age", "win_odds"])


def test_common_beta_shrinkage_handles_zero_and_small_samples():
    frame = pd.DataFrame(
        {
            "days_since_prev": [10, 10, 10],
            "career_top3": [0, 1, 5],
            "career_starts": [0, 1, 10],
            "turf_top3": [0, 1, 5],
            "turf_starts": [0, 1, 10],
            "same_distance_top3": [0, 1, 5],
            "same_distance_starts": [0, 1, 10],
            "same_course_top3": [0, 1, 5],
            "same_course_starts": [0, 1, 10],
        }
    )
    out = engineer_features(frame, prior_mean=0.2, prior_strength=6.0)
    assert out.loc[0, "career_top3_shrunk"] == pytest.approx(0.2)
    assert out.loc[1, "career_top3_shrunk"] < 1.0
    assert out.loc[2, "career_top3_shrunk"] > out.loc[0, "career_top3_shrunk"]


def test_race_adjustment_sums_to_three_and_preserves_order():
    frame = pd.DataFrame({"race_id": ["A"] * 5 + ["B"] * 6})
    raw = np.array(
        [0.05, 0.10, 0.20, 0.40, 0.60, 0.03, 0.08, 0.12, 0.22, 0.33, 0.50]
    )
    adjusted = enforce_race_top3_sum(frame, raw)
    sums = pd.Series(adjusted).groupby(frame["race_id"]).sum()
    assert sums["A"] == pytest.approx(3.0, abs=1e-9)
    assert sums["B"] == pytest.approx(3.0, abs=1e-9)
    assert list(np.argsort(raw[:5])) == list(np.argsort(adjusted[:5]))
    assert list(np.argsort(raw[5:])) == list(np.argsort(adjusted[5:]))


def test_feature_block_ablation_is_explicit():
    full_numeric, full_cat = feature_columns()
    no_history_numeric, no_history_cat = feature_columns(["history"])
    assert "career_top3_shrunk" in full_numeric
    assert "career_top3_shrunk" not in no_history_numeric
    assert full_cat == no_history_cat
