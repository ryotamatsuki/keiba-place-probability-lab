import numpy as np
import pandas as pd
import pytest

from keiba_place_lab.scope_features import augment_distance_history, distance_regime, surface_base


def records():
    return pd.DataFrame({
        "horse_id": ["a"] * 6,
        "race_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-03-01", "2020-04-01", "2020-05-01"]),
        "surface": ["turf", "turf", "dirt", "turf", "turf", "turf"],
        "distance_m": [1000, 1400, 1200, 1200, 1200, 1600],
        "racecourse": ["A", "A", "A", "A", "A", "B"],
        "top3_label": [1, 0, 1, 1, 0, 1],
    })


def test_future_and_same_day_labels_cannot_change_history():
    original = records()
    first = augment_distance_history(original)
    changed = original.copy()
    changed.loc[3:, "top3_label"] = 1 - changed.loc[3:, "top3_label"]
    changed = pd.concat([changed, pd.DataFrame({
        "horse_id": ["a"], "race_date": [pd.Timestamp("2030-01-01")],
        "surface": ["turf"], "distance_m": [1200], "racecourse": ["A"], "top3_label": [1],
    })], ignore_index=True)
    second = augment_distance_history(changed)
    cols = [c for c in first if c.endswith(("_starts", "_top3"))]
    pd.testing.assert_frame_equal(first.loc[:3, cols], second.loc[:3, cols])
    assert first.loc[3, "near_starts"] == 2
    assert first.loc[4, "near_starts"] == 3
    assert first.loc[4, "near_top3"] == 2
    assert first.loc[4, "course_distance_starts"] == 1
    assert first.loc[5, "near_starts"] == 1


def test_unknown_history_distinct_from_known_zero_and_dirt_replacement():
    frame = records()
    frame.loc[0, "distance_m"] = np.nan
    out = augment_distance_history(frame)
    assert np.isnan(out.loc[0, "near_starts"])
    assert out.loc[2, "same_surface_starts"] == 0
    dirt = out.loc[[2]].assign(turf_starts=100, turf_top3=99)
    adapted = surface_base(dirt, "dirt")
    assert adapted.turf_starts.iloc[0] == 0
    assert adapted.turf_top3.iloc[0] == 0
    with pytest.raises(ValueError, match="Mixed surface"):
        surface_base(out, "dirt")


def test_band_boundaries():
    assert distance_regime(np.array([1400, 1401, 2000, 2001, 2600, 2601])).tolist() == [0, 1, 1, 2, 2, 3]
