"""Safeguards for the exploratory historical rider-distance experiment."""
import numpy as np
import pandas as pd

from research_jockey_distance_effect_v1 import (
    BANDS, add_rider_features, asof_rates, date_bootstrap, distance_band,
)


def test_rider_prior_skips_same_calendar_date():
    x = pd.DataFrame({
        "jockey_name": ["A", "A", "A", "B"],
        "race_date": pd.to_datetime(["2024-01-01", "2024-01-02",
                                     "2024-01-02", "2024-01-03"]),
        "top3_label": [1, 0, 1, 1],
    })
    a = asof_rates(x, ["jockey_name"], "j", .2)
    assert list(a.j_starts) == [0, 1, 1, 0]
    assert a.j_rate.iloc[1] == a.j_rate.iloc[2]
    assert np.isclose(a.j_rate.iloc[1], (1 + 15 * .2) / 16)


def test_distance_band_boundaries():
    vals = distance_band(pd.Series([1200, 1600, 1601, 2200, 2399, 2400, 3200]))
    assert vals.tolist() == [BANDS[0], BANDS[0], BANDS[1], BANDS[1],
                             BANDS[1], BANDS[2], BANDS[2]]


def test_band_specific_history_no_same_day_and_no_other_band():
    data = pd.DataFrame({
        "race_date": pd.to_datetime(["2020-01-01", "2020-01-02",
                                     "2020-01-02", "2020-01-03"]),
        "race_id": ["001", "002", "003", "004"],
        "horse_no": [1, 1, 1, 1],
        "horse_id": ["H1", "H2", "H3", "H4"],
        "jockey_name": ["A"] * 4,
        "surface": ["turf"] * 4,
        "distance_m": [1200, 2500, 2500, 2500],
        "top3_label": [1, 0, 1, 1],
    })
    x = add_rider_features(data, .2)
    assert list(x.band_prior_starts) == [0, 0, 0, 2]
    assert list(x.overall_prior_starts) == [0, 1, 1, 3]
    assert x.rider_band_top3_rate.iloc[1] == x.rider_band_top3_rate.iloc[2]


def test_date_bootstrap_paired_contrast_is_consistent():
    x = pd.DataFrame({
        "race_date": pd.to_datetime(["2024-01-01", "2024-01-01"] * 30),
        "band": [BANDS[0], BANDS[2]] * 30,
        "improvement": [0.01, 0.02] * 30,
    })
    ans = date_bootstrap(x, "toy", draws=100)
    assert ans["status"] != "OK"  # no middle-distance observations
