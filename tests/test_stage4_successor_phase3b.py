import numpy as np
import pandas as pd
import pytest

from keiba_place_lab.stage4_successor_phase3b import (
    TREND_FEATURES,
    build_recent_trend_table,
)


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "race_date": "2022-01-01",
                "race_id": "R1",
                "horse_id": "H1",
                "recent3_finish_pct_mean": 0.50,
                "recent3_relative_time_mean": 0.030,
                "recent3_top3_count": 0.0,
                "recent4_early_pos_pct_mean": 0.60,
            },
            {
                "race_date": "2022-02-01",
                "race_id": "R2",
                "horse_id": "H1",
                "recent3_finish_pct_mean": 0.40,
                "recent3_relative_time_mean": 0.020,
                "recent3_top3_count": 1.0,
                "recent4_early_pos_pct_mean": 0.50,
            },
            {
                "race_date": "2022-03-01",
                "race_id": "R3",
                "horse_id": "H1",
                "recent3_finish_pct_mean": 0.30,
                "recent3_relative_time_mean": 0.010,
                "recent3_top3_count": 2.0,
                "recent4_early_pos_pct_mean": 0.40,
            },
            {
                "race_date": "2022-01-15",
                "race_id": "R4",
                "horse_id": "H2",
                "recent3_finish_pct_mean": 0.20,
                "recent3_relative_time_mean": 0.015,
                "recent3_top3_count": 2.0,
                "recent4_early_pos_pct_mean": 0.30,
            },
        ]
    )


def test_recent_trend_uses_immediately_previous_panel_row():
    trend = build_recent_trend_table(_history())
    row = trend.loc[trend["race_id"] == "R3"].iloc[0]

    assert row["trend_recent3_finish_improvement"] == pytest.approx(0.10)
    assert row["trend_recent3_time_improvement"] == pytest.approx(0.010)
    assert row["trend_recent3_top3_count_change"] == pytest.approx(1.0)
    assert row["trend_recent4_early_forward_change"] == pytest.approx(0.10)


def test_first_horse_start_has_missing_trends():
    trend = build_recent_trend_table(_history())
    row = trend.loc[trend["race_id"] == "R1"].iloc[0]
    assert all(pd.isna(row[feature]) for feature in TREND_FEATURES)


def test_missing_current_or_previous_summary_keeps_trend_missing():
    history = _history()
    history.loc[history["race_id"] == "R2", "recent3_relative_time_mean"] = np.nan
    trend = build_recent_trend_table(history)
    r2 = trend.loc[trend["race_id"] == "R2"].iloc[0]
    r3 = trend.loc[trend["race_id"] == "R3"].iloc[0]
    assert pd.isna(r2["trend_recent3_time_improvement"])
    assert pd.isna(r3["trend_recent3_time_improvement"])


def test_non_chronological_duplicate_race_horse_is_rejected():
    history = pd.concat([_history(), _history().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        build_recent_trend_table(history)
