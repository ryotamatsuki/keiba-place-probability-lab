import numpy as np
import pandas as pd
import pytest

from keiba_place_lab.historical_panel import (
    assert_strict_history,
    assign_temporal_split,
    build_historical_panel,
    find_forbidden_market_columns,
    select_phase_a_cohort,
)


def _synthetic_rows() -> pd.DataFrame:
    rows = []
    dates = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"])
    for race_index, race_date in enumerate(dates):
        for horse_no in range(1, 9):
            finish = ((horse_no + race_index - 2) % 8) + 1
            rows.append(
                {
                    "race_id": f"r{race_index}",
                    "race_date": race_date,
                    "horse_id": f"h{horse_no}",
                    "horse_name": f"H{horse_no}",
                    "horse_no": horse_no,
                    "field_size": 8,
                    "declared_field_size": 8,
                    "sex": "M",
                    "age": 4,
                    "assigned_weight_kg": 55.0 + 0.5 * race_index,
                    "distance_m": 1400 if race_index == 1 else 1200,
                    "surface": "turf",
                    "racecourse": "Kyoto",
                    "turn_direction": "right",
                    "race_class": "Open",
                    "handicap_indicator": 1,
                    "finish_position": finish,
                    "race_time_seconds": 70.0 + 0.2 * finish,
                    "early_position": finish,
                    "is_open_plus": 1,
                    "is_graded": 0,
                }
            )
    return pd.DataFrame(rows)


def test_market_columns_rejected() -> None:
    assert find_forbidden_market_columns(["age", "win_odds", "market_rank"]) == [
        "win_odds",
        "market_rank",
    ]


def test_target_race_does_not_enter_own_history() -> None:
    panel = build_historical_panel(_synthetic_rows())
    h1 = panel.loc[panel["horse_id"].eq("h1")].reset_index(drop=True)

    assert h1.loc[0, "career_starts"] == 0
    assert h1.loc[0, "career_top3"] == 0

    # h1 finished 8th in race 0, so race 1 history still has zero top-three finishes.
    assert h1.loc[1, "career_top3"] == 0

    # h1 then won race 1, so race 2 history includes exactly one top-three finish.
    assert h1.loc[2, "career_top3"] == 1


def test_recent_features_are_shifted() -> None:
    panel = build_historical_panel(_synthetic_rows())
    h1 = panel.loc[panel["horse_id"].eq("h1")].reset_index(drop=True)

    assert np.isnan(h1.loc[0, "recent3_top3_count"])
    assert h1.loc[1, "recent3_top3_count"] == 0
    assert h1.loc[2, "recent3_top3_count"] == 1


def test_distance_history_is_condition_specific() -> None:
    panel = build_historical_panel(_synthetic_rows())
    h1 = panel.loc[panel["horse_id"].eq("h1")].reset_index(drop=True)

    assert h1.loc[0, "same_distance_starts"] == 0
    assert h1.loc[1, "same_distance_starts"] == 0
    assert h1.loc[2, "same_distance_starts"] == 1


def test_weight_delta_uses_previous_start() -> None:
    panel = build_historical_panel(_synthetic_rows())
    h1 = panel.loc[panel["horse_id"].eq("h1")].reset_index(drop=True)

    assert np.isnan(h1.loc[0, "assigned_weight_delta_from_prev_kg"])
    assert h1.loc[1, "assigned_weight_delta_from_prev_kg"] == pytest.approx(0.5)


def test_front_share_uses_prior_running_history() -> None:
    panel = build_historical_panel(_synthetic_rows())
    first_race = panel.loc[panel["race_id"].eq("r0")]
    second_race = panel.loc[panel["race_id"].eq("r1")]

    assert first_race["front_forward_share"].isna().all()
    assert second_race["front_forward_share"].notna().all()


def test_temporal_split_contract() -> None:
    frame = pd.DataFrame(
        {"race_date": pd.to_datetime(["2015-01-01", "2020-01-01", "2024-01-01", "2025-01-01", "2026-01-01"])}
    )
    assert assign_temporal_split(frame).tolist() == [
        "warmup",
        "train",
        "validation",
        "test",
        "future",
    ]


def test_phase_a_requires_history_and_eight_runner_field() -> None:
    panel = build_historical_panel(_synthetic_rows())
    selected = select_phase_a_cohort(panel, min_prior_starts=3)
    assert set(selected["race_id"]) == {"r3"}


def test_strict_history_passes_synthetic_panel() -> None:
    panel = build_historical_panel(_synthetic_rows())
    assert_strict_history(panel)


def test_scratched_number_gap_does_not_break_draw_normalization() -> None:
    rows = _synthetic_rows()
    race = rows[rows["race_id"].eq("r0") & ~rows["horse_no"].eq(7)].copy()
    race["field_size"] = 7
    race["declared_field_size"] = 8
    panel = build_historical_panel(race)
    assert panel["draw_pct"].max() == 1.0
