import pandas as pd
import pytest

from keiba_place_lab.stage4_successor_phase3c import (
    INTERACTION_FEATURES,
    add_phase3c_features,
)


def _row(
    race_id,
    horse_id,
    horse_no,
    *,
    draw_pct,
    early,
    pressure,
    days_since_prev,
    age,
):
    return {
        "race_id": race_id,
        "race_date": "2022-01-01",
        "horse_id": horse_id,
        "horse_name": horse_id,
        "horse_no": horse_no,
        "field_size": 4,
        "sex": "M",
        "age": age,
        "assigned_weight_kg": 56.0,
        "assigned_weight_delta_from_prev_kg": 0.0,
        "days_since_prev": days_since_prev,
        "distance_change_from_prev_m": 0.0,
        "surface_changed_from_prev": 0.0,
        "career_top3": 2,
        "career_starts": 6,
        "turf_top3": 2,
        "turf_starts": 6,
        "same_distance_top3": 1,
        "same_distance_starts": 3,
        "same_course_top3": 1,
        "same_course_starts": 3,
        "recent3_finish_pct_mean": 0.4,
        "recent3_top3_count": 1.0,
        "recent3_open_plus_count": 0.0,
        "recent3_graded_count": 0.0,
        "recent3_relative_time_mean": 0.02,
        "recent4_early_pos_pct_mean": early,
        "front_forward_share": pressure,
        "draw_pct": draw_pct,
        "distance_m": 1200,
        "racecourse": "京都",
        "race_class": "Open",
        "top3_label": 0,
    }


def _context() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("R1", "H1", 1, draw_pct=0.0, early=0.2, pressure=0.5, days_since_prev=20, age=3),
            _row("R1", "H2", 2, draw_pct=0.33, early=0.4, pressure=0.5, days_since_prev=30, age=4),
            _row("R1", "H3", 3, draw_pct=0.67, early=0.6, pressure=0.5, days_since_prev=40, age=5),
            _row("R1", "H4", 4, draw_pct=1.0, early=0.8, pressure=0.5, days_since_prev=50, age=6),
        ]
    )


def test_phase3c_interactions_match_preregistered_formulas():
    context = _context()
    eligible = context.iloc[:2].copy()
    out, coverage = add_phase3c_features(
        eligible,
        context,
        prior_mean=0.2,
        prior_strength=6.0,
    )
    row = out.loc[out["horse_id"] == "H2"].iloc[0]

    assert row["interaction_draw_x_front_tendency"] == pytest.approx(
        0.33 * (1.0 - 0.4)
    )
    assert row["interaction_front_tendency_x_race_pressure"] == pytest.approx(
        (1.0 - 0.4) * 0.5
    )
    assert row["interaction_rest_x_age"] == pytest.approx(
        row["log_days_since_prev"] * 4.0
    )

    interaction = coverage.loc[coverage["block"] == "interaction"]
    assert set(interaction["feature"]) == set(INTERACTION_FEATURES)
    assert interaction["eligible_coverage"].eq(1.0).all()


def test_phase3c_missing_source_keeps_interaction_missing():
    context = _context()
    context.loc[context["horse_id"] == "H1", "recent4_early_pos_pct_mean"] = pd.NA
    eligible = context.iloc[:2].copy()
    out, _ = add_phase3c_features(
        eligible,
        context,
        prior_mean=0.2,
        prior_strength=6.0,
    )
    row = out.loc[out["horse_id"] == "H1"].iloc[0]
    assert pd.isna(row["interaction_draw_x_front_tendency"])
    assert pd.isna(row["interaction_front_tendency_x_race_pressure"])
