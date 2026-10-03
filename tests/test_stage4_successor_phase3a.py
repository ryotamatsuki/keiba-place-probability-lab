import numpy as np
import pandas as pd
import pytest

from keiba_place_lab.stage4_successor_phase3a import (
    RELATIVE_FEATURES,
    add_relative_ability_features,
    assert_full_field_context,
)


def _row(race_id, horse_id, field_size, career_starts, career_top3, recent_finish, recent_time):
    return {
        "race_id": race_id,
        "race_date": "2022-01-01",
        "horse_id": horse_id,
        "horse_name": horse_id,
        "horse_no": int(horse_id[-1]),
        "field_size": field_size,
        "sex": "M",
        "age": 4,
        "assigned_weight_kg": 56.0,
        "assigned_weight_delta_from_prev_kg": 0.0,
        "days_since_prev": 28.0,
        "distance_change_from_prev_m": 0.0,
        "surface_changed_from_prev": 0.0,
        "career_top3": career_top3,
        "career_starts": career_starts,
        "turf_top3": career_top3,
        "turf_starts": career_starts,
        "same_distance_top3": career_top3,
        "same_distance_starts": career_starts,
        "same_course_top3": career_top3,
        "same_course_starts": career_starts,
        "recent3_finish_pct_mean": recent_finish,
        "recent3_top3_count": 1.0,
        "recent3_open_plus_count": 0.0,
        "recent3_graded_count": 0.0,
        "recent3_relative_time_mean": recent_time,
        "recent4_early_pos_pct_mean": 0.5,
        "front_forward_share": 0.4,
        "distance_m": 1200,
        "racecourse": "京都",
        "race_class": "Open",
        "top3_label": 0,
    }


def _context():
    return pd.DataFrame(
        [
            _row("R1", "H1", 4, 10, 5, 0.20, -0.02),
            _row("R1", "H2", 4, 10, 4, 0.30, -0.01),
            _row("R1", "H3", 4, 10, 3, 0.40, 0.00),
            _row("R1", "H4", 4, 10, 2, 0.50, 0.01),
        ]
    )


def test_full_field_context_rejects_partial_race():
    context = _context().iloc[:3].copy()
    with pytest.raises(ValueError, match="complete starter field"):
        assert_full_field_context(context)


def test_relative_features_use_all_starters_not_only_eligible_subset():
    context = _context()
    eligible = context.iloc[:2].copy()
    enriched, coverage = add_relative_ability_features(
        eligible,
        context,
        prior_mean=0.2,
        prior_strength=6.0,
    )

    h1 = enriched.loc[enriched["horse_id"] == "H1"].iloc[0]
    # Career shrunk values with prior=0.2, strength=6:
    # H1=6.2/16, H2=5.2/16, H3=4.2/16, H4=3.2/16.
    expected_h1 = (6.2 / 16.0) - np.mean([5.2 / 16.0, 4.2 / 16.0, 3.2 / 16.0])
    assert h1["rel_career_top3_vs_others"] == pytest.approx(expected_h1)

    # Lower recent finish percentile and relative time are better, so signs reverse.
    assert h1["rel_recent3_finish_vs_others"] == pytest.approx(
        np.mean([0.30, 0.40, 0.50]) - 0.20
    )
    assert h1["rel_recent3_time_vs_others"] == pytest.approx(
        np.mean([-0.01, 0.00, 0.01]) - (-0.02)
    )
    assert set(coverage["feature"]) == set(RELATIVE_FEATURES)
    assert coverage["eligible_rows"].eq(2).all()


def test_missing_runner_source_stays_missing_for_relative_feature():
    context = _context()
    context.loc[context["horse_id"] == "H1", "recent3_relative_time_mean"] = np.nan
    eligible = context.iloc[:2].copy()
    enriched, _ = add_relative_ability_features(
        eligible,
        context,
        prior_mean=0.2,
        prior_strength=6.0,
    )
    h1 = enriched.loc[enriched["horse_id"] == "H1"].iloc[0]
    assert pd.isna(h1["rel_recent3_time_vs_others"])
