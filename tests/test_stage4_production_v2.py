import pandas as pd
import pytest

from keiba_place_lab.stage4_production_v2 import prepare_target_context


def _target(field_size=8):
    rows = []
    for horse_no in range(1, field_size + 1):
        rows.append(
            {
                "horse_no": horse_no,
                "horse_name": f"H{horse_no}",
                "draw_pct": (horse_no - 1) / (field_size - 1),
                "sex": "M",
                "age": 4,
                "assigned_weight_kg": 56.0,
                "assigned_weight_delta_from_prev_kg": 0.0,
                "days_since_prev": 28,
                "distance_change_from_prev_m": 0,
                "surface_changed_from_prev": 0,
                "career_starts": 6,
                "career_top3": 2,
                "turf_starts": 6,
                "turf_top3": 2,
                "same_distance_starts": 4,
                "same_distance_top3": 1,
                "same_course_starts": 2,
                "same_course_top3": 1,
                "recent3_finish_pct_mean": 0.4,
                "recent3_top3_count": 1,
                "recent3_open_plus_count": 0,
                "recent3_graded_count": 0,
                "recent3_relative_time_mean": 0.01,
                "recent4_early_pos_pct_mean": 0.4,
                "front_forward_share": 0.3,
                "field_size": field_size,
                "racecourse": "Kyoto",
                "surface": "turf",
                "distance_m": 1200,
                "course_layout": "inner",
                "race_class": "Listed_open",
                "handicap_indicator": 1,
            }
        )
    return pd.DataFrame(rows)


def test_prepare_target_infers_declared_field_when_no_scratch():
    active, eligible = prepare_target_context(
        _target(),
        race_id="TARGET",
        race_date="2026-10-03",
    )
    assert active["declared_field_size"].eq(8).all()
    assert active["field_size"].eq(8).all()
    assert len(eligible) == 8
    assert active["racecourse"].eq("京都").all()
    assert active["race_class"].eq("Open").all()


def test_prepare_target_requires_declared_field_after_scratch():
    frame = _target(9)
    frame = frame.loc[frame["horse_no"] != 4].copy()
    frame["field_size"] = 8
    with pytest.raises(ValueError, match="declared_field_size is required"):
        prepare_target_context(
            frame,
            race_id="TARGET",
            race_date="2026-10-03",
        )


def test_prepare_target_preserves_declared_draw_after_scratch():
    frame = _target(9)
    frame = frame.loc[frame["horse_no"] != 4].copy()
    frame["field_size"] = 8
    frame["declared_field_size"] = 9
    active, eligible = prepare_target_context(
        frame,
        race_id="TARGET",
        race_date="2026-10-03",
    )
    assert len(active) == 8
    assert len(eligible) == 8
    assert active["declared_field_size"].eq(9).all()


def test_prepare_target_rejects_sum_three_scope_extrapolation_inputs():
    frame = _target()
    frame["distance_m"] = 1400
    with pytest.raises(ValueError, match="1200m"):
        prepare_target_context(
            frame,
            race_id="TARGET",
            race_date="2026-10-03",
        )


def test_prepare_target_can_return_partial_eligible_subset_without_dropping_context():
    frame = _target()
    frame.loc[frame["horse_no"] == 8, "career_starts"] = 2
    active, eligible = prepare_target_context(
        frame,
        race_id="TARGET",
        race_date="2026-10-03",
    )
    assert len(active) == 8
    assert len(eligible) == 7
    assert 8 not in set(eligible["horse_no"])
