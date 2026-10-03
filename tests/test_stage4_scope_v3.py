import numpy as np
import pandas as pd
import pytest

from keiba_place_lab.historical_panel import build_historical_panel
from keiba_place_lab.scope_features import scope_features
from keiba_place_lab.stage4_production_v3 import (
    build_live_context,
    route_model,
    score_scope_target,
    validate_roster,
)


def history_and_roster():
    rows = []
    for day in range(1, 6):
        for horse in range(1, 9):
            rows.append({
                "race_id": f"R{day}", "race_date": pd.Timestamp(f"2026-01-{day:02}"),
                "horse_id": f"H{horse}", "horse_name": f"horse{horse}", "horse_no": horse,
                "field_size": 8, "declared_field_size": 8, "surface": "turf", "distance_m": 1200,
                "racecourse": "京都", "race_class": "Open", "turn_direction": "right",
                "age": 4, "sex": "M", "assigned_weight_kg": 56.,
                "finish_position": horse, "early_position": horse,
                "race_time_seconds": 70. + horse, "is_open_plus": 1, "is_graded": 0,
            })
    raw = pd.DataFrame(rows)
    history = raw.loc[raw.race_id.ne("R5")].copy()
    roster = raw.loc[raw.race_id.eq("R5")].drop(columns=["finish_position", "early_position", "race_time_seconds", "is_open_plus", "is_graded"])
    return history, roster, raw


def test_live_reconstruction_matches_historical_definitions():
    history, roster, raw = history_and_roster()
    live = build_live_context(history, roster)
    expected = build_historical_panel(raw).tail(8)
    cols = ["career_starts", "career_top3", "turf_starts", "turf_top3", "same_distance_top3",
            "same_course_starts", "days_since_prev", "distance_change_from_prev_m",
            "recent3_finish_pct_mean", "recent3_top3_count", "recent3_relative_time_mean",
            "recent4_early_pos_pct_mean", "front_forward_share", "draw_pct"]
    np.testing.assert_allclose(live[cols].to_numpy(float), expected[cols].to_numpy(float), equal_nan=True)
    assert live.near_starts.eq(4).all()
    with pytest.raises(ValueError, match="complete starter field"):
        build_live_context(history.iloc[1:], roster)
    with pytest.raises(ValueError, match="strictly before"):
        build_live_context(raw, roster)
    roster["race_date"] = "2026-10-01"
    with pytest.raises(ValueError, match="Stale history"):
        build_live_context(history, roster)


def test_new_block_uses_all_peers_and_distinguishes_missing():
    history, roster, _ = history_and_roster()
    context = build_live_context(history, roster)
    context["near_starts"] = [10, 10, 10, 10, 0, 0, 0, np.nan]
    context["near_top3"] = [5, 4, 3, 2, 0, 0, 0, np.nan]
    featured = scope_features(context.iloc[:2], context, prior_mean=.2, blocks=["near"])
    rates = (context.near_top3 + 1.2) / (context.near_starts + 6)
    assert featured.near_relative.iloc[0] == pytest.approx(rates.iloc[0] - rates.iloc[1:].mean())
    all_features = scope_features(context, context, prior_mean=.2, blocks=["near"])
    assert all_features.near_zero_experience.iloc[4] == 1
    assert all_features.near_history_missing.iloc[4] == 0
    assert all_features.near_top3_shrunk.iloc[4] == pytest.approx(.2)
    assert all_features.near_history_missing.iloc[7] == 1
    assert np.isnan(all_features.near_relative.iloc[7])


def test_draw_survives_cancellation_and_unsupported_routes_market_only():
    _, roster, _ = history_and_roster()
    roster = roster.loc[roster.horse_no.ne(2)].copy()
    roster["field_size"] = 7
    context = validate_roster(roster)
    assert context.loc[context.horse_no.eq(3), "draw_pct"].iloc[0] == pytest.approx(2 / 7)
    routing = {"surfaces": {"turf": {"routes": [{"min_distance_m": 1000, "max_distance_m": 2600, "model_id": "G0"}]}}}
    assert route_model(context, routing) is None
    scored = score_scope_target({"routing": routing, "models": {}}, context)
    assert scored.route.eq("market-only").all() and scored.p_nonmarket.isna().all()
    roster["finish_position"] = 1
    with pytest.raises(ValueError, match="outcomes"):
        validate_roster(roster)


def test_frozen_bundle_routes_turf_dirt_and_rejects_training_lookahead():
    import hashlib
    import json
    from pathlib import Path

    import joblib

    pytest.importorskip("xgboost")
    root = Path(__file__).resolve().parents[1] / "models/stage4_scope_v3"
    manifest = json.loads((root / "manifest.json").read_text())
    assert hashlib.sha256((root / "bundle.joblib").read_bytes()).hexdigest() == manifest["bundle_sha256"]
    bundle = joblib.load(root / "bundle.joblib")
    history, roster, _ = history_and_roster()
    context = build_live_context(history, roster)
    context.loc[0, "career_starts"] = 0
    context.loc[0, "career_top3"] = 0
    scored = score_scope_target(bundle, context)
    assert np.isfinite(scored.p_nonmarket.iloc[1:]).all()
    assert scored.model_id.iloc[1] == "turf_1200_override"
    assert scored.route.iloc[0] == "market-only" and np.isnan(scored.p_nonmarket.iloc[0])
    context["surface"] = "dirt"
    assert score_scope_target(bundle, context).model_id.iloc[1] == "dirt_G0"
    context["race_date"] = "2025-12-28"
    with pytest.raises(ValueError, match="future training"):
        score_scope_target(bundle, context)
