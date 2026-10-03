import json

import numpy as np
import pandas as pd
import pytest

from keiba_place_lab.prospective_scope import make_lock, review_predictions


def lock_input():
    return pd.DataFrame({
        "race_id": ["race1"] * 8, "race_date": ["2026-10-04"] * 8,
        "horse_id": [f"H{i}" for i in range(8)], "horse_no": range(1, 9),
        "field_size": [8] * 8, "surface": ["turf"] * 8, "distance_m": [1200] * 8,
        "turn_direction": ["right"] * 8, "model_id": ["turf_1200_override"] * 7 + [""],
        "p_market": [.2] * 8, "p_nonmarket": [.4] * 7 + [np.nan],
        "route": ["nonmarket-shadow"] * 7 + ["market-only"],
    })


def params():
    return {"off_at": "2026-10-04T06:00:00Z", "market_asof": "2026-10-04T05:49:00Z",
            "now": "2026-10-04T05:50:00Z", "model_hash": "model", "protocol_hash": "protocol"}


def test_lock_common_timestamp_fallback_and_immutability_hash():
    import hashlib
    from io import BytesIO

    payload, meta = make_lock(lock_input(), **params())
    saved = pd.read_csv(BytesIO(payload))
    assert saved.p_w025.iloc[0] == pytest.approx(.25)
    assert saved.p_w100.iloc[-1] == pytest.approx(.2)
    assert np.isnan(saved.p_nonmarket.iloc[-1])
    assert meta["payload_sha256"] == hashlib.sha256(payload).hexdigest()
    assert json.loads(json.dumps(meta))["stage5_canonical"] == "market-only"


@pytest.mark.parametrize("change", [
    {"now": "2026-10-04T06:01:00Z"},
    {"now": "2026-10-04T05:00:00Z"},
    {"market_asof": "2026-10-04T05:40:00Z"},
    {"market_asof": "2026-10-04T05:51:00Z"},
    {"off_at": "2026-10-04T06:00:00"},
])
def test_reject_postoff_early_stale_future_and_naive_clocks(change):
    with pytest.raises(ValueError):
        make_lock(lock_input(), **{**params(), **change})


def test_no_interim_scores_before_precommitted_review_thresholds():
    frame = pd.DataFrame({"race_id": ["R1", "R2"], "race_date": ["2026-01-01", "2026-04-01"]})
    report, scores = review_predictions(frame)
    assert report["status"] == "collecting" and scores is None
    assert report["production_changed"] is False


def test_review_gate_can_select_candidate_without_changing_production():
    rows = 1000 * 8
    index = np.arange(rows)
    frame = pd.DataFrame({
        "race_id": (index // 8).astype(str),
        "race_date": pd.Timestamp("2026-01-01") + pd.to_timedelta((index // 8 % 64) * 2, unit="D"),
        "top3_label": (index % 8 < 3).astype(int),
    })
    from keiba_place_lab.prospective_scope import WEIGHTS

    market = np.full(rows, 3 / 8)
    candidate = np.where(frame.top3_label, .9, .05)
    for weight in WEIGHTS:
        frame[f"p_w{int(weight * 100):03}"] = (1 - weight) * market + weight * candidate
    report, scores = review_predictions(frame)
    assert report["status"] == "review_ready"
    assert report["decision"]["winner"] == "w100"
    assert scores is not None
    assert report["production_changed"] is False


def test_push_event_after_off_cannot_be_used_as_publication_proof(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_scope_prospective as cli

    _, manifest = make_lock(lock_input(), **params())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(cli, "get_json", lambda url: {
        "event": "push", "repository": {"full_name": cli.REPOSITORY},
        "head_branch": "main",
        "created_at": "2026-10-04T06:01:00Z", "head_sha": "backdated_commit",
    })
    with pytest.raises(ValueError, match="between lock and off"):
        cli.verify_publication(tmp_path, 123, "prospective/locks/race1")
