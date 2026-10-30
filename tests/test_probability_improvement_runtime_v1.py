"""Frozen C-window/decay and licensed market ledger regression tests."""

from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_capture = run_path(str(_SCRIPTS / "capture_market_odds_v1.py"))
ingest, select = _capture["ingest"], _capture["select"]
training_weights = run_path(
    str(_SCRIPTS / "run_probability_improvement_c_v1.py")
)["training_weights"]


def test_c0_c1_c2_target_windows_keep_history_separate():
    dates = pd.to_datetime([
        "2016-01-03", "2018-01-01", "2020-06-30", "2022-12-31",
    ])
    frame = pd.DataFrame({"race_date": dates, "top3_label": [0, 1, 0, 1]})
    cutoff = pd.Timestamp("2022-12-31")
    w0, d0 = training_weights(frame, cutoff, "C0")
    w1, d1 = training_weights(frame, cutoff, "C1")
    w2, d2 = training_weights(frame, cutoff, "C2")
    assert len(d0) == 4 and len(w0) == 4
    assert len(d1) == 3 and len(w1) == 3
    assert d1.race_date.min() == pd.Timestamp("2018-01-01")
    assert len(d2) == len(d0)
    assert np.isclose(w2[-1], 1.0)
    assert (np.diff(w2) > 0).all()
    assert not np.array_equal(w2, w0)


def test_c_future_labels_rejected():
    data = pd.DataFrame({"race_date": pd.to_datetime(["2023-01-01"]),
                         "top3_label": [1]})
    with pytest.raises(ValueError, match="future"):
        training_weights(data, pd.Timestamp("2022-12-31"), "C2")


def test_licensed_odds_persist_and_replay(tmp_path):
    captured = "2026-11-01T15:29:00+09:00"
    updated = "2026-11-01T15:28:00+09:00"
    off = "2026-11-01T15:40:00+09:00"
    def row(horse):
        return {"race_id": "race_1", "horse_id": horse, "bet_type": "win",
                "odds": 3.5, "place_odds_low": np.nan,
                "place_odds_high": np.nan, "provider_updated_at": updated,
                "captured_at": captured, "scheduled_off_at": off,
                "source": "licensed_fixture", "snapshot_id": "real-observed-id-fixture"}
    source = tmp_path/"provider.csv"
    pd.DataFrame([row("h1"), row("h2")]).to_csv(source, index=False)
    ledger = ingest(source, tmp_path/"market")
    assert ledger.exists()
    assert pd.read_csv(ledger).shape[0] == 2
    assert ledger.with_suffix(".json").exists()
    with pytest.raises(ValueError, match="Append-only"):
        ingest(source, tmp_path/"market")
    roster = tmp_path/"roster.csv"
    pd.DataFrame([{"race_id":"race_1","horse_id":"h1"},
                  {"race_id":"race_1","horse_id":"h2"}]).to_csv(roster,index=False)
    selected = tmp_path/"lock.csv"
    select(ledger, roster, 10, selected)
    assert len(pd.read_csv(selected)) == 2
    with pytest.raises(FileExistsError):
        select(ledger, roster, 10, selected)


def test_offset_serialization_roundtrip(tmp_path):
    import joblib
    from keiba_place_lab.probability_improvement_v1 import (
        fit_market_offset, predict_market_offset,
    )

    rng = np.random.default_rng(20261031)
    p = rng.uniform(0.15, 0.75, size=60)
    x = rng.normal(size=(60, 5))
    y = rng.integers(0, 2, size=60)
    dates = np.array(["2023-01-01"] * 60)
    fitted = fit_market_offset(p, x, y, dates, max_training_date="2024-01-01")
    expected = predict_market_offset(p, x, fitted)
    path = tmp_path/"offset_model.joblib"
    joblib.dump(fitted, path)
    loaded = joblib.load(path)
    actual = predict_market_offset(p, x, loaded)
    np.testing.assert_array_equal(expected, actual)


def test_joint_selection_rejects_mismatched_starters():
    same_evaluation_keys = run_path(
        str(_SCRIPTS / "run_probability_improvement_joint_v1.py")
    )["same_evaluation_keys"]
    a = pd.DataFrame({
        "race_date": ["2024-01-01"] * 2, "race_id": ["r"] * 2,
        "horse_id": ["h1", "h2"], "horse_no": [1, 2],
        "top3_label": [0, 1],
    })
    b = a.copy()
    b.loc[1, "horse_id"] = "h3"
    with pytest.raises(ValueError, match="evaluation keys"):
        same_evaluation_keys(a, b, "synthetic test")
    b = a.copy()
    b.loc[1, "top3_label"] = 0
    with pytest.raises(ValueError, match="outcome"):
        same_evaluation_keys(a, b, "synthetic test")
