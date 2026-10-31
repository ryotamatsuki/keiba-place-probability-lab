"""No invented T-10 times, partial fields, stale odds or feature history."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest

from keiba_place_lab.ensemble import LogitCalibrator
from keiba_place_lab.tennosho_forward import (
    MARKET_FIELD,
    MARKET_SPLITS,
    TARGET_RACE,
    _prior_features,
    market_context,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
CAPTURE = run_path(str(SCRIPTS / "tennosho_preoff_capture.py"))
ensure_lock_clock = CAPTURE["ensure_lock_clock"]
ensure_provider_clock = CAPTURE["ensure_provider_clock"]
strict_same_market = CAPTURE["strict_same_market"]
strict_roster = CAPTURE["strict_roster"]
verified_provider_time = CAPTURE["verified_provider_time"]


@pytest.fixture
def entrant_odds() -> pd.DataFrame:
    n = 17
    return pd.DataFrame({
        "race_id": [TARGET_RACE] * n,
        "horse_id": [f"2018{i:06}" for i in range(n)],
        "horse_no": list(range(1, n + 1)),
        "horse_name": [f"dummy_name_{i}" for i in range(1, n + 1)],
        "win_odds": np.geomspace(2.5, 300, n),
    })


def test_market_field_all_starters_and_one_calibration(entrant_odds):
    fit = LogitCalibrator(intercept=0.08, slope=0.94)
    result, overround = market_context(
        entrant_odds, fit,
        required_horses=set(entrant_odds.horse_id.astype(str)),
    )
    assert len(result) == 17
    assert set((*MARKET_FIELD, *MARKET_SPLITS)).issubset(result.columns)
    assert np.isclose(result.market_p_win.sum(), 1, atol=1e-10)
    assert np.isclose(result.market_p_top3.sum(), 3, atol=1e-10)
    assert np.isfinite(result[list(MARKET_FIELD)].to_numpy(float)).all()
    assert (result.market_calibrated_top3_p > 0).all()
    assert (result.market_calibrated_top3_p < 1).all()
    assert overround > 0
    bad = entrant_odds.iloc[:-1]
    with pytest.raises(ValueError, match="full-starter"):
        market_context(
            bad, fit, required_horses=set(entrant_odds.horse_id),
        )


def test_real_2026_11_01_1540_lock_minute_only():
    jst = timedelta(hours=9)
    off = datetime.fromisoformat("2026-11-01T15:40:00+09:00")
    ensure_lock_clock(off, datetime.fromisoformat(
        "2026-11-01T15:30:00+09:00",
    ))
    ensure_lock_clock(off, datetime.fromisoformat(
        "2026-11-01T15:30:59.999999+09:00",
    ))
    for observed in (
        "2026-11-01T15:29:59+09:00",
        "2026-11-01T15:31:00+09:00",
        "2026-11-01T15:40:00+09:00",
        "2026-10-31T15:30:00+09:00",
    ):
        with pytest.raises(ValueError, match="T-minus-10"):
            ensure_lock_clock(off, datetime.fromisoformat(observed))
    assert jst.total_seconds() == 32400


def test_market_source_time_distinct_and_fresh():
    off = datetime.fromisoformat("2026-11-01T15:40:00+09:00")
    lock = off - timedelta(seconds=600)
    acquired = lock - timedelta(seconds=2)
    provider = lock - timedelta(seconds=240)
    ensure_provider_clock(provider, acquired, lock, off)
    with pytest.raises(ValueError, match="older"):
        ensure_provider_clock(lock - timedelta(seconds=301), acquired, lock, off)
    with pytest.raises(ValueError, match="order"):
        ensure_provider_clock(lock + timedelta(seconds=1), acquired, lock, off)
    with pytest.raises(ValueError):
        ensure_provider_clock(
            datetime(2026, 11, 1, 15, 27), acquired, lock, off,
        )


def test_explicit_provider_timezone_mandatory():
    aware = datetime.fromisoformat("2026-11-01T15:25:00+09:00")
    value = '<time class="hr-horseResult__update" datetime="2026-11-01T15:25:00+09:00"></time>'
    verified_provider_time(value, aware)
    with pytest.raises(ValueError, match="timezone"):
        verified_provider_time(
            value.replace("+09:00", ""), aware,
        )
    with pytest.raises(ValueError, match="missing"):
        verified_provider_time("<div>15:25 update</div>", aware)
    with pytest.raises(ValueError, match="disagrees"):
        verified_provider_time(
            value.replace("15:25:00", "15:26:00"), aware,
        )


def test_roster_identity_and_own_keys(entrant_odds):
    roster = entrant_odds.drop(columns="win_odds")
    cfg = {
        "race_id": TARGET_RACE,
        "expected_starters": len(roster),
        "horse_no_to_name": {
            str(row.horse_no): row.horse_name
            for row in roster.itertuples(index=False)
        },
    }
    assert strict_roster(roster, cfg, TARGET_RACE) == set(roster.horse_id)
    match = strict_same_market(roster, entrant_odds)
    assert len(match) == len(roster)
    assert match.horse_no.is_monotonic_increasing
    with pytest.raises(ValueError, match="mismatch"):
        strict_same_market(roster, entrant_odds.iloc[:-1])
    cfg["horse_no_to_name"]["1"] = "changed_roster"
    with pytest.raises(ValueError, match="changed"):
        strict_roster(roster, cfg, TARGET_RACE)


def test_f3_historical_source_requires_proven_complete_before_race(entrant_odds):
    frame = entrant_odds.drop(columns="win_odds").copy()
    frame["race_date"] = "2026-11-01"
    frame["career_top3_shrunk"] = .2
    meta = {
        "race_id": TARGET_RACE,
        "history_complete_for_all_entrants": True,
        "no_future_rows": True,
        "history_cutoff": "2026-10-31",
        "history_sha256": "a" * 64,
    }
    # Missing sex fails; market variables are supplied after live odds merge.
    with pytest.raises(ValueError, match="incomplete prior features"):
        _prior_features(
            frame, meta,
            required_horses=set(frame.horse_id),
            expected_numeric=["career_top3_shrunk", "market_win_probability"],
            expected_categorical=["sex"],
        )
    frame["sex"] = "M"
    good = _prior_features(
        frame, meta, required_horses=set(frame.horse_id),
        expected_numeric=["career_top3_shrunk", "market_win_probability"],
        expected_categorical=["sex"],
    )
    assert len(good) == 17
    frame["top3_label"] = 1
    with pytest.raises(ValueError, match="Postrace"):
        _prior_features(
            frame, meta, required_horses=set(frame.horse_id),
            expected_numeric=["career_top3_shrunk", "market_win_probability"],
            expected_categorical=["sex"],
        )
    frame = frame.drop(columns="top3_label")
    meta["history_cutoff"] = "2026-11-01"
    with pytest.raises(ValueError, match="unverified"):
        _prior_features(
            frame, meta, required_horses=set(frame.horse_id),
            expected_numeric=["career_top3_shrunk", "market_win_probability"],
            expected_categorical=["sex"],
        )
