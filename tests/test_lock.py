import pandas as pd
import pytest

from keiba_place_lab.lock import REQUIRED_LOCK_COLUMNS, validate_lock_frame


def valid_frame() -> pd.DataFrame:
    n = 18
    p = [3.0 / n] * n
    return pd.DataFrame(
        {
            "horse_no": range(1, n + 1),
            "horse_name": [f"H{i}" for i in range(1, n + 1)],
            "as_of_time": ["2026-10-03T08:35:00+09:00"] * n,
            "p_market": p,
            "p_model_raw": p,
            "p_model_calibrated": p,
            "p_ensemble": p,
            "uncertainty_low": [x - 0.01 for x in p],
            "uncertainty_high": [x + 0.01 for x in p],
            "rank": range(1, n + 1),
            "model_version": ["stage5-ensemble-v1"] * n,
            "locked_commit_sha": ["abc123"] * n,
        }
    )[REQUIRED_LOCK_COLUMNS]


def test_valid_lock_frame_passes():
    validate_lock_frame(valid_frame())


def test_lock_rejects_wrong_probability_sum():
    frame = valid_frame()
    frame.loc[0, "p_ensemble"] += 0.1
    with pytest.raises(ValueError, match="sum to 3"):
        validate_lock_frame(frame)


def test_lock_rejects_duplicate_runner():
    frame = valid_frame()
    frame.loc[1, "horse_no"] = frame.loc[0, "horse_no"]
    with pytest.raises(ValueError, match="18 unique runners"):
        validate_lock_frame(frame)
