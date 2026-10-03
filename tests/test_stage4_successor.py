import numpy as np
import pandas as pd
import pytest

from keiba_place_lab.stage4_successor import (
    add_diagnostic_bands,
    assert_outer_year_contract,
    calibration_intercept_slope,
    field_size_baseline,
    subgroup_diagnostics,
)


def _frame() -> pd.DataFrame:
    rows = []
    for race_no in range(8):
        year = 2023 if race_no < 4 else 2024
        race_date = f"{year}-01-{1 + race_no:02d}"
        field_size = 12 + (race_no % 4) * 2
        for runner, y in enumerate([1, 1, 1, 0, 0, 0], start=1):
            rows.append(
                {
                    "race_id": f"R{race_no}",
                    "race_date": race_date,
                    "top3_label": y,
                    "field_size": field_size,
                    "race_class": "Open" if race_no % 2 else "3Win",
                    "racecourse": "京都" if race_no % 2 else "中山",
                    "days_since_prev": 14 + runner * 7,
                    "career_starts": 3 + runner * 3,
                }
            )
    return pd.DataFrame(rows)


def test_diagnostic_bands_are_outcome_free_and_expected():
    frame = _frame()
    out = add_diagnostic_bands(frame)
    assert set(out["year"]) == {"2023", "2024"}
    assert out["field_size_band"].notna().all()
    assert out["rest_interval_band"].notna().all()
    assert out["career_starts_band"].notna().all()


def test_field_size_baseline_is_three_over_field():
    frame = _frame().iloc[:3].copy()
    frame["field_size"] = [12, 15, 18]
    baseline = field_size_baseline(frame)
    np.testing.assert_allclose(baseline, [0.25, 0.2, 1.0 / 6.0])


def test_outer_year_contract_rejects_current_year_training_rows():
    train = _frame().loc[lambda x: pd.to_datetime(x["race_date"]).dt.year == 2023]
    evaluation = _frame().loc[
        lambda x: pd.to_datetime(x["race_date"]).dt.year == 2024
    ]
    assert_outer_year_contract(train, evaluation, outer_year=2024)

    contaminated = pd.concat([train, evaluation.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="current/future-year"):
        assert_outer_year_contract(contaminated, evaluation, outer_year=2024)


def test_calibration_intercept_slope_returns_finite_values():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.25, 0.3, 0.55, 0.6, 0.7, 0.8, 0.35, 0.65, 0.4, 0.75])
    intercept, slope = calibration_intercept_slope(y, p)
    assert np.isfinite(intercept)
    assert np.isfinite(slope)


def test_subgroup_diagnostics_include_preregistered_dimensions():
    frame = _frame()
    p = np.full(len(frame), 0.5)
    out = subgroup_diagnostics(frame, p)
    assert {"overall", "year", "field_size_band", "race_class", "racecourse"}.issubset(
        set(out["dimension"])
    )
    overall = out.loc[out["dimension"] == "overall"].iloc[0]
    assert overall["rows"] == len(frame)
    assert overall["races"] == frame["race_id"].nunique()
    assert np.isfinite(overall["race_macro_brier"])
