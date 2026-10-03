from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from keiba_place_lab.ensemble import (
    LogitCalibrator,
    apply_logit_calibrator,
    convex_blend,
    fit_logit_calibrator,
    nearby_weights,
    select_blend_weight,
)
from keiba_place_lab.historical_market import reconstruct_historical_market


def test_logit_calibrator_returns_finite_probabilities():
    p = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90])
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    calibrator = fit_logit_calibrator(p, y)
    out = apply_logit_calibrator(p, calibrator)
    assert np.isfinite(out).all()
    assert ((out > 0) & (out < 1)).all()
    assert calibrator.slope > 0


def test_apply_known_calibrator():
    p = np.array([0.2, 0.5, 0.8])
    out = apply_logit_calibrator(p, LogitCalibrator(intercept=0.0, slope=1.0))
    assert np.allclose(out, p)


def test_convex_blend_and_weight_selection():
    frame = pd.DataFrame(
        {
            "race_id": ["A"] * 4 + ["B"] * 4,
            "top3_label": [1, 1, 1, 0, 1, 1, 1, 0],
        }
    )
    market = np.array([0.8, 0.7, 0.6, 0.2, 0.8, 0.7, 0.6, 0.2])
    nonmarket = np.array([0.6, 0.5, 0.4, 0.4, 0.6, 0.5, 0.4, 0.4])
    selected, grid = select_blend_weight(
        frame,
        market,
        nonmarket,
        weights=np.array([0.0, 0.5, 1.0]),
    )
    assert selected == 0.0
    assert grid.iloc[0]["brier"] <= grid.iloc[-1]["brier"]
    assert np.allclose(convex_blend(market, nonmarket, 0.0), market)


def test_nearby_weights_include_endpoints_and_selected():
    values = nearby_weights(0.35, radius=0.10)
    assert values == pytest.approx([0.0, 0.25, 0.35, 0.45, 1.0])


def test_historical_market_reconstruction_uses_complete_starter_field(tmp_path: Path):
    path = tmp_path / "results.csv"
    path.write_text(
        "race_id,number,rank,odds\n"
        "R1,1,1,2.0\n"
        "R1,2,2,3.0\n"
        "R1,3,3,4.0\n"
        "R1,4,4,8.0\n"
        "R1,5,除,---\n"
        "R2,1,1,2.0\n"
        "R2,2,2,---\n"
        "R2,3,3,4.0\n"
        "R2,4,4,8.0\n",
        encoding="utf-8",
    )
    market, diagnostics = reconstruct_historical_market(path, {"R1", "R2"})
    assert market["race_id"].unique().tolist() == ["R1"]
    assert market["horse_no"].tolist() == [1, 2, 3, 4]
    assert market["market_p_top3"].sum() == pytest.approx(3.0)
    assert diagnostics["complete_market_races"] == 1
    assert diagnostics["incomplete_market_races"] == 1
