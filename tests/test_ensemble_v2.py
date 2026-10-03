import numpy as np
import pandas as pd
import pytest

from keiba_place_lab.ensemble_v2 import blend_registry, select_blend_one_se


def _frame(races=12):
    rows = []
    for race in range(races):
        date = f"2024-01-{1 + race // 2:02d}"
        for y in [1, 1, 1, 0, 0, 0]:
            rows.append(
                {
                    "race_id": f"R{race}",
                    "race_date": date,
                    "top3_label": y,
                }
            )
    return pd.DataFrame(rows)


def test_registry_includes_market_only_endpoint():
    registry = blend_registry()
    assert len(registry) == 21
    assert registry.iloc[0]["nonmarket_weight"] == pytest.approx(0.0)
    assert registry.iloc[-1]["nonmarket_weight"] == pytest.approx(1.0)


def test_market_only_retained_when_blend_gain_is_within_one_se():
    frame = _frame()
    market = np.tile([0.70, 0.65, 0.60, 0.30, 0.25, 0.20], 12)
    nonmarket = market.copy()
    nonmarket[0] += 0.01
    decision, table, _ = select_blend_one_se(
        frame,
        market,
        nonmarket,
        weights=np.array([0.0, 0.5, 1.0]),
    )
    assert decision.winner == "blend_w0.00"
    assert decision.incumbent_retained
    assert set(table["candidate"]) == {
        "blend_w0.00",
        "blend_w0.50",
        "blend_w1.00",
    }


def test_clear_blend_can_displace_market_only():
    frame = _frame()
    market = np.full(len(frame), 0.5)
    nonmarket = np.tile([0.9, 0.85, 0.8, 0.2, 0.15, 0.1], 12)
    decision, _, _ = select_blend_one_se(
        frame,
        market,
        nonmarket,
        weights=np.array([0.0, 0.5, 1.0]),
    )
    assert decision.winner != "blend_w0.00"
    assert not decision.incumbent_retained
    assert decision.selected_nonmarket_weight > 0.0
