import pytest

from keiba_place_lab.feature_validation import (
    forbidden_columns,
    normalized_finish_position,
    relative_time_loss,
)


def test_market_columns_are_detected() -> None:
    assert forbidden_columns(["age", "win_odds", "market_rank"]) == ["win_odds", "market_rank"]


def test_clean_columns_pass() -> None:
    assert forbidden_columns(["age", "days_since_prev", "recent3_finish_pct_mean"]) == []


def test_normalized_finish_position() -> None:
    assert normalized_finish_position(1, 18) == 0.0
    assert normalized_finish_position(18, 18) == 1.0


def test_relative_time_loss_for_winner() -> None:
    assert relative_time_loss(68.2, 0.0) == 0.0


def test_relative_time_loss_rejects_invalid_margin() -> None:
    with pytest.raises(ValueError):
        relative_time_loss(68.2, 68.2)
