"""Validation for the canonical non-market feature matrix."""

from __future__ import annotations

from collections.abc import Iterable

FORBIDDEN_MARKET_TOKENS = (
    "odds",
    "market",
    "popularity",
    "payout",
)


def forbidden_columns(columns: Iterable[str]) -> list[str]:
    """Return columns whose names suggest market information."""
    bad: list[str] = []
    for column in columns:
        lower = column.lower()
        if any(token in lower for token in FORBIDDEN_MARKET_TOKENS):
            bad.append(column)
    return bad


def validate_probability_like_unit_interval(value: float, name: str) -> None:
    """Validate a normalized feature that must lie in [0, 1]."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def normalized_finish_position(finish_position: int, field_size: int) -> float:
    """Normalize finish position to [0,1], lower is better."""
    if field_size < 2:
        raise ValueError("field_size must be at least 2")
    if not 1 <= finish_position <= field_size:
        raise ValueError("finish_position must be within the field")
    return (finish_position - 1) / (field_size - 1)


def relative_time_loss(horse_time_seconds: float, margin_seconds: float) -> float:
    """Return margin divided by estimated winner time."""
    if horse_time_seconds <= 0:
        raise ValueError("horse_time_seconds must be positive")
    if margin_seconds < 0 or margin_seconds >= horse_time_seconds:
        raise ValueError("margin_seconds must be >=0 and smaller than horse time")
    winner_time = horse_time_seconds - margin_seconds
    return margin_seconds / winner_time
