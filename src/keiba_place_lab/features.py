"""Helpers for auditable, market-free race features."""

from __future__ import annotations


def normalized_early_position(position: int, field_size: int) -> float:
    """Map a running position to [0, 1], where 0 is the front of the field."""
    if field_size < 2:
        raise ValueError("field_size must be at least 2")
    if position < 1 or position > field_size:
        raise ValueError("position must be between 1 and field_size")
    return (position - 1) / (field_size - 1)


def pace_style_from_mean_position(mean_position_pct: float) -> str:
    """Return a descriptive heuristic label; not a learned model output."""
    if not 0.0 <= mean_position_pct <= 1.0:
        raise ValueError("mean_position_pct must be in [0, 1]")
    if mean_position_pct <= 0.20:
        return "front"
    if mean_position_pct <= 0.45:
        return "forward_mid"
    if mean_position_pct <= 0.70:
        return "mid_rear"
    return "deep_rear"
