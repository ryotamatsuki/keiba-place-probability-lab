"""Scope-expansion helpers for wider JRA turf training populations."""

from __future__ import annotations

import pandas as pd

from .historical_panel import select_phase_a_cohort
from .stage4_successor_phase3a import assert_full_field_context

STRAIGHT_TOKENS = ("straight", "直線")


def straight_course_mask(frame: pd.DataFrame) -> pd.Series:
    """Identify straight-course rows without using outcomes."""
    if "turn_direction" not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    text = frame["turn_direction"].astype("string").fillna("").str.lower()
    mask = pd.Series(False, index=frame.index, dtype=bool)
    for token in STRAIGHT_TOKENS:
        mask |= text.str.contains(token.lower(), regex=False)
    return mask


def select_turf_scope(
    panel: pd.DataFrame,
    *,
    min_distance_m: int,
    max_distance_m: int,
    exclude_straight: bool = True,
) -> pd.DataFrame:
    """Apply the frozen Phase-A eligibility rule to a turf distance band."""
    out = select_phase_a_cohort(
        panel,
        surface="turf",
        min_field_size=8,
        min_prior_starts=3,
        min_distance_m=min_distance_m,
        max_distance_m=max_distance_m,
    )
    if exclude_straight:
        out = out.loc[~straight_course_mask(out)].copy()
    return out


def full_context_for_races(
    full_panel: pd.DataFrame,
    race_ids: set[str],
) -> pd.DataFrame:
    """Return and validate complete actual-starter context for selected races."""
    normalized = full_panel.copy()
    normalized["race_id"] = normalized["race_id"].astype("string")
    context = normalized.loc[normalized["race_id"].isin({str(x) for x in race_ids})].copy()
    assert_full_field_context(context)
    return context
