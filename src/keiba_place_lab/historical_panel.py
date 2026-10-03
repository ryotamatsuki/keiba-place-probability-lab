"""Leakage-safe historical horse-racing panel construction.

The input is a standardized event-row table. Every historical feature is derived
with an explicit one-race lag before it can be used for a later target row.
Market/odds variables are not required or consumed.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "race_id",
    "race_date",
    "horse_id",
    "horse_name",
    "horse_no",
    "field_size",
    "sex",
    "age",
    "assigned_weight_kg",
    "distance_m",
    "surface",
    "racecourse",
    "course_layout",
    "race_class",
    "handicap_indicator",
    "finish_position",
    "race_time_seconds",
    "early_position",
    "is_open_plus",
    "is_graded",
}

FORBIDDEN_INPUT_TOKENS = ("odds", "popularity", "payout", "market")


def find_forbidden_market_columns(columns: Iterable[str]) -> list[str]:
    """Return columns whose names suggest Stage-4-prohibited market data."""
    bad: list[str] = []
    for column in columns:
        lower = str(column).lower()
        if any(token in lower for token in FORBIDDEN_INPUT_TOKENS):
            bad.append(str(column))
    return bad


def _prior_rolling(
    frame: pd.DataFrame,
    source: str,
    window: int,
    op: str,
) -> pd.Series:
    """Compute a rolling statistic from strictly prior starts for each horse."""
    shifted = frame.groupby("horse_id", sort=False)[source].shift(1)
    rolling = shifted.groupby(frame["horse_id"], sort=False).rolling(window, min_periods=1)
    result = getattr(rolling, op)().reset_index(level=0, drop=True)
    return result.reindex(frame.index)


def build_historical_panel(rows: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe, horse-by-race historical features.

    The returned table keeps outcome columns for model evaluation, but all
    feature columns are constructed from current entry information and/or
    strictly earlier races for that horse.
    """
    missing = REQUIRED_COLUMNS - set(rows.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    forbidden = find_forbidden_market_columns(rows.columns)
    if forbidden:
        raise ValueError(
            "standardized Stage 3.6 input must exclude market columns: "
            + ", ".join(forbidden)
        )

    x = rows.copy()
    x["race_date"] = pd.to_datetime(x["race_date"], errors="raise")
    x = x.sort_values(["race_date", "race_id", "horse_no"], kind="stable").reset_index(drop=True)

    if x.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("duplicate race_id × horse_id rows detected")
    if (x["field_size"] < 2).any():
        raise ValueError("field_size must be >= 2")
    if ((x["horse_no"] < 1) | (x["horse_no"] > x["field_size"])).any():
        raise ValueError("horse_no must be within field_size")

    # Current-race outcomes. These are labels / historical source values only.
    x["top3_label"] = x["finish_position"].between(1, 3).astype("int8")
    x["finish_pct_current"] = (x["finish_position"] - 1) / (x["field_size"] - 1)

    winner_time = x.groupby("race_id", sort=False)["race_time_seconds"].transform("min")
    x["relative_time_loss_current"] = (x["race_time_seconds"] - winner_time) / winner_time
    invalid_time = x["race_time_seconds"].isna() | winner_time.isna() | winner_time.le(0)
    x.loc[invalid_time, "relative_time_loss_current"] = np.nan

    x["early_pos_pct_current"] = (x["early_position"] - 1) / (x["field_size"] - 1)
    x.loc[x["early_position"].isna(), "early_pos_pct_current"] = np.nan

    x["is_turf_current"] = x["surface"].astype(str).str.lower().eq("turf").astype("int8")
    x["turf_top3_current"] = (x["is_turf_current"] * x["top3_label"]).astype("int8")

    horse = x.groupby("horse_id", sort=False)

    # Strictly prior cumulative history.
    x["career_starts"] = horse.cumcount()
    x["career_top3"] = horse["top3_label"].cumsum() - x["top3_label"]
    x["turf_starts"] = horse["is_turf_current"].cumsum() - x["is_turf_current"]
    x["turf_top3"] = horse["turf_top3_current"].cumsum() - x["turf_top3_current"]

    same_distance = x.groupby(["horse_id", "distance_m", "surface"], sort=False)
    x["same_distance_starts"] = same_distance.cumcount()
    x["same_distance_top3"] = same_distance["top3_label"].cumsum() - x["top3_label"]

    same_course = x.groupby(["horse_id", "racecourse", "surface"], sort=False)
    x["same_course_starts"] = same_course.cumcount()
    x["same_course_top3"] = same_course["top3_label"].cumsum() - x["top3_label"]

    # Change from immediately previous start.
    x["prev_race_date"] = horse["race_date"].shift(1)
    x["days_since_prev"] = (x["race_date"] - x["prev_race_date"]).dt.days

    x["prev_distance_m"] = horse["distance_m"].shift(1)
    x["distance_change_from_prev_m"] = x["distance_m"] - x["prev_distance_m"]

    x["prev_surface"] = horse["surface"].shift(1)
    x["surface_changed_from_prev"] = x["surface"].ne(x["prev_surface"]).astype(float)
    x.loc[x["prev_surface"].isna(), "surface_changed_from_prev"] = np.nan

    x["prev_assigned_weight_kg"] = horse["assigned_weight_kg"].shift(1)
    x["assigned_weight_delta_from_prev_kg"] = (
        x["assigned_weight_kg"] - x["prev_assigned_weight_kg"]
    )

    # Recent form. shift(1) is the key leakage barrier.
    specs = (
        ("finish_pct_current", "recent3_finish_pct_mean", 3, "mean"),
        ("top3_label", "recent3_top3_count", 3, "sum"),
        ("is_open_plus", "recent3_open_plus_count", 3, "sum"),
        ("is_graded", "recent3_graded_count", 3, "sum"),
        ("relative_time_loss_current", "recent3_relative_time_mean", 3, "mean"),
        ("early_pos_pct_current", "recent4_early_pos_pct_mean", 4, "mean"),
    )
    for source, output, window, operation in specs:
        x[output] = _prior_rolling(x, source, window, operation)

    # Current entry / race context.
    x["draw_pct"] = (x["horse_no"] - 1) / (x["field_size"] - 1)

    def _front_share(series: pd.Series) -> float:
        observed = series.dropna()
        if observed.empty:
            return np.nan
        return float(observed.le(0.25).mean())

    x["front_forward_share"] = x.groupby("race_id", sort=False)[
        "recent4_early_pos_pct_mean"
    ].transform(_front_share)

    return x


def assign_temporal_split(panel: pd.DataFrame) -> pd.Series:
    """Assign the pre-registered first-pass chronological split."""
    years = pd.to_datetime(panel["race_date"], errors="raise").dt.year
    split = pd.Series("outside", index=panel.index, dtype="object")
    split.loc[years.between(2010, 2015)] = "warmup"
    split.loc[years.between(2016, 2022)] = "train"
    split.loc[years.between(2023, 2024)] = "validation"
    split.loc[years.eq(2025)] = "test"
    split.loc[years.ge(2026)] = "future"
    return split


def select_phase_a_cohort(
    panel: pd.DataFrame,
    *,
    surface: str = "turf",
    min_field_size: int = 8,
    min_prior_starts: int = 3,
    min_distance_m: int | None = None,
    max_distance_m: int | None = None,
) -> pd.DataFrame:
    """Select a predeclared Phase-A cohort without using model outcomes."""
    mask = (
        panel["surface"].astype(str).str.lower().eq(surface.lower())
        & panel["field_size"].ge(min_field_size)
        & panel["career_starts"].ge(min_prior_starts)
    )
    if min_distance_m is not None:
        mask &= panel["distance_m"].ge(min_distance_m)
    if max_distance_m is not None:
        mask &= panel["distance_m"].le(max_distance_m)

    out = panel.loc[mask].copy()
    out["temporal_split"] = assign_temporal_split(out)
    return out


def assert_strict_history(panel: pd.DataFrame) -> None:
    """Fail when a reconstructed previous date is not strictly earlier."""
    observed = panel["prev_race_date"].notna()
    bad = observed & (panel["prev_race_date"] >= panel["race_date"])
    if bad.any():
        raise ValueError("non-strict historical date detected")
