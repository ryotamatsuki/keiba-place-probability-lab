"""Production helpers for the frozen Stage-4 successor v2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .nonmarket import canonicalize_target_context, validate_market_free
from .stage4_successor_phase3a import (
    fit_relative_candidate,
    predict_relative_candidate,
)

TARGET_OUTCOME_COLUMNS = {
    "top3_label",
    "finish_position",
    "finish_time",
    "payout",
    "payoff",
    "result",
    "rank_result",
}


@dataclass(frozen=True)
class ProductionFit:
    model: object
    prior_mean: float
    fit_rows: int
    fit_races: int
    fit_date_min: str
    fit_date_max: str


def _normalize_target_ids(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["race_id"] = out["race_id"].astype("string")
    if "horse_id" not in out.columns:
        out["horse_id"] = (
            "target-" + pd.to_numeric(out["horse_no"], errors="raise").astype(int).astype(str)
        )
    out["horse_id"] = out["horse_id"].astype("string")
    return out


def prepare_target_context(
    frame: pd.DataFrame,
    *,
    race_id: str,
    race_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate a live target roster and return all-active and eligible frames."""
    validate_market_free(frame.columns)
    bad_outcome = sorted(TARGET_OUTCOME_COLUMNS.intersection(frame.columns))
    if bad_outcome:
        raise ValueError(f"Target input contains outcome columns: {bad_outcome}")

    out = canonicalize_target_context(frame).copy()
    out["race_id"] = str(race_id)
    out["race_date"] = pd.Timestamp(race_date)

    required = {
        "horse_no",
        "horse_name",
        "field_size",
        "surface",
        "distance_m",
        "career_starts",
        "draw_pct",
    }
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"Missing target production columns: {sorted(missing)}")

    if out["horse_no"].duplicated().any():
        raise ValueError("Target horse_no must be unique")
    if out["horse_name"].duplicated().any():
        raise ValueError("Target horse_name must be unique")

    field_sizes = pd.to_numeric(out["field_size"], errors="raise")
    if field_sizes.nunique() != 1:
        raise ValueError("field_size must be constant across active target starters")
    active_field_size = int(field_sizes.iloc[0])
    if active_field_size != len(out):
        raise ValueError(
            f"Active target rows ({len(out)}) must equal field_size ({active_field_size})"
        )
    if active_field_size < 8:
        raise ValueError("Stage-4 successor v2 is frozen only for field_size >= 8")

    surface = out["surface"].astype(str).str.lower()
    if not surface.eq("turf").all():
        raise ValueError("Stage-4 successor v2 is frozen only for turf races")
    distance = pd.to_numeric(out["distance_m"], errors="raise")
    if not distance.eq(1200).all():
        raise ValueError("Stage-4 successor v2 is frozen only for 1200m races")

    horse_no = pd.to_numeric(out["horse_no"], errors="raise").astype(int)
    if "declared_field_size" in out.columns:
        declared = pd.to_numeric(out["declared_field_size"], errors="raise").astype(int)
        if declared.nunique() != 1:
            raise ValueError("declared_field_size must be constant within target race")
    else:
        # Safe inference is possible only when there is no scratch gap.
        max_horse_no = int(horse_no.max())
        expected_numbers = set(range(1, active_field_size + 1))
        if set(horse_no.tolist()) != expected_numbers or max_horse_no != active_field_size:
            raise ValueError(
                "declared_field_size is required when the active roster contains a scratch/gap"
            )
        declared = pd.Series(active_field_size, index=out.index, dtype=int)
        out["declared_field_size"] = declared

    declared_size = int(declared.iloc[0])
    if declared_size < active_field_size:
        raise ValueError("declared_field_size cannot be smaller than active field_size")
    if horse_no.lt(1).any() or horse_no.gt(declared_size).any():
        raise ValueError("horse_no must lie within declared_field_size")

    expected_draw = (horse_no.astype(float) - 1.0) / float(declared_size - 1)
    observed_draw = pd.to_numeric(out["draw_pct"], errors="raise").to_numpy(dtype=float)
    if not np.allclose(observed_draw, expected_draw.to_numpy(dtype=float), atol=1e-6):
        raise ValueError(
            "draw_pct does not match frozen (horse_no-1)/(declared_field_size-1) definition"
        )

    out["field_size"] = active_field_size
    out["declared_field_size"] = declared_size
    out = _normalize_target_ids(out)

    eligible = out.loc[
        pd.to_numeric(out["career_starts"], errors="coerce").ge(3)
    ].copy()
    if eligible.empty:
        raise ValueError("No target runners satisfy frozen career_starts >= 3 eligibility")

    return out, eligible


def fit_production_successor(
    eligible_history: pd.DataFrame,
    full_context: pd.DataFrame,
    *,
    target_date: str,
) -> ProductionFit:
    """Fit the frozen successor on all allowed history strictly before target_date."""
    validate_market_free(eligible_history.columns)
    validate_market_free(full_context.columns)

    train = eligible_history.copy()
    context = full_context.copy()
    train["race_date"] = pd.to_datetime(train["race_date"], errors="raise")
    context["race_date"] = pd.to_datetime(context["race_date"], errors="raise")
    cutoff = pd.Timestamp(target_date)
    if train["race_date"].ge(cutoff).any():
        raise ValueError("Production fit contains rows on/after target date")
    if context["race_date"].ge(cutoff).any():
        raise ValueError("Production context contains rows on/after target date")

    train_races = set(train["race_id"].astype("string"))
    context = context.loc[
        context["race_id"].astype("string").isin(train_races)
    ].copy()

    model, prior_mean, _ = fit_relative_candidate(train, context)
    return ProductionFit(
        model=model,
        prior_mean=float(prior_mean),
        fit_rows=len(train),
        fit_races=int(train["race_id"].nunique()),
        fit_date_min=str(train["race_date"].min().date()),
        fit_date_max=str(train["race_date"].max().date()),
    )


def score_production_target(
    fit: ProductionFit,
    active_context: pd.DataFrame,
    eligible_target: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score eligible runners using all active starters as relative-feature context."""
    probability, coverage = predict_relative_candidate(
        fit.model,
        eligible_target,
        active_context,
        prior_mean=fit.prior_mean,
    )
    result = eligible_target[
        ["horse_no", "horse_name", "horse_id", "field_size", "declared_field_size"]
    ].copy()
    result["p_top3_stage4_v2"] = probability
    result["rank_eligible"] = (
        result["p_top3_stage4_v2"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    result = result.sort_values("rank_eligible", kind="stable").reset_index(drop=True)
    return result, coverage
