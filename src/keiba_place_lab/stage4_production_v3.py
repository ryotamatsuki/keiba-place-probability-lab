"""Versioned scope routing and live as-of history generation; V2 stays intact."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .nonmarket import canonicalize_target_context, validate_market_free
from .scope_expansion import straight_course_mask
from .scope_features import augment_distance_history, predict_scope_candidate
from .stage4_production_v2 import TARGET_OUTCOME_COLUMNS
from .stage4_successor_phase3a import assert_full_field_context


def validate_roster(roster: pd.DataFrame) -> pd.DataFrame:
    validate_market_free(roster.columns)
    if TARGET_OUTCOME_COLUMNS.intersection(roster.columns):
        raise ValueError("Target roster contains outcomes")
    required = {"race_id", "race_date", "horse_id", "horse_no", "horse_name", "field_size",
                "declared_field_size", "surface", "distance_m", "racecourse", "race_class",
                "age", "sex", "assigned_weight_kg", "turn_direction"}
    if required.difference(roster.columns):
        raise ValueError(f"Missing roster columns: {sorted(required.difference(roster.columns))}")
    out = canonicalize_target_context(roster).reset_index(drop=True)
    for col in ("race_id", "horse_id"):
        if out[col].isna().any():
            raise ValueError("Missing roster identity")
        out[col] = out[col].astype("string")
    out["race_date"] = pd.to_datetime(out.race_date, errors="raise")
    if out.empty or out.race_id.nunique() != 1:
        raise ValueError("Provide one complete race per prediction")
    for col in ("race_date", "surface", "distance_m", "declared_field_size", "racecourse", "race_class", "turn_direction"):
        if out[col].isna().any() or out[col].nunique() != 1:
            raise ValueError(f"Invalid race-level roster field: {col}")
    assert_full_field_context(out)
    if not pd.to_numeric(out.field_size, errors="raise").eq(len(out)).all():
        raise ValueError("Active field_size must equal roster length")
    numbers = pd.to_numeric(out.horse_no, errors="raise").to_numpy(float)
    declared = float(out.declared_field_size.iloc[0])
    if (not np.isfinite(numbers).all() or not np.equal(numbers, numbers.astype(int)).all()
            or out.horse_no.duplicated().any() or declared != int(declared)
            or declared < len(out) or np.any(numbers < 1) or np.any(numbers > declared)):
        raise ValueError("Invalid declared roster/draw positions")
    out["draw_pct"] = (numbers - 1) / max(declared - 1, 1)
    return out


def build_live_context(history: pd.DataFrame, roster: pd.DataFrame, *, max_history_age_days: int = 7):
    """Recompute target histories from completed standardized prior records.

    The history feed must be complete through its last date. Caller specifies the
    allowed freshness bound; the shipped 2025 feed fails a current 2026 prediction.
    Past winner-time context uses all historical starters before horse filtering.
    """
    target = validate_roster(roster)
    validate_market_free(history.columns)
    date = target.race_date.iloc[0]
    history = canonicalize_target_context(history).copy()
    history["race_date"] = pd.to_datetime(history.race_date, errors="raise")
    if history.empty or history.race_date.isna().any() or history.race_date.ge(date).any():
        raise ValueError("History must be nonempty and strictly before target date")
    if max_history_age_days < 0 or (date - history.race_date.max()).days > max_history_age_days:
        raise ValueError("Stale history: update completed races before live prediction")
    required = {"race_id", "horse_id", "horse_no", "finish_position", "race_time_seconds", "early_position",
                "field_size", "surface", "distance_m", "racecourse", "assigned_weight_kg",
                "is_open_plus", "is_graded"}
    if required.difference(history.columns) or history.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Incomplete/duplicate standardized history")
    assert_full_field_context(history)
    if history.finish_position.isna().any() and "outcome_confirmed" not in history:
        raise ValueError("Unresolved prior outcomes; official DNF needs outcome_confirmed")
    if "outcome_confirmed" in history and not history.outcome_confirmed.eq(True).all():
        raise ValueError("Unresolved prior outcomes")
    history["top3_label"] = history.finish_position.between(1, 3).astype(int)
    winners = history.race_time_seconds.where(history.finish_position.eq(1)).groupby(history.race_id).transform("min")
    history["relative_time"] = ((history.race_time_seconds - winners) / winners).where(winners.gt(0))
    history["finish_pct"] = (history.finish_position - 1) / (history.field_size - 1)
    history["early_pct"] = ((history.early_position - 1) / (history.field_size - 1)).where(
        history.early_position.between(1, history.field_size))
    history = history.sort_values(["race_date", "race_id", "horse_no"], kind="stable")
    groups = {str(k): g for k, g in history.groupby("horse_id", sort=False)}
    rows = []
    for _, row in target.iterrows():
        r = row.to_dict()
        past = groups.get(str(row.horse_id), history.iloc[:0])
        for prefix, mask in (
            ("career", pd.Series(True, index=past.index)),
            ("turf", past.surface.eq("turf")),
            ("same_distance", past.surface.eq(row.surface) & past.distance_m.eq(row.distance_m)),
            ("same_course", past.surface.eq(row.surface) & past.racecourse.eq(row.racecourse)),
        ):
            r[f"{prefix}_starts"] = int(mask.sum())
            r[f"{prefix}_top3"] = int(past.loc[mask, "top3_label"].sum())
        prev = past.iloc[-1] if len(past) else None
        r["prev_race_date"] = prev.race_date if prev is not None else pd.NaT
        r["days_since_prev"] = (date - prev.race_date).days if prev is not None else np.nan
        r["distance_change_from_prev_m"] = row.distance_m - prev.distance_m if prev is not None else np.nan
        r["surface_changed_from_prev"] = float(row.surface != prev.surface) if prev is not None else np.nan
        r["assigned_weight_delta_from_prev_kg"] = row.assigned_weight_kg - prev.assigned_weight_kg if prev is not None else np.nan
        recent = past.tail(3)
        for src, name, op in (
            ("finish_pct", "recent3_finish_pct_mean", "mean"),
            ("top3_label", "recent3_top3_count", "sum"),
            ("is_open_plus", "recent3_open_plus_count", "sum"),
            ("is_graded", "recent3_graded_count", "sum"),
            ("relative_time", "recent3_relative_time_mean", "mean"),
        ):
            r[name] = getattr(recent[src], op)() if len(recent) else np.nan
        r["recent4_early_pos_pct_mean"] = past.tail(4).early_pct.mean()
        rows.append(r)
    context = pd.DataFrame(rows)
    known_style = context.recent4_early_pos_pct_mean.dropna()
    context["front_forward_share"] = known_style.le(.25).mean() if len(known_style) else np.nan
    keys = ["horse_id", "race_date", "surface", "distance_m", "racecourse", "top3_label"]
    prior = history.loc[history.horse_id.astype(str).isin(target.horse_id.astype(str)), keys]
    current = context.drop(columns=[c for c in context if c.endswith(("_starts", "_top3"))], errors="ignore")
    enriched = augment_distance_history(pd.concat([prior, current], ignore_index=True, sort=False)).tail(len(context))
    for col in ["same_surface_starts", "same_surface_top3", *(f"{b}_{s}" for b in ("near", "regime", "course_distance") for s in ("starts", "top3"))]:
        context[col] = enriched[col].to_numpy()
    return context


def route_model(context, routing):
    surface = str(context.surface.iloc[0])
    distance = float(context.distance_m.iloc[0])
    if len(context) < 8 or straight_course_mask(context).any() or surface not in routing["surfaces"]:
        return None
    policy = routing["surfaces"][surface]
    if distance == 1200 and "1200_override" in policy:
        return policy["1200_override"]["model_id"]
    for band in policy["routes"]:
        if band["min_distance_m"] <= distance <= band["max_distance_m"]:
            return band["model_id"]
    return None


def score_scope_target(bundle, context):
    assert_full_field_context(context)
    if context.race_id.nunique() != 1 or context.surface.nunique() != 1 or context.distance_m.nunique() != 1:
        raise ValueError("Prediction requires one consistent race")
    model_id = route_model(context, bundle["routing"])
    result = context[["race_id", "race_date", "horse_id", "horse_no", "horse_name",
                      "field_size", "declared_field_size", "surface", "distance_m", "turn_direction"]].copy()
    result["p_nonmarket"] = np.nan
    result["route"] = "market-only"
    result["model_id"] = ""
    if model_id is None:
        return result
    fit = bundle["models"][model_id]
    if pd.Timestamp(fit["train_max_date"]) >= pd.Timestamp(context.race_date.iloc[0]):
        raise ValueError("Production model contains target/future training dates")
    eligible = context.career_starts.ge(3).to_numpy()
    if not eligible.any():
        return result
    p = predict_scope_candidate(fit["model"], context.loc[eligible], context,
                                prior_mean=fit["prior_mean"], blocks=fit["blocks"], surface=fit["surface"])
    result.loc[eligible, "p_nonmarket"] = p
    result.loc[eligible, "route"] = "nonmarket-shadow"
    result.loc[eligible, "model_id"] = model_id
    return result
