"""Immutable pre-off marginal-probability locks and one prospective review."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .model_selection import select_with_incumbent_one_se
from .scope_expansion import straight_course_mask

WEIGHTS = (0.0, 0.10, 0.25, 0.50, 1.0)
MIN_RACES, MIN_DATES, MIN_DAYS = 1000, 60, 90


def utc_timestamp(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("Timezone-aware timestamps required")
    return stamp.tz_convert("UTC")


def make_lock(frame, *, off_at, market_asof, model_hash, protocol_hash, now=None):
    now = utc_timestamp(now if now is not None else datetime.now(UTC))
    off = utc_timestamp(off_at)
    market = utc_timestamp(market_asof)
    remaining = (off - now).total_seconds()
    if not 540 <= remaining <= 600:
        raise ValueError("Lock must occur in the 10-to-9-minute pre-off window")
    if not 0 <= (now - market).total_seconds() <= 300:
        raise ValueError("Market snapshot must be <=5 minutes old and not future")
    required = {"race_id", "race_date", "horse_id", "horse_no", "field_size", "surface",
                "distance_m", "p_market", "p_nonmarket", "route", "model_id"}
    if required.difference(frame.columns):
        raise ValueError("Missing lock columns")
    work = frame.reset_index(drop=True).copy()
    if work.empty or work.race_id.nunique() != 1 or work.horse_id.isna().any() or work.horse_id.duplicated().any() or work.horse_no.duplicated().any():
        raise ValueError("Lock must contain one complete unique roster")
    if not work.field_size.eq(len(work)).all() or len(work) < 8:
        raise ValueError("Incomplete lock roster")
    if work.surface.nunique() != 1 or not work.surface.isin(["turf", "dirt"]).all() or work.distance_m.nunique() != 1 or not work.distance_m.between(1000, 2600).all():
        raise ValueError("Lock outside supported scope")
    if "turn_direction" not in work or straight_course_mask(work).any():
        raise ValueError("Straight/unknown course routing is unsupported")
    if not pd.to_datetime(work.race_date).dt.date.eq(off.tz_convert("Asia/Tokyo").date()).all():
        raise ValueError("Race date and scheduled off disagree")
    if any(c in work for c in ("top3_label", "finish_position")):
        raise ValueError("Outcomes forbidden in pre-off lock")
    market_p = pd.to_numeric(work.p_market, errors="raise").to_numpy(float)
    nonmarket_p = pd.to_numeric(work.p_nonmarket, errors="raise").to_numpy(float)
    eligible = work.route.eq("nonmarket-shadow").to_numpy()
    if not eligible.any() or not work.route.isin(["market-only", "nonmarket-shadow"]).all():
        raise ValueError("No supported nonmarket candidate")
    if (not np.isfinite(market_p).all() or ((market_p < 0) | (market_p > 1)).any()
            or not np.isfinite(nonmarket_p[eligible]).all()
            or ((nonmarket_p[eligible] < 0) | (nonmarket_p[eligible] > 1)).any()
            or np.isfinite(nonmarket_p[~eligible]).any()):
        raise ValueError("Invalid lock probabilities/routing")
    fallback = np.where(eligible, nonmarket_p, market_p)
    for weight in WEIGHTS:
        work[f"p_w{int(weight * 100):03}"] = (1 - weight) * market_p + weight * fallback
    payload = work.to_csv(index=False, lineterminator="\n").encode()
    manifest = {"version": "stage5_scope_v3_prospective", "race_id": str(work.race_id.iloc[0]),
                "off_at": off.isoformat(), "locked_at": now.isoformat(), "market_asof": market.isoformat(),
                "model_hash": model_hash, "protocol_hash": protocol_hash,
                "payload_sha256": hashlib.sha256(payload).hexdigest(), "rows": len(work),
                "weights": WEIGHTS, "stage5_canonical": "market-only"}
    return payload, manifest


def review_predictions(frame):
    """Do not expose interim scores; adoption is never automatic."""
    dates = pd.to_datetime(frame.race_date)
    races, distinct_dates = frame.race_id.nunique(), dates.nunique()
    elapsed = (dates.max() - dates.min()).days if len(frame) else 0
    counts = {"races": races, "dates": distinct_dates, "elapsed_days": elapsed,
              "required": {"races": MIN_RACES, "dates": MIN_DATES, "elapsed_days": MIN_DAYS}}
    if races < MIN_RACES or distinct_dates < MIN_DATES or elapsed < MIN_DAYS:
        return {"status": "collecting", **counts, "production_changed": False}, None
    # Use the first complete calendar day that meets every threshold, even if
    # collection continued. Later outcomes cannot change this period's decision.
    daily = frame.assign(_date=dates).groupby("_date").race_id.nunique().sort_index()
    cumulative = daily.cumsum()
    for index, date in enumerate(daily.index):
        if cumulative.iloc[index] >= MIN_RACES and index + 1 >= MIN_DATES and (date - daily.index[0]).days >= MIN_DAYS:
            frame = frame.loc[dates.le(date)].copy()
            counts.update({"races": frame.race_id.nunique(), "dates": index + 1,
                           "elapsed_days": (date - daily.index[0]).days,
                           "window_end_date": str(date.date())})
            break
    predictions = {f"w{int(w * 100):03}": frame[f"p_w{int(w * 100):03}"].to_numpy(float) for w in WEIGHTS}
    decision, scores = select_with_incumbent_one_se(frame, predictions, incumbent="w000")
    return {"status": "review_ready", **counts, "decision": decision.__dict__,
            "production_changed": False, "requires_recorded_adoption": True}, scores
