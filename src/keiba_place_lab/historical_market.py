"""Historical win-market reconstruction for Stage 5."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .market import harville_top_k_probabilities, normalized_win_probabilities

NONSTARTER_STATUSES = {"", "取", "除"}


def _parse_positive_float(value: object) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")
    return parsed if np.isfinite(parsed) and parsed > 1.0 else float("nan")


def _parse_horse_no(value: object) -> int:
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def reconstruct_historical_market(
    results_path: Path,
    race_ids: set[str],
) -> tuple[pd.DataFrame, dict]:
    """Reconstruct win-market Harville P(top3) on complete starter fields.

    The raw Kaggle result file contains final win odds. Market probabilities are
    calculated using every starter in a race before any Stage-4 eligibility filter
    is applied. Races with a missing/invalid starter odds value are excluded as a
    whole rather than partially normalized.
    """
    requested = {str(x) for x in race_ids}
    by_race: dict[str, list[dict]] = {rid: [] for rid in requested}
    malformed_requested = 0

    with Path(results_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"race_id", "number", "rank", "odds"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Historical result file missing columns: {sorted(missing)}")

        for row in reader:
            rid = (row.get("race_id") or "").strip()
            if rid not in requested:
                continue
            if None in row or any(value is None for key, value in row.items() if key is not None):
                malformed_requested += 1
                continue
            rank = (row.get("rank") or "").strip()
            if rank in NONSTARTER_STATUSES:
                continue
            by_race[rid].append(
                {
                    "race_id": rid,
                    "horse_no": _parse_horse_no(row.get("number")),
                    "historical_win_odds": _parse_positive_float(row.get("odds")),
                }
            )

    records: list[dict] = []
    incomplete: list[dict] = []
    for rid in sorted(requested):
        starters = pd.DataFrame(by_race.get(rid, []))
        if starters.empty:
            incomplete.append({"race_id": rid, "reason": "no_source_starters"})
            continue
        if starters["horse_no"].le(0).any() or starters["horse_no"].duplicated().any():
            incomplete.append({"race_id": rid, "reason": "invalid_or_duplicate_horse_no"})
            continue
        if len(starters) < 4:
            incomplete.append({"race_id": rid, "reason": "field_too_small"})
            continue
        if starters["historical_win_odds"].isna().any():
            incomplete.append({"race_id": rid, "reason": "missing_or_invalid_win_odds"})
            continue

        p_win, overround = normalized_win_probabilities(
            starters["historical_win_odds"].to_numpy(dtype=float)
        )
        p_top3 = harville_top_k_probabilities(p_win, k=3)
        for idx, row in enumerate(starters.itertuples(index=False)):
            records.append(
                {
                    "race_id": rid,
                    "horse_no": int(row.horse_no),
                    "historical_win_odds": float(row.historical_win_odds),
                    "market_p_win": float(p_win[idx]),
                    "market_p_top3": float(p_top3[idx]),
                    "market_overround": float(overround),
                    "market_field_size": int(len(starters)),
                }
            )

    frame = pd.DataFrame(records)
    if not frame.empty:
        sums = frame.groupby("race_id")["market_p_top3"].sum()
        if not np.allclose(sums.to_numpy(), 3.0, atol=1e-9):
            raise ValueError("Historical market P(top3) does not sum to 3")
    diagnostics = {
        "requested_races": len(requested),
        "complete_market_races": int(frame["race_id"].nunique()) if not frame.empty else 0,
        "incomplete_market_races": len(incomplete),
        "malformed_requested_rows": malformed_requested,
        "incomplete_reason_counts": (
            pd.DataFrame(incomplete)["reason"].value_counts().to_dict()
            if incomplete
            else {}
        ),
    }
    return frame, diagnostics
