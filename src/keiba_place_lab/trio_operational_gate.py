"""Fail-closed eligibility for offering 3-renpuku tickets in an operational UI.

Research predictions are NOT bets. A credible deployment additionally needs
matching-model untouched holdout calibration, real timestamped pre-off 3-renpuku
prices, and bettor-defined exposure policy. Missing evidence -> NO_BET.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def validate_complete_trio_quotes(predictions: pd.DataFrame,
                                  quotes: pd.DataFrame,
                                  *, expected_horse_numbers: set[int],
                                  expected_race_id: str) -> pd.DataFrame:
    """An incomplete or mixed-snapshot odds set is never priced as complete."""
    keys = ["horse_1", "horse_2", "horse_3"]
    needed = set(keys + ["odds", "race_id", "source_updated_at", "capture_at"])
    if not needed.issubset(quotes):
        raise ValueError("3-renpuku quote source/time columns are missing")
    if not set(keys + ["probability"]).issubset(predictions):
        raise ValueError("Trio probabilities are incomplete")
    full = {
        tuple(c) for c in combinations(sorted(expected_horse_numbers), 3)
    }
    if len(full) != len(predictions) or len(full) != len(quotes):
        raise ValueError("Not all exact 3-horse tickets have prices and probabilities")
    def clean(frame: pd.DataFrame) -> pd.DataFrame:
        v = frame.copy()
        for col in keys:
            v[col] = pd.to_numeric(v[col], errors="raise").astype(int)
        triplets = [tuple(sorted(x)) for x in v[keys].to_numpy()]
        if len(set(triplets)) != len(v) or set(triplets) != full:
            raise ValueError("Duplicate, wrong-field, or missing 3-renpuku combinations")
        v[keys] = np.asarray(triplets, dtype=int)
        return v
    pp = clean(predictions)
    qq = clean(quotes)
    if set(qq.race_id.astype(str)) != {str(expected_race_id)}:
        raise ValueError("Odds are from another race or a mixed event")
    updates = pd.to_datetime(qq.source_updated_at, errors="raise", utc=True)
    captures = pd.to_datetime(qq.capture_at, errors="raise", utc=True)
    if updates.isna().any() or captures.isna().any():
        raise ValueError("Quote snapshot lacks verifiable timestamps")
    if updates.nunique() != 1 or captures.nunique() != 1:
        raise ValueError("All 3-renpuku prices must come from one identical timestamp")
    lag = (captures.iloc[0] - updates.iloc[0]).total_seconds()
    if lag < 0 or lag > 180:
        raise ValueError("Quotes are older than three minutes or from the future")
    odds = pd.to_numeric(qq.odds, errors="raise")
    if not np.isfinite(odds).all() or odds.le(1).any():
        raise ValueError("Missing/invalid Japanese total-return decimal odds")
    probs = pd.to_numeric(pp.probability, errors="raise")
    if not np.isfinite(probs).all() or probs.le(0).any() or not np.isclose(
            probs.sum(), 1, atol=1e-9):
        raise ValueError("Invalid joint trio probabilities")
    combined = pp.merge(qq[keys+["odds"]], on=keys, validate="one_to_one")
    combined["model_ev_per_yen_at_snapshot"] = (
        combined.probability * combined.odds - 1
    )
    return combined


def evaluate_readiness(
    *,
    matching_frozen_ensemble_holdout: bool,
    independent_prospective_t10_validation: bool,
    empirical_trio_calibration_approved: bool,
    full_same_snapshot_trio_quotes: bool,
    price_slippage_assessed: bool,
    approved_risk_limits: bool,
) -> dict:
    requirements = {
        "matching_frozen_ensemble_holdout": matching_frozen_ensemble_holdout,
        "independent_prospective_t10_validation": independent_prospective_t10_validation,
        "empirical_trio_calibration_approved": empirical_trio_calibration_approved,
        "full_same_snapshot_trio_quotes": full_same_snapshot_trio_quotes,
        "price_slippage_assessed": price_slippage_assessed,
        "approved_risk_limits": approved_risk_limits,
    }
    if any(type(v) is not bool for v in requirements.values()):
        raise TypeError("Production gates require explicit verified Boolean evidence")
    blockers = [k for k,v in requirements.items() if not v]
    return {
        "decision": "NO_BET" if blockers else "ELIGIBLE_FOR_MANUAL_REVIEW",
        "evidence": requirements,
        "blockers": blockers,
        "never_auto_place_wagers": True,
    }
