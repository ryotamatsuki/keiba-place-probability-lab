"""Optional REAL three-horse tote snapshot benchmark/combiner, fail-closed.

Real JRA 3-renpuku odds are NOT available in the historical Kaggle FINAL WIN
odds dataset. This module trains nothing unless full verified archived quotes
and original observation timestamps are provided by an authorized source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .trio_operational_gate import validate_complete_trio_quotes


def toto_market_distribution(odds: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    """All-ticket tote implied market shares; not a declared profitable strategy."""
    o = np.asarray(odds, dtype=float)
    if (o.ndim != 1 or len(o) < 4 or not np.isfinite(o).all()
            or np.any(o <= 1) or not np.isfinite(temperature)
            or temperature <= 0):
        raise ValueError("Real three-horse tote odds must be valid on all combinations")
    logp = -float(temperature)*np.log(o)
    return np.exp(logp - logsumexp(logp))



def validate_archived_tote_snapshot(
    predictions: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    expected_horse_numbers: set[int],
    expected_race_id: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Match every ticket to a verified, single-time real JRA 3-renpuku quote.

    Uses the strict timestamp, completeness and odds assertions already
    implemented for the production NO_BET gate. No missing price imputation.
    """
    full = validate_complete_trio_quotes(
        predictions, quotes, expected_horse_numbers=expected_horse_numbers,
        expected_race_id=expected_race_id,
    )
    implied = toto_market_distribution(full.odds.to_numpy(float))
    return full, implied

def geometric_joint_blend(f3: np.ndarray, real_market: np.ndarray,
                          *, market_weight: float) -> np.ndarray:
    """Logarithmic pooling of two joint distributions, not win market PL."""
    a = np.asarray(f3, dtype=float)
    b = np.asarray(real_market, dtype=float)
    if (a.shape != b.shape or a.ndim != 1 or
            not np.isfinite(a).all() or not np.isfinite(b).all() or
            np.any(a<=0) or np.any(b<=0) or
            not np.isclose(a.sum(),1,atol=1e-9) or
            not np.isclose(b.sum(),1,atol=1e-9) or
            not 0 <= market_weight <= 1):
        raise ValueError("Only complete matched 3-renpuku distributions can be combined")
    logp = (1-market_weight)*np.log(a) + market_weight*np.log(b)
    return np.exp(logp-logsumexp(logp))


def fit_market_pool(training: list[dict], *, fit_end_year: int) -> dict:
    """Fit on chronologically PRIOR years, never on target race year labels."""
    if len(training) < 80:
        raise ValueError("Insufficient verified historical real trio market snapshots")
    if any(item["race_year"] > fit_end_year or
           item["quote_year"] > fit_end_year or
           item["quote_year"] != item["race_year"] for item in training):
        raise ValueError("Quote years or racing labels leak into later validation")
    # Optimize ONE market weight and ONE tote-market power against historical
    # unordered combination outcomes. No 2024 or 2025 model selection.
    p_f3 = [np.asarray(r["f3_joint"],dtype=float) for r in training]
    market_odds = [np.asarray(r["real_trio_odds"],dtype=float) for r in training]
    labels = [int(r["actual_index"]) for r in training]
    for f,o,ix in zip(p_f3,market_odds,labels,strict=True):
        if len(f)!=len(o) or not 0 <= ix < len(o):
            raise ValueError("Complete 3-horse quotes not aligned with outcome")
        toto_market_distribution(o)
        geometric_joint_blend(f,toto_market_distribution(o),market_weight=.5)

    def mean_loss(alpha: float, temp: float) -> float:
        losses = []
        for f,odds,actual in zip(p_f3,market_odds,labels,strict=True):
            q = geometric_joint_blend(f,toto_market_distribution(odds,temperature=temp),
                                      market_weight=alpha)
            losses.append(-np.log(q[actual]))
        return float(np.mean(losses))

    # Small ex-ante grid avoids excessive flexibility and opaque optimization.
    candidates=[]
    for temp in (.8,1.0,1.2):
        result=minimize_scalar(lambda alpha, current_temp=temp: mean_loss(float(alpha),current_temp),
                               bounds=(0,1),method="bounded")
        if not result.success:
            raise ValueError("Historical real tote combination calibration failed")
        candidates.append((float(result.fun),float(result.x),temp))
    loss,weight,power=min(candidates,key=lambda z:z[0])
    return {
        "source":"verified REAL JRA 3-renpuku odds only",
        "training_races":len(training),"training_end_year":fit_end_year,
        "market_weight":weight,"market_power":power,
        "train_joint_logloss":loss,
        "may_apply_only_to_future_races":True,
        "profitability_not_established":True,
    }
