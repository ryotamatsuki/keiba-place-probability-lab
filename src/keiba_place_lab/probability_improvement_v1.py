"""Research-only probability improvements; frozen production policies are untouched."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .ensemble import probability_logit


def corrected_harville_top3(q: np.ndarray, gamma: float = 1.0,
                           delta: float = 1.0) -> np.ndarray:
    """Sequential top-three marginal with q, q**gamma and q**delta.

    Each successive conditional choice is normalized on all remaining runners.
    O(n^2) marginalization avoids enumerating every ordered triple.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or q.size < 4 or not np.isfinite(q).all():
        raise ValueError("Expected >=4 finite runner probabilities")
    if (q <= 0).any() or not np.isclose(q.sum(), 1.0, atol=1e-10):
        raise ValueError("q must be positive and sum to one")
    if not (gamma > 0 and delta > 0 and np.isfinite([gamma, delta]).all()):
        raise ValueError("Positive finite exponents required")
    a = q ** gamma
    b = q ** delta
    a /= a.sum()
    b /= b.sum()
    second = np.zeros_like(q)
    third = np.zeros_like(q)
    for i in range(len(q)):
        mask = np.arange(len(q)) != i
        denom2 = 1.0 - a[i]
        if denom2 <= 0:
            raise ValueError("Invalid second-place normalization")
        second[mask] += q[i] * a[mask] / denom2
        # For fixed winner i, integrate over all second-place runners j.
        # Third-place conditional denominator excludes i and j.
        t = np.zeros_like(q)
        denom3 = 1.0 - b[i] - b
        good = mask & (denom3 > 0)
        if not np.array_equal(good, mask):
            raise ValueError("Invalid third-place normalization")
        t[mask] = a[mask] / denom3[mask]
        third[mask] += q[i] * b[mask] * (t.sum() - t[mask]) / denom2
    result = q + second + third
    if not np.all(np.isfinite(result)) or np.any(result < -1e-10) or np.any(result > 1+1e-10):
        raise ValueError("Non-probabilistic corrected Harville result")
    if not np.isclose(result.sum(), 3.0, atol=1e-8):
        raise ValueError("Top-three sum is not 3")
    return np.clip(result, 0.0, 1.0)


def euclidean_top3_projection(p: np.ndarray) -> np.ndarray:
    """Project onto [0,1]^n intersect sum(x)=3, not proportional scaling."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError("Expected a finite full-field vector of at least 3")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Input probabilities must lie in [0,1]")
    if np.isclose(p.sum(), 3.0, atol=1e-12):
        return p.copy()
    lo, hi = float(p.min()-1), float(p.max())
    for _ in range(85):
        pivot = (lo + hi) / 2
        if np.clip(p-pivot, 0, 1).sum() > 3.0:
            lo = pivot
        else:
            hi = pivot
    result = np.clip(p-(lo+hi)/2, 0, 1)
    if not np.isclose(result.sum(), 3.0, atol=1e-10):
        raise ValueError("Projection failed")
    return result


@dataclass(frozen=True)
class OffsetFit:
    standardizer: StandardScaler
    coefficients: np.ndarray
    reference: str
    fitted_max_date: str


def fit_market_offset(market: np.ndarray, features: np.ndarray, labels: np.ndarray,
                      dates: np.ndarray, *, max_training_date: str,
                      penalty: float = 1.0) -> OffsetFit:
    """Penalized logit-offset model. Zero coefficients recover market exactly."""
    dates = pd.to_datetime(dates)
    if len(dates) == 0 or dates.max() >= pd.Timestamp(max_training_date):
        raise ValueError("Offset training includes evaluation date or later")
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    p = np.asarray(market, dtype=float)
    if X.ndim != 2 or len(X) != len(y) or len(y) != len(p):
        raise ValueError("Offset features/labels/market misaligned")
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid offset training data")
    scale = StandardScaler().fit(X)
    z = np.column_stack([np.ones(len(X)), scale.transform(X)])
    baseline = probability_logit(p)

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        logit = baseline + z @ w
        loss = np.logaddexp(0, logit) - y * logit
        grad = z.T @ (expit(logit) - y) / len(y)
        return float(loss.mean() + penalty * np.dot(w, w) / 2), grad + penalty*w

    fitted = minimize(objective, np.zeros(z.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 500, "ftol": 1e-12})
    if not fitted.success:
        raise ValueError(f"Offset optimizer failed: {fitted.message}")
    return OffsetFit(scale, np.asarray(fitted.x, dtype=float),
                     "offset_to_market", str(dates.max().date()))


def predict_market_offset(market: np.ndarray, features: np.ndarray, fitted: OffsetFit
                          ) -> np.ndarray:
    X = np.asarray(features, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("Invalid prediction features")
    z = np.column_stack([np.ones(len(X)), fitted.standardizer.transform(X)])
    return expit(probability_logit(np.asarray(market)) + z @ fitted.coefficients)


def fit_logit_stack(market: np.ndarray, nonmarket: np.ndarray,
                    labels: np.ndarray, dates: np.ndarray,
                    *, max_training_date: str) -> LogisticRegression:
    dates = pd.to_datetime(dates)
    if not len(dates) or dates.max() >= pd.Timestamp(max_training_date):
        raise ValueError("Meta training crosses outer fold")
    X = np.column_stack([probability_logit(market), probability_logit(nonmarket)])
    y = np.asarray(labels, dtype=int)
    if not np.isfinite(X).all() or len(set(y)) < 2:
        raise ValueError("Invalid OOF meta training")
    fitted = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs",
                                max_iter=2000, random_state=20261031)
    fitted.fit(X, y)
    return fitted


def predict_logit_stack(model: LogisticRegression, market: np.ndarray,
                        nonmarket: np.ndarray) -> np.ndarray:
    X = np.column_stack([probability_logit(market), probability_logit(nonmarket)])
    return model.predict_proba(X)[:, 1]


SNAPSHOT_COLUMNS = (
    "race_id", "horse_id", "bet_type", "odds", "place_odds_low",
    "place_odds_high", "provider_updated_at", "captured_at",
    "scheduled_off_at", "source", "snapshot_id",
)


def validate_snapshot_rows(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(SNAPSHOT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Snapshot missing columns: {sorted(missing)}")
    data = frame.copy()
    for column in ("provider_updated_at", "captured_at", "scheduled_off_at"):
        # utc=True explicitly handles offsets; forbid naive timestamps first.
        parsed = pd.to_datetime(data[column], errors="raise")
        if any(pd.Timestamp(t).tzinfo is None for t in parsed):
            raise ValueError(f"{column} must include UTC offset")
        data[column] = pd.to_datetime(data[column], utc=True)
    if data[["race_id", "horse_id", "source", "snapshot_id"]].isna().any().any():
        raise ValueError("Snapshot identity/source missing")
    if data.duplicated(["race_id", "horse_id", "bet_type", "snapshot_id"]).any():
        raise ValueError("Duplicate snapshot odds row")
    if not data["bet_type"].isin(["win", "place"]).all():
        raise ValueError("Unknown bet type")
    if (data["provider_updated_at"] > data["captured_at"]).any():
        raise ValueError("Provider update is later than capture")
    for row in data.itertuples():
        if row.bet_type == "win":
            if not np.isfinite(float(row.odds)) or float(row.odds) <= 1:
                raise ValueError("Invalid win odds")
        elif (not np.isfinite(float(row.place_odds_low))
              or not np.isfinite(float(row.place_odds_high))
              or float(row.place_odds_low) <= 0
              or float(row.place_odds_high) < float(row.place_odds_low)):
            raise ValueError("Invalid place-odds range")
    return data


def select_snapshot_at(frame: pd.DataFrame, *, minutes_before: int,
                       expected_horse_ids: set[str]) -> pd.DataFrame:
    """Choose one complete timestamped capture no later than the target time."""
    if minutes_before not in (60, 30, 10):
        raise ValueError("Frozen checkpoints are 60/30/10 minutes")
    data = validate_snapshot_rows(frame)
    if data["race_id"].nunique() != 1 or data["scheduled_off_at"].nunique() != 1:
        raise ValueError("One race and scheduled off time required")
    off = data["scheduled_off_at"].iloc[0]
    target = off - pd.Timedelta(minutes=minutes_before)
    max_age = pd.Timedelta(minutes=5 if minutes_before == 10 else 10)
    valid = data.loc[
        (data["captured_at"] <= target)
        & (data["provider_updated_at"] <= target)
        & ((target - data["provider_updated_at"]) <= max_age)
    ]
    # A complete snapshot must be one ID / publication, not stitched horse rows.
    for _, block in valid.sort_values("captured_at", ascending=False).groupby(
        "snapshot_id", sort=False
    ):
        ids = set(block.loc[block.bet_type.eq("win"), "horse_id"].astype(str))
        if ids == set(expected_horse_ids) and len(block.loc[block.bet_type.eq("win")]) == len(ids):
            return block.sort_values(["horse_id", "bet_type"]).reset_index(drop=True)
    raise ValueError("No complete fresh snapshot at this prediction checkpoint")
