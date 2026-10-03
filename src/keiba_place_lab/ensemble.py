"""Transparent Stage 5 calibration and market/non-market ensemble helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


@dataclass(frozen=True)
class ProbabilityMetrics:
    brier: float
    log_loss: float
    rows: int
    races: int


@dataclass(frozen=True)
class LogitCalibrator:
    intercept: float
    slope: float


def _clip_probability(probability: np.ndarray) -> np.ndarray:
    p = np.asarray(probability, dtype=float)
    if p.ndim != 1:
        raise ValueError("probability must be one-dimensional")
    if p.size == 0:
        raise ValueError("probability must be non-empty")
    if np.any(~np.isfinite(p)):
        raise ValueError("probability contains non-finite values")
    if np.any((p <= 0.0) | (p >= 1.0)):
        p = np.clip(p, 1e-9, 1 - 1e-9)
    return p


def probability_logit(probability: np.ndarray) -> np.ndarray:
    """Return finite logits for marginal probabilities."""
    p = _clip_probability(probability)
    return np.log(p / (1.0 - p))


def fit_logit_calibrator(
    probability: np.ndarray,
    target: np.ndarray,
) -> LogitCalibrator:
    """Fit transparent intercept+slope recalibration on logit(probability)."""
    p = _clip_probability(probability)
    y = np.asarray(target, dtype=int)
    if y.ndim != 1 or len(y) != len(p):
        raise ValueError("target must be a 1D vector aligned to probability")
    if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) < 2:
        raise ValueError("target must contain both binary classes")
    x = probability_logit(p).reshape(-1, 1)
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=2000,
        random_state=20261003,
    )
    model.fit(x, y)
    return LogitCalibrator(
        intercept=float(model.intercept_[0]),
        slope=float(model.coef_[0, 0]),
    )


def apply_logit_calibrator(
    probability: np.ndarray,
    calibrator: LogitCalibrator,
) -> np.ndarray:
    """Apply intercept+slope logit calibration."""
    z = calibrator.intercept + calibrator.slope * probability_logit(probability)
    return 1.0 / (1.0 + np.exp(-z))


def convex_blend(
    market_probability: np.ndarray,
    nonmarket_probability: np.ndarray,
    nonmarket_weight: float,
) -> np.ndarray:
    """Blend calibrated market and non-market marginal probabilities."""
    if not 0.0 <= nonmarket_weight <= 1.0:
        raise ValueError("nonmarket_weight must be in [0, 1]")
    market = _clip_probability(market_probability)
    nonmarket = _clip_probability(nonmarket_probability)
    if market.shape != nonmarket.shape:
        raise ValueError("market and nonmarket probability shape mismatch")
    return (1.0 - nonmarket_weight) * market + nonmarket_weight * nonmarket


def evaluate_probability(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    target_col: str = "top3_label",
    race_col: str = "race_id",
) -> ProbabilityMetrics:
    """Evaluate marginal probabilities on paired eligible rows."""
    if target_col not in frame or race_col not in frame:
        raise ValueError("evaluation frame missing target/race columns")
    p = _clip_probability(probability)
    if len(frame) != len(p):
        raise ValueError("frame and probability length mismatch")
    y = frame[target_col].astype(int).to_numpy()
    return ProbabilityMetrics(
        brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        rows=int(len(frame)),
        races=int(frame[race_col].nunique()),
    )


def select_blend_weight(
    frame: pd.DataFrame,
    market_probability: np.ndarray,
    nonmarket_probability: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """Select convex weight by Brier, then log loss, then smaller weight."""
    if weights is None:
        weights = np.linspace(0.0, 1.0, 21)
    rows: list[dict] = []
    for weight in np.asarray(weights, dtype=float):
        blended = convex_blend(market_probability, nonmarket_probability, float(weight))
        metrics = evaluate_probability(frame, blended)
        rows.append(
            {
                "nonmarket_weight": float(weight),
                "market_weight": float(1.0 - weight),
                "brier": metrics.brier,
                "log_loss": metrics.log_loss,
                "rows": metrics.rows,
                "races": metrics.races,
            }
        )
    grid = pd.DataFrame(rows).sort_values(
        ["brier", "log_loss", "nonmarket_weight"],
        kind="mergesort",
    ).reset_index(drop=True)
    return float(grid.iloc[0]["nonmarket_weight"]), grid


def nearby_weights(selected_weight: float, radius: float = 0.10) -> list[float]:
    """Return a deterministic local blend-weight sensitivity set."""
    if not 0.0 <= selected_weight <= 1.0:
        raise ValueError("selected_weight must be in [0, 1]")
    values = {
        0.0,
        1.0,
        selected_weight,
        max(0.0, selected_weight - radius),
        min(1.0, selected_weight + radius),
    }
    return sorted(float(v) for v in values)
