"""Evaluation metrics for pre-race place-probability forecasts."""

from __future__ import annotations

import numpy as np


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    """Return the mean Brier score for binary place outcomes."""
    y = np.asarray(y_true, dtype=float)
    prob = np.asarray(p, dtype=float)
    if y.shape != prob.shape:
        raise ValueError("y_true and p must have the same shape")
    if np.any((prob < 0.0) | (prob > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")
    return float(np.mean((prob - y) ** 2))
