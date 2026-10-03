"""Market-only probability baselines.

The primary Stage 2 baseline removes the overround from win odds and then
uses the Harville / Plackett-Luce sequential ranking assumption to derive
top-k probabilities.

Place-odds midpoint scores are diagnostic strength proxies only. They are
not labelled as implied probabilities because JRA place payouts depend on
multiple winning horses and are displayed as ranges.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np


def normalized_win_probabilities(win_odds: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert decimal win odds to normalized market win probabilities."""
    odds = np.asarray(win_odds, dtype=float)
    if odds.ndim != 1 or odds.size < 2:
        raise ValueError("win_odds must be a one-dimensional array with at least two runners")
    if np.any(~np.isfinite(odds)) or np.any(odds <= 1.0):
        raise ValueError("all decimal win odds must be finite and > 1")
    raw = 1.0 / odds
    total = float(raw.sum())
    return raw / total, total


def harville_top_k_probabilities(
    win_probabilities: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """Return marginal probability that each runner finishes in the top k."""
    p = np.asarray(win_probabilities, dtype=float)
    if p.ndim != 1 or p.size < k:
        raise ValueError("probabilities must be one-dimensional with at least k runners")
    if not 1 <= k < p.size:
        raise ValueError("k must satisfy 1 <= k < number of runners")
    if np.any(~np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("all probabilities must be finite and positive")
    if not np.isclose(float(p.sum()), 1.0, atol=1e-10):
        raise ValueError("win probabilities must sum to 1")

    result = np.zeros_like(p)

    for order in permutations(range(p.size), k):
        remaining_mass = 1.0
        probability = 1.0
        for runner in order:
            probability *= p[runner] / remaining_mass
            remaining_mass -= p[runner]
        for runner in order:
            result[runner] += probability

    return result


def place_odds_strength_proxy(
    place_odds_low: np.ndarray,
    place_odds_high: np.ndarray,
    target_sum: float = 3.0,
) -> np.ndarray:
    """Return a rank diagnostic from place-odds interval midpoints.

    This is not an implied probability.
    """
    low = np.asarray(place_odds_low, dtype=float)
    high = np.asarray(place_odds_high, dtype=float)
    if low.shape != high.shape or low.ndim != 1:
        raise ValueError("low/high arrays must be one-dimensional and have the same shape")
    if np.any(~np.isfinite(low)) or np.any(~np.isfinite(high)):
        raise ValueError("place odds must be finite")
    if np.any(low <= 0) or np.any(high < low):
        raise ValueError("place odds must satisfy 0 < low <= high")
    midpoint = (low + high) / 2.0
    raw = 1.0 / midpoint
    return target_sum * raw / raw.sum()
