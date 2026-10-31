"""Research-only exactly-three unordered trio distribution from F3 marginal P(top3).

The inverse conditional-Bernoulli transform preserves racewise projected
marginals BEFORE optional historical trio-temperature calibration.
Never alters, refits, or serializes the frozen F3 bundle.
"""
from __future__ import annotations

from itertools import combinations, permutations

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit, logsumexp

from .three_subset_v1 import (
    K,
    _normalizer_and_prefix,
    top3_subset_marginals,
)

EPS = 1e-6


def validate_probability_field(p: np.ndarray) -> np.ndarray:
    a = np.asarray(p, dtype=float)
    if a.ndim != 1 or len(a) < K + 1:
        raise ValueError("Trio requires at least four actual starters")
    if not np.all(np.isfinite(a)) or np.any((a <= 0) | (a >= 1)):
        raise ValueError("Full-field marginal probabilities must be strictly between zero and one")
    return a


def project_exactly_three(p: np.ndarray) -> np.ndarray:
    """Minimal common logit shift, so sum(marginal P(top3)) = exactly 3."""
    a = np.clip(validate_probability_field(p), EPS, 1 - EPS)
    logits = logit(a)
    if abs(a.sum() - K) < 1e-11:
        return a
    shift = brentq(lambda z: float(expit(logits + z).sum() - K), -100.0, 100.0)
    q = expit(logits + shift)
    if abs(q.sum() - K) > 1e-9:
        raise ValueError("Unable to project all starters onto exactly three slots")
    return q


def invert_subset_marginals(q: np.ndarray) -> np.ndarray:
    """Maximum-entropy exactly-three subset log weights matching given marginals.

    Uses convex dual logZ(theta)-sum(q*theta). One gauge degree is fixed.
    For n starters, computes all marginals via elementary symmetric polynomials
    in O(3*n) per optimization iteration.
    """
    a = validate_probability_field(q)
    if abs(a.sum() - K) > 1e-8:
        raise ValueError("Input subset marginals must sum to three")
    x0 = logit(np.clip(a, EPS, 1 - EPS))
    x0 -= x0[-1]

    def loss_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        w = np.r_[x, 0.0]
        logz, _, _ = _normalizer_and_prefix(w)
        marginal = top3_subset_marginals(w)
        return float(logz - np.dot(a, w)), marginal[:-1] - a[:-1]

    result = minimize(
        loss_and_grad, x0[:-1], method="L-BFGS-B", jac=True,
        options={"maxiter": 550, "ftol": 1e-13, "gtol": 1e-10},
    )
    w = np.r_[result.x, 0.0]
    error = float(np.max(np.abs(top3_subset_marginals(w) - a)))
    if not np.isfinite(w).all() or error > 3e-5:
        raise ValueError(f"Inverse trio marginals did not converge: {error:.5g}")
    return w - w.mean()


def trio_distribution(log_weights: np.ndarray, horse_numbers: np.ndarray,
                      *, temperature: float = 1.0) -> pd.DataFrame:
    """Exactly n choose 3 combinations, sorted highest probability first."""
    w = np.asarray(log_weights, dtype=float)
    horses = np.asarray(horse_numbers)
    if (len(w) != len(horses) or len(set(map(str, horses))) != len(horses)
            or not np.all(np.isfinite(w)) or not np.isfinite(temperature)
            or temperature <= 0):
        raise ValueError("Invalid trio contenders, logits, or temperature")
    if len(w) < 4:
        raise ValueError("At least four horses required")
    w = w * float(temperature)
    indices = np.asarray(list(combinations(range(len(w)), K)), dtype=int)
    logp = w[indices].sum(axis=1)
    p = np.exp(logp - logsumexp(logp))
    out = pd.DataFrame({
        "horse_1": horses[indices[:, 0]],
        "horse_2": horses[indices[:, 1]],
        "horse_3": horses[indices[:, 2]],
        "probability": p,
    })
    if not np.isclose(p.sum(), 1.0, atol=1e-12):
        raise ValueError("Unordered joint distribution must sum to exactly one")
    return out.sort_values(
        ["probability", "horse_1", "horse_2", "horse_3"],
        ascending=[False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)


def plackett_luce_trio(win_probabilities: np.ndarray, horse_numbers: np.ndarray) -> pd.DataFrame:
    """Win-market sequential ordering summed over the six trio permutations."""
    p = np.asarray(win_probabilities, dtype=float)
    horses = np.asarray(horse_numbers)
    if (p.ndim != 1 or len(p) != len(horses) or len(p) < K + 1
            or not np.all(np.isfinite(p)) or np.any(p <= 0)
            or not np.isclose(p.sum(), 1.0, atol=1e-8)):
        raise ValueError("PL market input must be a full field of normalized positive shares")
    records = []
    for a, b, c in combinations(range(len(p)), K):
        prob = 0.0
        for i, j, k in permutations((a, b, c), K):
            prob += (p[i] * p[j] / (1 - p[i])
                     * p[k] / (1 - p[i] - p[j]))
        records.append((horses[a], horses[b], horses[c], prob))
    df = pd.DataFrame(records, columns=["horse_1", "horse_2", "horse_3", "probability"])
    if not np.isclose(df.probability.sum(), 1.0, atol=1e-9):
        raise ValueError("PL trio combinations do not sum to one")
    return df.sort_values("probability", ascending=False, kind="stable").reset_index(drop=True)


def historical_trio_temperature(
    weights: list[np.ndarray], outcomes: list[tuple[int, int, int]],
) -> dict:
    """One scalar tuned on PREVIOUS YEAR's held-out trio outcomes only."""
    if len(weights) != len(outcomes) or len(weights) < 40:
        raise ValueError("Too few prior-year valid races to calibrate trio temperature")
    prepared = []
    for w, trio in zip(weights, outcomes, strict=True):
        if len(trio) != K or len(set(trio)) != K:
            raise ValueError("Exactly three finishing horses are required")
        prepared.append((np.asarray(w, dtype=float), np.asarray(trio, dtype=int)))

    def mean_nll(log_temp: float) -> float:
        t = float(np.exp(log_temp))
        losses = []
        for w, selected in prepared:
            logz, _, _ = _normalizer_and_prefix(w * t)
            losses.append(float(logz - t * w[selected].sum()))
        return float(np.mean(losses))

    from scipy.optimize import minimize_scalar

    fitted = minimize_scalar(mean_nll, bounds=(np.log(.25), np.log(4)),
                             method="bounded", options={"xatol": 1e-5})
    if not fitted.success:
        raise ValueError("Historical trio temperature calibration failed")
    return {
        "temperature": float(np.exp(fitted.x)),
        "train_races": len(prepared),
        "calibration_nll_pre": mean_nll(0.0),
        "calibration_nll_post": mean_nll(float(fitted.x)),
        "temperature_bounds": [0.25, 4.0],
    }
