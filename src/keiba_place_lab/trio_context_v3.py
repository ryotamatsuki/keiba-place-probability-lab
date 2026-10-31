"""Research-only contextual unordered-trio conditional residual, fixed 2023 specification.

All race-level context from entry, prior 4 race pace, course & FINAL win market.
No target-race pace or result used as features. Unordered 3-subset softmax.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .trio_residual_v2 import _prepare, score_probabilities

V2_IDS = (0, 1, 2)
# Main contextual specification is determined before examining new fold scores.
CONTEXT_NAMES = (
    "pair_strength_x_front_count",
    "pace_similarity_x_front_pressure",
    "pace_similarity_x_turf",
    "pace_similarity_x_long",
    "draw_similarity_x_short",
    "draw_similarity_x_front_pressure",
)
CANDIDATES = {
    "v1": (),
    "v2_matched": (0, 1, 2),
    "v3_context": tuple(range(9)),
}
RIDGE = 0.10


def contextual_features(
    *,
    horse_numbers: np.ndarray,
    logits: np.ndarray,
    projected_top3: np.ndarray,
    recent_early_pct: np.ndarray,
    draw_pct: np.ndarray,
    market_joint: np.ndarray,
    market_trios: np.ndarray,
    surface: str,
    distance_m: int,
) -> dict:
    """Nine per-subset columns: existing V2 three + six conditional interactions."""
    prepared = _prepare(
        horse_numbers, logits, projected_top3, recent_early_pct,
        draw_pct, market_joint, market_trios,
    )
    pace = np.asarray(recent_early_pct, dtype=float)
    good = np.isfinite(pace) & (pace >= 0) & (pace <= 1)
    valid = pace[good]
    # Running-position < 0.35 indicates previous starts near the front.
    # Missing prior pace is NOT silently classified as a confirmed front runner.
    front_count = float(np.sum(valid <= 0.35))
    pressure = float(np.clip((front_count - 2.5) / 3.0, -1.0, 1.0))
    turf = float(surface == "turf")
    long = float(distance_m >= 2000)
    short = float(distance_m <= 1600)
    if surface not in {"turf", "dirt"} or distance_m < 1000:
        raise ValueError("Invalid race context for conditional trio")
    base = prepared["features"][:, :3]
    strength, pace_similarity, draw_similarity = base.T
    new = np.column_stack((
        strength * pressure,
        pace_similarity * pressure,
        pace_similarity * (turf - 0.5),
        pace_similarity * (long - 0.5),
        draw_similarity * (short - 0.5),
        draw_similarity * pressure,
    ))
    new -= new.mean(axis=0)
    prepared["context_matrix"] = np.column_stack((base, new))
    prepared["context_names"] = (
        "pair_strength", "pace_similarity", "draw_similarity", *CONTEXT_NAMES
    )
    prepared["pace_pressure"] = pressure
    prepared["front_count_prior"] = front_count
    if not np.isfinite(prepared["context_matrix"]).all():
        raise ValueError("Invalid contextual trio features")
    return prepared


def joint_probability(item: dict, beta: np.ndarray, selected: tuple[int, ...]) -> np.ndarray:
    b = np.asarray(beta, dtype=float)
    if b.shape != (len(selected),) or not np.isfinite(b).all():
        raise ValueError("Wrong contextual coefficients")
    x = item["context_matrix"]
    if any(k < 0 or k >= x.shape[1] for k in selected):
        raise ValueError("Unavailable contextual feature index")
    logp = item["base_logp"].copy()
    if selected:
        logp += x[:, selected] @ b
    p = np.exp(logp - logsumexp(logp))
    if not np.isclose(np.sum(p), 1.0, atol=1e-10):
        raise ValueError("Contextual three-subset distribution is not normalized")
    return p


def fit_contextual(races: list[dict], selected: tuple[int, ...],
                   *, ridge: float = RIDGE) -> dict:
    if len(races) < 100 or not selected or ridge < 0:
        raise ValueError("Need complete prior-year candidate training with ridge")
    features = [r["context_matrix"][:, selected] for r in races]
    baselines = [r["base_logp"] for r in races]
    results = [r["actual_trio_idx"] for r in races]
    n = len(races)

    def objective(b: np.ndarray) -> tuple[float, np.ndarray]:
        loss, grad = 0.0, np.zeros(len(selected))
        for x, prior, actual in zip(features, baselines, results, strict=True):
            score = prior + x @ b
            z = logsumexp(score)
            p = np.exp(score - z)
            loss += z - score[actual]
            grad += p @ x - x[actual]
        return (loss / n + 0.5 * ridge * (b @ b),
                grad / n + ridge * b)

    m = minimize(objective, np.zeros(len(selected)), method="L-BFGS-B", jac=True,
                 bounds=[(-3.0, 3.0)] * len(selected),
                 options={"maxiter": 180, "ftol": 1e-11, "gtol": 1e-7})
    if not m.success or not np.isfinite(m.x).all():
        raise ValueError("Chronological contextual trio fit failed")
    return {
        "coefficients": m.x.tolist(),
        "feature_names": [races[0]["context_names"][k] for k in selected],
        "training_races": n,
        "training_joint_loss_regularized": float(m.fun),
        "ridge": ridge,
        "converged": bool(m.success),
    }
