"""Trio v2: regularized unordered-top-three conditional-logit residual model.

Uses ONLY race-preknown F3/marginal, original WIN odds, horse prior pace
and entry draw. No current-race results become features. A single 2023-only
fit can be evaluated in untouched 2024 out-of-time rows.

P(S) = softmax(log P_F3_maxent(S) + X(S) beta), |S|=3.
This preserves 3-horse joint normalization, but residual adjustment will
change individual top-three marginals; never describe those as F3-preserved.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

FEATURE_NAMES = ("pair_strength", "pace_similarity", "draw_similarity", "market_pl_residual")
CANDIDATE_SETS = {
    "f3_raw": (),
    "pair_strength": (0,),
    "race_interactions": (0, 1, 2),
    "market_hybrid": (0, 1, 2, 3),
}
RIDGE = 0.05
MAX_LOGIT = 45.0


def _prepare(horses: np.ndarray, log_weights: np.ndarray,
             projected_top3: np.ndarray, pace_prior: np.ndarray,
             draw: np.ndarray, market_joint: np.ndarray,
             market_trios: np.ndarray) -> dict:
    """Return trio outcome enumeration and preknown per-ticket features."""
    hs = np.asarray(horses)
    w = np.asarray(log_weights, float)
    q = np.asarray(projected_top3, float)
    pace = np.asarray(pace_prior, float)
    dr = np.asarray(draw, float)
    if (len(hs) < 4 or len(hs) != len(set(hs.tolist())) or
            not all(len(a) == len(hs) for a in (w, q, pace, dr)) or
            not np.isfinite(w).all() or not np.isfinite(q).all() or
            np.any(q <= 0) or np.any(q >= 1) or not np.isfinite(dr).all() or
            np.any(dr < 0) or np.any(dr > 1) or
            not np.isclose(q.sum(), 3, atol=1e-7)):
        raise ValueError("All actual runners and prior-only feature vectors required")
    n = len(hs)
    triplets = np.array(list(combinations(range(n), 3)), dtype=np.int16)
    rows = np.array([tuple(sorted(row)) for row in hs[triplets]])
    market = {tuple(sorted(v)): float(z)
              for v, z in zip(market_trios, market_joint, strict=True)}
    if len(market) != len(rows) or len(market) != len(set(map(tuple, rows))):
        raise ValueError("Market/F3 trio sets disagree")
    mp = np.array([market[tuple(v)] for v in rows], dtype=float)
    if np.any(mp <= 0) or not np.isclose(mp.sum(), 1.0, atol=1e-9):
        raise ValueError("Invalid corresponding full-field market PL joint")
    base_score = w[triplets].sum(axis=1)
    base_logp = base_score - logsumexp(base_score)
    # Favorite-vs-longshot pairs, with scaling determined only by this race.
    z = np.log(q / (1 - q))
    z = np.clip((z - z.mean()) / max(float(z.std()), 1.0), -3.0, 3.0)
    i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    pair_strength = (z[i] * z[j] + z[i] * z[k] + z[j] * z[k]) / 3.0
    # A priori running-position similarity. No THIS-race pace/finishing data.
    observed = np.isfinite(pace) & (pace >= 0) & (pace <= 1)
    pace = pace.copy()
    pace[~observed] = 0.5
    def similarity(a: np.ndarray) -> np.ndarray:
        return (np.exp(-np.abs(a[i] - a[j]) / .16) +
                np.exp(-np.abs(a[i] - a[k]) / .16) +
                np.exp(-np.abs(a[j] - a[k]) / .16)) / 3.0
    pace_similarity = similarity(pace)
    # Draw similarity has potential path-based competition effects.
    draw_similarity = similarity(dr)
    # A market-only alternative, not the actual trio tote implied odds.
    market_residual = np.log(mp) - base_logp
    x = np.column_stack((pair_strength, pace_similarity,
                         draw_similarity, market_residual))
    # Per-race centering has no effect on softmax but stabilizes optimization.
    x -= x.mean(axis=0)
    if not np.isfinite(x).all():
        raise ValueError("Invalid interaction features")
    return {
        "trios": rows,
        "base_logp": base_logp,
        "features": x,
        "pace_missing_count": int((~observed).sum()),
    }


def residual_distribution(item: dict, beta: np.ndarray,
                          included: tuple[int, ...]) -> np.ndarray:
    b = np.asarray(beta, float)
    if b.shape != (len(included),) or not np.isfinite(b).all():
        raise ValueError("Incompatible residual model coefficient")
    if len(set(included)) != len(included) or any(i not in range(4) for i in included):
        raise ValueError("Invalid feature subset")
    x = item["features"]
    score = item["base_logp"] + x[:, included] @ b if included else item["base_logp"]
    logp = score - logsumexp(score)
    probabilities = np.exp(logp)
    if (not np.isfinite(probabilities).all() or np.any(probabilities <= 0) or
            not np.isclose(probabilities.sum(), 1, atol=1e-10)):
        raise ValueError("Unnormalized or invalid trio residual distribution")
    return probabilities


def fit_residual(train: list[dict], included: tuple[int, ...],
                 *, ridge: float = RIDGE) -> dict:
    if len(train) < 100 or not included or not np.isfinite(ridge) or ridge < 0:
        raise ValueError("Need 2023 full-race training data and fixed regularization")
    xs = [r["features"][:, included] for r in train]
    baselines = [r["base_logp"] for r in train]
    true_index = [r["actual_trio_idx"] for r in train]
    if any(t >= len(x) or t < 0 for t, x in zip(true_index, xs, strict=True)):
        raise ValueError("Invalid training trio labels")

    def objective(beta):
        value = 0.0
        grad = np.zeros(len(included), dtype=float)
        for x, b, truth in zip(xs, baselines, true_index, strict=True):
            score = b + x @ beta
            z = logsumexp(score)
            pp = np.exp(score - z)
            value += z - score[truth]
            grad += pp @ x - x[truth]
        count = len(train)
        value = value / count + .5 * ridge * float(beta @ beta)
        grad = grad / count + ridge * beta
        return value, grad

    result = minimize(
        objective, np.zeros(len(included), float), method="L-BFGS-B", jac=True,
        bounds=[(-4.0, 4.0)] * len(included),
        options={"maxiter": 175, "ftol": 1e-11, "gtol": 1e-7},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("Joint conditional logistic training failed")
    return {
        "beta": result.x.tolist(),
        "train_races": len(train), "ridge": ridge,
        "training_joint_logloss_regularized": float(result.fun),
        "features": [FEATURE_NAMES[i] for i in included],
        "optimizer_success": bool(result.success),
        "optimizer_iterations": int(result.nit),
    }


def score_probabilities(p: np.ndarray, actual_index: int, top_k: int = 10) -> dict:
    if not 0 <= actual_index < len(p):
        raise ValueError("Missing winning trio")
    q = float(p[actual_index])
    ordered = np.argsort(-p, kind="stable")
    return {
        "joint_logloss": float(-np.log(q)),
        "joint_brier": float(1.0 - 2 * q + np.dot(p, p)),
        "top1_hit": int(ordered[0] == actual_index),
        "top5_hit": int(actual_index in ordered[:5]),
        "top10_hit": int(actual_index in ordered[:top_k]),
        "top10_mass": float(p[ordered[:top_k]].sum()),
        "top1_mass": float(p[ordered[0]]),
    }
