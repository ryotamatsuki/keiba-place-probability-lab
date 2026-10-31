"""Unordered joint probability arithmetic pool: fit exclusively prior-year OOF."""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.special import logsumexp

from .trio_bridge_v1 import invert_subset_marginals, project_exactly_three


def v1_joint(probability: np.ndarray, horses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All unordered triplets, preserving projected (not raw) horse marginals."""
    h = np.asarray(horses, dtype=int)
    if h.ndim != 1 or len(set(h)) != len(h) or len(h) < 4:
        raise ValueError("Incomplete full starter field")
    q = project_exactly_three(np.asarray(probability, dtype=float))
    w = invert_subset_marginals(q)
    idx = np.asarray(list(combinations(range(len(h)), 3)), dtype=int)
    logp = np.sum(w[idx], axis=1)
    joint = np.exp(logp - logsumexp(logp))
    return h[idx], joint


def checked_pool(probabilities: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Nonnegative arithmetic joint mixture, exact sum=1, same ticket ordering."""
    w = np.asarray(weights, dtype=float)
    if len(probabilities) != len(w) or w.ndim != 1 or np.any(~np.isfinite(w)):
        raise ValueError("Bad mixture family")
    if np.any(w < 0) or not np.isclose(w.sum(), 1, atol=1e-12):
        raise ValueError("Mixture must be convex")
    ref = np.asarray(probabilities[0], dtype=float)
    for p in probabilities:
        z = np.asarray(p, dtype=float)
        if z.shape != ref.shape or not np.isfinite(z).all() or np.any(z <= 0):
            raise ValueError("Incompatible distribution or omitted tickets")
        if not np.isclose(z.sum(), 1, atol=1e-10):
            raise ValueError("Unnormalized joint component")
    output = np.average(np.stack(probabilities), axis=0, weights=w)
    if not np.isclose(output.sum(), 1, atol=1e-10):
        raise ValueError("Arithmetic mixture no longer normalized")
    return output


def weight_grid_two(races: list[dict], left: str, right: str,
                    *, step: float = 0.05) -> dict:
    """Minimize mean true-trio NLL only on prior-year OOF; endpoints included."""
    if len(races) < 100 or not 0 < step <= 1:
        raise ValueError("Invalid chronological training cohort")
    candidate = []
    for w in np.round(np.arange(0, 1 + step/2, step), 10):
        nll = []
        for race in races:
            p = checked_pool([race[left], race[right]], [w, 1-w])
            nll.append(-np.log(p[race["actual_idx"]]))
        candidate.append((float(np.mean(nll)), float(w)))
    # Small improvements favor the left endpoint (first component, w=1).
    best = min(candidate, key=lambda x: (x[0], -x[1]))
    return {"left": left, "right": right, "weight_left": best[1],
            "weight_right": 1-best[1], "train_joint_loss": best[0],
            "grid": candidate, "year_labels_used": [2023]}
