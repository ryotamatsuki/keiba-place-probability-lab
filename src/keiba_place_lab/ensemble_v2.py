"""Stage 5 v2 helpers for market-only incumbent and paired one-SE blend selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ensemble import convex_blend
from .model_selection import candidate_score, paired_delta


@dataclass(frozen=True)
class BlendDecision:
    winner: str
    point_brier_best: str
    incumbent_retained: bool
    selected_nonmarket_weight: float
    reason: str


def blend_registry(weights: np.ndarray | None = None) -> pd.DataFrame:
    if weights is None:
        weights = np.linspace(0.0, 1.0, 21)
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a non-empty 1D array")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("weights must lie in [0, 1]")
    values = np.unique(np.round(values, 10))
    if not np.any(np.isclose(values, 0.0)):
        raise ValueError("weight grid must include market-only endpoint w=0")
    return pd.DataFrame(
        {
            "candidate": [f"blend_w{value:.2f}" for value in values],
            "nonmarket_weight": values,
            "market_weight": 1.0 - values,
        }
    )


def select_blend_one_se(
    frame: pd.DataFrame,
    market_probability: np.ndarray,
    nonmarket_probability: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> tuple[BlendDecision, pd.DataFrame, dict[str, np.ndarray]]:
    """Select blend with market-only retention using race-macro Brier."""
    registry = blend_registry(weights)
    predictions: dict[str, np.ndarray] = {}
    score_rows: list[dict[str, object]] = []

    for row in registry.itertuples(index=False):
        probability = convex_blend(
            market_probability,
            nonmarket_probability,
            float(row.nonmarket_weight),
        )
        predictions[str(row.candidate)] = probability
        score = candidate_score(frame, probability, name=str(row.candidate))
        score_rows.append(
            {
                "candidate": str(row.candidate),
                "nonmarket_weight": float(row.nonmarket_weight),
                **score.__dict__,
            }
        )

    scores = pd.DataFrame(score_rows).sort_values(
        ["race_macro_brier", "race_macro_log_loss", "nonmarket_weight", "candidate"],
        kind="stable",
    )
    best_name = str(scores.iloc[0]["candidate"])
    incumbent_name = "blend_w0.00"

    rows = []
    for row in scores.itertuples(index=False):
        name = str(row.candidate)
        delta = paired_delta(
            frame,
            predictions[incumbent_name],
            predictions[name],
            candidate_name=incumbent_name,
            reference_name=name,
        )
        best_delta = paired_delta(
            frame,
            predictions[name],
            predictions[best_name],
            candidate_name=name,
            reference_name=best_name,
        )
        rows.append(
            {
                **row._asdict(),
                "mean_incumbent_minus_candidate_brier": delta.mean_brier_delta,
                "se_incumbent_minus_candidate_brier": delta.se_brier_delta,
                "mean_brier_delta_vs_best": best_delta.mean_brier_delta,
                "se_brier_delta_vs_best": best_delta.se_brier_delta,
                "within_one_se_of_best": (
                    best_delta.mean_brier_delta
                    <= best_delta.se_brier_delta + 1e-15
                ),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["race_macro_brier", "race_macro_log_loss", "nonmarket_weight", "candidate"],
        kind="stable",
    )

    incumbent_vs_best = paired_delta(
        frame,
        predictions[incumbent_name],
        predictions[best_name],
        candidate_name=incumbent_name,
        reference_name=best_name,
    )
    incumbent_retained = (
        incumbent_vs_best.mean_brier_delta
        <= incumbent_vs_best.se_brier_delta + 1e-15
    )
    if incumbent_retained:
        winner = incumbent_name
        reason = (
            "market-only retained: improvement to point-Brier-best does not exceed "
            "one paired race-date-clustered SE"
        )
    else:
        winner = best_name
        reason = (
            "market-only displaced: point-Brier-best improvement exceeds one paired "
            "race-date-clustered SE"
        )

    selected_weight = float(
        registry.loc[registry["candidate"].eq(winner), "nonmarket_weight"].iloc[0]
    )
    return (
        BlendDecision(
            winner=winner,
            point_brier_best=best_name,
            incumbent_retained=incumbent_retained,
            selected_nonmarket_weight=selected_weight,
            reason=reason,
        ),
        table.reset_index(drop=True),
        predictions,
    )
