"""Stage 7 post-race evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


@dataclass(frozen=True)
class RaceEvaluation:
    brier: float
    log_loss: float
    actual_top3_probability_mass: float
    actual_top3_forecast_ranks: tuple[int, int, int]


def validate_outcome(outcome: pd.DataFrame) -> None:
    required = {
        "finish_rank",
        "horse_no",
        "horse_name",
        "top3_label",
        "official_source_url",
    }
    missing = sorted(required.difference(outcome.columns))
    if missing:
        raise ValueError(f"Missing outcome columns: {missing}")
    if len(outcome) != 18 or outcome["horse_no"].nunique() != 18:
        raise ValueError("Outcome must contain exactly 18 unique runners")
    if set(outcome["finish_rank"]) != set(range(1, 19)):
        raise ValueError("finish_rank must be exactly 1..18")
    expected = outcome["finish_rank"].le(3).astype(int)
    if not expected.equals(outcome["top3_label"].astype(int)):
        raise ValueError("top3_label is inconsistent with finish_rank")
    if int(outcome["top3_label"].sum()) != 3:
        raise ValueError("Outcome must contain exactly three top3 labels")


def evaluate_race(
    frame: pd.DataFrame,
    probability_col: str,
    *,
    target_col: str = "top3_label",
) -> RaceEvaluation:
    if probability_col not in frame or target_col not in frame:
        raise ValueError("Evaluation frame missing probability or target column")
    probability = pd.to_numeric(frame[probability_col], errors="coerce").to_numpy()
    target = frame[target_col].astype(int).to_numpy()
    if len(probability) != 18 or np.any(~np.isfinite(probability)):
        raise ValueError("Expected 18 finite probabilities")
    if np.any((probability <= 0) | (probability >= 1)):
        raise ValueError("Probabilities must be strictly inside (0, 1)")

    ranks = pd.Series(probability).rank(method="first", ascending=False).astype(int)
    top3_mask = target == 1
    return RaceEvaluation(
        brier=float(brier_score_loss(target, probability)),
        log_loss=float(log_loss(target, probability, labels=[0, 1])),
        actual_top3_probability_mass=float(probability[top3_mask].sum()),
        actual_top3_forecast_ranks=tuple(
            int(value) for value in ranks.loc[top3_mask].sort_values().to_list()
        ),
    )
