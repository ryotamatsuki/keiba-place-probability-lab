"""Deterministic model-selection helpers for marginal P(top3) forecasts.

The v2 selection contract uses race-macro proper scores and a paired,
date-clustered one-standard-error incumbent gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateScore:
    name: str
    race_macro_brier: float
    race_macro_log_loss: float
    runner_micro_brier: float
    runner_micro_log_loss: float
    races: int
    rows: int


@dataclass(frozen=True)
class PairedDelta:
    candidate: str
    reference: str
    mean_brier_delta: float
    se_brier_delta: float
    mean_log_loss_delta: float
    se_log_loss_delta: float


@dataclass(frozen=True)
class SelectionDecision:
    winner: str
    point_brier_best: str
    incumbent: str | None
    incumbent_retained: bool
    reason: str


def _validate_probability(probability: np.ndarray, rows: int) -> np.ndarray:
    p = np.asarray(probability, dtype=float)
    if p.ndim != 1 or len(p) != rows:
        raise ValueError("probability must be a 1D array aligned to frame rows")
    if not np.isfinite(p).all():
        raise ValueError("probability contains non-finite values")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probability must lie in [0, 1]")
    return p


def _row_losses(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    brier = (p - y) ** 2
    logloss = -(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))
    return brier, logloss


def race_loss_table(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    race_col: str = "race_id",
    block_col: str = "race_date",
    target_col: str = "top3_label",
) -> pd.DataFrame:
    """Return one row per race with race-macro Brier and log loss components."""
    required = {race_col, block_col, target_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing selection columns: {sorted(missing)}")

    p = _validate_probability(probability, len(frame))
    y = pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=float)
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("target must be binary 0/1")

    work = frame[[race_col, block_col]].copy()
    brier, logloss = _row_losses(y, p)
    work["_brier"] = brier
    work["_log_loss"] = logloss

    block_counts = work.groupby(race_col)[block_col].nunique(dropna=False)
    if block_counts.gt(1).any():
        raise ValueError("block_col must be constant within each race")

    out = (
        work.groupby(race_col, sort=False, dropna=False)
        .agg(
            block=(block_col, "first"),
            rows=("_brier", "size"),
            brier=("_brier", "mean"),
            log_loss=("_log_loss", "mean"),
        )
        .reset_index()
    )
    if out["block"].isna().any():
        raise ValueError("block_col contains missing values")
    return out


def candidate_score(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    name: str,
    race_col: str = "race_id",
    block_col: str = "race_date",
    target_col: str = "top3_label",
) -> CandidateScore:
    """Compute race-macro primary scores plus runner-micro continuity metrics."""
    p = _validate_probability(probability, len(frame))
    y = pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=float)
    race = race_loss_table(
        frame,
        p,
        race_col=race_col,
        block_col=block_col,
        target_col=target_col,
    )
    row_brier, row_logloss = _row_losses(y, p)
    return CandidateScore(
        name=name,
        race_macro_brier=float(race["brier"].mean()),
        race_macro_log_loss=float(race["log_loss"].mean()),
        runner_micro_brier=float(row_brier.mean()),
        runner_micro_log_loss=float(row_logloss.mean()),
        races=len(race),
        rows=len(frame),
    )


def _clustered_mean_and_se(values: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    """Return the mean and CR1-style cluster-robust SE for an intercept-only mean."""
    x = np.asarray(values, dtype=float)
    g = np.asarray(clusters)
    if x.ndim != 1 or g.ndim != 1 or len(x) != len(g):
        raise ValueError("values and clusters must be aligned 1D arrays")
    if len(x) == 0:
        raise ValueError("cannot summarize an empty difference vector")
    mean = float(x.mean())
    unique = pd.unique(g)
    if len(unique) < 2:
        return mean, float("inf")

    centered = x - mean
    cluster_scores = np.array(
        [centered[g == label].sum() for label in unique],
        dtype=float,
    )
    variance = (len(unique) / (len(unique) - 1.0)) * float(
        np.sum(cluster_scores**2)
    ) / (len(x) ** 2)
    return mean, sqrt(max(0.0, variance))


def paired_delta(
    frame: pd.DataFrame,
    candidate_probability: np.ndarray,
    reference_probability: np.ndarray,
    *,
    candidate_name: str,
    reference_name: str,
    race_col: str = "race_id",
    block_col: str = "race_date",
    target_col: str = "top3_label",
) -> PairedDelta:
    """Compare two candidates on identical races using date-clustered paired losses."""
    cand = race_loss_table(
        frame,
        candidate_probability,
        race_col=race_col,
        block_col=block_col,
        target_col=target_col,
    ).rename(columns={"brier": "cand_brier", "log_loss": "cand_log"})
    ref = race_loss_table(
        frame,
        reference_probability,
        race_col=race_col,
        block_col=block_col,
        target_col=target_col,
    ).rename(columns={"brier": "ref_brier", "log_loss": "ref_log"})

    merged = cand.merge(
        ref[[race_col, "block", "ref_brier", "ref_log"]],
        on=[race_col, "block"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(cand) or len(merged) != len(ref):
        raise ValueError("candidate and reference must cover identical races")

    db = (merged["cand_brier"] - merged["ref_brier"]).to_numpy(dtype=float)
    dl = (merged["cand_log"] - merged["ref_log"]).to_numpy(dtype=float)
    blocks = merged["block"].to_numpy()
    mean_b, se_b = _clustered_mean_and_se(db, blocks)
    mean_l, se_l = _clustered_mean_and_se(dl, blocks)
    return PairedDelta(
        candidate=candidate_name,
        reference=reference_name,
        mean_brier_delta=mean_b,
        se_brier_delta=se_b,
        mean_log_loss_delta=mean_l,
        se_log_loss_delta=se_l,
    )


def select_with_incumbent_one_se(
    frame: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    *,
    incumbent: str | None,
    one_se_multiplier: float = 1.0,
    race_col: str = "race_id",
    block_col: str = "race_date",
    target_col: str = "top3_label",
) -> tuple[SelectionDecision, pd.DataFrame]:
    """Select by race-macro Brier with a paired one-SE incumbent-retention gate.

    The point-Brier-best candidate is the reference. The incumbent is retained only
    when it is within one paired clustered SE of that candidate on both Brier and
    log loss. If there is no incumbent, or the incumbent fails either gate, the
    point-Brier-best candidate wins.
    """
    if not predictions:
        raise ValueError("predictions must contain at least one candidate")
    if one_se_multiplier < 0:
        raise ValueError("one_se_multiplier must be non-negative")
    if incumbent is not None and incumbent not in predictions:
        raise ValueError("incumbent must be present in predictions")

    scores = [
        candidate_score(
            frame,
            probability,
            name=name,
            race_col=race_col,
            block_col=block_col,
            target_col=target_col,
        )
        for name, probability in predictions.items()
    ]
    score_df = pd.DataFrame([score.__dict__ for score in scores]).sort_values(
        ["race_macro_brier", "race_macro_log_loss", "name"],
        kind="stable",
    )
    best = str(score_df.iloc[0]["name"])

    diagnostics: list[dict[str, object]] = []
    for name, probability in predictions.items():
        delta = paired_delta(
            frame,
            probability,
            predictions[best],
            candidate_name=name,
            reference_name=best,
            race_col=race_col,
            block_col=block_col,
            target_col=target_col,
        )
        diagnostics.append(
            {
                "name": name,
                "point_brier_best": name == best,
                "mean_brier_delta_vs_best": delta.mean_brier_delta,
                "se_brier_delta_vs_best": delta.se_brier_delta,
                "mean_log_loss_delta_vs_best": delta.mean_log_loss_delta,
                "se_log_loss_delta_vs_best": delta.se_log_loss_delta,
                "within_one_se_brier": (
                    delta.mean_brier_delta
                    <= one_se_multiplier * delta.se_brier_delta + 1e-15
                ),
                "within_one_se_log_loss": (
                    delta.mean_log_loss_delta
                    <= one_se_multiplier * delta.se_log_loss_delta + 1e-15
                ),
            }
        )

    diag_df = pd.DataFrame(diagnostics)
    table = score_df.merge(diag_df, on="name", how="left", validate="one_to_one")

    retained = False
    if incumbent is not None:
        row = table.loc[table["name"] == incumbent].iloc[0]
        retained = bool(row["within_one_se_brier"] and row["within_one_se_log_loss"])

    if retained:
        winner = incumbent
        reason = (
            "incumbent retained: within paired date-clustered one-SE of the "
            "point-Brier-best on both Brier and log loss"
        )
    else:
        winner = best
        reason = (
            "point-Brier-best selected: no incumbent was supplied or the incumbent "
            "failed the paired one-SE Brier/log-loss retention gate"
        )

    decision = SelectionDecision(
        winner=winner,
        point_brier_best=best,
        incumbent=incumbent,
        incumbent_retained=retained,
        reason=reason,
    )
    return decision, table.sort_values(
        ["race_macro_brier", "race_macro_log_loss", "name"],
        kind="stable",
    ).reset_index(drop=True)
