"""Stage 4 successor development helpers.

Phase 1 reconstructs leakage-safe incumbent OOF predictions and audits where the
current non-market model is weak before any challenger is executed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .model_selection import candidate_score


DIAGNOSTIC_DIMENSIONS = (
    "year",
    "field_size_band",
    "race_class",
    "racecourse",
    "rest_interval_band",
    "career_starts_band",
)


def add_diagnostic_bands(frame: pd.DataFrame) -> pd.DataFrame:
    """Add preregistered, outcome-free diagnostic strata."""
    required = {
        "race_date",
        "field_size",
        "race_class",
        "racecourse",
        "days_since_prev",
        "career_starts",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing diagnostic columns: {sorted(missing)}")

    out = frame.copy()
    out["race_date"] = pd.to_datetime(out["race_date"], errors="raise")
    out["year"] = out["race_date"].dt.year.astype(str)
    out["field_size_band"] = pd.cut(
        pd.to_numeric(out["field_size"], errors="raise"),
        bins=[0, 11, 14, 16, np.inf],
        labels=["<=11", "12-14", "15-16", "17+"],
        include_lowest=True,
    ).astype("string")
    out["rest_interval_band"] = pd.cut(
        pd.to_numeric(out["days_since_prev"], errors="coerce"),
        bins=[-np.inf, 21, 35, 56, 90, np.inf],
        labels=["<=21", "22-35", "36-56", "57-90", "91+"],
    ).astype("string")
    out["career_starts_band"] = pd.cut(
        pd.to_numeric(out["career_starts"], errors="coerce"),
        bins=[-np.inf, 5, 10, 20, np.inf],
        labels=["3-5", "6-10", "11-20", "21+"],
    ).astype("string")
    for col in ("race_class", "racecourse"):
        out[col] = out[col].astype("string").fillna("UNKNOWN")
    for col in ("field_size_band", "rest_interval_band", "career_starts_band"):
        out[col] = out[col].fillna("UNKNOWN")
    return out


def field_size_baseline(frame: pd.DataFrame) -> np.ndarray:
    """Return the exchangeable pre-race baseline P(top3)=3/field_size."""
    if "field_size" not in frame.columns:
        raise ValueError("field_size is required")
    field_size = pd.to_numeric(frame["field_size"], errors="raise").to_numpy(dtype=float)
    if np.any(field_size <= 3):
        raise ValueError("field_size must exceed three for a top-3 probability baseline")
    return np.clip(3.0 / field_size, 1e-12, 1.0 - 1e-12)


def calibration_intercept_slope(
    y_true,
    probability,
) -> tuple[float, float]:
    """Estimate logistic calibration intercept and slope.

    Fits logit(E[y]) = intercept + slope * logit(p). Returns NaN pair when
    the stratum has only one outcome class or too few finite observations.
    """
    y = np.asarray(list(y_true) if not isinstance(y_true, np.ndarray) else y_true, dtype=int)
    p = np.asarray(
        list(probability) if not isinstance(probability, np.ndarray) else probability,
        dtype=float,
    )
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("y_true and probability must be aligned 1D arrays")
    finite = np.isfinite(p)
    y = y[finite]
    p = p[finite]
    if len(y) < 10 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    logit_p = np.log(p / (1.0 - p)).reshape(-1, 1)
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=2000,
        random_state=20261003,
    )
    model.fit(logit_p, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def diagnostic_row(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    dimension: str,
    level: str,
) -> dict[str, object]:
    """Summarize proper-score and calibration diagnostics for one stratum."""
    if len(frame) != len(probability):
        raise ValueError("frame and probability length mismatch")
    model_score = candidate_score(frame, probability, name="incumbent")
    baseline = field_size_baseline(frame)
    baseline_score = candidate_score(frame, baseline, name="field_size_baseline")
    y = frame["top3_label"].astype(int).to_numpy()
    p = np.asarray(probability, dtype=float)
    intercept, slope = calibration_intercept_slope(y, p)
    baseline_brier = baseline_score.race_macro_brier
    skill = (
        1.0 - model_score.race_macro_brier / baseline_brier
        if baseline_brier > 0
        else float("nan")
    )
    return {
        "dimension": dimension,
        "level": level,
        "rows": len(frame),
        "races": int(frame["race_id"].nunique()),
        "race_macro_brier": model_score.race_macro_brier,
        "race_macro_log_loss": model_score.race_macro_log_loss,
        "runner_micro_brier": model_score.runner_micro_brier,
        "runner_micro_log_loss": model_score.runner_micro_log_loss,
        "baseline_race_macro_brier": baseline_score.race_macro_brier,
        "baseline_race_macro_log_loss": baseline_score.race_macro_log_loss,
        "brier_skill_vs_field_size_baseline": skill,
        "observed_top3_rate": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "calibration_gap_observed_minus_predicted": float(y.mean() - p.mean()),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def subgroup_diagnostics(
    frame: pd.DataFrame,
    probability: np.ndarray,
) -> pd.DataFrame:
    """Create the preregistered Phase-1 subgroup diagnostics."""
    work = add_diagnostic_bands(frame).reset_index(drop=True)
    p = np.asarray(probability, dtype=float)
    if len(work) != len(p):
        raise ValueError("frame and probability length mismatch")

    rows: list[dict[str, object]] = [
        diagnostic_row(work, p, dimension="overall", level="all")
    ]
    for dimension in DIAGNOSTIC_DIMENSIONS:
        for level, group in work.groupby(dimension, sort=True, observed=True):
            positions = group.index.to_numpy(dtype=int)
            rows.append(
                diagnostic_row(
                    group,
                    p[positions],
                    dimension=dimension,
                    level=str(level),
                )
            )
    return pd.DataFrame(rows)


def assert_outer_year_contract(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    outer_year: int,
) -> None:
    """Hard-fail if an outer prediction can see its own/future year."""
    train_dates = pd.to_datetime(train["race_date"], errors="raise")
    eval_dates = pd.to_datetime(evaluation["race_date"], errors="raise")
    if train_dates.dt.year.ge(outer_year).any():
        raise ValueError("outer training data contains current/future-year rows")
    if not eval_dates.dt.year.eq(outer_year).all():
        raise ValueError("outer evaluation data does not match outer_year")
    train_races = set(train["race_id"].astype(str))
    eval_races = set(evaluation["race_id"].astype(str))
    if train_races.intersection(eval_races):
        raise ValueError("outer train/evaluation race overlap")
