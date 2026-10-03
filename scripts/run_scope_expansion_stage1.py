"""Run Scope Expansion Stage 1 A/B/C training-population comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from keiba_place_lab.model_selection import (
    candidate_score,
    paired_delta,
    race_loss_table,
    select_with_incumbent_one_se,
)
from keiba_place_lab.nonmarket import (
    calibration_table,
    expected_calibration_error,
    validate_market_free,
)
from keiba_place_lab.scope_expansion import (
    full_context_for_races,
    select_turf_scope,
)
from keiba_place_lab.stage4_successor import calibration_intercept_slope
from keiba_place_lab.stage4_successor_phase3a import (
    fit_relative_candidate,
    predict_relative_candidate,
)

OUTER_YEARS = (2023, 2024)
INCUMBENT = "A_current_1200"
EXPECTED_EVAL_ROWS = 6313
EXPECTED_EVAL_RACES = 495
EXPECTED_EVAL_FINGERPRINT = (
    "a72fa10cc9adcd1ceee4c5b5eacd0dfdcdfc33fc984bf0b7866ec70b617bfd85"
)
EXPECTED_A_BRIER = 0.1489585929248138
PANEL_SHA256 = "cee9ae9a099f521f12b1bcdd371c25a3d7b9ba3555fbcf5a46f2c9098f59a33d"

CANDIDATES = {
    "A_current_1200": (1200, 1200),
    "B_turf_sprint_1000_1400": (1000, 1400),
    "C_turf_global_1000_2600": (1000, 2600),
}


def load_panel(base: Path, name: str) -> pd.DataFrame:
    path = base / name
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    validate_market_free(frame.columns)
    frame = frame.copy()
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="raise")
    frame["race_id"] = frame["race_id"].astype("string")
    frame["horse_id"] = frame["horse_id"].astype("string")
    return frame


def fingerprint(frame: pd.DataFrame) -> str:
    columns = ["race_date", "race_id"]
    for col in ("horse_id", "horse_no", "horse_name"):
        if col in frame.columns:
            columns.append(col)
    keys = frame[columns].copy()
    keys["race_date"] = pd.to_datetime(keys["race_date"]).dt.strftime("%Y-%m-%d")
    keys = keys.sort_values(columns, kind="stable").reset_index(drop=True)
    payload = keys.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def diagnostic_row(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    name: str,
) -> dict[str, object]:
    score = candidate_score(frame, probability, name=name)
    intercept, slope = calibration_intercept_slope(
        frame["top3_label"].astype(int).to_numpy(),
        probability,
    )
    reliability = calibration_table(frame, probability, bins=10)
    return {
        **score.__dict__,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece_10bin": expected_calibration_error(reliability),
    }


def format_table(frame: pd.DataFrame, columns: list[str]) -> str:
    show = frame[columns].copy()
    for col in show.select_dtypes(include=["number"]).columns:
        show[col] = show[col].map(
            lambda value: f"{value:.9f}" if pd.notna(value) else ""
        )
    return show.to_markdown(index=False)


def attach_frozen_phase3a_reference(
    oof: pd.DataFrame,
    reference_path: Path,
) -> np.ndarray:
    frozen = pd.read_csv(reference_path)
    frozen["race_id"] = frozen["race_id"].astype("string")
    frozen["horse_id"] = frozen["horse_id"].astype("string")
    required = {"race_id", "horse_id", "outer_year", "p_challenger"}
    missing = required.difference(frozen.columns)
    if missing:
        raise ValueError(f"Frozen Phase3A predictions missing columns: {sorted(missing)}")
    ref = frozen[
        ["race_id", "horse_id", "outer_year", "p_challenger"]
    ].rename(columns={"p_challenger": "p_frozen_phase3a"})
    joined = oof[
        ["race_id", "horse_id", "outer_year"]
    ].merge(
        ref,
        on=["race_id", "horse_id", "outer_year"],
        how="left",
        validate="one_to_one",
    )
    if joined["p_frozen_phase3a"].isna().any():
        raise ValueError("Frozen Phase3A reference does not cover common evaluation rows")
    return joined["p_frozen_phase3a"].to_numpy(dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--phase3a-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_panel = pd.concat(
        [
            load_panel(args.historical_dir, "jra_flat_warmup_v1.parquet"),
            load_panel(args.historical_dir, "jra_flat_train_v1.parquet"),
            load_panel(args.historical_dir, "jra_flat_validation_v1.parquet"),
            load_panel(args.historical_dir, "jra_flat_test_v1.parquet"),
        ],
        ignore_index=True,
        sort=False,
    )
    full_panel = full_panel.sort_values(
        ["race_date", "race_id", "horse_no"],
        kind="stable",
    ).reset_index(drop=True)

    common_1200 = select_turf_scope(
        full_panel,
        min_distance_m=1200,
        max_distance_m=1200,
        exclude_straight=True,
    )
    common_eval = common_1200.loc[
        common_1200["race_date"].dt.year.isin(OUTER_YEARS)
    ].copy()
    common_eval = common_eval.sort_values(
        ["race_date", "race_id", "horse_no"],
        kind="stable",
    ).reset_index(drop=True)

    fp = fingerprint(common_eval)
    if len(common_eval) != EXPECTED_EVAL_ROWS:
        raise ValueError(f"Evaluation rows mismatch: {len(common_eval)}")
    if common_eval["race_id"].nunique() != EXPECTED_EVAL_RACES:
        raise ValueError(
            f"Evaluation races mismatch: {common_eval['race_id'].nunique()}"
        )
    if fp != EXPECTED_EVAL_FINGERPRINT:
        raise ValueError(f"Evaluation fingerprint mismatch: {fp}")
    if common_eval["race_date"].dt.year.eq(2025).any():
        raise ValueError("2025 must not enter Stage 1 comparison")

    eval_context = full_context_for_races(
        full_panel,
        set(common_eval["race_id"].astype("string")),
    )

    base_cols = [
        "race_date",
        "race_id",
        "horse_id",
        "horse_no",
        "horse_name",
        "top3_label",
    ]
    oof = common_eval[base_cols].copy()
    oof["outer_year"] = oof["race_date"].dt.year
    coverage_parts: list[pd.DataFrame] = []
    training_rows: list[dict[str, object]] = []

    for candidate, (min_distance, max_distance) in CANDIDATES.items():
        cohort = select_turf_scope(
            full_panel,
            min_distance_m=min_distance,
            max_distance_m=max_distance,
            exclude_straight=True,
        )
        candidate_predictions = np.full(len(common_eval), np.nan, dtype=float)

        for outer_year in OUTER_YEARS:
            train = cohort.loc[
                cohort["race_date"].dt.year.between(2016, outer_year - 1)
            ].copy()
            eval_mask = common_eval["race_date"].dt.year.eq(outer_year).to_numpy()
            evaluation = common_eval.loc[eval_mask].copy()

            if train.empty or evaluation.empty:
                raise ValueError(f"Empty fold for {candidate} outer {outer_year}")
            if train["race_date"].dt.year.max() >= outer_year:
                raise ValueError("Training chronology violation")
            if train["race_date"].dt.year.eq(2025).any():
                raise ValueError("2025 training rows prohibited in Stage 1")

            train_context = full_context_for_races(
                full_panel,
                set(train["race_id"].astype("string")),
            )
            evaluation_context = eval_context.loc[
                eval_context["race_id"].isin(set(evaluation["race_id"].astype("string")))
            ].copy()

            model, prior_mean, train_coverage = fit_relative_candidate(
                train,
                train_context,
            )
            probability, eval_coverage = predict_relative_candidate(
                model,
                evaluation,
                evaluation_context,
                prior_mean=prior_mean,
            )
            candidate_predictions[eval_mask] = probability

            train_cov = train_coverage.copy()
            train_cov.insert(0, "scope", "training")
            train_cov.insert(0, "outer_year", outer_year)
            train_cov.insert(0, "candidate", candidate)
            coverage_parts.append(train_cov)

            eval_cov = eval_coverage.copy()
            eval_cov.insert(0, "scope", "evaluation")
            eval_cov.insert(0, "outer_year", outer_year)
            eval_cov.insert(0, "candidate", candidate)
            coverage_parts.append(eval_cov)

            training_rows.append(
                {
                    "candidate": candidate,
                    "outer_year": outer_year,
                    "train_min_year": int(train["race_date"].dt.year.min()),
                    "train_max_year": int(train["race_date"].dt.year.max()),
                    "train_rows": len(train),
                    "train_races": int(train["race_id"].nunique()),
                    "train_context_rows": len(train_context),
                    "prior_mean": prior_mean,
                    "min_distance_m": min_distance,
                    "max_distance_m": max_distance,
                }
            )

        if not np.isfinite(candidate_predictions).all():
            raise ValueError(f"Missing/non-finite predictions for {candidate}")
        oof[f"p_{candidate}"] = candidate_predictions

    frozen_reference = attach_frozen_phase3a_reference(
        oof,
        args.phase3a_reference,
    )
    a_probability = oof[f"p_{INCUMBENT}"].to_numpy(dtype=float)
    max_abs_reference_diff = float(
        np.max(np.abs(a_probability - frozen_reference))
    )
    # The frozen Phase3A CSV was intentionally written at 9 decimal places.
    if max_abs_reference_diff > 5.1e-10:
        raise ValueError(
            "Candidate A exceeds the frozen Phase3A CSV rounding bound: "
            f"max_abs={max_abs_reference_diff}"
        )

    predictions = {
        candidate: oof[f"p_{candidate}"].to_numpy(dtype=float)
        for candidate in CANDIDATES
    }
    decision, selection = select_with_incumbent_one_se(
        oof,
        predictions,
        incumbent=INCUMBENT,
    )

    metrics_rows = []
    year_rows = []
    reliability_parts = []
    race_loss_parts = []

    for candidate, probability in predictions.items():
        metrics_rows.append(
            diagnostic_row(oof, probability, name=candidate)
        )
        reliability = calibration_table(oof, probability, bins=10)
        reliability.insert(0, "candidate", candidate)
        reliability_parts.append(reliability)

        race_loss = race_loss_table(oof, probability)
        race_loss.insert(1, "candidate", candidate)
        race_loss_parts.append(race_loss)

        for outer_year in OUTER_YEARS:
            mask = oof["outer_year"].eq(outer_year).to_numpy()
            row = diagnostic_row(
                oof.loc[mask],
                probability[mask],
                name=candidate,
            )
            year_rows.append({"outer_year": outer_year, **row})

    metrics = pd.DataFrame(metrics_rows).sort_values(
        ["race_macro_brier", "race_macro_log_loss", "name"],
        kind="stable",
    )
    year_metrics = pd.DataFrame(year_rows).sort_values(
        ["outer_year", "race_macro_brier", "name"],
        kind="stable",
    )
    reliability = pd.concat(reliability_parts, ignore_index=True)
    race_losses = pd.concat(race_loss_parts, ignore_index=True)
    coverage = pd.concat(coverage_parts, ignore_index=True)
    training = pd.DataFrame(training_rows)

    incumbent_vs_rows = []
    for candidate in CANDIDATES:
        if candidate == INCUMBENT:
            continue
        delta = paired_delta(
            oof,
            predictions[INCUMBENT],
            predictions[candidate],
            candidate_name=INCUMBENT,
            reference_name=candidate,
        )
        incumbent_vs_rows.append(
            {
                "incumbent": INCUMBENT,
                "challenger": candidate,
                "mean_incumbent_minus_challenger_brier": delta.mean_brier_delta,
                "se_incumbent_minus_challenger_brier": delta.se_brier_delta,
                "mean_incumbent_minus_challenger_log_loss": delta.mean_log_loss_delta,
                "se_incumbent_minus_challenger_log_loss": delta.se_log_loss_delta,
            }
        )
    incumbent_pairwise = pd.DataFrame(incumbent_vs_rows)

    a_score = candidate_score(oof, a_probability, name=INCUMBENT)
    if abs(a_score.race_macro_brier - EXPECTED_A_BRIER) > 1e-12:
        raise ValueError(
            "Candidate A Brier does not reproduce frozen Stage4 successor: "
            f"{a_score.race_macro_brier} vs {EXPECTED_A_BRIER}"
        )

    oof.to_csv(
        args.output_dir / "scope_stage1_outer_predictions.csv",
        index=False,
        float_format="%.12f",
    )
    metrics.to_csv(
        args.output_dir / "scope_stage1_outer_metrics.csv",
        index=False,
    )
    year_metrics.to_csv(
        args.output_dir / "scope_stage1_outer_year_metrics.csv",
        index=False,
    )
    selection.to_csv(
        args.output_dir / "scope_stage1_selection.csv",
        index=False,
    )
    race_losses.to_csv(
        args.output_dir / "scope_stage1_race_losses.csv",
        index=False,
    )
    coverage.to_csv(
        args.output_dir / "scope_stage1_relative_coverage.csv",
        index=False,
    )
    reliability.to_csv(
        args.output_dir / "scope_stage1_reliability.csv",
        index=False,
    )
    training.to_csv(
        args.output_dir / "scope_stage1_training_folds.csv",
        index=False,
    )
    incumbent_pairwise.to_csv(
        args.output_dir / "scope_stage1_incumbent_pairwise.csv",
        index=False,
    )

    manifest = {
        "status": "PASS",
        "stage": "scope_expansion_stage1_comparison",
        "source_commit_sha": os.environ.get("GITHUB_SHA"),
        "historical_panel_sha256": PANEL_SHA256,
        "evaluation_fingerprint_sha256": fp,
        "evaluation_rows": len(oof),
        "evaluation_races": int(oof["race_id"].nunique()),
        "outer_years": list(OUTER_YEARS),
        "uses_2025_rows": False,
        "uses_market_fields": False,
        "probability_sum_to_three_adjustment": False,
        "feature_definition_changed": False,
        "hyperparameters_changed": False,
        "candidate_A_matches_frozen_phase3a_predictions_within_csv_rounding": True,
        "candidate_A_max_abs_diff_vs_9dp_reference": max_abs_reference_diff,
        "candidate_A_race_macro_brier": a_score.race_macro_brier,
        "incumbent": INCUMBENT,
        "winner": decision.winner,
        "point_brier_best": decision.point_brier_best,
        "incumbent_retained": decision.incumbent_retained,
        "selection_reason": decision.reason,
        "candidate_training_populations": training.to_dict(orient="records"),
    }
    (args.output_dir / "scope_stage1_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = [
        "# Scope Expansion Stage 1 — A/B/C Comparison",
        "",
        "Status: **PASS — frozen training-population comparison completed**",
        "",
        "## Question",
        "",
        "Does broader turf training data improve prediction on the existing 1200m evaluation rows",
        "without changing features or XGB01 hyperparameters?",
        "",
        "## Decision",
        "",
        f"- incumbent: **{INCUMBENT}**",
        f"- point-Brier-best: **{decision.point_brier_best}**",
        f"- winner under frozen paired one-SE rule: **{decision.winner}**",
        f"- incumbent retained: **{decision.incumbent_retained}**",
        f"- reason: {decision.reason}",
        "",
        "This is a model-selection rule, not a statistical-significance claim.",
        "",
        "## Common 2023-2024 evaluation",
        "",
        f"- rows: **{len(oof):,}**",
        f"- races: **{oof['race_id'].nunique():,}**",
        f"- fingerprint: {fp}",
        "- Candidate A reproduces frozen Phase3A predictions within 9-decimal CSV rounding: **yes**",
        f"- max absolute A/reference difference: {max_abs_reference_diff:.12g}",
        "",
        "## Overall metrics",
        "",
        format_table(
            metrics,
            [
                "name",
                "race_macro_brier",
                "race_macro_log_loss",
                "runner_micro_brier",
                "runner_micro_log_loss",
                "calibration_intercept",
                "calibration_slope",
                "ece_10bin",
                "races",
                "rows",
            ],
        ),
        "",
        "## Selection table",
        "",
        format_table(
            selection,
            [
                "name",
                "race_macro_brier",
                "race_macro_log_loss",
                "mean_brier_delta_vs_best",
                "se_brier_delta_vs_best",
                "within_one_se_brier",
            ],
        ),
        "",
        "## Direct incumbent/challenger paired differences",
        "",
        incumbent_pairwise.to_markdown(index=False),
        "",
        "## Outer-year metrics",
        "",
        format_table(
            year_metrics,
            [
                "outer_year",
                "name",
                "race_macro_brier",
                "race_macro_log_loss",
                "calibration_intercept",
                "calibration_slope",
                "races",
                "rows",
            ],
        ),
        "",
        "## Training folds",
        "",
        training.to_markdown(index=False),
        "",
        "## Interpretation boundary",
        "",
        "This experiment changes training-population width only. It does not validate B or C for",
        "prediction on non-1200 races. Cross-distance deployment remains prohibited until the later",
        "distance-wide evaluation stage.",
    ]
    (args.output_dir / "SCOPE_EXPANSION_STAGE1_REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
