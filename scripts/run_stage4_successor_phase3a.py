"""Run Phase 3A: add full-field relative-ability features to frozen XGB01."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from keiba_place_lab.model_selection import (
    candidate_score,
    select_with_incumbent_one_se,
)
from keiba_place_lab.nonmarket import (
    calibration_table,
    expected_calibration_error,
    validate_market_free,
)
from keiba_place_lab.stage4_successor import calibration_intercept_slope
from keiba_place_lab.stage4_successor_phase3a import (
    fit_relative_candidate,
    predict_relative_candidate,
)

OUTER_YEARS = (2023, 2024)
INCUMBENT = "xgboost_phase2_xgb01"
CHALLENGER = "xgboost_xgb01_plus_relative_ability"
EXPECTED_FINGERPRINT = (
    "a72fa10cc9adcd1ceee4c5b5eacd0dfdcdfc33fc984bf0b7866ec70b617bfd85"
)
EXPECTED_PHASE2_BRIER = 0.15149388455442392
PANEL_SHA256 = "cee9ae9a099f521f12b1bcdd371c25a3d7b9ba3555fbcf5a46f2c9098f59a33d"


def load_parquet(base: Path, name: str) -> pd.DataFrame:
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


def attach_phase2_incumbent(
    frame: pd.DataFrame,
    path: Path,
) -> np.ndarray:
    frozen = pd.read_csv(path)
    frozen["race_id"] = frozen["race_id"].astype("string")
    frozen["horse_id"] = frozen["horse_id"].astype("string")
    current = frame[["race_id", "horse_id"]].copy()
    joined = current.merge(
        frozen[["race_id", "horse_id", "p_xgboost"]],
        on=["race_id", "horse_id"],
        how="left",
        validate="one_to_one",
    )
    if joined["p_xgboost"].isna().any() or len(joined) != len(frame):
        raise ValueError("Frozen Phase-2 XGBoost predictions do not align to Phase 3A rows")
    probability = joined["p_xgboost"].to_numpy(dtype=float)
    score = candidate_score(frame, probability, name=INCUMBENT)
    if abs(score.race_macro_brier - EXPECTED_PHASE2_BRIER) > 1e-8:
        raise ValueError(
            "Frozen Phase-2 incumbent Brier mismatch: "
            f"{score.race_macro_brier} vs {EXPECTED_PHASE2_BRIER}"
        )
    return probability


def fmt(frame: pd.DataFrame, columns: list[str]) -> str:
    show = frame[columns].copy()
    for col in show.select_dtypes(include=["number"]).columns:
        show[col] = show[col].map(
            lambda value: f"{value:.6f}" if pd.notna(value) else ""
        )
    return show.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--phase2-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    eligible_train = load_parquet(
        args.historical_dir,
        "phase_a_turf_1200_train_v1.parquet",
    )
    eligible_validation = load_parquet(
        args.historical_dir,
        "phase_a_turf_1200_validation_v1.parquet",
    )
    context_train = load_parquet(args.historical_dir, "jra_flat_train_v1.parquet")
    context_validation = load_parquet(
        args.historical_dir,
        "jra_flat_validation_v1.parquet",
    )

    if (
        eligible_train["race_date"].dt.year.ge(2025).any()
        or eligible_validation["race_date"].dt.year.ge(2025).any()
        or context_train["race_date"].dt.year.ge(2025).any()
        or context_validation["race_date"].dt.year.ge(2025).any()
    ):
        raise ValueError("Phase 3A inputs must not contain 2025 rows")

    eligible = pd.concat(
        [eligible_train, eligible_validation],
        ignore_index=True,
        sort=False,
    )
    context = pd.concat(
        [context_train, context_validation],
        ignore_index=True,
        sort=False,
    )
    eligible_year = eligible["race_date"].dt.year
    context_year = context["race_date"].dt.year

    outer_parts: list[pd.DataFrame] = []
    coverage_parts: list[pd.DataFrame] = []

    for outer_year in OUTER_YEARS:
        train = eligible.loc[
            eligible_year.between(2016, outer_year - 1)
        ].copy()
        evaluation = eligible.loc[
            eligible_year.eq(outer_year)
        ].copy().reset_index(drop=True)

        train_races = set(train["race_id"])
        eval_races = set(evaluation["race_id"])
        if train_races.intersection(eval_races):
            raise ValueError("Outer train/evaluation race overlap")
        if train["race_date"].dt.year.ge(outer_year).any():
            raise ValueError("Outer training contains current/future year")
        if not evaluation["race_date"].dt.year.eq(outer_year).all():
            raise ValueError("Outer evaluation year mismatch")

        train_context = context.loc[
            context["race_id"].isin(train_races)
            & context_year.between(2016, outer_year - 1)
        ].copy()
        eval_context = context.loc[
            context["race_id"].isin(eval_races)
            & context_year.eq(outer_year)
        ].copy()

        model, prior_mean, train_coverage = fit_relative_candidate(
            train,
            train_context,
        )
        probability, eval_coverage = predict_relative_candidate(
            model,
            evaluation,
            eval_context,
            prior_mean=prior_mean,
        )

        train_coverage.insert(0, "outer_year", outer_year)
        train_coverage.insert(1, "scope", "train")
        eval_coverage.insert(0, "outer_year", outer_year)
        eval_coverage.insert(1, "scope", "evaluation")
        coverage_parts.extend([train_coverage, eval_coverage])

        out = evaluation[
            [
                col
                for col in [
                    "race_date",
                    "race_id",
                    "horse_id",
                    "horse_no",
                    "horse_name",
                    "top3_label",
                ]
                if col in evaluation.columns
            ]
        ].copy()
        out["outer_year"] = outer_year
        out["p_challenger"] = probability
        outer_parts.append(out)

    oof = pd.concat(outer_parts, ignore_index=True, sort=False)
    fp = fingerprint(oof)
    if fp != EXPECTED_FINGERPRINT:
        raise ValueError(f"Phase 3A evaluation fingerprint mismatch: {fp}")

    incumbent_probability = attach_phase2_incumbent(oof, args.phase2_oof)
    challenger_probability = oof["p_challenger"].to_numpy(dtype=float)
    oof["p_incumbent"] = incumbent_probability

    predictions = {
        INCUMBENT: incumbent_probability,
        CHALLENGER: challenger_probability,
    }
    decision, selection = select_with_incumbent_one_se(
        oof,
        predictions,
        incumbent=INCUMBENT,
    )

    metrics_rows = []
    year_rows = []
    reliability_parts = []
    for name, probability in predictions.items():
        score = candidate_score(oof, probability, name=name)
        intercept, slope = calibration_intercept_slope(
            oof["top3_label"].astype(int).to_numpy(),
            probability,
        )
        rel = calibration_table(oof, probability, bins=10)
        ece = expected_calibration_error(rel)
        metrics_rows.append(
            {
                **score.__dict__,
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "ece_10bin": ece,
            }
        )
        rel.insert(0, "name", name)
        reliability_parts.append(rel)

        for year in OUTER_YEARS:
            mask = oof["outer_year"].eq(year).to_numpy()
            year_score = candidate_score(
                oof.loc[mask],
                probability[mask],
                name=name,
            )
            year_rows.append({"outer_year": year, **year_score.__dict__})

    metrics = pd.DataFrame(metrics_rows).sort_values(
        ["race_macro_brier", "race_macro_log_loss", "name"],
        kind="stable",
    )
    year_metrics = pd.DataFrame(year_rows)
    coverage = pd.concat(coverage_parts, ignore_index=True)
    reliability = pd.concat(reliability_parts, ignore_index=True)

    oof.to_csv(
        args.output_dir / "phase3a_outer_predictions.csv",
        index=False,
        float_format="%.9f",
    )
    metrics.to_csv(args.output_dir / "phase3a_outer_metrics.csv", index=False)
    year_metrics.to_csv(
        args.output_dir / "phase3a_outer_year_metrics.csv",
        index=False,
    )
    selection.to_csv(
        args.output_dir / "phase3a_selection_table.csv",
        index=False,
    )
    coverage.to_csv(
        args.output_dir / "phase3a_relative_feature_coverage.csv",
        index=False,
    )
    reliability.to_csv(
        args.output_dir / "phase3a_reliability.csv",
        index=False,
    )

    accepted = decision.winner == CHALLENGER
    manifest = {
        "status": "PASS",
        "phase": "stage4_successor_phase3a",
        "historical_panel_sha256": PANEL_SHA256,
        "evaluation_fingerprint_sha256": fp,
        "evaluation_rows": len(oof),
        "evaluation_races": int(oof["race_id"].nunique()),
        "outer_years": list(OUTER_YEARS),
        "uses_2025_rows": False,
        "uses_market_fields": False,
        "model_hyperparameters_retuned": False,
        "posthoc_calibration": "none",
        "incumbent": INCUMBENT,
        "challenger": CHALLENGER,
        "winner": decision.winner,
        "point_brier_best": decision.point_brier_best,
        "incumbent_retained": decision.incumbent_retained,
        "challenger_accepted": accepted,
        "selection_reason": decision.reason,
    }
    (args.output_dir / "phase3a_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = [
        "# Stage 4 Successor Phase 3A — Relative Ability",
        "",
        "Status: **PASS — preregistered feature-block comparison completed**",
        "",
        "## Decision",
        "",
        f"- incumbent: **{INCUMBENT}**",
        f"- challenger: **{CHALLENGER}**",
        f"- winner under frozen v2 rule: **{decision.winner}**",
        f"- challenger accepted: **{accepted}**",
        f"- reason: {decision.reason}",
        "",
        "This is a preregistered replacement decision, not a significance-test claim.",
        "",
        "## Outer metrics",
        "",
        fmt(
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
        fmt(
            selection,
            [
                "name",
                "race_macro_brier",
                "mean_brier_delta_vs_best",
                "se_brier_delta_vs_best",
                "within_one_se_brier",
            ],
        ),
        "",
        "## Outer-year stability",
        "",
        fmt(
            year_metrics.sort_values(["outer_year", "race_macro_brier"]),
            [
                "outer_year",
                "name",
                "race_macro_brier",
                "race_macro_log_loss",
                "races",
                "rows",
            ],
        ),
        "",
        "## Relative-feature coverage",
        "",
        fmt(
            coverage.loc[coverage["scope"].eq("evaluation")],
            [
                "outer_year",
                "feature",
                "relative_coverage",
                "eligible_coverage",
                "eligible_rows",
            ],
        ),
        "",
        "## Audit",
        "",
        f"- evaluation rows: {len(oof):,}",
        f"- evaluation races: {int(oof['race_id'].nunique()):,}",
        f"- evaluation-row SHA256: {fp}",
        "- all relative features use full starter fields before eligible-row filtering",
        "- 2025 rows loaded: **no**",
        "- market fields loaded: **no**",
        "- XGB01 hyperparameters retuned: **no**",
        "- post-hoc calibration: **none**",
        "",
        (
            "The Phase-3A winner becomes the frozen incumbent for Phase 3B."
            if accepted
            else "Phase-2 XGB01 remains the frozen incumbent for Phase 3B."
        ),
    ]
    (args.output_dir / "PHASE3A_RELATIVE_ABILITY.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
