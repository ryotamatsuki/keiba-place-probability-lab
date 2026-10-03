"""Run Stage 4 successor Phase 2 model-family comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from keiba_place_lab.model_selection import (
    candidate_score,
    paired_delta,
    select_with_incumbent_one_se,
)
from keiba_place_lab.nonmarket import (
    calibration_table,
    expected_calibration_error,
    validate_market_free,
)
from keiba_place_lab.stage4_successor import (
    assert_outer_year_contract,
    calibration_intercept_slope,
)
from keiba_place_lab.stage4_successor_phase2 import (
    FAMILY_CONFIGS,
    INCUMBENT_NAME,
    fit_challenger,
    fit_predict_incumbent,
    predict_challenger,
    registry_frame,
    select_inner_config,
)

OUTER_YEARS = (2023, 2024)
INNER_YEARS = {
    2023: (2021, 2022),
    2024: (2021, 2022, 2023),
}
EXPECTED_PHASE1_FINGERPRINT = (
    "a72fa10cc9adcd1ceee4c5b5eacd0dfdcdfc33fc984bf0b7866ec70b617bfd85"
)
PANEL_SHA256 = "cee9ae9a099f521f12b1bcdd371c25a3d7b9ba3555fbcf5a46f2c9098f59a33d"


def load_turf_1200(base: Path, split: str) -> pd.DataFrame:
    path = base / f"phase_a_turf_1200_{split}_v1.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    validate_market_free(frame.columns)
    frame = frame.copy()
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="raise")
    return frame


def evaluation_fingerprint(frame: pd.DataFrame) -> str:
    columns = ["race_date", "race_id"]
    for col in ("horse_id", "horse_no", "horse_name"):
        if col in frame.columns:
            columns.append(col)
    keys = frame[columns].copy()
    keys["race_date"] = pd.to_datetime(keys["race_date"]).dt.strftime("%Y-%m-%d")
    keys = keys.sort_values(columns, kind="stable").reset_index(drop=True)
    payload = keys.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_phase1_incumbent(oof: pd.DataFrame, phase1_path: Path) -> float:
    frozen = pd.read_csv(phase1_path)
    join_keys = (
        ["race_id", "horse_id"]
        if "horse_id" in frozen.columns
        else ["race_id", "horse_no"]
    )
    current = oof[join_keys + ["p_incumbent"]].copy()
    merged = current.merge(
        frozen[join_keys + ["p_incumbent"]],
        on=join_keys,
        how="inner",
        suffixes=("_phase2", "_phase1"),
        validate="one_to_one",
    )
    if len(merged) != len(oof) or len(merged) != len(frozen):
        raise ValueError("Phase-1 and Phase-2 incumbent rows do not align")
    max_abs = float(
        np.max(
            np.abs(
                merged["p_incumbent_phase2"].to_numpy()
                - merged["p_incumbent_phase1"].to_numpy()
            )
        )
    )
    if max_abs > 1e-10:
        raise ValueError(
            f"Phase-2 incumbent does not reproduce Phase 1: max_abs={max_abs}"
        )
    return max_abs


def fmt(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    show = frame[columns].copy()
    for col in show.select_dtypes(include=["number"]).columns:
        show[col] = show[col].map(
            lambda value: f"{value:.{digits}f}" if pd.notna(value) else ""
        )
    return show.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--phase1-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_2016_2022 = load_turf_1200(args.historical_dir, "train")
    validation_2023_2024 = load_turf_1200(args.historical_dir, "validation")
    if validation_2023_2024["race_date"].dt.year.max() > 2024:
        raise ValueError("Phase 2 must not load post-2024 validation rows")

    development = pd.concat(
        [train_2016_2022, validation_2023_2024],
        ignore_index=True,
        sort=False,
    )
    years = development["race_date"].dt.year

    outer_parts: list[pd.DataFrame] = []
    inner_tables: list[pd.DataFrame] = []
    selected_rows: list[dict[str, object]] = []

    for outer_year in OUTER_YEARS:
        outer_train = development.loc[years.between(2016, outer_year - 1)].copy()
        outer_eval = development.loc[years.eq(outer_year)].copy().reset_index(drop=True)
        assert_outer_year_contract(outer_train, outer_eval, outer_year=outer_year)

        outer = outer_eval.copy()
        outer["outer_year"] = outer_year
        outer["p_incumbent"] = fit_predict_incumbent(outer_train, outer_eval)

        for family, configs in FAMILY_CONFIGS.items():
            inner_frame_parts: list[pd.DataFrame] = []
            inner_predictions: dict[str, list[np.ndarray]] = {
                cfg.config_id: [] for cfg in configs
            }

            for inner_year in INNER_YEARS[outer_year]:
                inner_train = development.loc[
                    years.between(2016, inner_year - 1)
                ].copy()
                inner_eval = development.loc[
                    years.eq(inner_year)
                ].copy().reset_index(drop=True)
                assert_outer_year_contract(
                    inner_train,
                    inner_eval,
                    outer_year=inner_year,
                )
                inner_frame_parts.append(inner_eval)

                for cfg in configs:
                    model, prior_mean = fit_challenger(inner_train, cfg)
                    probability = predict_challenger(
                        model,
                        inner_eval,
                        prior_mean=prior_mean,
                    )
                    inner_predictions[cfg.config_id].append(probability)

            inner_frame = pd.concat(inner_frame_parts, ignore_index=True, sort=False)
            concatenated_predictions = {
                config_id: np.concatenate(parts)
                for config_id, parts in inner_predictions.items()
            }
            selected_cfg, tuning_table = select_inner_config(
                inner_frame,
                concatenated_predictions,
                configs,
            )
            tuning_table.insert(0, "outer_year", outer_year)
            tuning_table.insert(1, "family", family)
            inner_tables.append(tuning_table)

            final_model, final_prior = fit_challenger(outer_train, selected_cfg)
            outer_probability = predict_challenger(
                final_model,
                outer_eval,
                prior_mean=final_prior,
            )
            outer[f"p_{family}"] = outer_probability
            selected_rows.append(
                {
                    "outer_year": outer_year,
                    "family": family,
                    "config_id": selected_cfg.config_id,
                    "complexity_rank": selected_cfg.complexity_rank,
                    "params_json": json.dumps(
                        selected_cfg.params,
                        sort_keys=True,
                        ensure_ascii=False,
                    ),
                }
            )

        outer_parts.append(outer)

    oof = pd.concat(outer_parts, ignore_index=True, sort=False)
    if oof["race_date"].dt.year.ge(2025).any():
        raise ValueError("2025+ rows are forbidden in Phase 2")

    fingerprint = evaluation_fingerprint(oof)
    if fingerprint != EXPECTED_PHASE1_FINGERPRINT:
        raise ValueError(
            f"Phase-2 evaluation fingerprint mismatch: {fingerprint}"
        )
    max_incumbent_abs_diff = verify_phase1_incumbent(oof, args.phase1_oof)

    prediction_columns = {
        INCUMBENT_NAME: "p_incumbent",
        "random_forest": "p_random_forest",
        "hist_gradient_boosting": "p_hist_gradient_boosting",
        "xgboost": "p_xgboost",
    }
    predictions = {
        name: oof[column].to_numpy(dtype=float)
        for name, column in prediction_columns.items()
    }
    decision, selection_table = select_with_incumbent_one_se(
        oof,
        predictions,
        incumbent=INCUMBENT_NAME,
    )

    versus_incumbent = []
    for name, probability in predictions.items():
        delta = paired_delta(
            oof,
            probability,
            predictions[INCUMBENT_NAME],
            candidate_name=name,
            reference_name=INCUMBENT_NAME,
        )
        versus_incumbent.append(
            {
                "name": name,
                "mean_brier_delta_vs_incumbent": delta.mean_brier_delta,
                "se_brier_delta_vs_incumbent": delta.se_brier_delta,
                "mean_log_loss_delta_vs_incumbent": delta.mean_log_loss_delta,
                "se_log_loss_delta_vs_incumbent": delta.se_log_loss_delta,
            }
        )
    selection_table = selection_table.merge(
        pd.DataFrame(versus_incumbent),
        on="name",
        how="left",
        validate="one_to_one",
    )

    metric_rows = []
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
        metric_rows.append(
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
            score_year = candidate_score(
                oof.loc[mask],
                probability[mask],
                name=name,
            )
            year_rows.append({"outer_year": year, **score_year.__dict__})

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["race_macro_brier", "race_macro_log_loss", "name"],
        kind="stable",
    )
    year_metrics = pd.DataFrame(year_rows)
    reliability = pd.concat(reliability_parts, ignore_index=True)
    inner_tuning = pd.concat(inner_tables, ignore_index=True)
    selected_configs = pd.DataFrame(selected_rows)

    registry_frame().to_csv(
        args.output_dir / "phase2_candidate_registry.csv", index=False
    )
    inner_tuning.to_csv(
        args.output_dir / "phase2_inner_tuning.csv", index=False
    )
    selected_configs.to_csv(
        args.output_dir / "phase2_selected_configs.csv", index=False
    )
    selection_table.to_csv(
        args.output_dir / "phase2_selection_table.csv", index=False
    )
    metrics.to_csv(args.output_dir / "phase2_outer_metrics.csv", index=False)
    year_metrics.to_csv(
        args.output_dir / "phase2_outer_year_metrics.csv", index=False
    )
    reliability.to_csv(
        args.output_dir / "phase2_reliability.csv", index=False
    )

    output_cols = [
        col
        for col in [
            "race_date",
            "race_id",
            "horse_id",
            "horse_no",
            "horse_name",
            "top3_label",
            "outer_year",
            *prediction_columns.values(),
        ]
        if col in oof.columns
    ]
    oof[output_cols].to_csv(
        args.output_dir / "phase2_outer_predictions.csv",
        index=False,
        float_format="%.9f",
    )

    manifest = {
        "status": "PASS",
        "phase": "stage4_successor_phase2",
        "historical_panel_sha256": PANEL_SHA256,
        "phase1_evaluation_fingerprint_sha256": EXPECTED_PHASE1_FINGERPRINT,
        "phase2_evaluation_fingerprint_sha256": fingerprint,
        "phase1_incumbent_max_abs_probability_diff": max_incumbent_abs_diff,
        "uses_2025_rows": False,
        "uses_market_fields": False,
        "posthoc_calibration": "none",
        "outer_years": list(OUTER_YEARS),
        "inner_years": {str(k): list(v) for k, v in INNER_YEARS.items()},
        "evaluation_rows": len(oof),
        "evaluation_races": int(oof["race_id"].nunique()),
        "winner": decision.winner,
        "point_brier_best": decision.point_brier_best,
        "incumbent_retained": decision.incumbent_retained,
        "selection_reason": decision.reason,
        "selected_configs": selected_rows,
    }
    (args.output_dir / "phase2_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    winner_row = selection_table.loc[
        selection_table["name"] == decision.winner
    ].iloc[0]
    report = [
        "# Stage 4 Successor Phase 2 — Model-family Comparison",
        "",
        "Status: **PASS — preregistered outer 2023/2024 comparison completed**",
        "",
        "## Decision",
        "",
        f"- winner under frozen v2 rule: **{decision.winner}**",
        f"- point-Brier-best: **{decision.point_brier_best}**",
        f"- incumbent retained: **{decision.incumbent_retained}**",
        f"- reason: {decision.reason}",
        f"- winner race-macro Brier: {float(winner_row['race_macro_brier']):.6f}",
        "",
        "This is a preregistered replacement decision, not a claim of statistical proof.",
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
        "## Frozen v2 selection table",
        "",
        fmt(
            selection_table,
            [
                "name",
                "race_macro_brier",
                "mean_brier_delta_vs_best",
                "se_brier_delta_vs_best",
                "within_one_se_brier",
                "mean_brier_delta_vs_incumbent",
                "se_brier_delta_vs_incumbent",
            ],
        ),
        "",
        "## Selected inner configurations",
        "",
        selected_configs.to_markdown(index=False),
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
        "## Audit",
        "",
        f"- evaluation rows: {len(oof):,}",
        f"- evaluation races: {int(oof['race_id'].nunique()):,}",
        f"- evaluation-row SHA256: {fingerprint}",
        f"- incumbent reproduction max absolute probability difference vs Phase 1: {max_incumbent_abs_diff:.3e}",
        "- 2025 rows loaded: **no**",
        "- market fields loaded: **no**",
        "- post-hoc calibration: **none**",
        "",
        "Phase 3 must not start until this Phase-2 result is frozen.",
    ]
    (args.output_dir / "PHASE2_MODEL_COMPARISON.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
