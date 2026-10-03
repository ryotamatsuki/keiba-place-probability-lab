#!/usr/bin/env python3
"""Run Stage 5 historical market calibration and market/non-market ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from keiba_place_lab.ensemble import (
    apply_logit_calibrator,
    convex_blend,
    evaluate_probability,
    fit_logit_calibrator,
    nearby_weights,
    select_blend_weight,
)
from keiba_place_lab.historical_market import reconstruct_historical_market
from keiba_place_lab.nonmarket import (
    enforce_race_top3_sum,
    fit_model,
    predict_raw_probability,
)

EXPECTED_RESULTS_SHA256 = "fb9345273b21a7c23d41260dc77e45134a1bc2fe756899f3416b24ac0dd51de3"
SELECTED_C = 0.1
PRIOR_STRENGTH = 6.0
TARGET_RACE_ID = "2026-10-03_KYOTO_11R"
TARGET_MARKET_AS_OF = "2026-10-03T08:35:00+09:00"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(base: Path, split: str) -> pd.DataFrame:
    path = base / f"phase_a_turf_1200_{split}_v1.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="raise")
    return frame


def predict_nonmarket(model, frame: pd.DataFrame, prior_mean: float) -> np.ndarray:
    _, probability = predict_raw_probability(
        model,
        frame,
        prior_mean=prior_mean,
        prior_strength=PRIOR_STRENGTH,
    )
    return probability


def metric_table(
    frame: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for name, probability in probabilities.items():
        metrics = evaluate_probability(frame, probability)
        rows.append(
            {
                "model": name,
                "brier": metrics.brier,
                "log_loss": metrics.log_loss,
                "rows": metrics.rows,
                "races": metrics.races,
            }
        )
    return pd.DataFrame(rows).sort_values(["brier", "log_loss", "model"]).reset_index(drop=True)


def target_consistent_probability(probability: np.ndarray, n: int) -> np.ndarray:
    frame = pd.DataFrame({"race_id": [TARGET_RACE_ID] * n})
    return enforce_race_top3_sum(frame, np.asarray(probability, dtype=float))


def markdown_table(frame: pd.DataFrame, digits: int = 6) -> str:
    show = frame.copy()
    for col in show.select_dtypes(include=["number"]).columns:
        show[col] = show[col].map(
            lambda value: f"{value:.{digits}f}" if pd.notna(value) else ""
        )
    return show.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--target-market", type=Path, required=True)
    parser.add_argument("--target-nonmarket", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_sha = sha256_file(args.results_csv)
    if results_sha != EXPECTED_RESULTS_SHA256:
        raise ValueError(
            f"Unexpected keiba_results.csv SHA256: {results_sha}; "
            f"expected {EXPECTED_RESULTS_SHA256}"
        )

    train = load_split(args.historical_dir, "train")
    validation = load_split(args.historical_dir, "validation")
    test = load_split(args.historical_dir, "test")

    val_2023 = validation.loc[validation["race_date"].dt.year.eq(2023)].copy()
    val_2024 = validation.loc[validation["race_date"].dt.year.eq(2024)].copy()
    if val_2023.empty or val_2024.empty:
        raise ValueError("Validation split must contain both 2023 and 2024")

    validation_model, validation_prior = fit_model(
        train,
        c_value=SELECTED_C,
        prior_strength=PRIOR_STRENGTH,
    )
    p_validation = predict_nonmarket(validation_model, validation, validation_prior)

    fit_2016_2024 = pd.concat([train, validation], ignore_index=True)
    production_model, production_prior = fit_model(
        fit_2016_2024,
        c_value=SELECTED_C,
        prior_strength=PRIOR_STRENGTH,
    )
    p_test = predict_nonmarket(production_model, test, production_prior)

    requested_race_ids = set(validation["race_id"]).union(set(test["race_id"]))
    market, market_diagnostics = reconstruct_historical_market(
        args.results_csv,
        requested_race_ids,
    )

    validation_with_pred = validation.copy()
    validation_with_pred["p_nonmarket_raw"] = p_validation
    test_with_pred = test.copy()
    test_with_pred["p_nonmarket_raw"] = p_test

    paired_validation = validation_with_pred.merge(
        market,
        on=["race_id", "horse_no"],
        how="inner",
        validate="many_to_one",
    )
    paired_test = test_with_pred.merge(
        market,
        on=["race_id", "horse_no"],
        how="inner",
        validate="many_to_one",
    )
    paired_2023 = paired_validation.loc[
        paired_validation["race_date"].dt.year.eq(2023)
    ].copy()
    paired_2024 = paired_validation.loc[
        paired_validation["race_date"].dt.year.eq(2024)
    ].copy()
    if paired_2023.empty or paired_2024.empty or paired_test.empty:
        raise ValueError("Insufficient paired historical market rows")

    market_cal_2023 = fit_logit_calibrator(
        paired_2023["market_p_top3"].to_numpy(),
        paired_2023["top3_label"].to_numpy(),
    )
    nonmarket_cal_2023 = fit_logit_calibrator(
        paired_2023["p_nonmarket_raw"].to_numpy(),
        paired_2023["top3_label"].to_numpy(),
    )
    market_2024_cal = apply_logit_calibrator(
        paired_2024["market_p_top3"].to_numpy(),
        market_cal_2023,
    )
    nonmarket_2024_cal = apply_logit_calibrator(
        paired_2024["p_nonmarket_raw"].to_numpy(),
        nonmarket_cal_2023,
    )

    selected_weight, blend_grid = select_blend_weight(
        paired_2024,
        market_2024_cal,
        nonmarket_2024_cal,
    )

    selection_metrics = metric_table(
        paired_2024,
        {
            "market_raw": paired_2024["market_p_top3"].to_numpy(),
            "market_calibrated": market_2024_cal,
            "nonmarket_raw": paired_2024["p_nonmarket_raw"].to_numpy(),
            "nonmarket_calibrated": nonmarket_2024_cal,
            "selected_blend": convex_blend(
                market_2024_cal,
                nonmarket_2024_cal,
                selected_weight,
            ),
        },
    )

    market_cal_final = fit_logit_calibrator(
        paired_validation["market_p_top3"].to_numpy(),
        paired_validation["top3_label"].to_numpy(),
    )
    nonmarket_cal_final = fit_logit_calibrator(
        paired_validation["p_nonmarket_raw"].to_numpy(),
        paired_validation["top3_label"].to_numpy(),
    )

    market_test_cal = apply_logit_calibrator(
        paired_test["market_p_top3"].to_numpy(),
        market_cal_final,
    )
    nonmarket_test_cal = apply_logit_calibrator(
        paired_test["p_nonmarket_raw"].to_numpy(),
        nonmarket_cal_final,
    )
    ensemble_test = convex_blend(
        market_test_cal,
        nonmarket_test_cal,
        selected_weight,
    )

    test_metrics = metric_table(
        paired_test,
        {
            "market_raw": paired_test["market_p_top3"].to_numpy(),
            "market_calibrated": market_test_cal,
            "nonmarket_raw": paired_test["p_nonmarket_raw"].to_numpy(),
            "nonmarket_calibrated": nonmarket_test_cal,
            "selected_blend": ensemble_test,
        },
    )

    target_market = pd.read_csv(args.target_market)
    target_nonmarket = pd.read_csv(args.target_nonmarket)
    target = target_market[
        ["horse_no", "horse_name", "p_top3_winmarket_harville"]
    ].merge(
        target_nonmarket[
            ["horse_no", "horse_name", "raw_probability", "p_top3"]
        ],
        on=["horse_no", "horse_name"],
        how="inner",
        validate="one_to_one",
    )
    if len(target) != 18 or target["horse_no"].nunique() != 18:
        raise ValueError("Target market/non-market merge must contain exactly 18 runners")

    target["p_market_stage2"] = target["p_top3_winmarket_harville"]
    target["p_nonmarket_stage4"] = target["p_top3"]
    target["p_market_calibrated_raw"] = apply_logit_calibrator(
        target["p_market_stage2"].to_numpy(),
        market_cal_final,
    )
    target["p_nonmarket_calibrated_raw"] = apply_logit_calibrator(
        target["raw_probability"].to_numpy(),
        nonmarket_cal_final,
    )
    target_blend_raw = convex_blend(
        target["p_market_calibrated_raw"].to_numpy(),
        target["p_nonmarket_calibrated_raw"].to_numpy(),
        selected_weight,
    )
    target["p_ensemble"] = target_consistent_probability(target_blend_raw, len(target))

    sensitivity_probabilities = []
    sensitivity_weights = nearby_weights(selected_weight, radius=0.10)
    for weight in sensitivity_weights:
        raw = convex_blend(
            target["p_market_calibrated_raw"].to_numpy(),
            target["p_nonmarket_calibrated_raw"].to_numpy(),
            weight,
        )
        sensitivity_probabilities.append(
            target_consistent_probability(raw, len(target))
        )
    sensitivity_matrix = np.column_stack(sensitivity_probabilities)
    target["uncertainty_low"] = sensitivity_matrix.min(axis=1)
    target["uncertainty_high"] = sensitivity_matrix.max(axis=1)
    target["component_abs_gap"] = np.abs(
        target["p_market_stage2"] - target["p_nonmarket_stage4"]
    )
    target["rank"] = (
        target["p_ensemble"].rank(method="first", ascending=False).astype(int)
    )
    target["as_of_time"] = TARGET_MARKET_AS_OF
    target["model_version"] = "stage5-ensemble-v1"
    target["selected_nonmarket_weight"] = selected_weight
    target = target.sort_values("rank").reset_index(drop=True)

    if not np.isclose(target["p_ensemble"].sum(), 3.0, atol=1e-10):
        raise ValueError("Target ensemble does not sum to 3")

    validation_market_races = int(paired_validation["race_id"].nunique())
    test_market_races = int(paired_test["race_id"].nunique())
    coverage = pd.DataFrame(
        [
            {
                "period": "2023",
                "eligible_races": int(val_2023["race_id"].nunique()),
                "paired_market_races": int(paired_2023["race_id"].nunique()),
                "eligible_rows": len(val_2023),
                "paired_rows": len(paired_2023),
            },
            {
                "period": "2024",
                "eligible_races": int(val_2024["race_id"].nunique()),
                "paired_market_races": int(paired_2024["race_id"].nunique()),
                "eligible_rows": len(val_2024),
                "paired_rows": len(paired_2024),
            },
            {
                "period": "2025",
                "eligible_races": int(test["race_id"].nunique()),
                "paired_market_races": test_market_races,
                "eligible_rows": len(test),
                "paired_rows": len(paired_test),
            },
        ]
    )
    coverage["race_coverage"] = (
        coverage["paired_market_races"] / coverage["eligible_races"]
    )
    coverage["row_coverage"] = coverage["paired_rows"] / coverage["eligible_rows"]

    target_columns = [
        "rank",
        "horse_no",
        "horse_name",
        "p_market_stage2",
        "p_nonmarket_stage4",
        "p_market_calibrated_raw",
        "p_nonmarket_calibrated_raw",
        "p_ensemble",
        "uncertainty_low",
        "uncertainty_high",
        "component_abs_gap",
        "selected_nonmarket_weight",
        "as_of_time",
        "model_version",
    ]
    target[target_columns].to_csv(
        args.output_dir / "stage5_ensemble.csv",
        index=False,
        float_format="%.9f",
    )
    blend_grid.to_csv(args.output_dir / "blend_grid_2024.csv", index=False)
    selection_metrics.to_csv(args.output_dir / "selection_metrics_2024.csv", index=False)
    test_metrics.to_csv(args.output_dir / "test_metrics_2025.csv", index=False)
    coverage.to_csv(args.output_dir / "historical_market_coverage.csv", index=False)

    metrics = {
        "status": "PASS",
        "source_results_sha256": results_sha,
        "market_source_semantics": "historical final win odds",
        "target_market_as_of": TARGET_MARKET_AS_OF,
        "stage4_spec": {"cohort": "turf_1200", "C": SELECTED_C},
        "calibration_fit_2023": {
            "market": market_cal_2023.__dict__,
            "nonmarket": nonmarket_cal_2023.__dict__,
            "rows": len(paired_2023),
            "races": int(paired_2023["race_id"].nunique()),
        },
        "blend_selection_2024": {
            "selected_nonmarket_weight": selected_weight,
            "selected_market_weight": 1.0 - selected_weight,
            "rows": len(paired_2024),
            "races": int(paired_2024["race_id"].nunique()),
            "metrics": selection_metrics.to_dict(orient="records"),
        },
        "final_calibration_fit_2023_2024": {
            "market": market_cal_final.__dict__,
            "nonmarket": nonmarket_cal_final.__dict__,
            "rows": len(paired_validation),
            "races": validation_market_races,
        },
        "test_2025": {
            "rows": len(paired_test),
            "races": test_market_races,
            "metrics": test_metrics.to_dict(orient="records"),
        },
        "historical_market_diagnostics": market_diagnostics,
        "coverage": coverage.to_dict(orient="records"),
        "target": {
            "runners": len(target),
            "sum_p_top3": float(target["p_ensemble"].sum()),
            "sensitivity_weights": sensitivity_weights,
            "target_outcome_loaded": False,
            "final_target_odds_loaded": False,
            "market_snapshot_replaced": False,
        },
        "transport_limit": (
            "Historical source odds are final win odds while target market is the frozen "
            "08:35 snapshot; historical blend performance is not perfectly time-matched."
        ),
    }
    (args.output_dir / "stage5_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    selected_2024 = selection_metrics.loc[
        selection_metrics["model"].eq("selected_blend")
    ].iloc[0]
    selected_2025 = test_metrics.loc[
        test_metrics["model"].eq("selected_blend")
    ].iloc[0]
    market_2025 = test_metrics.loc[
        test_metrics["model"].eq("market_raw")
    ].iloc[0]
    nonmarket_2025 = test_metrics.loc[
        test_metrics["model"].eq("nonmarket_raw")
    ].iloc[0]

    report = [
        "# Stage 5 — Calibration / Ensemble",
        "",
        "Status: **PASS — historical market comparison, calibration, blend selection, and 2025 test completed**",
        "",
        "## Timing",
        "",
        "This execution occurred after the scheduled start of the 2026-10-03 Kyoto 11R.",
        "It is a blind retrospective reconstruction from inputs frozen before the race, not a",
        "pre-start prediction lock. No target outcome or later/final target odds are loaded.",
        "",
        "## Historical market reconstruction",
        "",
        f"- exact keiba_results.csv SHA256: `{results_sha}`",
        "- market input: historical win odds from the frozen Kaggle v1 source",
        "- reciprocal odds normalized on every starter in each complete race",
        "- top-3 marginal: same Harville / Plackett-Luce method as Stage 2",
        "- historical market probabilities are constructed before Phase-A runner filtering",
        "",
        markdown_table(coverage),
        "",
        "## Calibration and selection protocol",
        "",
        "- 2023: fit logit intercept/slope calibration separately for market and non-market",
        "- 2024: choose convex blend weight on paired rows only",
        "- candidate non-market weight: 0.00 to 1.00 by 0.05",
        "- selection: Brier, then log loss, then smaller non-market weight",
        "- after weight freeze: refit calibration parameters on 2023-2024",
        "- 2025: one-time held-out Stage 5 test; no weight tuning",
        "",
        "### 2023 calibration parameters used for 2024 selection",
        "",
        f"- market: intercept `{market_cal_2023.intercept:.6f}`, slope `{market_cal_2023.slope:.6f}`",
        f"- non-market: intercept `{nonmarket_cal_2023.intercept:.6f}`, slope `{nonmarket_cal_2023.slope:.6f}`",
        "",
        "## 2024 selection result",
        "",
        f"- selected market weight: `{1.0 - selected_weight:.2f}`",
        f"- selected non-market weight: `{selected_weight:.2f}`",
        f"- selected blend Brier: `{selected_2024['brier']:.6f}`",
        f"- selected blend log loss: `{selected_2024['log_loss']:.6f}`",
        "",
        markdown_table(selection_metrics),
        "",
        "## 2025 held-out test",
        "",
        f"- paired races: {test_market_races:,}",
        f"- paired rows: {len(paired_test):,}",
        f"- raw market Brier / log loss: `{market_2025['brier']:.6f}` / `{market_2025['log_loss']:.6f}`",
        f"- raw non-market Brier / log loss: `{nonmarket_2025['brier']:.6f}` / `{nonmarket_2025['log_loss']:.6f}`",
        f"- selected ensemble Brier / log loss: `{selected_2025['brier']:.6f}` / `{selected_2025['log_loss']:.6f}`",
        "",
        markdown_table(test_metrics),
        "",
        "## Target 18-runner Stage 5 output",
        "",
        f"Exact probability-sum check: **{target['p_ensemble'].sum():.12f}**.",
        "",
        markdown_table(
            target[
                [
                    "rank",
                    "horse_no",
                    "horse_name",
                    "p_market_stage2",
                    "p_nonmarket_stage4",
                    "p_ensemble",
                    "uncertainty_low",
                    "uncertainty_high",
                    "component_abs_gap",
                ]
            ]
        ),
        "",
        "The uncertainty range is a model-sensitivity range across the selected blend, +/-0.10",
        "non-market weight, and both component endpoints. It is not a confidence interval.",
        "",
        "## Critical transport limitation",
        "",
        "The historical Kaggle odds are effectively final historical win odds, but the target",
        "market vector is the frozen **08:35** snapshot. Therefore historical test performance",
        "measures blending against a mature market and is not perfectly time-matched to the target.",
        "No later target odds are substituted to remove this mismatch.",
        "",
        "## Stage boundary",
        "",
        "Stage 5 compares/calibrates/ensembles the frozen Stage 2 and Stage 4 signals.",
        "It does not create a pre-start lock retroactively. Stage 6 must preserve that distinction.",
    ]
    (args.output_dir / "STAGE5_ENSEMBLE.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
