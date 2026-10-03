"""Productionize the frozen Stage-4 successor v2 and score the frozen target snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from keiba_place_lab.nonmarket import validate_market_free
from keiba_place_lab.stage4_production_v2 import (
    fit_production_successor,
    prepare_target_context,
    score_production_target,
)

TARGET_RACE_ID = "2026-10-03_KYOTO_11R"
TARGET_DATE = "2026-10-03"
PANEL_SHA256 = "cee9ae9a099f521f12b1bcdd371c25a3d7b9ba3555fbcf5a46f2c9098f59a33d"
SUCCESSOR_SPEC = "XGB01 + full-field relative ability"


def load_parquet(base: Path, name: str) -> pd.DataFrame:
    path = base / name
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    validate_market_free(frame.columns)
    frame = frame.copy()
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="raise")
    frame["race_id"] = frame["race_id"].astype("string")
    if "horse_id" in frame.columns:
        frame["horse_id"] = frame["horse_id"].astype("string")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)

    eligible = pd.concat(
        [
            load_parquet(
                args.historical_dir,
                "phase_a_turf_1200_train_v1.parquet",
            ),
            load_parquet(
                args.historical_dir,
                "phase_a_turf_1200_validation_v1.parquet",
            ),
            load_parquet(
                args.historical_dir,
                "phase_a_turf_1200_test_v1.parquet",
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    context = pd.concat(
        [
            load_parquet(args.historical_dir, "jra_flat_train_v1.parquet"),
            load_parquet(args.historical_dir, "jra_flat_validation_v1.parquet"),
            load_parquet(args.historical_dir, "jra_flat_test_v1.parquet"),
        ],
        ignore_index=True,
        sort=False,
    )

    years = eligible["race_date"].dt.year
    if not years.between(2016, 2025).all():
        raise ValueError("Production eligible fit must contain only 2016-2025")
    if context["race_date"].dt.year.gt(2025).any():
        raise ValueError("Production context must not contain post-2025 rows")

    target_input = pd.read_csv(args.target)
    active_target, eligible_target = prepare_target_context(
        target_input,
        race_id=TARGET_RACE_ID,
        race_date=TARGET_DATE,
    )

    fit = fit_production_successor(
        eligible,
        context,
        target_date=TARGET_DATE,
    )
    result, coverage = score_production_target(
        fit,
        active_target,
        eligible_target,
    )

    result.to_csv(
        args.output_dir / "stage4_successor_v2.csv",
        index=False,
        float_format="%.9f",
    )
    coverage.to_csv(
        args.output_dir / "stage4_successor_v2_relative_coverage.csv",
        index=False,
    )

    joblib.dump(
        {
            "model": fit.model,
            "prior_mean": fit.prior_mean,
            "specification": SUCCESSOR_SPEC,
            "fit_date_min": fit.fit_date_min,
            "fit_date_max": fit.fit_date_max,
            "fit_rows": fit.fit_rows,
            "fit_races": fit.fit_races,
            "panel_sha256": PANEL_SHA256,
        },
        args.model_output,
    )

    raw_sum = float(result["p_top3_stage4_v2"].sum())
    active_count = len(active_target)
    eligible_count = len(eligible_target)
    complete_coverage = active_count == eligible_count

    manifest = {
        "status": "PASS",
        "mode": "RETROSPECTIVE_PRODUCTION_REHEARSAL",
        "successor_specification": SUCCESSOR_SPEC,
        "historical_panel_sha256": PANEL_SHA256,
        "production_fit": {
            "period": "2016-2025",
            "fit_date_min": fit.fit_date_min,
            "fit_date_max": fit.fit_date_max,
            "rows": fit.fit_rows,
            "races": fit.fit_races,
            "prior_mean": fit.prior_mean,
            "prior_strength": 6.0,
            "uses_2025_as_training_data": True,
            "uses_2025_as_untouched_test": False,
        },
        "target": {
            "race_id": TARGET_RACE_ID,
            "race_date": TARGET_DATE,
            "active_starters": active_count,
            "eligible_starters": eligible_count,
            "complete_stage4_coverage": complete_coverage,
            "declared_field_size": int(active_target["declared_field_size"].iloc[0]),
            "active_field_size": int(active_target["field_size"].iloc[0]),
            "raw_probability_sum": raw_sum,
            "sum_to_three_adjustment_applied": False,
            "market_columns_loaded": False,
            "target_outcome_loaded": False,
        },
        "scratch_contract": {
            "relative_peer_set": "active starters at lock",
            "draw_pct_denominator": "declared_field_size - 1",
            "horse_numbers_renumbered_after_scratch": False,
            "post_lock_scratch_mutates_lock": False,
        },
    }
    (args.output_dir / "stage4_successor_v2_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    display = result[
        [
            "rank_eligible",
            "horse_no",
            "horse_name",
            "p_top3_stage4_v2",
        ]
    ].copy()
    report = [
        "# Stage 4 Successor v2 — Production Rehearsal",
        "",
        "Status: **PASS — frozen successor refit through 2025 and target scoring completed**",
        "",
        "This is a retrospective production rehearsal using the already frozen 2026-10-03",
        "pre-race target matrix. It is not a genuine pre-start v2 lock and the target outcome is not",
        "loaded.",
        "",
        "## Frozen production specification",
        "",
        f"- model: **{SUCCESSOR_SPEC}**",
        "- XGB01 hyperparameters are unchanged from the development freeze",
        "- five full-field leave-one-out relative-ability features are included",
        "- Phase-3B recent-trend features: **not included**",
        "- Phase-3C explicit interactions: **not included**",
        "- post-hoc calibration: **none**",
        "- market information: **none**",
        "",
        "## Production refit",
        "",
        f"- fit period: {fit.fit_date_min} through {fit.fit_date_max}",
        f"- eligible rows: {fit.fit_rows:,}",
        f"- races: {fit.fit_races:,}",
        f"- production prior mean: {fit.prior_mean:.9f}",
        "- 2025 is used as historical training data, not as an untouched test",
        "",
        "## Target coverage",
        "",
        f"- active starters: {active_count}",
        f"- Stage-4 eligible starters: {eligible_count}",
        f"- complete Stage-4 coverage: **{complete_coverage}**",
        f"- declared field size: {int(active_target['declared_field_size'].iloc[0])}",
        f"- active field size: {int(active_target['field_size'].iloc[0])}",
        f"- raw marginal probability sum: **{raw_sum:.9f}**",
        "- canonical sum-to-three adjustment applied: **no**",
        "",
        "## Canonical Stage-4 v2 probabilities",
        "",
        display.to_markdown(index=False),
        "",
        "These are raw marginal probabilities. They are intentionally not transformed to sum to 3.",
        "",
        "## Scratch policy",
        "",
        "For a scratch before lock, remove the horse from the active relative-feature peer set and",
        "reduce active field_size, but preserve official horse numbers and declared_field_size.",
        "draw_pct remains based on the declared field. A scratch after Stage-6 lock does not mutate",
        "the locked prediction file.",
        "",
        "## Stage 5 boundary",
        "",
        "This production refit is not used to reconstruct 2023-2024 Stage-5 history. Stage 5 v2 must",
        "use out-of-time non-market predictions for each historical evaluation year and retain",
        "market-only as the mandatory incumbent/reference.",
    ]
    (args.output_dir / "STAGE4_SUCCESSOR_PRODUCTION_V2.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
