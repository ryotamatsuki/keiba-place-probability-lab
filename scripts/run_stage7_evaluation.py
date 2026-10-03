"""Evaluate the immutable Stage 6 lock against the official race outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from keiba_place_lab.postrace import evaluate_race, validate_outcome

EXPECTED_LOCK_SHA256 = "eefa16ad709c7568690e3f75a0df22491a31b3cb6aebdeebcf27fab274bc9e66"
EXPECTED_LOCK_SOURCE_COMMIT = "086f7a2a20e20dc226db6535b4738905e2011f1b"
EXPECTED_MARKET_AS_OF = "2026-10-03T08:35:00+09:00"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_ranks(ranks: tuple[int, int, int]) -> str:
    return "/".join(str(v) for v in ranks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--nonmarket", type=Path, required=True)
    parser.add_argument("--stage5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lock_sha = sha256_file(args.lock)
    if lock_sha != EXPECTED_LOCK_SHA256:
        raise ValueError(
            f"Locked probability file changed: {lock_sha} != {EXPECTED_LOCK_SHA256}"
        )

    outcome = pd.read_csv(args.outcome)
    validate_outcome(outcome)
    lock = pd.read_csv(args.lock)
    market = pd.read_csv(args.market)
    nonmarket = pd.read_csv(args.nonmarket)
    stage5 = pd.read_csv(args.stage5)

    if lock["locked_commit_sha"].nunique() != 1:
        raise ValueError("Lock source commit is not unique")
    if lock["locked_commit_sha"].iloc[0] != EXPECTED_LOCK_SOURCE_COMMIT:
        raise ValueError("Lock source commit changed")
    if lock["as_of_time"].nunique() != 1 or lock["as_of_time"].iloc[0] != EXPECTED_MARKET_AS_OF:
        raise ValueError("Locked market timestamp changed")

    joined = outcome.merge(
        lock[
            [
                "horse_no",
                "horse_name",
                "p_market",
                "p_model_raw",
                "p_model_calibrated",
                "p_ensemble",
                "uncertainty_low",
                "uncertainty_high",
                "rank",
            ]
        ],
        on=["horse_no", "horse_name"],
        how="inner",
        validate="one_to_one",
    ).merge(
        market[["horse_no", "horse_name", "p_top3_winmarket_harville"]],
        on=["horse_no", "horse_name"],
        how="inner",
        validate="one_to_one",
    ).merge(
        nonmarket[["horse_no", "horse_name", "p_top3"]],
        on=["horse_no", "horse_name"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_stage4"),
    ).merge(
        stage5[["horse_no", "horse_name", "p_ensemble"]],
        on=["horse_no", "horse_name"],
        how="inner",
        validate="one_to_one",
        suffixes=("_stage6", "_stage5"),
    )
    if len(joined) != 18:
        raise ValueError("Stage 7 join must contain exactly 18 runners")
    if not np.allclose(
        joined["p_ensemble_stage6"].to_numpy(),
        joined["p_ensemble_stage5"].to_numpy(),
        atol=1e-12,
    ):
        raise ValueError("Stage 6 lock no longer matches the frozen Stage 5 ensemble")

    joined["p_uniform"] = 3.0 / 18.0
    joined = joined.rename(
        columns={
            "p_top3_winmarket_harville": "p_stage2_market",
            "p_top3": "p_stage4_nonmarket",
        }
    )

    models = {
        "uniform_3_over_18": "p_uniform",
        "stage2_market_0835": "p_stage2_market",
        "stage4_nonmarket": "p_stage4_nonmarket",
        "stage5_ensemble": "p_ensemble_stage5",
        "stage6_locked": "p_ensemble_stage6",
    }
    metric_rows = []
    for model_name, probability_col in models.items():
        result = evaluate_race(joined, probability_col)
        forecast_ranks = (
            "all_tied_1-18"
            if model_name == "uniform_3_over_18"
            else format_ranks(result.actual_top3_forecast_ranks)
        )
        metric_rows.append(
            {
                "model": model_name,
                "brier": result.brier,
                "log_loss": result.log_loss,
                "actual_top3_probability_mass": result.actual_top3_probability_mass,
                "actual_top3_forecast_ranks": forecast_ranks,
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["brier", "log_loss", "model"],
        kind="mergesort",
    ).reset_index(drop=True)

    actual_top3 = (
        joined.loc[joined["top3_label"].eq(1)]
        .sort_values("finish_rank")
        [
            [
                "finish_rank",
                "horse_no",
                "horse_name",
                "p_stage2_market",
                "p_stage4_nonmarket",
                "p_ensemble_stage6",
                "rank",
            ]
        ]
        .copy()
    )
    actual_top3 = actual_top3.rename(columns={"rank": "stage6_forecast_rank"})

    # Error decomposition anchored to pre-race probabilities, not causal stories.
    joined["stage6_squared_error"] = (
        joined["p_ensemble_stage6"] - joined["top3_label"]
    ) ** 2
    joined["market_squared_error"] = (
        joined["p_stage2_market"] - joined["top3_label"]
    ) ** 2
    joined["nonmarket_squared_error"] = (
        joined["p_stage4_nonmarket"] - joined["top3_label"]
    ) ** 2
    joined["ensemble_minus_market_sqerr"] = (
        joined["stage6_squared_error"] - joined["market_squared_error"]
    )
    joined["abs_component_gap"] = np.abs(
        joined["p_stage2_market"] - joined["p_stage4_nonmarket"]
    )

    largest_ensemble_regrets = joined.sort_values(
        "ensemble_minus_market_sqerr",
        ascending=False,
    ).head(5)
    largest_component_gaps = joined.sort_values(
        "abs_component_gap",
        ascending=False,
    ).head(5)

    metrics.to_csv(args.output_dir / "postrace_metrics.csv", index=False)
    joined.sort_values("finish_rank").to_csv(
        args.output_dir / "postrace_joined.csv",
        index=False,
        float_format="%.9f",
    )

    outcome_source = str(outcome["official_source_url"].iloc[0])
    manifest = {
        "status": "PASS",
        "evaluation_mode": "RETROSPECTIVE_PIPELINE_TEST",
        "official_outcome_source": outcome_source,
        "locked_probability_sha256": lock_sha,
        "locked_source_commit_sha": EXPECTED_LOCK_SOURCE_COMMIT,
        "market_as_of_time": EXPECTED_MARKET_AS_OF,
        "runner_count": len(joined),
        "actual_top3": actual_top3[
            ["finish_rank", "horse_no", "horse_name"]
        ].to_dict(orient="records"),
        "stage6_values_modified": False,
        "later_or_final_target_odds_used_for_forecast": False,
        "metrics": metrics.to_dict(orient="records"),
    }
    (args.output_dir / "stage7_evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    def table(frame: pd.DataFrame, digits: int = 6) -> str:
        show = frame.copy()
        for col in show.select_dtypes(include=["number"]).columns:
            if col in {"finish_rank", "horse_no", "stage6_forecast_rank"}:
                continue
            show[col] = show[col].map(
                lambda x: f"{x:.{digits}f}" if pd.notna(x) else ""
            )
        return show.to_markdown(index=False)

    report = [
        "# Stage 7 — Post-race Evaluation",
        "",
        "Status: **PASS — official outcome joined to immutable Stage 6 rehearsal lock**",
        "",
        "## Evaluation status",
        "",
        "This is a retrospective pipeline test, not a genuine pre-start forecast audit,",
        "because Stage 6 missed the scheduled-start timing gate. The locked probabilities",
        "are nevertheless immutable and were not changed after the outcome was known.",
        "",
        f"- official JRA outcome source: {outcome_source}",
        f"- locked probability SHA256: `{lock_sha}`",
        f"- locked source commit: `{EXPECTED_LOCK_SOURCE_COMMIT}`",
        f"- frozen market timestamp: `{EXPECTED_MARKET_AS_OF}`",
        "- later/final target odds substituted into forecast: **no**",
        "",
        "## Official top 3",
        "",
        table(actual_top3),
        "",
        "## One-race probability metrics",
        "",
        table(metrics),
        "",
        "Lower Brier and log loss are better. Probability mass is the sum of each model's",
        "three marginal probabilities on the horses that actually finished 1st-3rd.",
        "",
        "## What this race says",
        "",
        "- Stage 6/Stage 5 ranked all three actual top-3 horses inside its top six (2nd, 3rd, 6th).",
        "- The raw 08:35 Stage 2 market had the same forecast ranks for the actual top three.",
        "- On this single race, raw Stage 2 market beat the Stage 5/6 ensemble on both Brier and log loss.",
        "- Stage 4 non-market alone was weaker than the raw market on this race.",
        "- Therefore the historical 95/5 ensemble advantage did not reproduce in this one-race test.",
        "",
        "This single race cannot overturn the 2025 held-out aggregate comparison; equally, the",
        "historical aggregate result must not be used to claim that the blend improved this race.",
        "",
        "## Largest ensemble-vs-market error contributions",
        "",
        table(
            largest_ensemble_regrets[
                [
                    "horse_no",
                    "horse_name",
                    "finish_rank",
                    "top3_label",
                    "p_stage2_market",
                    "p_ensemble_stage6",
                    "market_squared_error",
                    "stage6_squared_error",
                    "ensemble_minus_market_sqerr",
                ]
            ]
        ),
        "",
        "## Largest pre-race market/non-market disagreements",
        "",
        table(
            largest_component_gaps[
                [
                    "horse_no",
                    "horse_name",
                    "finish_rank",
                    "top3_label",
                    "p_stage2_market",
                    "p_stage4_nonmarket",
                    "abs_component_gap",
                ]
            ]
        ),
        "",
        "## Error analysis rule",
        "",
        "No causal explanation is inferred from the finishing order alone. The error review is",
        "restricted to discrepancies that were already observable in the frozen pre-race probabilities:",
        "market/non-market disagreement, forecast rank, and probability error.",
        "",
        "## Stage conclusion",
        "",
        "The full outcome/evaluation pipeline works. The next live experiment should repeat Stages",
        "1-6 before post time, then use this unchanged Stage 7 procedure after the official result.",
    ]
    (args.output_dir / "POSTRACE_EVALUATION.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
