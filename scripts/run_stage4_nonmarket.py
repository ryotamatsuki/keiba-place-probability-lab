#!/usr/bin/env python3
"""Fit, validate, test, and score the frozen Stage 4 non-market baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from keiba_place_lab.nonmarket import (
    ALL_BLOCKS,
    canonicalize_target_context,
    complete_race_subset,
    enforce_race_top3_sum,
    evaluate,
    evaluate_binary,
    fit_model,
    predict_raw_probability,
    validate_market_free,
)

COHORTS = ("all_turf", "turf_sprint_1000_1400", "turf_1200")
C_VALUES = (0.1, 1.0, 10.0)
PRIOR_STRENGTH = 6.0
TARGET_RACE_ID = "2026-10-03_KYOTO_11R"


def load_split(base: Path, cohort: str, split: str) -> pd.DataFrame:
    path = base / f"phase_a_{cohort}_{split}_v1.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    validate_market_free(frame.columns)
    return frame


def predict_adjusted(model, frame, prior_mean):
    raw_logit, raw_probability = predict_raw_probability(
        model,
        frame,
        prior_mean=prior_mean,
        prior_strength=PRIOR_STRENGTH,
    )
    adjusted = enforce_race_top3_sum(frame, raw_probability)
    return raw_logit, raw_probability, adjusted


def metric_row(cohort: str, c_value: float, ev) -> dict:
    return {
        "training_cohort": cohort,
        "C": float(c_value),
        "brier": ev.brier,
        "log_loss": ev.log_loss,
        "rows": ev.rows,
        "races": ev.races,
    }


def top_coefficients(model, limit: int = 12) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    preprocess = model.named_steps["preprocess"]
    names = preprocess.get_feature_names_out()
    coef = model.named_steps["model"].coef_[0]
    pairs = sorted(zip(names, coef, strict=True), key=lambda item: item[1])
    negative = [(str(n), float(v)) for n, v in pairs[:limit]]
    positive = [(str(n), float(v)) for n, v in pairs[-limit:][::-1]]
    return positive, negative


def fmt_table(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    show = frame[columns].copy()
    for col in show.select_dtypes(include=["number"]).columns:
        show[col] = show[col].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "")
    return show.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = canonicalize_target_context(pd.read_csv(args.target))
    validate_market_free(target.columns)
    if len(target) != 18 or target["horse_no"].nunique() != 18:
        raise ValueError("Canonical target matrix must contain exactly 18 unique runners")
    target = target.copy()
    target["race_id"] = TARGET_RACE_ID

    validation_eval = load_split(args.historical_dir, "turf_1200", "validation")
    test_eval = load_split(args.historical_dir, "turf_1200", "test")

    grid_rows: list[dict] = []
    train_cache: dict[str, pd.DataFrame] = {}
    validation_cache: dict[str, pd.DataFrame] = {}
    for cohort in COHORTS:
        train = load_split(args.historical_dir, cohort, "train")
        train_cache[cohort] = train
        validation_cache[cohort] = load_split(args.historical_dir, cohort, "validation")
        for c_value in C_VALUES:
            model, prior_mean = fit_model(
                train,
                c_value=c_value,
                prior_strength=PRIOR_STRENGTH,
            )
            _, raw_probability, _ = predict_adjusted(model, validation_eval, prior_mean)
            grid_rows.append(
                metric_row(
                    cohort,
                    c_value,
                    evaluate_binary(validation_eval, raw_probability),
                )
            )

    grid = pd.DataFrame(grid_rows).sort_values(
        ["brier", "log_loss", "C", "training_cohort"], kind="mergesort"
    ).reset_index(drop=True)
    selected = grid.iloc[0]
    selected_cohort = str(selected["training_cohort"])
    selected_c = float(selected["C"])

    sensitivity_rows: list[dict] = []
    selected_train = train_cache[selected_cohort]
    for excluded in ALL_BLOCKS:
        model, prior_mean = fit_model(
            selected_train,
            c_value=selected_c,
            prior_strength=PRIOR_STRENGTH,
            excluded_blocks=[excluded],
        )
        _, raw_probability, _ = predict_adjusted(model, validation_eval, prior_mean)
        ev = evaluate_binary(validation_eval, raw_probability)
        sensitivity_rows.append(
            {
                "excluded_block": excluded,
                "validation_brier": ev.brier,
                "validation_log_loss": ev.log_loss,
                "delta_brier_vs_primary": ev.brier - float(selected["brier"]),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows).sort_values("validation_brier")

    # Freeze the selected specification, then refit on 2016-2024 only.
    fit_2016_2024 = pd.concat(
        [selected_train, validation_cache[selected_cohort]], ignore_index=True
    )
    final_model, final_prior_mean = fit_model(
        fit_2016_2024,
        c_value=selected_c,
        prior_strength=PRIOR_STRENGTH,
    )

    # Held-out 2025 historical test evaluation. The Phase-A panel is runner-filtered,
    # so marginal probabilities are scored on all eligible rows without forcing race sums.
    _, test_raw_probability, _ = predict_adjusted(
        final_model, test_eval, final_prior_mean
    )
    test_metrics = evaluate_binary(test_eval, test_raw_probability)

    # Separate race-consistency diagnostic on races where every starter remains in
    # the eligibility-filtered panel. This filter uses field_size, never outcomes.
    test_complete = complete_race_subset(test_eval)
    _, _, test_complete_adjusted = predict_adjusted(
        final_model, test_complete, final_prior_mean
    )
    test_complete_metrics = evaluate(test_complete, test_complete_adjusted)

    raw_logit, raw_probability, target_adjusted = predict_adjusted(
        final_model, target, final_prior_mean
    )
    result = target[["horse_no", "horse_name"]].copy()
    result["raw_logit"] = raw_logit
    result["raw_probability"] = raw_probability
    result["p_top3"] = target_adjusted
    result["rank"] = result["p_top3"].rank(method="first", ascending=False).astype(int)

    # Target sensitivity uses the selected cohort/C and 2016-2024 fitting data only.
    p_variants = {"primary": target_adjusted}
    rank_variants = {"primary": result["rank"].to_numpy()}
    for excluded in ALL_BLOCKS:
        model, prior_mean = fit_model(
            fit_2016_2024,
            c_value=selected_c,
            prior_strength=PRIOR_STRENGTH,
            excluded_blocks=[excluded],
        )
        _, _, p = predict_adjusted(model, target, prior_mean)
        p_variants[f"drop_{excluded}"] = p
        rank_variants[f"drop_{excluded}"] = (
            pd.Series(p).rank(method="first", ascending=False).astype(int).to_numpy()
        )

    p_matrix = np.column_stack(list(p_variants.values()))
    rank_matrix = np.column_stack(list(rank_variants.values()))
    result["sensitivity_p_min"] = p_matrix.min(axis=1)
    result["sensitivity_p_max"] = p_matrix.max(axis=1)
    result["sensitivity_rank_best"] = rank_matrix.min(axis=1)
    result["sensitivity_rank_worst"] = rank_matrix.max(axis=1)
    result = result.sort_values("rank").reset_index(drop=True)

    positive, negative = top_coefficients(final_model)

    grid.to_csv(args.output_dir / "validation_grid.csv", index=False)
    sensitivity.to_csv(args.output_dir / "sensitivity.csv", index=False)
    result.to_csv(args.output_dir / "nonmarket_baseline.csv", index=False, float_format="%.9f")

    metrics = {
        "status": "PASS",
        "stage36_panel_sha256": "cee9ae9a099f521f12b1bcdd371c25a3d7b9ba3555fbcf5a46f2c9098f59a33d",
        "selection": {
            "training_cohort": selected_cohort,
            "C": selected_c,
            "metric_basis": "raw marginal P(top3) on all eligible 2023-2024 turf-1200 rows",
            "validation_brier": float(selected["brier"]),
            "validation_log_loss": float(selected["log_loss"]),
            "validation_rows": int(selected["rows"]),
            "validation_races": int(selected["races"]),
        },
        "test_2025": {
            **test_metrics.__dict__,
            "metric_basis": "raw marginal P(top3) on all eligible turf-1200 rows",
        },
        "race_consistency_diagnostic_2025_complete_races": test_complete_metrics.__dict__,
        "production_fit": {
            "period": "2016-2024",
            "rows": len(fit_2016_2024),
            "races": int(fit_2016_2024["race_id"].nunique()),
            "prior_mean": float(final_prior_mean),
            "prior_strength": PRIOR_STRENGTH,
            "uses_2025_outcomes": False,
        },
        "target": {
            "runners": len(result),
            "sum_p_top3": float(result["p_top3"].sum()),
            "racecourse_after_canonicalization": str(target["racecourse"].iloc[0]),
            "race_class_after_canonicalization": str(target["race_class"].iloc[0]),
            "market_columns_loaded": False,
            "target_outcome_loaded": False,
            "generated_after_scheduled_start": True,
        },
        "qa_history": {
            "discarded_run": 1,
            "reason": "run 1 incorrectly imposed sum P(top3)=3 on historically runner-filtered partial race cohorts",
            "test_performance_used_for_model_selection": False,
        },
        "top_positive_coefficients": positive,
        "top_negative_coefficients": negative,
    }
    (args.output_dir / "stage4_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Stage 4 Non-market P(top3) Baseline",
        "",
        "Status: **PASS — historical fit / validation / held-out-2025 test completed**",
        "",
        "## Timing and interpretation",
        "",
        "This Stage 4 run was generated after the scheduled start time of the 2026-10-03 Kyoto 11R.",
        "It is therefore a **blind retrospective reconstruction from the frozen pre-race matrix**, not a",
        "claim that this Stage 4 probability table was locked before the start. No target outcome, final",
        "odds, popularity, payout, or Stage 2 market probability is loaded by the Stage 4 code.",
        "",
        "## Frozen model selection",
        "",
        f"- selected training cohort: `{selected_cohort}`",
        f"- selected L2 logistic C: `{selected_c:g}`",
        f"- validation set for every candidate: 2023-2024 turf 1200m ({int(selected['races']):,} races / {int(selected['rows']):,} rows)",
        f"- validation marginal Brier: `{float(selected['brier']):.6f}`",
        f"- validation marginal log loss: `{float(selected['log_loss']):.6f}`",
        f"- empirical-Bayes prior: fitting-sample top3 prevalence, strength `{PRIOR_STRENGTH:g}`",
        "- missing values: fitting-sample numeric median + indicators; categorical `UNKNOWN`",
        "- historical validation/test metric: raw marginal P(top3), because Phase-A filters runner rows",
        "- target transform: common logit intercept with exact `sum P(top3)=3`",
        "- target category aliases: `Kyoto -> 京都`, `Listed_open -> Open`",
        "",
        "Candidate models were trained on different predeclared cohorts but evaluated on the same",
        "target-like validation population. 2025 was not used for cohort, feature-block, or C selection.",
        "Run #1 was discarded at QA because it imposed a three-slot race constraint on partial historical",
        "race cohorts after the career-start eligibility filter. The correction is structural and does not",
        "use 2025 performance to choose the model.",
        "",
        "## Validation grid",
        "",
        fmt_table(grid, ["training_cohort", "C", "brier", "log_loss"]),
        "",
        "## Held-out 2025 test",
        "",
        "After selection, the chosen specification was refit on its 2016-2024 train+validation cohort",
        "and evaluated once on the frozen 2025 turf-1200 test set.",
        "",
        f"- races: {test_metrics.races:,}",
        f"- rows: {test_metrics.rows:,}",
        f"- marginal Brier: `{test_metrics.brier:.6f}`",
        f"- marginal log loss: `{test_metrics.log_loss:.6f}`",
        "- 2025 outcomes used in target fit: **no**",
        "",
        "Race-sum QA is evaluated separately only on 2025 races where every starter survives the",
        "pre-race eligibility filter:",
        "",
        f"- complete races: {test_complete_metrics.races:,}",
        f"- complete-race rows: {test_complete_metrics.rows:,}",
        f"- adjusted Brier: `{test_complete_metrics.brier:.6f}`",
        f"- adjusted log loss: `{test_complete_metrics.log_loss:.6f}`",
        f"- mean race probability sum: `{test_complete_metrics.mean_race_sum:.12f}`",
        f"- max |race sum - 3|: `{test_complete_metrics.max_abs_race_sum_error:.3e}`",
        "",
        "## Sensitivity — leave one feature block out",
        "",
        fmt_table(sensitivity, ["excluded_block", "validation_brier", "validation_log_loss", "delta_brier_vs_primary"]),
        "",
        "These ablations are diagnostic. They do not change the primary model after the 2025 test is viewed.",
        "",
        "## 18-runner non-market output",
        "",
        f"Probability sum check: **{result['p_top3'].sum():.12f}**.",
        "",
        fmt_table(
            result,
            [
                "rank",
                "horse_no",
                "horse_name",
                "raw_logit",
                "raw_probability",
                "p_top3",
                "sensitivity_p_min",
                "sensitivity_p_max",
                "sensitivity_rank_best",
                "sensitivity_rank_worst",
            ],
        ),
        "",
        "`p_top3` is the Stage 4 non-market probability after race-level consistency adjustment.",
        "It is not blended with Stage 2 market information; that remains Stage 5 work.",
        "",
        "## Largest standardized model coefficients",
        "",
        "Positive coefficients increase the raw logistic top-3 logit, holding other encoded variables fixed.",
        "Negative coefficients decrease it. Coefficients are descriptive model parameters, not causal effects.",
        "",
        "### Positive",
        "",
        *[f"- `{name}`: {value:+.6f}" for name, value in positive],
        "",
        "### Negative",
        "",
        *[f"- `{name}`: {value:+.6f}" for name, value in negative],
        "",
        "## Reproducibility",
        "",
        "Run `scripts/run_stage4_nonmarket.py` against the frozen Stage 3.6 Actions artifact and",
        "`analysis/2026-10-03_kyoto11_opal/canonical_nonmarket_features_v1.csv`.",
        "The workflow `.github/workflows/stage4-nonmarket.yml` pins the Stage 3.6 artifact run.",
    ]
    (args.output_dir / "NONMARKET_BASELINE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
