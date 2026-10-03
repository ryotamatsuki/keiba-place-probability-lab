#!/usr/bin/env python3
"""Run the pre-registered Stage 4 successor development experiment.

Selection uses only 2016-2024. The 2025 split is loaded only after the development
winner and production recipe have been frozen; it is then used for production refit,
never for candidate ranking or evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from keiba_place_lab.model_selection import candidate_score, select_with_incumbent_one_se
from keiba_place_lab.nonmarket import (
    calibration_table,
    canonicalize_target_context,
    enforce_race_top3_sum,
    expected_calibration_error,
    validate_market_free,
)
from keiba_place_lab.successor import (
    FEATURE_VARIANTS,
    MODEL_GRIDS,
    config_id,
    fit_forecaster,
    predict_probability,
)

OUTER_YEARS = (2023, 2024)
INNER_YEARS = {2023: (2021, 2022), 2024: (2022, 2023)}
PRODUCTION_TUNING_YEARS = (2023, 2024)
FAMILIES = ("logistic", "random_forest", "hist_gradient_boosting", "xgboost")
INCUMBENT = "incumbent_logistic"
TARGET_RACE_ID = "2026-10-03_KYOTO_11R_SUCCESSOR"


def load_split(base: Path, split: str) -> pd.DataFrame:
    path = base / f"phase_a_turf_1200_{split}_v1.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    validate_market_free(frame.columns)
    frame = frame.copy()
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="raise")
    frame["year"] = frame["race_date"].dt.year.astype(int)
    return frame.sort_values(
        ["race_date", "race_id", "horse_no"], kind="stable"
    ).reset_index(drop=True)


def evaluate_prediction(
    frame: pd.DataFrame, probability: np.ndarray, name: str
) -> dict[str, Any]:
    score = candidate_score(frame, probability, name=name)
    table = calibration_table(frame, probability, bins=10)
    ece = expected_calibration_error(table)
    y = frame["top3_label"].astype(int).to_numpy()
    p = np.clip(np.asarray(probability, dtype=float), 1e-9, 1.0 - 1e-9)
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    cal.fit(logit, y)
    return {
        **score.__dict__,
        "ece_10bin": float(ece),
        "calibration_intercept": float(cal.intercept_[0]),
        "calibration_slope": float(cal.coef_[0, 0]),
        "roc_auc": float(roc_auc_score(y, p)),
    }


def fit_predict_year(
    dev: pd.DataFrame,
    *,
    train_before_year: int,
    eval_year: int,
    family: str,
    config: dict[str, Any],
    feature_variant: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    train = dev.loc[dev["year"].lt(train_before_year)].copy()
    evaluation = dev.loc[dev["year"].eq(eval_year)].copy()
    if train.empty or evaluation.empty:
        raise ValueError(f"Empty temporal split for eval year {eval_year}")
    if train["race_date"].max() >= evaluation["race_date"].min():
        raise ValueError("Temporal leakage in successor split")
    fitted = fit_forecaster(
        train,
        family=family,
        config=config,
        feature_variant=feature_variant,
    )
    return evaluation, predict_probability(fitted, evaluation)


def oof_predictions(
    dev: pd.DataFrame,
    *,
    years: tuple[int, ...],
    family: str,
    config: dict[str, Any],
    feature_variant: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    frame_parts: list[pd.DataFrame] = []
    probability_parts: list[np.ndarray] = []
    for year in years:
        evaluation, p = fit_predict_year(
            dev,
            train_before_year=year,
            eval_year=year,
            family=family,
            config=config,
            feature_variant=feature_variant,
        )
        frame_parts.append(evaluation)
        probability_parts.append(p)
    frame = pd.concat(frame_parts, ignore_index=True)
    return frame, np.concatenate(probability_parts)


def tune_family(
    dev: pd.DataFrame,
    *,
    years: tuple[int, ...],
    family: str,
    feature_variant: str = "base",
) -> tuple[int, dict[str, Any], pd.DataFrame, pd.DataFrame, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    common_frame: pd.DataFrame | None = None
    rows: list[dict[str, Any]] = []
    configs = MODEL_GRIDS[family]
    for index, config in enumerate(configs):
        frame, p = oof_predictions(
            dev,
            years=years,
            family=family,
            config=config,
            feature_variant=feature_variant,
        )
        if common_frame is None:
            common_frame = frame
        elif not frame[["race_id", "horse_id"]].equals(
            common_frame[["race_id", "horse_id"]]
        ):
            raise ValueError("Inner candidate rows are not aligned")
        name = config_id(family, index)
        predictions[name] = p
        score = candidate_score(frame, p, name=name)
        rows.append(
            {
                **score.__dict__,
                "config_index": index,
                "config": json.dumps(config, sort_keys=True),
            }
        )
    assert common_frame is not None
    default_name = config_id(family, 0)
    decision, selection = select_with_incumbent_one_se(
        common_frame,
        predictions,
        incumbent=default_name,
    )
    chosen_name = decision.winner
    chosen_index = int(chosen_name.rsplit("__", 1)[1])
    table = pd.DataFrame(rows).merge(
        selection[
            [
                "name",
                "point_brier_best",
                "mean_brier_delta_vs_best",
                "se_brier_delta_vs_best",
                "within_one_se_brier",
            ]
        ],
        on="name",
        how="left",
        validate="one_to_one",
    )
    return (
        chosen_index,
        dict(configs[chosen_index]),
        table,
        common_frame,
        predictions[chosen_name],
    )


def choose_family(
    dev: pd.DataFrame,
    *,
    years: tuple[int, ...],
) -> tuple[str, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    family_predictions: dict[str, np.ndarray] = {}
    family_configs: dict[str, dict[str, Any]] = {}
    family_tables: list[pd.DataFrame] = []
    common_frame: pd.DataFrame | None = None
    for family in FAMILIES:
        index, config, tuning_table, frame, p = tune_family(
            dev,
            years=years,
            family=family,
            feature_variant="base",
        )
        tuning_table.insert(0, "family", family)
        tuning_table.insert(1, "chosen_config", tuning_table["config_index"].eq(index))
        family_tables.append(tuning_table)
        if common_frame is None:
            common_frame = frame
        elif not frame[["race_id", "horse_id"]].equals(
            common_frame[["race_id", "horse_id"]]
        ):
            raise ValueError("Inner family rows are not aligned")
        family_predictions[family] = p
        family_configs[family] = config
    assert common_frame is not None
    decision, family_selection = select_with_incumbent_one_se(
        common_frame,
        family_predictions,
        incumbent="logistic",
    )
    chosen_family = decision.winner
    family_selection["chosen_family"] = family_selection["name"].eq(chosen_family)
    all_tuning = pd.concat(family_tables, ignore_index=True)
    all_tuning["inner_years"] = ",".join(map(str, years))
    family_selection["inner_years"] = ",".join(map(str, years))
    return (
        chosen_family,
        family_configs[chosen_family],
        all_tuning,
        family_selection,
    )


def choose_feature_variant(
    dev: pd.DataFrame,
    *,
    years: tuple[int, ...],
    family: str,
    config: dict[str, Any],
) -> tuple[str, pd.DataFrame]:
    predictions: dict[str, np.ndarray] = {}
    common_frame: pd.DataFrame | None = None
    for variant in FEATURE_VARIANTS:
        frame, p = oof_predictions(
            dev,
            years=years,
            family=family,
            config=config,
            feature_variant=variant,
        )
        if common_frame is None:
            common_frame = frame
        elif not frame[["race_id", "horse_id"]].equals(
            common_frame[["race_id", "horse_id"]]
        ):
            raise ValueError("Inner feature rows are not aligned")
        predictions[variant] = p
    assert common_frame is not None
    decision, table = select_with_incumbent_one_se(
        common_frame,
        predictions,
        incumbent="base",
    )
    table["chosen_feature_variant"] = table["name"].eq(decision.winner)
    table["family"] = family
    table["inner_years"] = ",".join(map(str, years))
    return decision.winner, table


def subgroup_diagnostics(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    model_name: str,
) -> pd.DataFrame:
    work = frame.copy()
    p = np.asarray(probability, dtype=float)
    y = work["top3_label"].astype(float).to_numpy()
    naive = np.minimum(
        3.0 / pd.to_numeric(work["field_size"]).to_numpy(dtype=float),
        1.0,
    )
    work["_model_brier"] = (p - y) ** 2
    work["_naive_brier"] = (naive - y) ** 2
    work["_pred"] = p
    work["_y"] = y
    work["field_size_band"] = pd.cut(
        work["field_size"],
        bins=[0, 11, 14, 16, 99],
        labels=["<=11", "12-14", "15-16", "17-18"],
    ).astype(str)
    work["layoff_band"] = pd.cut(
        work["days_since_prev"],
        bins=[-np.inf, 21, 35, 56, 90, np.inf],
        labels=["<=21", "22-35", "36-56", "57-90", "91+"],
    ).astype(str)
    work["experience_band"] = pd.cut(
        work["career_starts"],
        bins=[-np.inf, 5, 10, 20, np.inf],
        labels=["3-5", "6-10", "11-20", "21+"],
    ).astype(str)
    work["year_group"] = work["race_date"].dt.year.astype(str)

    dimensions = {
        "field_size": "field_size_band",
        "race_class": "race_class",
        "racecourse": "racecourse",
        "layoff": "layoff_band",
        "career_starts": "experience_band",
        "year": "year_group",
    }
    rows: list[dict[str, Any]] = []
    for dimension, column in dimensions.items():
        for value, group in work.groupby(column, dropna=False, observed=True):
            race = group.groupby("race_id", sort=False).agg(
                model_brier=("_model_brier", "mean"),
                naive_brier=("_naive_brier", "mean"),
            )
            model_brier = float(race["model_brier"].mean())
            naive_brier = float(race["naive_brier"].mean())
            rows.append(
                {
                    "model": model_name,
                    "dimension": dimension,
                    "group": str(value),
                    "rows": len(group),
                    "races": int(group["race_id"].nunique()),
                    "race_macro_brier": model_brier,
                    "naive_race_macro_brier": naive_brier,
                    "brier_skill_vs_3_over_field": (
                        1.0 - model_brier / naive_brier if naive_brier > 0 else np.nan
                    ),
                    "mean_predicted": float(group["_pred"].mean()),
                    "observed_top3_rate": float(group["_y"].mean()),
                    "calibration_gap": float(
                        group["_y"].mean() - group["_pred"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def fmt_table(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    show = frame[columns].copy()
    for column in show.select_dtypes(include=["number"]).columns:
        show[column] = show[column].map(
            lambda value: f"{value:.{digits}f}" if pd.notna(value) else ""
        )
    return show.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(args.historical_dir, "train")
    validation = load_split(args.historical_dir, "validation")
    dev = pd.concat([train, validation], ignore_index=True).sort_values(
        ["race_date", "race_id", "horse_no"], kind="stable"
    ).reset_index(drop=True)
    if dev["year"].max() != 2024:
        raise ValueError("Development data must stop at 2024")

    outer_parts: list[pd.DataFrame] = []
    inner_tuning_parts: list[pd.DataFrame] = []
    family_selection_parts: list[pd.DataFrame] = []
    feature_selection_parts: list[pd.DataFrame] = []
    outer_choices: list[dict[str, Any]] = []

    for outer_year in OUTER_YEARS:
        inner_years = INNER_YEARS[outer_year]
        chosen_family, chosen_config, tuning_table, family_table = choose_family(
            dev,
            years=inner_years,
        )
        feature_variant, feature_table = choose_feature_variant(
            dev,
            years=inner_years,
            family=chosen_family,
            config=chosen_config,
        )
        tuning_table["outer_year"] = outer_year
        family_table["outer_year"] = outer_year
        feature_table["outer_year"] = outer_year
        inner_tuning_parts.append(tuning_table)
        family_selection_parts.append(family_table)
        feature_selection_parts.append(feature_table)

        outer = dev.loc[dev["year"].eq(outer_year)].copy()
        train_outer = dev.loc[dev["year"].lt(outer_year)].copy()
        if train_outer["race_date"].max() >= outer["race_date"].min():
            raise ValueError("Outer chronology failure")

        for family in FAMILIES:
            family_rows = tuning_table.loc[
                (tuning_table["family"] == family)
                & tuning_table["chosen_config"]
            ]
            if len(family_rows) != 1:
                raise ValueError(f"No unique tuned config for {family}")
            config_index = int(family_rows.iloc[0]["config_index"])
            config = dict(MODEL_GRIDS[family][config_index])
            fitted = fit_forecaster(
                train_outer,
                family=family,
                config=config,
                feature_variant="base",
            )
            column = INCUMBENT if family == "logistic" else f"{family}_base"
            outer[column] = predict_probability(fitted, outer)

        adaptive_fit = fit_forecaster(
            train_outer,
            family=chosen_family,
            config=chosen_config,
            feature_variant=feature_variant,
        )
        outer["adaptive_successor"] = predict_probability(adaptive_fit, outer)
        outer["outer_year"] = outer_year
        outer_parts.append(outer)
        outer_choices.append(
            {
                "outer_year": outer_year,
                "inner_years": list(inner_years),
                "adaptive_family": chosen_family,
                "adaptive_config": chosen_config,
                "adaptive_feature_variant": feature_variant,
                "outer_train_end": outer_year - 1,
            }
        )

    outer_all = pd.concat(outer_parts, ignore_index=True)
    candidate_columns = [
        INCUMBENT,
        "random_forest_base",
        "hist_gradient_boosting_base",
        "xgboost_base",
        "adaptive_successor",
    ]
    predictions = {
        name: outer_all[name].to_numpy(dtype=float)
        for name in candidate_columns
    }
    final_decision, selection_table = select_with_incumbent_one_se(
        outer_all,
        predictions,
        incumbent=INCUMBENT,
    )
    diagnostic_rows = [
        evaluate_prediction(outer_all, predictions[name], name)
        for name in candidate_columns
    ]
    diagnostics = pd.DataFrame(diagnostic_rows)
    selection_table = selection_table.merge(
        diagnostics[
            [
                "name",
                "ece_10bin",
                "calibration_intercept",
                "calibration_slope",
                "roc_auc",
            ]
        ],
        on="name",
        how="left",
        validate="one_to_one",
    )

    subgroup = subgroup_diagnostics(
        outer_all,
        predictions[INCUMBENT],
        model_name=INCUMBENT,
    )
    if final_decision.winner != INCUMBENT:
        subgroup = pd.concat(
            [
                subgroup,
                subgroup_diagnostics(
                    outer_all,
                    predictions[final_decision.winner],
                    model_name=final_decision.winner,
                ),
            ],
            ignore_index=True,
        )

    # Production recipe is finalized only after the outer 2023-2024 decision.
    if final_decision.winner == INCUMBENT:
        production_family = "logistic"
        production_config = dict(MODEL_GRIDS["logistic"][0])
        production_feature_variant = "base"
    elif final_decision.winner.endswith("_base"):
        production_family = final_decision.winner.removesuffix("_base")
        chosen_index, production_config, production_tuning, _, _ = tune_family(
            dev,
            years=PRODUCTION_TUNING_YEARS,
            family=production_family,
            feature_variant="base",
        )
        production_tuning.insert(
            0, "production_family", production_family
        )
        production_tuning["production_chosen"] = production_tuning[
            "config_index"
        ].eq(chosen_index)
        inner_tuning_parts.append(production_tuning)
        production_feature_variant = "base"
    else:
        (
            production_family,
            production_config,
            production_tuning,
            production_family_table,
        ) = choose_family(dev, years=PRODUCTION_TUNING_YEARS)
        production_feature_variant, production_feature_table = (
            choose_feature_variant(
                dev,
                years=PRODUCTION_TUNING_YEARS,
                family=production_family,
                config=production_config,
            )
        )
        production_tuning["outer_year"] = "production"
        production_family_table["outer_year"] = "production"
        production_feature_table["outer_year"] = "production"
        inner_tuning_parts.append(production_tuning)
        family_selection_parts.append(production_family_table)
        feature_selection_parts.append(production_feature_table)

    # Only now may 2025 be read, and only for production refit. No 2025 score is computed.
    test_2025 = load_split(args.historical_dir, "test")
    production_train = pd.concat(
        [dev, test_2025], ignore_index=True
    ).sort_values(["race_date", "race_id", "horse_no"], kind="stable")
    target = canonicalize_target_context(pd.read_csv(args.target))
    if len(target) != 18 or target["horse_no"].nunique() != 18:
        raise ValueError("Canonical target must contain 18 unique runners")
    target = target.copy()
    target["race_id"] = TARGET_RACE_ID
    production_fit = fit_forecaster(
        production_train,
        family=production_family,
        config=production_config,
        feature_variant=production_feature_variant,
    )
    target_raw = predict_probability(production_fit, target)
    target_adjusted = enforce_race_top3_sum(target, target_raw)
    target_output = target[["horse_no", "horse_name"]].copy()
    target_output["raw_probability"] = target_raw
    target_output["p_top3"] = target_adjusted
    target_output["rank"] = target_output["p_top3"].rank(
        method="first", ascending=False
    ).astype(int)
    target_output = target_output.sort_values("rank").reset_index(drop=True)

    inner_tuning = pd.concat(inner_tuning_parts, ignore_index=True)
    family_selection = pd.concat(family_selection_parts, ignore_index=True)
    feature_selection = pd.concat(feature_selection_parts, ignore_index=True)

    outer_export_cols = [
        "race_date",
        "race_id",
        "horse_id",
        "horse_no",
        "top3_label",
        "field_size",
        "racecourse",
        "race_class",
        "days_since_prev",
        "career_starts",
        *candidate_columns,
    ]
    outer_all[outer_export_cols].to_csv(
        args.output_dir / "outer_predictions_2023_2024.csv",
        index=False,
    )
    selection_table.to_csv(
        args.output_dir / "candidate_selection.csv", index=False
    )
    diagnostics.to_csv(
        args.output_dir / "candidate_diagnostics.csv", index=False
    )
    inner_tuning.to_csv(
        args.output_dir / "inner_hyperparameter_tuning.csv", index=False
    )
    family_selection.to_csv(
        args.output_dir / "inner_family_selection.csv", index=False
    )
    feature_selection.to_csv(
        args.output_dir / "inner_feature_selection.csv", index=False
    )
    subgroup.to_csv(
        args.output_dir / "subgroup_diagnostics.csv", index=False
    )
    target_output.to_csv(
        args.output_dir / "successor_target_probabilities.csv",
        index=False,
        float_format="%.9f",
    )

    metrics = {
        "status": "PASS",
        "selection_data_end": "2024-12-31",
        "outer_years": list(OUTER_YEARS),
        "inner_years": {
            str(k): list(v) for k, v in INNER_YEARS.items()
        },
        "winner_metric": "race-macro Brier",
        "winner_rule": (
            "paired race-date-clustered one-SE incumbent retention"
        ),
        "calibration_transform": (
            "none; calibration is diagnostic in this first "
            "family/feature experiment"
        ),
        "historical_race_sum_adjustment": (
            "none on runner-filtered partial historical cohorts"
        ),
        "incumbent": INCUMBENT,
        "decision": final_decision.__dict__,
        "outer_choices": outer_choices,
        "production_recipe": {
            "family": production_family,
            "config": production_config,
            "feature_variant": production_feature_variant,
            "fit_period": (
                "2016-2025 after development winner freeze"
            ),
            "2025_used_for_selection": False,
            "2025_evaluation_performed": False,
        },
        "target": {
            "runners": len(target_output),
            "sum_p_top3": float(target_output["p_top3"].sum()),
            "outcome_used": False,
            "interpretation": (
                "retrospective application only; "
                "not evidence for model selection"
            ),
        },
    }
    (args.output_dir / "stage4_successor_metrics.json").write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    incumbent_sub = subgroup.loc[
        subgroup["model"].eq(INCUMBENT)
    ].sort_values("brier_skill_vs_3_over_field")
    lines = [
        "# Stage 4 Successor Development v1",
        "",
        "Status: **PASS — nested temporal development run completed**",
        "",
        "## Frozen interpretation",
        "",
        (
            "This is a development experiment, not an untouched final test. "
            "Selection uses only 2016-2024."
        ),
        (
            "2025 was already known before this experiment and is therefore not "
            "scored. It is loaded only"
        ),
        (
            "after the development decision, for production refitting before "
            "applying the frozen recipe to"
        ),
        (
            "the already-frozen 2026-10-03 target features. That target application "
            "is retrospective and is"
        ),
        "not used as evidence of improvement.",
        "",
        (
            "No market variables are loaded. No historical partial race receives "
            "a forced sum P(top3)=3."
        ),
        (
            "The target has all 18 runners, so the final target marginals receive "
            "the common-logit sum-to-3"
        ),
        "adjustment used by the existing Stage 4 contract.",
        "",
        "## Final outer 2023-2024 candidate comparison",
        "",
        fmt_table(
            selection_table,
            [
                "name",
                "race_macro_brier",
                "race_macro_log_loss",
                "mean_brier_delta_vs_best",
                "se_brier_delta_vs_best",
                "within_one_se_brier",
                "ece_10bin",
                "calibration_intercept",
                "calibration_slope",
                "roc_auc",
            ],
        ),
        "",
        f"- point-Brier-best: `{final_decision.point_brier_best}`",
        f"- incumbent retained: **{final_decision.incumbent_retained}**",
        f"- development winner: `{final_decision.winner}`",
        f"- reason: {final_decision.reason}",
        "",
        (
            "The one-SE gate is a pre-registered model-selection rule, "
            "not a significance test."
        ),
        "",
        "## Adaptive inner choices",
        "",
        fmt_table(
            pd.DataFrame(outer_choices),
            [
                "outer_year",
                "adaptive_family",
                "adaptive_feature_variant",
                "outer_train_end",
            ],
        ),
        "",
        "## Incumbent condition-level diagnostics",
        "",
        (
            "Groups below are ordered by Brier skill versus an exchangeable "
            "`3 / field_size` predictor."
        ),
        (
            "Lower skill marks conditions where the incumbent has less "
            "incremental predictive value."
        ),
        "",
        fmt_table(
            incumbent_sub.head(18),
            [
                "dimension",
                "group",
                "rows",
                "races",
                "race_macro_brier",
                "naive_race_macro_brier",
                "brier_skill_vs_3_over_field",
                "mean_predicted",
                "observed_top3_rate",
                "calibration_gap",
            ],
        ),
        "",
        "## Production recipe frozen after development decision",
        "",
        f"- family: `{production_family}`",
        f"- feature variant: `{production_feature_variant}`",
        f"- config: `{json.dumps(production_config, sort_keys=True)}`",
        "- post-hoc calibration transform: none",
        "- fit data after freeze: 2016-2025",
        "- 2025 candidate scoring: **not performed**",
        "",
        "## Retrospective target output",
        "",
        f"sum P(top3) = `{target_output['p_top3'].sum():.12f}`",
        "",
        fmt_table(
            target_output,
            [
                "rank",
                "horse_no",
                "horse_name",
                "raw_probability",
                "p_top3",
            ],
        ),
        "",
        (
            "This target output must not be used to claim validation because "
            "the race outcome is already known."
        ),
    ]
    (args.output_dir / "STAGE4_SUCCESSOR_DEVELOPMENT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
