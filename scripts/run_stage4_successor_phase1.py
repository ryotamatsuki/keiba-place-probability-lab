"""Phase 1: leakage-safe diagnosis of the frozen Stage 4 incumbent."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from keiba_place_lab.model_selection import candidate_score
from keiba_place_lab.nonmarket import (
    fit_model,
    predict_raw_probability,
    validate_market_free,
)
from keiba_place_lab.stage4_successor import (
    assert_outer_year_contract,
    field_size_baseline,
    subgroup_diagnostics,
)

INCUMBENT_NAME = "l2_logistic_c0.1"
INCUMBENT_C = 0.1
PRIOR_STRENGTH = 6.0
OUTER_YEARS = (2023, 2024)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_2016_2022 = load_turf_1200(args.historical_dir, "train")
    validation_2023_2024 = load_turf_1200(args.historical_dir, "validation")
    if validation_2023_2024["race_date"].dt.year.max() > 2024:
        raise ValueError("Phase 1 must not load post-2024 validation rows")

    development = pd.concat(
        [train_2016_2022, validation_2023_2024],
        ignore_index=True,
        sort=False,
    )
    development_year = development["race_date"].dt.year

    outer_parts: list[pd.DataFrame] = []
    fold_manifest: list[dict[str, object]] = []

    for outer_year in OUTER_YEARS:
        train = development.loc[
            development_year.between(2016, outer_year - 1)
        ].copy()
        evaluation = development.loc[development_year.eq(outer_year)].copy()
        if train.empty or evaluation.empty:
            raise ValueError(f"Empty outer fold {outer_year}")
        assert_outer_year_contract(train, evaluation, outer_year=outer_year)

        model, prior_mean = fit_model(
            train,
            c_value=INCUMBENT_C,
            prior_strength=PRIOR_STRENGTH,
        )
        _, probability = predict_raw_probability(
            model,
            evaluation,
            prior_mean=prior_mean,
            prior_strength=PRIOR_STRENGTH,
        )

        out = evaluation.copy()
        out["outer_year"] = outer_year
        out["incumbent_name"] = INCUMBENT_NAME
        out["p_incumbent"] = probability
        out["p_field_size_baseline"] = field_size_baseline(evaluation)
        outer_parts.append(out)

        fold_score = candidate_score(
            evaluation,
            probability,
            name=INCUMBENT_NAME,
        )
        baseline_score = candidate_score(
            evaluation,
            out["p_field_size_baseline"].to_numpy(dtype=float),
            name="field_size_baseline",
        )
        fold_manifest.append(
            {
                "outer_year": outer_year,
                "train_date_min": str(train["race_date"].min().date()),
                "train_date_max": str(train["race_date"].max().date()),
                "train_rows": len(train),
                "train_races": int(train["race_id"].nunique()),
                "evaluation_rows": len(evaluation),
                "evaluation_races": int(evaluation["race_id"].nunique()),
                "prior_mean": float(prior_mean),
                "race_macro_brier": fold_score.race_macro_brier,
                "race_macro_log_loss": fold_score.race_macro_log_loss,
                "baseline_race_macro_brier": baseline_score.race_macro_brier,
            }
        )

    oof = pd.concat(outer_parts, ignore_index=True, sort=False)
    if set(oof["race_date"].dt.year.unique()) != set(OUTER_YEARS):
        raise ValueError("OOF output contains unexpected years")
    if oof["race_date"].dt.year.ge(2025).any():
        raise ValueError("2025+ rows are forbidden in Phase 1")

    probability = oof["p_incumbent"].to_numpy(dtype=float)
    baseline_probability = oof["p_field_size_baseline"].to_numpy(dtype=float)
    overall = candidate_score(oof, probability, name=INCUMBENT_NAME)
    baseline_overall = candidate_score(
        oof,
        baseline_probability,
        name="field_size_baseline",
    )
    diagnostics = subgroup_diagnostics(oof, probability)

    overall_table = pd.DataFrame(
        [
            {
                "model": INCUMBENT_NAME,
                **overall.__dict__,
                "brier_skill_vs_field_size_baseline": (
                    1.0
                    - overall.race_macro_brier
                    / baseline_overall.race_macro_brier
                ),
            },
            {
                "model": "field_size_baseline",
                **baseline_overall.__dict__,
                "brier_skill_vs_field_size_baseline": 0.0,
            },
        ]
    )

    output_columns = [
        col
        for col in [
            "race_date",
            "race_id",
            "horse_id",
            "horse_no",
            "horse_name",
            "top3_label",
            "field_size",
            "race_class",
            "racecourse",
            "days_since_prev",
            "career_starts",
            "outer_year",
            "incumbent_name",
            "p_incumbent",
            "p_field_size_baseline",
        ]
        if col in oof.columns
    ]
    oof[output_columns].to_csv(
        args.output_dir / "incumbent_oof_2023_2024.csv",
        index=False,
        float_format="%.9f",
    )
    overall_table.to_csv(
        args.output_dir / "incumbent_overall_metrics.csv",
        index=False,
        float_format="%.9f",
    )
    diagnostics.to_csv(
        args.output_dir / "incumbent_subgroup_diagnostics.csv",
        index=False,
        float_format="%.9f",
    )

    fingerprint = evaluation_fingerprint(oof)
    manifest = {
        "status": "PASS",
        "phase": "stage4_successor_phase1",
        "incumbent": {
            "name": INCUMBENT_NAME,
            "C": INCUMBENT_C,
            "prior_strength": PRIOR_STRENGTH,
            "posthoc_calibration": "none",
            "historical_probability_transform": "raw marginal P(top3)",
        },
        "historical_panel_sha256": PANEL_SHA256,
        "development_years": [2016, 2024],
        "outer_years": list(OUTER_YEARS),
        "uses_2025_rows": False,
        "uses_market_fields": False,
        "uses_target_outcome": False,
        "evaluation_rows": len(oof),
        "evaluation_races": int(oof["race_id"].nunique()),
        "evaluation_row_fingerprint_sha256": fingerprint,
        "folds": fold_manifest,
        "overall": {
            **overall.__dict__,
            "baseline_race_macro_brier": baseline_overall.race_macro_brier,
            "brier_skill_vs_field_size_baseline": (
                1.0
                - overall.race_macro_brier
                / baseline_overall.race_macro_brier
            ),
        },
    }
    (args.output_dir / "phase1_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    stable = diagnostics.loc[
        (diagnostics["dimension"] != "overall")
        & (diagnostics["rows"] >= 100)
        & (diagnostics["races"] >= 20)
    ].sort_values(
        [
            "brier_skill_vs_field_size_baseline",
            "race_macro_brier",
            "dimension",
            "level",
        ],
        kind="stable",
    )
    lowest = stable.head(12)
    yearly = diagnostics.loc[diagnostics["dimension"] == "year"].sort_values("level")

    lines = [
        "# Stage 4 Successor Phase 1 — Incumbent Diagnostics",
        "",
        "Status: **PASS — leakage-safe 2023/2024 incumbent OOF diagnosis completed**",
        "",
        "No challenger model or feature block is selected in this phase.",
        "2025 is not loaded.",
        "",
        "## Outer walk-forward contract",
        "",
        "- 2023 forecast: trained through 2022 only.",
        "- 2024 forecast: refit through 2023 only.",
        "- incumbent: L2 Logistic Regression, C=0.1.",
        "- current Stage 4 feature engineering and prior strength 6.",
        "- historical score: raw marginal P(top3); no sum-to-three adjustment on partial cohorts.",
        "",
        "## Overall",
        "",
        fmt(
            overall_table,
            [
                "model",
                "race_macro_brier",
                "race_macro_log_loss",
                "runner_micro_brier",
                "runner_micro_log_loss",
                "brier_skill_vs_field_size_baseline",
                "races",
                "rows",
            ],
        ),
        "",
        "## By outer year",
        "",
        fmt(
            yearly,
            [
                "level",
                "race_macro_brier",
                "baseline_race_macro_brier",
                "brier_skill_vs_field_size_baseline",
                "observed_top3_rate",
                "mean_predicted",
                "calibration_intercept",
                "calibration_slope",
                "races",
                "rows",
            ],
        ),
        "",
        "## Lowest skill strata",
        "",
        "Shown only for preregistered strata with at least 100 runner rows and 20 races.",
        "These are diagnostic hypotheses, not evidence for changing a feature.",
        "",
        fmt(
            lowest,
            [
                "dimension",
                "level",
                "race_macro_brier",
                "baseline_race_macro_brier",
                "brier_skill_vs_field_size_baseline",
                "calibration_gap_observed_minus_predicted",
                "calibration_intercept",
                "calibration_slope",
                "races",
                "rows",
            ],
        ),
        "",
        "## Audit",
        "",
        f"- evaluation rows: {len(oof):,}",
        f"- evaluation races: {int(oof['race_id'].nunique()):,}",
        f"- common evaluation-row SHA256: {fingerprint}",
        "- 2025 rows loaded: **no**",
        "- market fields loaded: **no**",
        "- target-race outcome loaded: **no**",
        "",
        "The next phase may use these diagnostics to motivate hypotheses, but its candidate registry",
        "and tuning ranges must be frozen before Phase-2 model-family scores are generated.",
    ]
    (args.output_dir / "PHASE1_INCUMBENT_DIAGNOSTICS.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
