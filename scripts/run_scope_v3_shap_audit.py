"""Compute strict OOF XGBoost contribution importance for routed Scope V3."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from keiba_place_lab.historical_panel import select_phase_a_cohort
from keiba_place_lab.nonmarket import CATEGORICAL_BLOCKS, NUMERIC_BLOCKS
from keiba_place_lab.scope_expansion import full_context_for_races, straight_course_mask
from keiba_place_lab.scope_features import (
    BLOCKS,
    augment_distance_history,
    fit_scope_candidate,
    scope_features,
)

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["race_id", "horse_id"]
ROUTES = (
    "sprint_other_1000_1400",
    "turf_1200",
    "middle_1401_2000",
    "long_2001_2600",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort(panel: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    out = select_phase_a_cohort(
        panel, surface="turf", min_field_size=8, min_prior_starts=3,
        min_distance_m=lo, max_distance_m=hi,
    )
    return (
        out.loc[~straight_course_mask(out)]
        .sort_values(["race_date", "race_id", "horse_no"], kind="stable")
        .reset_index(drop=True)
    )


def route_name(distance: pd.Series) -> pd.Series:
    d = pd.to_numeric(distance, errors="raise")
    out = pd.Series(index=distance.index, dtype="string")
    out.loc[d.between(1000, 1400) & d.ne(1200)] = "sprint_other_1000_1400"
    out.loc[d.eq(1200)] = "turf_1200"
    out.loc[d.between(1401, 2000)] = "middle_1401_2000"
    out.loc[d.between(2001, 2600)] = "long_2001_2600"
    if out.isna().any():
        raise ValueError("Unrouted evaluation distances")
    return out


BASE_FAMILY: dict[str, str] = {}
for family, cols in NUMERIC_BLOCKS.items():
    for col in cols:
        BASE_FAMILY[col] = family
for family, cols in CATEGORICAL_BLOCKS.items():
    for col in cols:
        BASE_FAMILY[col] = family
for col in (
    "rel_career_top3_vs_others",
    "rel_same_distance_top3_vs_others",
    "rel_same_course_top3_vs_others",
    "rel_recent3_finish_vs_others",
    "rel_recent3_time_vs_others",
):
    BASE_FAMILY[col] = "relative_ability"
for block in BLOCKS:
    for suffix in (
        "top3_shrunk", "log_starts", "zero_experience", "history_missing", "relative",
    ):
        BASE_FAMILY[f"{block}_{suffix}"] = f"distance_{block}"


def source_feature(transformed: str, numeric_sources: list[str], categorical_sources: list[str]) -> str:
    raw = transformed.split("__", 1)[-1]
    if raw.startswith("missingindicator_"):
        raw = raw.removeprefix("missingindicator_")
    if raw in numeric_sources:
        return raw
    for cat in categorical_sources:
        if raw == cat or raw.startswith(cat + "_"):
            return cat
    return raw


class Aggregator:
    def __init__(self) -> None:
        self.data: dict[tuple[str, str, str, str], list[float]] = defaultdict(
            lambda: [0.0, 0.0, 0.0]
        )
        self.total_rows: dict[tuple[str, str], int] = defaultdict(int)

    def mark_rows(self, *, scope: str, year: str, rows: int) -> None:
        self.total_rows[(scope, year)] += int(rows)

    def add(self, *, scope: str, year: str, level: str, names: list[str], matrix: np.ndarray) -> None:
        if matrix.shape[1] != len(names):
            raise ValueError("Contribution/name width mismatch")
        for j, name in enumerate(names):
            values = matrix[:, j]
            rec = self.data[(scope, year, level, name)]
            rec[0] += float(np.abs(values).sum())
            rec[1] += float(values.sum())
            rec[2] += float(np.count_nonzero(values))

    def frame(self, level: str) -> pd.DataFrame:
        rows = []
        for (scope, year, lev, name), (sum_abs, sum_signed, nonzero) in self.data.items():
            if lev != level:
                continue
            count = self.total_rows[(scope, year)]
            if count <= 0:
                raise ValueError(f"Missing total row count for {scope}/{year}")
            rows.append({
                "scope": scope,
                "year": year,
                "name": name,
                "rows": int(count),
                "mean_abs_shap_logodds": sum_abs / count,
                "mean_shap_logodds": sum_signed / count,
                "nonzero_share": nonzero / count,
            })
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        denom = out.groupby(["scope", "year"])["mean_abs_shap_logodds"].transform("sum")
        out["share_of_total_abs"] = out["mean_abs_shap_logodds"] / denom
        return out.sort_values(
            ["scope", "year", "mean_abs_shap_logodds"],
            ascending=[True, True, False],
        ).reset_index(drop=True)


def contributions(model, engineered: pd.DataFrame):
    pre = model.named_steps["preprocess"]
    clf = model.named_steps["model"]
    x = pre.transform(engineered)
    transformed_names = [str(v) for v in pre.get_feature_names_out()]
    contrib = clf.get_booster().predict(xgb.DMatrix(x), pred_contribs=True)
    if contrib.shape[1] != len(transformed_names) + 1:
        raise ValueError("Unexpected pred_contribs width")
    probability = clf.predict_proba(x)[:, 1]
    reconstructed = 1.0 / (1.0 + np.exp(-contrib.sum(axis=1)))
    contribution_parity = float(np.max(np.abs(probability - reconstructed)))
    if contribution_parity > 5e-6:
        raise ValueError(
            f"SHAP contribution parity failure: {contribution_parity}"
        )
    numeric_sources = list(pre.transformers_[0][2])
    categorical_sources = list(pre.transformers_[1][2])
    sources = [
        source_feature(name, numeric_sources, categorical_sources)
        for name in transformed_names
    ]
    return contrib[:, :-1], probability, transformed_names, sources


def grouped_matrix(contrib: np.ndarray, groups: list[str]) -> tuple[np.ndarray, list[str]]:
    unique = list(dict.fromkeys(groups))
    index = {name: i for i, name in enumerate(unique)}
    out = np.zeros((len(contrib), len(unique)), dtype=float)
    for j, group in enumerate(groups):
        out[:, index[group]] += contrib[:, j]
    return out, unique


def add_chunk(agg: Aggregator, *, scope: str, year: int, model, engineered: pd.DataFrame) -> np.ndarray:
    contrib, probability, transformed, sources = contributions(model, engineered)
    source_matrix, source_names = grouped_matrix(contrib, sources)
    families = [BASE_FAMILY.get(name, "other") for name in source_names]
    family_matrix, family_names = grouped_matrix(source_matrix, families)

    for y in (str(year), "ALL"):
        for s in (scope, "ALL_ROUTED"):
            agg.mark_rows(scope=s, year=y, rows=len(engineered))
            agg.add(scope=s, year=y, level="transformed", names=transformed, matrix=contrib)
            agg.add(scope=s, year=y, level="source", names=source_names, matrix=source_matrix)
            agg.add(scope=s, year=y, level="family", names=family_names, matrix=family_matrix)
    return probability


def markdown_top(frame: pd.DataFrame, scope: str, n: int = 12) -> str:
    subset = frame.loc[(frame.scope == scope) & (frame.year == "ALL")].head(n).copy()
    if subset.empty:
        return "_No rows_\n"
    subset["mean_abs_shap_logodds"] = subset["mean_abs_shap_logodds"].map(lambda x: f"{x:.6f}")
    subset["share_of_total_abs"] = subset["share_of_total_abs"].map(lambda x: f"{100*x:.1f}%")
    return subset[["name", "mean_abs_shap_logodds", "share_of_total_abs"]].to_markdown(index=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/scope_v3_shap")
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    panel["race_date"] = pd.to_datetime(panel["race_date"])
    for key in KEYS:
        panel[key] = panel[key].astype("string")

    cache = ROOT / "data/historical_processed/scope_augmented.parquet"
    source_hash = digest(args.panel)
    cache_meta = cache.with_suffix(".json")
    code_hash = digest(ROOT / "src/keiba_place_lab/scope_features.py")
    if (
        cache.exists()
        and cache_meta.exists()
        and json.loads(cache_meta.read_text()) == {"source": source_hash, "code": code_hash}
    ):
        panel = pd.read_parquet(cache)
    else:
        print("Computing strictly prior-date distance histories", flush=True)
        panel = augment_distance_history(panel)
        cache.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(cache, index=False)
        cache_meta.write_text(json.dumps({"source": source_hash, "code": code_hash}, indent=2) + "\n")

    evaluation = cohort(panel, 1000, 2600)
    evaluation = evaluation.loc[evaluation.race_date.dt.year.isin([2023, 2024])].reset_index(drop=True)
    evaluation["route"] = route_name(evaluation["distance_m"])
    reference = pd.read_csv(
        ROOT / "analysis/scope_expansion_turf/predictions.csv.gz",
        dtype={key: "string" for key in KEYS},
    )
    reference = evaluation[KEYS].merge(
        reference[KEYS + ["p_routed"]], on=KEYS, validate="one_to_one"
    )
    if len(reference) != len(evaluation):
        raise ValueError("Stored routed OOF does not align to evaluation cohort")

    global_all = cohort(panel, 1000, 2600)
    sprint_all = cohort(panel, 1000, 1400)
    long_all = cohort(panel, 2001, 2600)

    agg = Aggregator()
    predictions = np.full(len(evaluation), np.nan)
    fold_rows: list[dict[str, object]] = []

    for year in (2023, 2024):
        year_mask = evaluation.race_date.dt.year.eq(year)
        print(f"Year {year}: {int(year_mask.sum())} evaluation rows", flush=True)

        global_train = global_all.loc[global_all.race_date.dt.year.between(2016, year - 1)]
        global_context = full_context_for_races(panel, set(global_train.race_id))
        global_model, global_prior = fit_scope_candidate(
            global_train, global_context, blocks=(), surface="turf"
        )

        sprint_train = sprint_all.loc[sprint_all.race_date.dt.year.between(2016, year - 1)]
        sprint_context = full_context_for_races(panel, set(sprint_train.race_id))
        sprint_model, sprint_prior = fit_scope_candidate(
            sprint_train, sprint_context, blocks=("regime",), surface="turf"
        )

        long_train = long_all.loc[long_all.race_date.dt.year.between(2016, year - 1)]
        long_context = full_context_for_races(panel, set(long_train.race_id))
        long_model, long_prior = fit_scope_candidate(
            long_train, long_context, blocks=BLOCKS, surface="turf"
        )

        models = {
            "sprint_other_1000_1400": (global_model, global_prior, (), len(global_train)),
            "turf_1200": (sprint_model, sprint_prior, ("regime",), len(sprint_train)),
            "middle_1401_2000": (global_model, global_prior, (), len(global_train)),
            "long_2001_2600": (long_model, long_prior, BLOCKS, len(long_train)),
        }
        for scope in ROUTES:
            mask = year_mask & evaluation.route.eq(scope)
            test = evaluation.loc[mask]
            model, prior, blocks, train_rows = models[scope]
            test_context = full_context_for_races(panel, set(test.race_id))
            engineered = scope_features(
                test, test_context, prior_mean=prior, blocks=blocks, surface="turf"
            )
            p = add_chunk(agg, scope=scope, year=year, model=model, engineered=engineered)
            predictions[mask.to_numpy()] = p
            fold_rows.append({
                "year": year,
                "scope": scope,
                "evaluation_rows": len(test),
                "evaluation_races": int(test.race_id.nunique()),
                "train_rows": train_rows,
                "prior_mean": prior,
                "blocks": ",".join(blocks),
            })

    if not np.isfinite(predictions).all():
        raise ValueError("Incomplete routed SHAP predictions")
    stored = reference["p_routed"].to_numpy(float)
    max_prediction_error = float(np.max(np.abs(predictions - stored)))
    if max_prediction_error > 1e-7:
        raise ValueError(f"Routed OOF parity failure: {max_prediction_error}")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    source = agg.frame("source")
    family = agg.frame("family")
    transformed = agg.frame("transformed")
    source.to_csv(out / "source_importance.csv", index=False)
    family.to_csv(out / "family_importance.csv", index=False)
    transformed.to_csv(out / "transformed_importance.csv.gz", index=False, compression="gzip")
    pd.DataFrame(fold_rows).to_csv(out / "folds.csv", index=False)

    parity = {
        "source_panel_sha256": source_hash,
        "stored_oof_sha256": digest(ROOT / "analysis/scope_expansion_turf/predictions.csv.gz"),
        "max_abs_prediction_error": max_prediction_error,
        "evaluation_rows": len(evaluation),
        "evaluation_races": int(evaluation.race_id.nunique()),
        "evaluation_dates": int(evaluation.race_date.nunique()),
        "years": [2023, 2024],
        "routes": {
            scope: {
                "rows": int(evaluation.route.eq(scope).sum()),
                "races": int(evaluation.loc[evaluation.route.eq(scope), "race_id"].nunique()),
            }
            for scope in ROUTES
        },
        "importance_definition": (
            "Transformed XGBoost contributions are summed per original source feature "
            "within each row before taking absolute values; mean absolute grouped "
            "contribution is averaged over strict OOF rows in log-odds space."
        ),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgb.__version__,
        },
    }
    (out / "parity.json").write_text(json.dumps(parity, ensure_ascii=False, indent=2) + "\n")

    labels = {
        "ALL_ROUTED": "All routed turf 1000-2600m",
        "turf_1200": "Turf exactly 1200m",
        "sprint_other_1000_1400": "Turf 1000-1400m except 1200m",
        "middle_1401_2000": "Turf 1401-2000m",
        "long_2001_2600": "Turf 2001-2600m",
    }
    report = [
        "# Scope V3 strict OOF SHAP audit",
        "",
        "Importance uses XGBoost contribution values in log-odds space on the frozen 2023-2024 outer folds.",
        "No 2025 outcomes or current-race outcomes are used. Routed predictions must reproduce the stored OOF output within 1e-7.",
        "",
        f"Prediction parity max absolute error: {max_prediction_error:.3g}.",
        "",
    ]
    for scope in ("ALL_ROUTED", *ROUTES):
        report += [
            f"## {labels[scope]}",
            "",
            "### Source features",
            "",
            markdown_top(source, scope, 15),
            "### Feature families",
            "",
            markdown_top(family, scope, 10),
        ]
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
