"""Audit Stage 3.6 for Scope Expansion Stage 1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from keiba_place_lab.nonmarket import REQUIRED_RAW_COLUMNS, validate_market_free
from keiba_place_lab.scope_expansion import (
    full_context_for_races,
    select_turf_scope,
    straight_course_mask,
)

EXPECTED_EVAL_ROWS = 6313
EXPECTED_EVAL_RACES = 495
EXPECTED_EVAL_FINGERPRINT = (
    "a72fa10cc9adcd1ceee4c5b5eacd0dfdcdfc33fc984bf0b7866ec70b617bfd85"
)
PANEL_SHA256 = "cee9ae9a099f521f12b1bcdd371c25a3d7b9ba3555fbcf5a46f2c9098f59a33d"


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


def aggregate_counts(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*group_cols, "races", "eligible_rows", "race_dates"])
    work = frame.copy()
    work["year"] = work["race_date"].dt.year
    grouped = (
        work.groupby(group_cols, dropna=False)
        .agg(
            races=("race_id", "nunique"),
            eligible_rows=("horse_id", "size"),
            race_dates=("race_date", "nunique"),
        )
        .reset_index()
    )
    return grouped.sort_values(group_cols, kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_names = [
        "jra_flat_warmup_v1.parquet",
        "jra_flat_train_v1.parquet",
        "jra_flat_validation_v1.parquet",
        "jra_flat_test_v1.parquet",
    ]
    full_panel = pd.concat(
        [load_panel(args.historical_dir, name) for name in split_names],
        ignore_index=True,
        sort=False,
    )
    full_panel = full_panel.sort_values(
        ["race_date", "race_id", "horse_no"],
        kind="stable",
    ).reset_index(drop=True)

    in_band = (
        full_panel["surface"].astype(str).str.lower().eq("turf")
        & pd.to_numeric(full_panel["distance_m"], errors="coerce").between(1000, 2600)
    )
    band_full = full_panel.loc[in_band].copy()
    band_full["is_straight_course"] = straight_course_mask(band_full)

    eligible_all = select_turf_scope(
        full_panel,
        min_distance_m=1000,
        max_distance_m=2600,
        exclude_straight=False,
    )
    eligible_all["is_straight_course"] = straight_course_mask(eligible_all)
    eligible_primary = eligible_all.loc[~eligible_all["is_straight_course"]].copy()

    eval_1200 = eligible_primary.loc[
        eligible_primary["race_date"].dt.year.isin([2023, 2024])
        & pd.to_numeric(eligible_primary["distance_m"], errors="coerce").eq(1200)
    ].copy()
    eval_fp = fingerprint(eval_1200)
    if len(eval_1200) != EXPECTED_EVAL_ROWS:
        raise ValueError(
            f"1200m evaluation rows mismatch: {len(eval_1200)} != {EXPECTED_EVAL_ROWS}"
        )
    if eval_1200["race_id"].nunique() != EXPECTED_EVAL_RACES:
        raise ValueError(
            "1200m evaluation race count mismatch: "
            f"{eval_1200['race_id'].nunique()} != {EXPECTED_EVAL_RACES}"
        )
    if eval_fp != EXPECTED_EVAL_FINGERPRINT:
        raise ValueError(
            f"1200m evaluation fingerprint mismatch: {eval_fp}"
        )

    primary_race_ids = set(eligible_primary["race_id"].astype("string"))
    context = full_context_for_races(full_panel, primary_race_ids)

    context_counts = (
        context.groupby("race_id", dropna=False)
        .agg(
            context_rows=("horse_id", "size"),
            field_size=("field_size", "first"),
            race_date=("race_date", "first"),
            distance_m=("distance_m", "first"),
            racecourse=("racecourse", "first"),
        )
        .reset_index()
    )
    context_counts["context_complete"] = (
        context_counts["context_rows"].astype(int)
        == context_counts["field_size"].astype(int)
    )
    if not context_counts["context_complete"].all():
        raise ValueError("Incomplete full-field context detected")

    eligible_counts = (
        eligible_primary.groupby("race_id", dropna=False)
        .size()
        .rename("eligible_rows")
        .reset_index()
    )
    context_qa = context_counts.merge(
        eligible_counts,
        on="race_id",
        how="left",
        validate="one_to_one",
    )
    context_qa["eligible_rows"] = context_qa["eligible_rows"].fillna(0).astype(int)

    by_year_distance = aggregate_counts(
        eligible_primary.assign(year=eligible_primary["race_date"].dt.year),
        ["year", "distance_m"],
    )
    by_year_course = aggregate_counts(
        eligible_primary.assign(year=eligible_primary["race_date"].dt.year),
        ["year", "racecourse"],
    )
    by_distance_course = aggregate_counts(
        eligible_primary,
        ["distance_m", "racecourse"],
    )

    missing_cols = sorted(col for col in REQUIRED_RAW_COLUMNS if col in eligible_primary.columns)
    missing_rows = []
    for col in missing_cols:
        missing_rows.append(
            {
                "feature": col,
                "rows": len(eligible_primary),
                "missing_rows": int(eligible_primary[col].isna().sum()),
                "missing_rate": float(eligible_primary[col].isna().mean()),
            }
        )
    missingness = pd.DataFrame(missing_rows).sort_values(
        ["missing_rate", "feature"],
        ascending=[False, True],
        kind="stable",
    )

    straight_summary = (
        eligible_all.loc[eligible_all["is_straight_course"]]
        .assign(year=lambda x: x["race_date"].dt.year)
        .groupby(["year", "distance_m", "racecourse"], dropna=False)
        .agg(
            races=("race_id", "nunique"),
            eligible_rows=("horse_id", "size"),
            race_dates=("race_date", "nunique"),
        )
        .reset_index()
        .sort_values(["year", "distance_m", "racecourse"], kind="stable")
    )

    candidate_rows = []
    for candidate, lo, hi in [
        ("A_current_1200", 1200, 1200),
        ("B_turf_sprint_1000_1400", 1000, 1400),
        ("C_turf_global_1000_2600", 1000, 2600),
    ]:
        cohort = select_turf_scope(
            full_panel,
            min_distance_m=lo,
            max_distance_m=hi,
            exclude_straight=True,
        )
        years = cohort["race_date"].dt.year
        candidate_rows.append(
            {
                "candidate": candidate,
                "min_distance_m": lo,
                "max_distance_m": hi,
                "eligible_rows_2016_2022": int(years.between(2016, 2022).sum()),
                "races_2016_2022": int(
                    cohort.loc[years.between(2016, 2022), "race_id"].nunique()
                ),
                "eligible_rows_2016_2023": int(years.between(2016, 2023).sum()),
                "races_2016_2023": int(
                    cohort.loc[years.between(2016, 2023), "race_id"].nunique()
                ),
                "eligible_rows_all_2010_2025": len(cohort),
                "races_all_2010_2025": int(cohort["race_id"].nunique()),
            }
        )
    candidate_sizes = pd.DataFrame(candidate_rows)

    by_year_distance.to_csv(
        args.output_dir / "scope_stage1_counts_year_distance.csv",
        index=False,
    )
    by_year_course.to_csv(
        args.output_dir / "scope_stage1_counts_year_course.csv",
        index=False,
    )
    by_distance_course.to_csv(
        args.output_dir / "scope_stage1_counts_distance_course.csv",
        index=False,
    )
    missingness.to_csv(
        args.output_dir / "scope_stage1_missingness.csv",
        index=False,
    )
    context_qa.to_csv(
        args.output_dir / "scope_stage1_context_qa.csv",
        index=False,
    )
    straight_summary.to_csv(
        args.output_dir / "scope_stage1_straight_course_counts.csv",
        index=False,
    )
    candidate_sizes.to_csv(
        args.output_dir / "scope_stage1_candidate_training_sizes.csv",
        index=False,
    )

    manifest = {
        "status": "PASS",
        "stage": "scope_expansion_stage1_audit",
        "historical_panel_sha256": PANEL_SHA256,
        "scope": {
            "surface": "turf",
            "distance_min_m": 1000,
            "distance_max_m": 2600,
            "min_field_size": 8,
            "min_prior_starts": 3,
            "primary_excludes_straight_course": True,
        },
        "full_band": {
            "starter_rows": len(band_full),
            "races": int(band_full["race_id"].nunique()),
            "race_dates": int(band_full["race_date"].nunique()),
        },
        "eligible_primary": {
            "rows": len(eligible_primary),
            "races": int(eligible_primary["race_id"].nunique()),
            "race_dates": int(eligible_primary["race_date"].nunique()),
        },
        "eligible_straight_separate": {
            "rows": int(eligible_all["is_straight_course"].sum()),
            "races": int(
                eligible_all.loc[
                    eligible_all["is_straight_course"], "race_id"
                ].nunique()
            ),
        },
        "evaluation_1200_2023_2024": {
            "rows": len(eval_1200),
            "races": int(eval_1200["race_id"].nunique()),
            "fingerprint_sha256": eval_fp,
            "matches_existing": True,
        },
        "full_context": {
            "rows": len(context),
            "races": int(context["race_id"].nunique()),
            "all_races_complete": bool(context_qa["context_complete"].all()),
        },
        "candidates": candidate_sizes.to_dict(orient="records"),
        "uses_2025_for_abc_comparison": False,
        "market_fields_loaded": False,
    }
    (args.output_dir / "scope_stage1_audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = [
        "# Scope Expansion Stage 1 — Data Audit",
        "",
        "Status: **PASS — Stage 3.6 supports the frozen A/B/C comparison**",
        "",
        "## Primary QA",
        "",
        f"- 1200m 2023-2024 evaluation rows: **{len(eval_1200):,}**",
        f"- 1200m 2023-2024 evaluation races: **{eval_1200['race_id'].nunique():,}**",
        f"- evaluation fingerprint: {eval_fp}",
        "- existing 1200m evaluation reproduced exactly: **yes**",
        f"- 1000-2600m eligible primary rows: **{len(eligible_primary):,}**",
        f"- 1000-2600m eligible primary races: **{eligible_primary['race_id'].nunique():,}**",
        f"- full-field context rows: **{len(context):,}**",
        "- all selected race contexts complete: **yes**",
        "",
        "## Candidate training sizes",
        "",
        candidate_sizes.to_markdown(index=False),
        "",
        "## Straight-course handling",
        "",
        f"- separately reported eligible straight-course rows: **{int(eligible_all['is_straight_course'].sum()):,}**",
        f"- separately reported straight-course races: **{eligible_all.loc[eligible_all['is_straight_course'], 'race_id'].nunique():,}**",
        "- straight-course races enter none of the A/B/C primary training populations",
        "",
        "## Highest missingness among frozen raw model inputs",
        "",
        missingness.head(20).to_markdown(index=False),
        "",
        "The comparison specification is frozen in docs/SCOPE_EXPANSION_STAGE1_SPEC.md.",
        "No A/B/C outer scores are generated by this audit.",
    ]
    (args.output_dir / "SCOPE_EXPANSION_STAGE1_AUDIT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
