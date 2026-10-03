"""Materialize and fail-closed QA the fixed 2010-2025 historical panel."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import requests
from materialize_historical_training_dataset import (
    SAFE_EXPORT_COLUMNS,
    VENUE_JP_TO_CODE,
    parse_early_position,
    parse_float,
    parse_rank,
    parse_sex_age,
    parse_time,
    sha256_file,
)

from keiba_place_lab.historical_panel import (
    assert_strict_history,
    assign_temporal_split,
    build_historical_panel,
    find_forbidden_market_columns,
    select_phase_a_cohort,
)

SOURCE = "noriyukifurufuru/japan-horse-racing-2010-2025"
OUT = Path("data/historical_processed/stage36_freeze_v1")
DERIVED = Path("data/derived")
DOCS = Path("docs")


def json_write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n")


def official_index() -> pd.DataFrame:
    files = [DERIVED / f"jra_official_{year}.csv" for year in range(2010, 2026)]
    absent = [str(p) for p in files if not p.exists()]
    if absent:
        raise ValueError(f"Missing official year extractions: {absent}")
    df = pd.concat([pd.read_csv(p, dtype={"race_id": str}) for p in files], ignore_index=True)
    if df.race_id.duplicated().any():
        raise ValueError("Duplicate official race ids")
    date = pd.to_datetime(df.actual_date, errors="raise")
    if not date.between("2010-01-01", "2025-12-31").all():
        raise ValueError("Official date outside frozen interval")
    if df.duplicated(["actual_date", "racecourse", "race_number"]).any():
        raise ValueError("Duplicate official date/course/race number")
    keys = ["year", "racecourse", "meeting_number", "meeting_day"]
    if df.groupby(keys).actual_date.nunique().gt(1).any():
        raise ValueError("Conflicting official day/date mappings")
    if df.race_kind.eq("unknown").any():
        raise ValueError(
            f"Unclassified official races: {df[df.race_kind.eq('unknown')].race_id.tolist()}"
        )
    return df.sort_values("race_id").reset_index(drop=True)


def standardized_results(
    path: Path, races: pd.DataFrame, all_jra_ids: set
) -> tuple[pd.DataFrame, dict]:
    meta = races.set_index("race_id")
    records = []
    malformed = []
    ranks = Counter()
    scratch = Counter()
    declared = {}
    total = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = len(reader.fieldnames)
        for line, row in enumerate(reader, 2):
            total += 1
            rid = row["race_id"]
            if None in row or any(v is None for v in row.values()):
                malformed.append(
                    {
                        "line": line,
                        "race_id": rid,
                        "jra": rid in all_jra_ids,
                        "flat": rid in meta.index,
                        "expected_fields": expected,
                        "actual_fields": sum(1 for k in row if k is not None)
                        + len(row.get(None, [])),
                    }
                )
                continue
            if rid not in meta.index:
                continue
            ranks[row["rank"]] += 1
            pos, status, started = parse_rank(row["rank"])
            if status.startswith("other:"):
                raise ValueError(f"Unhandled official placing {rid}: {row['rank']}")
            no = int(row["number"])
            declared[rid] = max(declared.get(rid, 0), no)
            if not started:
                scratch[row["rank"]] += 1
                continue
            sex, age = parse_sex_age(row["sex_age"])
            records.append(
                {
                    "race_id": rid,
                    "horse_id": row["horse_id"],
                    "horse_name": row["horse_name"],
                    "horse_no": no,
                    "sex": sex,
                    "age": age,
                    "assigned_weight_kg": parse_float(row["weight"]),
                    "finish_position": pos,
                    "finish_status": status,
                    "race_time_seconds": parse_time(row["time"])
                    if status == "finished"
                    else np.nan,
                    "early_position": parse_early_position(row["passing"]),
                }
            )
    bad = pd.DataFrame(malformed)
    bad.to_csv(DOCS / "HISTORICAL_MALFORMED_ROWS.csv", index=False)
    if any(r["jra"] for r in malformed):
        raise ValueError("Malformed JRA rows require repair, never silent skip")
    df = pd.DataFrame(records).merge(races, on="race_id", validate="many_to_one")
    df["field_size"] = df.groupby("race_id").horse_id.transform("size")
    df["declared_field_size"] = df.race_id.map(declared)
    df = df.sort_values(["race_date", "race_id", "horse_no"]).reset_index(drop=True)
    if set(meta.index) != set(df.race_id):
        raise ValueError("Selected flat races without results")
    if df.horse_id.eq("").any() or df.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Missing or duplicate horse identifiers")
    if (
        (~df.sex.isin(["M", "F", "G"])).any()
        or df.age.isna().any()
        or df.assigned_weight_kg.isna().any()
    ):
        raise ValueError("Invalid core entrant attributes")
    if (df.finish_position.notna() & ~df.finish_position.between(1, df.field_size)).any():
        raise ValueError("Final placing outside actual starter field")
    diag = {
        "source_result_rows": total,
        "malformed_all": len(malformed),
        "malformed_JRA": sum(r["jra"] for r in malformed),
        "malformed_flat": sum(r["flat"] for r in malformed),
        "rank_counts": dict(ranks),
        "nonstarter_counts": dict(scratch),
        "finish_status_counts": df.finish_status.value_counts().to_dict(),
    }
    return df, diag


def count_summary(x: pd.DataFrame) -> dict:
    fields = x.drop_duplicates("race_id").field_size
    return {
        "races": x.race_id.nunique(),
        "runner_rows": len(x),
        "horses": x.horse_id.nunique(),
        "top3_prevalence": x.top3_label.mean(),
        "field_size_distribution": fields.value_counts().sort_index().to_dict(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    official = official_index()
    md = requests.get(f"https://www.kaggle.com/api/v1/datasets/view/{SOURCE}", timeout=60)
    md.raise_for_status()
    md = md.json()
    metadata = {
        k: md[k]
        for k in [
            "id",
            "ref",
            "title",
            "currentVersionNumber",
            "licenseName",
            "lastUpdated",
            "totalBytes",
        ]
    }
    if metadata["licenseName"] != "CC0: Public Domain" or metadata["currentVersionNumber"] != 1:
        raise ValueError("Source version/license changed: requires separate review")
    root = Path(
        kagglehub.dataset_download(
            f"{SOURCE}/versions/1", output_dir="data/historical_raw/kaggle_v1"
        )
    )
    hashes = {p.name: sha256_file(p) for p in root.glob("keiba_*.csv")}
    raw = pd.read_csv(root / "keiba_races.csv", dtype={"race_id": str})
    jra = raw[raw.venue.isin(VENUE_JP_TO_CODE)].copy()
    if not jra.race_id.str.fullmatch(r"\d{12}").all() or jra.race_id.duplicated().any():
        raise ValueError("Invalid or duplicate source JRA ids")
    if not jra.race_id.str[:4].astype(int).between(2010, 2025).all():
        raise ValueError("Unexpected source year")
    if not jra.race_id.str[4:6].eq(jra.venue.map(VENUE_JP_TO_CODE)).all():
        raise ValueError("JRA venue/code conflict")
    joined = jra.merge(official, on="race_id", how="left", validate="one_to_one")
    unresolved = joined[joined.actual_date.isna()]
    unresolved.to_csv(DOCS / "HISTORICAL_UNRESOLVED_MAPPINGS.csv", index=False)
    if len(unresolved):
        raise ValueError(f"Unresolved mappings: {unresolved.race_id.tolist()}")
    if not joined.venue.eq(joined.racecourse).all():
        raise ValueError("Official course disagrees with source venue")
    mapping_cols = [
        "race_id",
        "year",
        "racecourse",
        "meeting_number",
        "meeting_day",
        "race_number",
        "actual_date",
        "official_source_url",
        "mapping_status",
    ]
    official[mapping_cols].to_csv(DERIVED / "jra_race_date_map_2010_2025.csv", index=False)
    absent = official[~official.race_id.isin(jra.race_id)].copy()
    absent.to_csv(DOCS / "HISTORICAL_OFFICIAL_RACES_MISSING_FROM_SOURCE.csv", index=False)
    selected = joined[joined.race_kind.eq("flat")].copy()
    races = selected[
        [
            "race_id",
            "actual_date",
            "racecourse",
            "official_surface",
            "official_distance_m",
            "official_turn",
            "course_layout",
            "handicap_indicator",
            "official_class",
            "is_open_plus",
            "is_graded",
        ]
    ].rename(
        columns={
            "actual_date": "race_date",
            "official_surface": "surface",
            "official_distance_m": "distance_m",
            "official_turn": "turn_direction",
            "official_class": "race_class",
        }
    )
    if races.distance_m.isna().any() or races.surface.isna().any():
        raise ValueError("Official core race conditions missing")
    rows, diag = standardized_results(root / "keiba_results.csv", races, set(jra.race_id))
    rows.to_parquet(
        OUT / "standardized_jra_flat_source_v1.parquet", index=False, compression="zstd"
    )
    print("Standardized source:", len(rows), flush=True)
    panel = build_historical_panel(rows)
    assert_strict_history(panel)
    panel["temporal_split"] = assign_temporal_split(panel)
    safe_cols = list(
        dict.fromkeys(
            [*SAFE_EXPORT_COLUMNS, "prev_race_date", "course_layout", "handicap_indicator"]
        )
    )
    safe = panel[safe_cols].copy()
    if find_forbidden_market_columns(safe.columns):
        raise ValueError("Market contamination")
    paths = []

    def export(x: pd.DataFrame, name: str) -> None:
        p = OUT / name
        x.to_parquet(p, index=False, compression="zstd")
        paths.append(
            {"file": name, **count_summary(x), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        )

    export(safe, "jra_flat_historical_panel_v1.parquet")
    splits = {}
    for split in ["warmup", "train", "validation", "test"]:
        part = safe[safe.temporal_split.eq(split)]
        splits[split] = count_summary(part)
        export(part, f"jra_flat_{split}_v1.parquet")
    cohort_counts = {}
    for name, lo, hi in [
        ("all_turf", None, None),
        ("turf_sprint", 1000, 1400),
        ("turf_1200", 1200, 1200),
    ]:
        c = select_phase_a_cohort(panel, min_distance_m=lo, max_distance_m=hi)[safe_cols]
        cohort_counts[name] = {}
        export(c, f"phase_a_{name}_v1.parquet")
        for split in ["warmup", "train", "validation", "test"]:
            part = c[c.temporal_split.eq(split)]
            cohort_counts[name][split] = count_summary(part)
            export(part, f"phase_a_{name}_{split}_v1.parquet")
    pd.DataFrame(paths).drop(columns=["field_size_distribution"]).to_csv(
        OUT / "MANIFEST.csv", index=False
    )
    predictive = [
        c
        for c in SAFE_EXPORT_COLUMNS
        if c
        not in [
            "race_date",
            "race_id",
            "horse_id",
            "horse_name",
            "horse_no",
            "top3_label",
            "temporal_split",
        ]
    ]
    unavailable = ["course_layout", "handicap_indicator"]
    coverage = {
        c: {
            "known": int(rows[c].notna().sum()),
            "total": len(rows),
            "coverage": rows[c].notna().mean(),
        }
        for c in ["race_class", "course_layout", "handicap_indicator", "is_open_plus", "is_graded"]
    }
    missing = [
        {
            "feature": c,
            "missing_count": int(safe[c].isna().sum()),
            "missing_rate": safe[c].isna().mean(),
        }
        for c in [*predictive, *unavailable]
    ]
    pd.DataFrame(missing).to_csv(DOCS / "HISTORICAL_FEATURE_MISSINGNESS.csv", index=False)
    yearly = []
    for year in range(2010, 2026):
        o = official[official.year.eq(year)]
        s = selected[selected.year.eq(year)]
        p = panel[panel.race_date.dt.year.eq(year)]
        yearly.append(
            {
                "year": year,
                "official_JRA_races": len(o),
                "source_JRA_races": int(joined.year.eq(year).sum()),
                "official_flat_races": int(o.race_kind.eq("flat").sum()),
                "source_flat_races": len(s),
                "official_obstacle_races": int(o.race_kind.eq("obstacle").sum()),
                "runner_rows": len(p),
            }
        )
    pd.DataFrame(yearly).to_csv(DOCS / "HISTORICAL_PANEL_YEAR_COUNTS.csv", index=False)
    # Counterfactual check on actual held-out race rows: target outcomes must not
    # change that same race's predictors. No cohort is selected by its outcome.
    sample_ids = panel.groupby(panel.race_date.dt.year).race_id.first().tolist()
    sample_horses = panel[panel.race_id.isin(sample_ids)].horse_id.unique()
    check_rows = rows[rows.horse_id.isin(sample_horses)].copy()
    baseline = build_historical_panel(check_rows)
    mutated = check_rows.copy()
    altered = mutated.race_id.isin(sample_ids)
    mutated.loc[altered, ["finish_position", "race_time_seconds", "early_position"]] = np.nan
    changed = build_historical_panel(mutated)
    comparison_cols = [c for c in predictive if c != "front_forward_share"]
    same = baseline.race_id.isin(sample_ids)
    contamination = int(
        (
            ~(
                baseline.loc[same, comparison_cols].eq(changed.loc[same, comparison_cols])
                | (
                    baseline.loc[same, comparison_cols].isna()
                    & changed.loc[same, comparison_cols].isna()
                )
            )
        )
        .sum()
        .sum()
    )
    # Full small-race counterfactual includes pace share for complete selected races.
    pace_contamination = int(
        (
            ~(
                baseline.loc[same, "front_forward_share"].eq(
                    changed.loc[same, "front_forward_share"]
                )
                | (
                    baseline.loc[same, "front_forward_share"].isna()
                    & changed.loc[same, "front_forward_share"].isna()
                )
            )
        ).sum()
    )
    chronology = int(
        (panel.prev_race_date.notna() & panel.prev_race_date.ge(panel.race_date)).sum()
    )
    dup = int(safe.duplicated(["race_id", "horse_id"]).sum())
    checks = {
        "chronology_violations": chronology,
        "duplicate_keys": dup,
        "market_leakage": len(find_forbidden_market_columns(safe.columns)),
        "future_year_rows": int(panel.race_date.dt.year.gt(2025).sum()),
        "same_race_predictor_contamination": contamination,
        "same_race_pace_contamination": pace_contamination,
        "unresolved_mappings": len(unresolved),
        "malformed_JRA_rows": diag["malformed_JRA"],
    }
    if any(checks.values()):
        raise ValueError(f"QA check failure: {checks}")
    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    freeze = {
        "status": "BLOCKED_PENDING_OFFICIAL_SOURCE_RECONCILIATION" if len(absent) else "PASS",
        "source": metadata,
        "source_files_sha256": hashes,
        "adapter_version": "official-JRA-adapter-v1",
        "code_commit": code_sha,
        "date_mapping_sha256": sha256_file(DERIVED / "jra_race_date_map_2010_2025.csv"),
        "mapping_mapped": len(joined),
        "mapping_total": len(jra),
        "mapping_coverage": joined.actual_date.notna().mean(),
        "data": count_summary(safe),
        "date_min": panel.race_date.min(),
        "date_max": panel.race_date.max(),
        "splits": splits,
        "cohorts": cohort_counts,
        "checks": checks,
        "result_diagnostics": diag,
        "official_source_missing_races": len(absent),
        "coverage": coverage,
        "historical_predictor_allowlist": predictive,
        "excluded_historical_predictors": unavailable,
        "artifacts": paths,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "generation_command": "python -u scripts/freeze_historical_panel.py",
    }
    json_write(OUT / "FREEZE.json", freeze)
    json_write(DOCS / "HISTORICAL_PANEL_QA.json", freeze)
    schema = [
        {
            "column": c,
            "dtype": str(safe[c].dtype),
            "role": "target"
            if c == "top3_label"
            else "predictor"
            if c in predictive
            else "audit_or_excluded",
        }
        for c in safe.columns
    ]
    pd.DataFrame(schema).to_csv(DOCS / "HISTORICAL_TRAINING_SCHEMA.csv", index=False)
    report = [
        "# Historical Panel QA Report",
        "",
        f"Status: **{freeze['status']}**",
        "",
        "All figures below are computed by Python from actual materialized artifacts.",
        "",
        "```json",
        json.dumps(
            {k: v for k, v in freeze.items() if k != "artifacts"},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        "```",
        "",
        "## Annual reconciliation",
        "",
        pd.DataFrame(yearly).to_markdown(index=False),
        "",
        "## Feature missingness",
        "",
        pd.DataFrame(missing).to_markdown(index=False),
        "",
        "Malformed non-JRA rows are enumerated in HISTORICAL_MALFORMED_ROWS.csv; no JRA malformed row was skipped.",
        "Official-only races are enumerated in HISTORICAL_OFFICIAL_RACES_MISSING_FROM_SOURCE.csv. They require investigation before closing Stage 3.6.",
        "Numeric final placings, including (降)/(再), use the numeric portion. 取/除 never enter history. 中/失 enter starts with label zero and missing placing/time.",
        "The source pseudo-date, source blank race names, market odds/popularity/payouts never enter model input.",
        "Counts mean observed JRA flat history since 2010, not lifetime starts or NAR/overseas starts.",
        "2025 has only structural QA; no feature/cohort/model selection or fitting has occurred.",
    ]
    (DOCS / "HISTORICAL_PANEL_QA_REPORT.md").write_text("\n".join(report) + "\n")
    (DOCS / "HISTORICAL_PANEL_FREEZE_V1.md").write_text(
        "# Historical Panel Freeze v1\n\n"
        + "\n".join(report[2:8])
        + "\n\n## Files\n\n"
        + pd.DataFrame(paths).drop(columns=["field_size_distribution"]).to_markdown(index=False)
        + "\n"
    )
    print(
        json.dumps(
            {
                k: freeze[k]
                for k in [
                    "status",
                    "mapping_mapped",
                    "data",
                    "checks",
                    "official_source_missing_races",
                ]
            },
            default=str,
        ),
        flush=True,
    )
    if len(absent):
        raise ValueError("Official/source reconciliation is unresolved; cannot close Stage 3.6")


if __name__ == "__main__":
    main()
