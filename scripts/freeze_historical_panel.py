"""Materialize and fail-closed QA the frozen 2010-2025 JRA historical panel."""

# ruff: noqa: UP032, RUF046

from __future__ import annotations

import csv
import json
import platform
import re
import subprocess
from collections import Counter
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import requests
from materialize_historical_training_dataset import (
    GRADE_RE,
    OBSTACLE_RE,
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
YEARS = range(2010, 2026)


def json_write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n")


def official_day_index() -> pd.DataFrame:
    files = [DERIVED / ("jra_official_{}.csv".format(year)) for year in YEARS]
    absent = [str(p) for p in files if not p.exists()]
    if absent:
        raise ValueError("Missing official year extractions: {}".format(absent))
    frames = [pd.read_csv(p, dtype={"race_day_key": str}) for p in files]
    df = pd.concat(frames, ignore_index=True)
    required = {
        "race_day_key",
        "year",
        "racecourse",
        "meeting_number",
        "meeting_day",
        "actual_date",
        "official_source_url",
        "mapping_status",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Official day-map columns missing: {}".format(sorted(missing)))
    df["race_day_key"] = df["race_day_key"].astype(str).str.zfill(10)
    date = pd.to_datetime(df["actual_date"], errors="raise")
    if not date.between("2010-01-01", "2025-12-31").all():
        raise ValueError("Official date outside frozen interval")
    if df.groupby("race_day_key").actual_date.nunique().gt(1).any():
        raise ValueError("Conflicting official race-day dates")
    keys = ["year", "racecourse", "meeting_number", "meeting_day"]
    if df.groupby(keys).actual_date.nunique().gt(1).any():
        raise ValueError("Conflicting (year, course, meeting, day) -> date mapping")
    df = df.sort_values(["race_day_key", "official_source_url"]).drop_duplicates(
        "race_day_key", keep="first"
    )
    return df.reset_index(drop=True)


def official_condition_index() -> pd.DataFrame:
    files = [DERIVED / ("jra_official_conditions_{}.csv".format(year)) for year in YEARS]
    available = [p for p in files if p.exists()]
    if not available:
        return pd.DataFrame()
    df = pd.concat(
        [pd.read_csv(p, dtype={"race_id": str}) for p in available],
        ignore_index=True,
    )
    if df.empty:
        return df
    df["race_id"] = df["race_id"].astype(str).str.zfill(12)
    if df.race_id.duplicated().any():
        dup = df.loc[df.race_id.duplicated(False), "race_id"].tolist()
        raise ValueError("Duplicate official condition race ids: {}".format(dup[:20]))
    return df.sort_values("race_id").reset_index(drop=True)


def conservative_class(raw_class: object, race_name: object) -> str | None:
    raw = "" if pd.isna(raw_class) else str(raw_class)
    name = "" if pd.isna(race_name) else str(race_name)
    text = raw + " " + name
    if "新馬" in text:
        return "Newcomer"
    if "未勝利" in text:
        return "Maiden"
    if "1勝クラス" in text or "500万" in text:
        return "Class1"
    if "2勝クラス" in text or "1000万" in text:
        return "Class2"
    if "3勝クラス" in text or "1600万" in text:
        return "Class3"
    if raw == "オープン" or "オープン" in text or bool(GRADE_RE.search(name)):
        return "Open"
    return None


def inspect_result_file(path: Path, jra_ids: set[str]) -> tuple[dict[str, float], dict]:
    """Collect obstacle evidence and independently audit malformed rows."""
    winner_last3f: dict[str, float] = {}
    malformed: list[dict] = []
    total = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        expected = len(header)
        idx = {name: i for i, name in enumerate(header)}
        for line, row in enumerate(reader, 2):
            total += 1
            rid = row[0].strip() if row else ""
            if len(row) != expected:
                malformed.append(
                    {
                        "line": line,
                        "race_id": rid,
                        "jra": rid in jra_ids,
                        "expected_fields": expected,
                        "actual_fields": len(row),
                    }
                )
                continue
            if rid not in jra_ids:
                continue
            rank = (row[idx["rank"]] or "").strip()
            if re.match(r"^1(?:\D|$)", rank):
                value = parse_float(row[idx["last_3f"]])
                if pd.notna(value):
                    winner_last3f[rid] = float(value)
    pd.DataFrame(malformed).to_csv(DOCS / "HISTORICAL_MALFORMED_ROWS.csv", index=False)
    return winner_last3f, {
        "source_result_rows": total,
        "malformed_all": len(malformed),
        "malformed_JRA": sum(int(x["jra"]) for x in malformed),
    }


def classify_and_standardize_races(
    raw: pd.DataFrame,
    days: pd.DataFrame,
    conditions: pd.DataFrame,
    winner_last3f: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    jra = raw.loc[raw.venue.isin(VENUE_JP_TO_CODE)].copy()
    jra["race_id"] = jra["race_id"].astype(str).str.strip()
    if not jra.race_id.str.fullmatch(r"\d{12}").all() or jra.race_id.duplicated().any():
        raise ValueError("Invalid or duplicate source JRA ids")

    jra["year"] = jra.race_id.str[:4].astype(int)
    if not jra.year.between(2010, 2025).all():
        raise ValueError("Unexpected JRA source year")
    jra["course_code"] = jra.race_id.str[4:6]
    jra["meeting_number"] = jra.race_id.str[6:8].astype(int)
    jra["meeting_day"] = jra.race_id.str[8:10].astype(int)
    jra["race_number_from_id"] = jra.race_id.str[10:12].astype(int)
    if not jra.course_code.eq(jra.venue.map(VENUE_JP_TO_CODE)).all():
        raise ValueError("JRA venue/code conflict")
    source_race_no = pd.to_numeric(jra.race_number, errors="coerce")
    if (source_race_no.notna() & source_race_no.ne(jra.race_number_from_id)).any():
        raise ValueError("JRA source race_number conflicts with race_id")

    jra["race_day_key"] = jra.race_id.str[:10]
    day_cols = [
        "race_day_key",
        "actual_date",
        "official_source_url",
        "mapping_status",
        "racecourse",
    ]
    joined = jra.merge(days[day_cols], on="race_day_key", how="left", validate="many_to_one")
    unresolved = joined.loc[joined.actual_date.isna()].copy()
    unresolved.to_csv(DOCS / "HISTORICAL_UNRESOLVED_MAPPINGS.csv", index=False)
    if len(unresolved):
        raise ValueError(
            "Unresolved official date mappings: {}".format(unresolved.race_id.tolist()[:50])
        )
    if not joined.venue.eq(joined.racecourse).all():
        raise ValueError("Official racecourse disagrees with source venue")

    if not conditions.empty:
        condition_cols = [
            c
            for c in [
                "race_id",
                "race_kind",
                "official_surface",
                "official_distance_m",
                "course_layout",
                "official_turn",
                "handicap_indicator",
                "official_class",
                "is_open_plus",
                "is_graded",
            ]
            if c in conditions.columns
        ]
        joined = joined.merge(
            conditions[condition_cols],
            on="race_id",
            how="left",
            validate="one_to_one",
        )
    for col in [
        "race_kind",
        "official_surface",
        "official_distance_m",
        "course_layout",
        "official_turn",
        "handicap_indicator",
        "official_class",
        "is_open_plus",
        "is_graded",
    ]:
        if col not in joined:
            joined[col] = np.nan

    source_surface = joined.course_type.map({"芝": "turf", "ダ": "dirt"})
    source_distance = pd.to_numeric(joined.distance, errors="coerce")
    source_distance = source_distance.where(source_distance.gt(0))
    official_distance = pd.to_numeric(joined.official_distance_m, errors="coerce")
    official_surface = joined.official_surface.where(
        joined.official_surface.isin(["turf", "dirt"])
    )

    surface_conflict = (
        official_surface.notna() & source_surface.notna() & official_surface.ne(source_surface)
    )
    distance_conflict = (
        official_distance.notna()
        & source_distance.notna()
        & official_distance.ne(source_distance)
    )
    if surface_conflict.any() or distance_conflict.any():
        examples = joined.loc[surface_conflict | distance_conflict, "race_id"].tolist()[:30]
        raise ValueError("Official/source race-condition conflict: {}".format(examples))

    joined["winner_last3f"] = joined.race_id.map(winner_last3f)
    name_obstacle = joined.race_name.fillna("").astype(str).str.contains(
        OBSTACLE_RE, regex=True
    )
    last3f_obstacle = joined.winner_last3f.lt(20)
    fallback_obstacle = name_obstacle | last3f_obstacle

    official_kind = joined.race_kind.where(joined.race_kind.isin(["flat", "obstacle"]))
    known_official = official_kind.notna()
    kind_conflict = known_official & (
        (official_kind.eq("obstacle") & joined.winner_last3f.notna() & ~last3f_obstacle)
        | (official_kind.eq("flat") & last3f_obstacle)
    )
    if kind_conflict.any():
        examples = joined.loc[kind_conflict, "race_id"].tolist()[:30]
        raise ValueError("Official/fallback obstacle classifier conflict: {}".format(examples))

    joined["surface"] = official_surface.combine_first(source_surface)
    joined["distance_m"] = official_distance.combine_first(source_distance)
    niigata_straight = (
        joined.venue.eq("新潟") & joined.distance_m.eq(1000) & ~fallback_obstacle
    )
    joined.loc[niigata_straight & joined.surface.isna(), "surface"] = "turf"

    explicit_flat = official_kind.eq("flat")
    explicit_obstacle = official_kind.eq("obstacle")
    fallback_flat = (
        official_kind.isna()
        & ~fallback_obstacle
        & joined.surface.notna()
        & joined.distance_m.notna()
    )
    joined["race_kind_final"] = np.where(
        explicit_obstacle | (official_kind.isna() & fallback_obstacle),
        "obstacle",
        np.where(explicit_flat | fallback_flat, "flat", "unknown"),
    )

    unknown = joined.loc[joined.race_kind_final.eq("unknown")].copy()
    unknown.to_csv(DOCS / "HISTORICAL_UNCLASSIFIED_RACES.csv", index=False)
    if len(unknown):
        raise ValueError(
            "Unclassified JRA races remain: {} {}".format(
                len(unknown), unknown.race_id.tolist()[:30]
            )
        )

    flat = joined.loc[joined.race_kind_final.eq("flat")].copy()
    missing_core = flat.surface.isna() | flat.distance_m.isna()
    if missing_core.any():
        bad = flat.loc[missing_core].copy()
        bad.to_csv(DOCS / "HISTORICAL_FLAT_CORE_METADATA_MISSING.csv", index=False)
        raise ValueError(
            "Flat races missing surface/distance: {}".format(bad.race_id.tolist()[:30])
        )

    source_turn = joined.turn.map({"右": "right", "左": "left"})
    flat["turn_direction"] = flat.official_turn.combine_first(source_turn.loc[flat.index])
    straight = flat.venue.eq("新潟") & flat.surface.eq("turf") & flat.distance_m.eq(1000)
    flat.loc[straight, "turn_direction"] = "straight"
    flat["turn_direction"] = flat.turn_direction.fillna("other")

    source_class = pd.Series(
        [
            conservative_class(rc, rn)
            for rc, rn in zip(flat.race_class, flat.race_name, strict=False)
        ],
        index=flat.index,
        dtype="object",
    )
    flat["race_class_norm"] = flat.official_class.combine_first(source_class)

    source_grade = flat.race_name.fillna("").astype(str).str.contains(GRADE_RE, regex=True)
    official_grade = pd.to_numeric(flat.is_graded, errors="coerce")
    flat["is_graded_final"] = official_grade
    flat.loc[source_grade, "is_graded_final"] = 1
    source_name_known = (
        flat.race_name.notna() & flat.race_name.astype(str).str.strip().ne("")
    )
    flat.loc[flat.is_graded_final.isna() & source_name_known, "is_graded_final"] = 0

    official_open = pd.to_numeric(flat.is_open_plus, errors="coerce")
    flat["is_open_plus_final"] = official_open
    flat.loc[
        flat.race_class_norm.eq("Open") | flat.is_graded_final.eq(1),
        "is_open_plus_final",
    ] = 1
    class_known = flat.race_class_norm.notna()
    flat.loc[flat.is_open_plus_final.isna() & class_known, "is_open_plus_final"] = 0

    flat["course_layout_final"] = flat.course_layout
    flat["handicap_indicator_final"] = pd.to_numeric(
        flat.handicap_indicator, errors="coerce"
    )

    mapping = joined[
        [
            "race_id",
            "year",
            "racecourse",
            "meeting_number",
            "meeting_day",
            "race_number_from_id",
            "actual_date",
            "official_source_url",
            "mapping_status",
        ]
    ].rename(columns={"race_number_from_id": "race_number"})
    mapping = mapping.sort_values("race_id").reset_index(drop=True)

    diagnostics = {
        "source_JRA_races": int(len(joined)),
        "flat_races": int(len(flat)),
        "obstacle_races": int(joined.race_kind_final.eq("obstacle").sum()),
        "unknown_races": int(joined.race_kind_final.eq("unknown").sum()),
        "official_race_condition_rows_matched": int(known_official.sum()),
        "fallback_obstacle_races": int(
            (official_kind.isna() & fallback_obstacle).sum()
        ),
        "niigata_straight_surface_restored": int(
            (niigata_straight & source_surface.isna()).sum()
        ),
        "official_surface_source_conflicts": int(surface_conflict.sum()),
        "official_distance_source_conflicts": int(distance_conflict.sum()),
        "official_fallback_kind_conflicts": int(kind_conflict.sum()),
    }
    return flat, mapping, diagnostics


def standardized_results(
    path: Path,
    races: pd.DataFrame,
    all_jra_ids: set[str],
) -> tuple[pd.DataFrame, dict]:
    meta = races.set_index("race_id")
    records: list[dict] = []
    malformed: list[dict] = []
    ranks = Counter()
    scratch = Counter()
    declared: dict[str, int] = {}
    total = 0

    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = len(reader.fieldnames or [])
        for line, row in enumerate(reader, 2):
            total += 1
            rid = (row.get("race_id") or "").strip()
            extras = row.get(None, [])
            malformed_row = (
                None in row
                or bool(extras)
                or any(v is None for k, v in row.items() if k is not None)
            )
            if malformed_row:
                malformed.append(
                    {
                        "line": line,
                        "race_id": rid,
                        "jra": rid in all_jra_ids,
                        "flat": rid in meta.index,
                        "expected_fields": expected,
                        "actual_fields": expected + len(extras or []),
                    }
                )
                continue
            if rid not in meta.index:
                continue

            ranks[row["rank"]] += 1
            pos, status, started = parse_rank(row["rank"])
            if status.startswith("other:"):
                raise ValueError("Unhandled placing status {}: {}".format(rid, row["rank"]))
            raw_no = parse_float(row["number"])
            if pd.isna(raw_no) or int(raw_no) <= 0:
                raise ValueError("Invalid horse number: {} {!r}".format(rid, row["number"]))
            no = int(raw_no)
            declared[rid] = max(declared.get(rid, 0), no)
            if not started:
                scratch[row["rank"]] += 1
                continue

            sex, age = parse_sex_age(row["sex_age"])
            records.append(
                {
                    "race_id": rid,
                    "horse_id": (row["horse_id"] or "").strip(),
                    "horse_name": row["horse_name"],
                    "horse_no": no,
                    "sex": sex,
                    "age": age,
                    "assigned_weight_kg": parse_float(row["weight"]),
                    "finish_position": pos,
                    "finish_status": status,
                    "race_time_seconds": (
                        parse_time(row["time"]) if status == "finished" else np.nan
                    ),
                    "early_position": parse_early_position(row["passing"]),
                }
            )

    pd.DataFrame(malformed).to_csv(DOCS / "HISTORICAL_MALFORMED_ROWS.csv", index=False)
    if any(r["jra"] for r in malformed):
        bad_jra = [r for r in malformed if r["jra"]]
        raise ValueError("Malformed JRA rows require repair: {}".format(bad_jra[:20]))

    race_cols = [
        "race_id",
        "actual_date",
        "racecourse",
        "surface",
        "distance_m",
        "turn_direction",
        "race_class_norm",
        "is_open_plus_final",
        "is_graded_final",
        "course_layout_final",
        "handicap_indicator_final",
    ]
    rmeta = races[race_cols].rename(
        columns={
            "actual_date": "race_date",
            "race_class_norm": "race_class",
            "is_open_plus_final": "is_open_plus",
            "is_graded_final": "is_graded",
            "course_layout_final": "course_layout",
            "handicap_indicator_final": "handicap_indicator",
        }
    )
    df = pd.DataFrame(records).merge(rmeta, on="race_id", validate="many_to_one")
    df["race_date"] = pd.to_datetime(df.race_date, errors="raise")
    df["field_size"] = df.groupby("race_id").horse_id.transform("size")
    df["declared_field_size"] = df.race_id.map(declared)
    df = df.sort_values(["race_date", "race_id", "horse_no"]).reset_index(drop=True)

    if set(meta.index) != set(df.race_id):
        missing = sorted(set(meta.index) - set(df.race_id))
        raise ValueError("Selected flat races without starter rows: {}".format(missing[:30]))
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
        "malformed_JRA": sum(int(r["jra"]) for r in malformed),
        "malformed_flat": sum(int(r["flat"]) for r in malformed),
        "rank_counts": dict(ranks),
        "nonstarter_counts": dict(scratch),
        "finish_status_counts": df.finish_status.value_counts().to_dict(),
    }
    return df, diag


def count_summary(x: pd.DataFrame) -> dict:
    fields = x.drop_duplicates("race_id").field_size
    return {
        "races": int(x.race_id.nunique()),
        "runner_rows": int(len(x)),
        "horses": int(x.horse_id.nunique()),
        "top3_prevalence": float(x.top3_label.mean()) if len(x) else None,
        "field_size_distribution": {
            str(k): int(v) for k, v in fields.value_counts().sort_index().items()
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    days = official_day_index()
    conditions = official_condition_index()

    response = requests.get(
        "https://www.kaggle.com/api/v1/datasets/view/{}".format(SOURCE), timeout=60
    )
    response.raise_for_status()
    md = response.json()
    metadata = {
        k: md.get(k)
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
            "{}/versions/1".format(SOURCE),
            output_dir="data/historical_raw/kaggle_v1",
        )
    )
    races_path = root / "keiba_races.csv"
    results_path = root / "keiba_results.csv"
    payouts_path = root / "keiba_payouts.csv"
    for path in [races_path, results_path, payouts_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    hashes = {
        path.name: sha256_file(path)
        for path in [races_path, results_path, payouts_path]
    }

    raw = pd.read_csv(races_path, dtype={"race_id": str}, low_memory=False)
    all_jra = raw.loc[raw.venue.isin(VENUE_JP_TO_CODE), "race_id"].astype(str)
    all_jra_ids = set(all_jra)
    winner_last3f, raw_result_diag = inspect_result_file(results_path, all_jra_ids)
    if raw_result_diag["malformed_JRA"]:
        raise ValueError(
            "Malformed JRA rows detected before standardization: {}".format(raw_result_diag)
        )

    flat, date_mapping, race_diag = classify_and_standardize_races(
        raw, days, conditions, winner_last3f
    )
    date_map_path = DERIVED / "jra_race_date_map_2010_2025.csv"
    date_mapping.to_csv(date_map_path, index=False)
    if date_mapping.race_id.duplicated().any():
        raise ValueError("Duplicate race_id in final date mapping")
    if not pd.to_datetime(date_mapping.actual_date).between(
        "2010-01-01", "2025-12-31"
    ).all():
        raise ValueError("Mapped date outside target interval")
    if date_mapping.duplicated(["actual_date", "racecourse", "race_number"]).any():
        raise ValueError("Duplicate actual_date/racecourse/race_number")
    map_keys = ["year", "racecourse", "meeting_number", "meeting_day"]
    if date_mapping.groupby(map_keys).actual_date.nunique().gt(1).any():
        raise ValueError("Non-unique meeting-day date mapping")

    day_recon_rows: list[dict] = []
    source_by_day = date_mapping.groupby(date_mapping.race_id.str[:10]).race_number.apply(
        lambda values: sorted(int(x) for x in values)
    )
    for row in days.itertuples(index=False):
        nums = source_by_day.get(str(row.race_day_key), [])
        missing = [number for number in range(1, 13) if number not in nums]
        day_recon_rows.append(
            {
                "year": int(row.year),
                "race_day_key": str(row.race_day_key),
                "racecourse": row.racecourse,
                "meeting_number": int(row.meeting_number),
                "meeting_day": int(row.meeting_day),
                "actual_date": row.actual_date,
                "source_race_count": len(nums),
                "source_race_numbers": ";".join(str(x) for x in nums),
                "unfilled_1_to_12_slots": ";".join(str(x) for x in missing),
                "official_source_url": row.official_source_url,
            }
        )
    day_recon = pd.DataFrame(day_recon_rows)
    day_recon.to_csv(DOCS / "HISTORICAL_JRA_DAY_RECONCILIATION.csv", index=False)

    source_day_keys = set(date_mapping.race_id.str[:10])
    official_day_keys = set(days.race_day_key.astype(str))
    extra_source_days = source_day_keys - official_day_keys
    if extra_source_days:
        raise ValueError(
            "Source race days absent from official archive: {}".format(
                sorted(extra_source_days)[:20]
            )
        )

    rows, result_diag = standardized_results(results_path, flat, all_jra_ids)
    if result_diag["malformed_all"] != raw_result_diag["malformed_all"]:
        raise ValueError("Malformed-row audit changed between independent passes")

    rows.to_parquet(
        OUT / "standardized_jra_flat_source_v1.parquet",
        index=False,
        compression="zstd",
    )
    panel = build_historical_panel(rows)
    assert_strict_history(panel)
    panel["temporal_split"] = assign_temporal_split(panel)

    audit_cols = ["prev_race_date", "course_layout", "handicap_indicator"]
    safe_cols = list(dict.fromkeys([*SAFE_EXPORT_COLUMNS, *audit_cols]))
    safe = panel[safe_cols].copy()
    forbidden = find_forbidden_market_columns(safe.columns)
    if forbidden:
        raise ValueError("Market contamination: {}".format(forbidden))

    artifacts: list[dict] = []

    def export(frame: pd.DataFrame, name: str) -> None:
        path = OUT / name
        frame.to_parquet(path, index=False, compression="zstd")
        artifacts.append(
            {
                "file": name,
                **count_summary(frame),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )

    export(safe, "jra_flat_historical_panel_v1.parquet")
    splits: dict[str, dict] = {}
    for split in ["warmup", "train", "validation", "test"]:
        part = safe.loc[safe.temporal_split.eq(split)].copy()
        splits[split] = count_summary(part)
        export(part, "jra_flat_{}_v1.parquet".format(split))

    cohort_counts: dict[str, dict] = {}
    for name, low, high in [
        ("all_turf", None, None),
        ("turf_sprint_1000_1400", 1000, 1400),
        ("turf_1200", 1200, 1200),
    ]:
        cohort = select_phase_a_cohort(
            panel,
            surface="turf",
            min_field_size=8,
            min_prior_starts=3,
            min_distance_m=low,
            max_distance_m=high,
        )[safe_cols]
        export(cohort, "phase_a_{}_v1.parquet".format(name))
        cohort_counts[name] = {}
        for split in ["warmup", "train", "validation", "test"]:
            part = cohort.loc[cohort.temporal_split.eq(split)].copy()
            cohort_counts[name][split] = count_summary(part)
            export(part, "phase_a_{}_{}_v1.parquet".format(name, split))

    manifest = pd.DataFrame(artifacts)
    manifest.drop(columns=["field_size_distribution"]).to_csv(
        OUT / "MANIFEST.csv", index=False
    )

    panel_file = OUT / "jra_flat_historical_panel_v1.parquet"
    panel_sha = sha256_file(panel_file)
    date_map_sha = sha256_file(date_map_path)

    predictive = [
        column
        for column in SAFE_EXPORT_COLUMNS
        if column
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
    coverage_cols = [
        "race_class",
        "course_layout",
        "handicap_indicator",
        "is_open_plus",
        "is_graded",
    ]
    coverage = {
        column: {
            "known": int(rows[column].notna().sum()),
            "total": int(len(rows)),
            "coverage": float(rows[column].notna().mean()),
        }
        for column in coverage_cols
    }
    missing_rows = [
        {
            "feature": column,
            "missing_count": int(safe[column].isna().sum()),
            "missing_rate": float(safe[column].isna().mean()),
        }
        for column in [*predictive, "course_layout", "handicap_indicator"]
    ]
    pd.DataFrame(missing_rows).to_csv(
        DOCS / "HISTORICAL_FEATURE_MISSINGNESS.csv", index=False
    )

    yearly_rows: list[dict] = []
    for year in YEARS:
        year_map = date_mapping.loc[date_mapping.year.eq(year)]
        year_flat = flat.loc[flat.year.eq(year)]
        year_rows = panel.loc[panel.race_date.dt.year.eq(year)]
        year_days = days.loc[days.year.eq(year)]
        yearly_rows.append(
            {
                "year": year,
                "official_race_days": int(len(year_days)),
                "official_12_race_slots": int(len(year_days) * 12),
                "source_JRA_races": int(len(year_map)),
                "unfilled_slots_vs_12_race_cards": int(len(year_days) * 12 - len(year_map)),
                "flat_races": int(year_flat.race_id.nunique()),
                "obstacle_races": int(len(year_map) - year_flat.race_id.nunique()),
                "runner_rows": int(len(year_rows)),
            }
        )
    yearly = pd.DataFrame(yearly_rows)
    yearly.to_csv(DOCS / "HISTORICAL_PANEL_YEAR_COUNTS.csv", index=False)

    sample_ids = panel.groupby(panel.race_date.dt.year).race_id.first().tolist()
    sample_horses = panel.loc[panel.race_id.isin(sample_ids), "horse_id"].unique()
    check_rows = rows.loc[rows.horse_id.isin(sample_horses)].copy()
    baseline = build_historical_panel(check_rows)
    mutated = check_rows.copy()
    altered = mutated.race_id.isin(sample_ids)
    mutated.loc[
        altered,
        ["finish_position", "race_time_seconds", "early_position"],
    ] = np.nan
    changed = build_historical_panel(mutated)

    comparison_cols = [
        column
        for column in predictive
        if column
        not in [
            "race_class",
            "turn_direction",
            "racecourse",
            "surface",
            "sex",
            "front_forward_share",
        ]
        and column in baseline.columns
    ]
    same = baseline.race_id.isin(sample_ids)
    equal = baseline.loc[same, comparison_cols].eq(changed.loc[same, comparison_cols])
    both_missing = (
        baseline.loc[same, comparison_cols].isna()
        & changed.loc[same, comparison_cols].isna()
    )
    contamination = int((~(equal | both_missing)).sum().sum())
    pace_equal = baseline.loc[same, "front_forward_share"].eq(
        changed.loc[same, "front_forward_share"]
    )
    pace_missing = (
        baseline.loc[same, "front_forward_share"].isna()
        & changed.loc[same, "front_forward_share"].isna()
    )
    pace_contamination = int((~(pace_equal | pace_missing)).sum())

    chronology = int(
        (panel.prev_race_date.notna() & panel.prev_race_date.ge(panel.race_date)).sum()
    )
    duplicate = int(safe.duplicated(["race_id", "horse_id"]).sum())
    duplicate_targets = int(
        safe.loc[
            safe.temporal_split.isin(["train", "validation", "test"])
        ].duplicated(["race_id", "horse_id"]).sum()
    )
    checks = {
        "chronology_violations": chronology,
        "duplicate_race_horse_keys": duplicate,
        "duplicate_target_rows": duplicate_targets,
        "market_leakage_columns": len(find_forbidden_market_columns(safe.columns)),
        "future_year_rows": int(panel.race_date.dt.year.gt(2025).sum()),
        "same_race_predictor_contamination": contamination,
        "same_race_pace_contamination": pace_contamination,
        "unresolved_date_mappings": int(date_mapping.actual_date.isna().sum()),
        "malformed_JRA_rows": int(result_diag["malformed_JRA"]),
        "unclassified_JRA_races": int(race_diag["unknown_races"]),
    }
    if any(checks.values()):
        raise ValueError("QA check failure: {}".format(checks))

    mapping_total = int(len(date_mapping))
    mapping_mapped = int(date_mapping.actual_date.notna().sum())
    mapping_coverage = float(mapping_mapped / mapping_total) if mapping_total else 0.0
    if mapping_coverage != 1.0:
        raise ValueError("Official date coverage is not 100%: {}".format(mapping_coverage))

    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    feature_exclusions = {
        "course_layout": {
            "coverage": coverage["course_layout"]["coverage"],
            "historical_model_input": False,
            "reason": "official PDF race-level extraction is incomplete across 2010-2025",
        },
        "handicap_indicator": {
            "coverage": coverage["handicap_indicator"]["coverage"],
            "historical_model_input": False,
            "reason": "official weight-clause extraction is incomplete across 2010-2025",
        },
    }

    freeze = {
        "status": "PASS",
        "source": metadata,
        "source_files_sha256": hashes,
        "adapter_version": "jra-historical-adapter-v2",
        "feature_spec_version": "historical-v1.1",
        "code_commit": code_sha,
        "generation_command": "python -u scripts/freeze_historical_panel.py",
        "date_mapping": {
            "mapped": mapping_mapped,
            "total": mapping_total,
            "coverage": mapping_coverage,
            "sha256": date_map_sha,
        },
        "data": {
            **count_summary(safe),
            "date_min": str(panel.race_date.min().date()),
            "date_max": str(panel.race_date.max().date()),
            "flat_races": int(flat.race_id.nunique()),
            "obstacle_races_excluded": int(race_diag["obstacle_races"]),
        },
        "splits": splits,
        "cohorts": cohort_counts,
        "checks": checks,
        "result_diagnostics": result_diag,
        "race_diagnostics": race_diag,
        "feature_coverage": coverage,
        "historical_model_predictors": predictive,
        "historical_feature_exclusions": feature_exclusions,
        "panel": {
            "file": "jra_flat_historical_panel_v1.parquet",
            "sha256": panel_sha,
            "bytes": int(panel_file.stat().st_size),
        },
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    json_write(OUT / "FREEZE.json", freeze)
    json_write(DOCS / "HISTORICAL_PANEL_QA.json", freeze)

    schema = [
        {
            "column": column,
            "dtype": str(safe[column].dtype),
            "role": (
                "target"
                if column == "top3_label"
                else "predictor"
                if column in predictive
                else "audit_or_excluded"
            ),
        }
        for column in safe.columns
    ]
    pd.DataFrame(schema).to_csv(DOCS / "HISTORICAL_TRAINING_SCHEMA.csv", index=False)

    report = [
        "# Historical Panel QA Report",
        "",
        "Status: PASS",
        "",
        "All numerical values below are computed by Python from materialized 2010-2025 data.",
        "",
        "## Source and date mapping",
        "",
        "- source: {}".format(metadata["title"]),
        "- license: {}".format(metadata["licenseName"]),
        "- lastUpdated: {}".format(metadata["lastUpdated"]),
        "- version: {}".format(metadata["currentVersionNumber"]),
        "- mapped race IDs: {:,} / {:,} ({:.6%})".format(
            mapping_mapped, mapping_total, mapping_coverage
        ),
        "- date-map SHA256: {}".format(date_map_sha),
        "",
        "The source date column is never used. Every JRA race_id is mapped through the official",
        "JRA annual-results PDF archive at year/course/meeting/meeting-day granularity.",
        "",
        "## Final JRA flat historical panel",
        "",
        "- races: {:,}".format(safe.race_id.nunique()),
        "- runner rows: {:,}".format(len(safe)),
        "- unique horses: {:,}".format(safe.horse_id.nunique()),
        "- actual date range: {} to {}".format(
            panel.race_date.min().date(), panel.race_date.max().date()
        ),
        "- obstacle races excluded: {:,}".format(race_diag["obstacle_races"]),
        "- malformed raw result rows: {:,}".format(result_diag["malformed_all"]),
        "- malformed JRA result rows: {:,}".format(result_diag["malformed_JRA"]),
        "",
        "## Annual JRA reconciliation",
        "",
        yearly.to_markdown(index=False),
        "",
        "official_12_race_slots is a schedule-capacity diagnostic, not a forced race count.",
        "Cancelled or short-card slots are never invented. Every realized source JRA race_id",
        "must map to an official venue-day; details are in HISTORICAL_JRA_DAY_RECONCILIATION.csv.",
        "",
        "## Temporal split",
        "",
        "| split | races | runner rows | top3 prevalence |",
        "|---|---:|---:|---:|",
    ]
    for split in ["warmup", "train", "validation", "test"]:
        item = splits[split]
        report.append(
            "| {} | {:,} | {:,} | {:.6f} |".format(
                split,
                item["races"],
                item["runner_rows"],
                item["top3_prevalence"],
            )
        )

    report += ["", "## Leakage and structural QA", ""]
    for key, value in checks.items():
        report.append("- {}: {}".format(key, value))

    report += [
        "",
        "## Feature coverage and historical-v1.1 decision",
        "",
        "| feature | known | total | coverage |",
        "|---|---:|---:|---:|",
    ]
    for column in coverage_cols:
        item = coverage[column]
        report.append(
            "| {} | {:,} | {:,} | {:.6%} |".format(
                column, item["known"], item["total"], item["coverage"]
            )
        )
    report += [
        "",
        "course_layout and handicap_indicator are retained as audit columns but excluded from",
        "the Stage 4 historical model allowlist because official-PDF extraction is incomplete",
        "across the full period. No fake/default value is imputed. The separate 18-runner",
        "current-race matrix remains unchanged.",
        "",
        "## Missingness",
        "",
        pd.DataFrame(missing_rows).to_markdown(index=False),
        "",
        "## Finish-status policy",
        "",
        "- 取 / 除: non-starters; excluded from entrant/start history.",
        "- 中 / 失: started; career start counted, top3_label=0, placing/time-derived values missing.",
        "- numeric ranks including (降) / (再): numeric official placing prefix is used.",
        "",
        "## Obstacle classification",
        "",
        "Official PDF race-condition classification is used where machine-readable. Else source",
        "race metadata is combined with obstacle-name markers and the source winner-last3F",
        "encoding below 20. That fallback is only a race-type classifier, never a predictor,",
        "and any official/fallback disagreement is a hard failure.",
        "",
        "## Final artifact",
        "",
        "- file: jra_flat_historical_panel_v1.parquet",
        "- SHA256: {}".format(panel_sha),
        "- bytes: {:,}".format(panel_file.stat().st_size),
        "- Actions artifact directory: data/historical_processed/stage36_freeze_v1/",
        "",
        "2025 is an untouched final historical test. No 2025 outcome is used for feature selection,",
        "hyperparameter tuning, or cohort selection. No 2026 race outcome is present.",
        "",
    ]
    (DOCS / "HISTORICAL_PANEL_QA_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    freeze_doc = [
        "# Historical Panel Freeze v1",
        "",
        "Status: PASS",
        "",
        "- source: {} version {}".format(SOURCE, metadata["currentVersionNumber"]),
        "- source license: {}".format(metadata["licenseName"]),
        "- source lastUpdated: {}".format(metadata["lastUpdated"]),
        "- adapter: jra-historical-adapter-v2",
        "- feature spec: historical-v1.1",
        "- panel-builder commit: {}".format(code_sha),
        "- date-map SHA256: {}".format(date_map_sha),
        "- panel file: jra_flat_historical_panel_v1.parquet",
        "- panel SHA256: {}".format(panel_sha),
        "- races: {:,}".format(safe.race_id.nunique()),
        "- runner rows: {:,}".format(len(safe)),
        "- horses: {:,}".format(safe.horse_id.nunique()),
        "- actual date range: {} to {}".format(
            panel.race_date.min().date(), panel.race_date.max().date()
        ),
        "",
        "Generation command: python -u scripts/freeze_historical_panel.py",
        "",
        "Raw Kaggle CSVs and JRA PDFs are not committed. The Parquet panel is retained as",
        "a GitHub Actions artifact; aggregate QA, mapping, schema, and fingerprints are committed.",
        "",
        "Historical model exclusions: course_layout and handicap_indicator because official",
        "coverage is incomplete. No synthetic/default fill is used.",
        "",
    ]
    (DOCS / "HISTORICAL_PANEL_FREEZE_V1.md").write_text(
        "\n".join(freeze_doc), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "status": freeze["status"],
                "date_mapping": freeze["date_mapping"],
                "data": freeze["data"],
                "splits": freeze["splits"],
                "checks": freeze["checks"],
                "panel": freeze["panel"],
            },
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
