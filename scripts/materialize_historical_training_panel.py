"""Materialize the Stage 3.6 leakage-safe historical training dataset.

Raw source files stay ephemeral / git-ignored. The committed outputs are QA,
provenance and cryptographic fingerprints; row-level Parquet files are uploaded
as a GitHub Actions artifact.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from urllib.request import urlopen

import kagglehub
import numpy as np
import pandas as pd

from keiba_place_lab.historical_panel import (
    assert_strict_history,
    build_historical_panel,
    find_forbidden_market_columns,
    select_phase_a_cohort,
)

PRIMARY = "noriyukifurufuru/japan-horse-racing-2010-2025"
SECONDARY = "takamotoki/jra-horse-racing-dataset"
RACEINFO_COMMIT = "da8eb65883495b4ae37fa65dc3a0d9c9076a9e3a"
DATE_HELPER_COMMIT = "3ca10446d8ae28a21ee67ceed6429bacc0ebba93"

JRA_VENUES = {
    "札幌": "Sapporo",
    "函館": "Hakodate",
    "福島": "Fukushima",
    "新潟": "Niigata",
    "東京": "Tokyo",
    "中山": "Nakayama",
    "中京": "Chukyo",
    "京都": "Kyoto",
    "阪神": "Hanshin",
    "小倉": "Kokura",
}
SEX_MAP = {"牡": "M", "牝": "F", "セ": "G"}
TURN_MAP = {"右": "right", "左": "left", "直": "straight"}

OUT_DIR = Path("data/historical_processed/stage36_v1")
QA_PATH = Path("docs/HISTORICAL_PANEL_QA.md")
FINGERPRINT_PATH = Path("docs/HISTORICAL_PANEL_FINGERPRINT.md")
YEAR_COUNTS_PATH = Path("docs/HISTORICAL_PANEL_YEAR_COUNTS.csv")


def download_dataset(slug: str, directory: str) -> Path:
    path = Path(kagglehub.dataset_download(slug, output_dir=directory, force_download=True))
    return path if path.is_dir() else path.parent


def parse_time_seconds(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.fullmatch(r"\s*(\d+):(\d{1,2}(?:\.\d+)?)\s*", str(value))
    if not match:
        return np.nan
    return 60.0 * float(match.group(1)) + float(match.group(2))


def parse_rank(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.match(r"^\s*(\d+)", str(value))
    return float(match.group(1)) if match else np.nan


def parse_first_position(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"\d+", str(value))
    return float(match.group(0)) if match else np.nan


def parse_sex_age(value: object) -> tuple[str | None, float]:
    if pd.isna(value):
        return None, np.nan
    match = re.match(r"^([牡牝セ])(\d+)", str(value))
    if not match:
        return None, np.nan
    return SEX_MAP[match.group(1)], float(match.group(2))


def load_secondary_dates(root: Path) -> pd.DataFrame:
    candidates = sorted(root.glob("*_race_result.csv"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one secondary race_result CSV, got {candidates}")
    frame = pd.read_csv(
        candidates[0],
        usecols=[1, 2],
        dtype="string",
        encoding="utf-8-sig",
        low_memory=False,
    )
    frame.columns = ["race_id", "race_date"]
    frame["race_id"] = frame["race_id"].str.strip()
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="coerce")
    frame = frame.dropna().drop_duplicates("race_id")
    return frame


def load_raceinfo_dates(year: int) -> pd.DataFrame:
    url = (
        "https://raw.githubusercontent.com/10matcho27/Keiba_AI/"
        f"{RACEINFO_COMMIT}/OUTPUT/RAW/raceInfo_{year}.csv"
    )
    frame = pd.read_csv(url, dtype="string", usecols=["race_ids", "race_date"])
    frame = frame.rename(columns={"race_ids": "race_id"})
    frame["race_id"] = frame["race_id"].str.strip()
    frame["race_date"] = pd.to_datetime(frame["race_date"], format="%Y%m%d", errors="coerce")
    return frame.dropna().drop_duplicates("race_id")


def load_2025_helper_dates(root: Path) -> pd.DataFrame:
    records: list[tuple[str, str]] = []
    for path in sorted(root.glob("*.csv")):
        date_text = path.stem
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "race_id" not in reader.fieldnames:
                continue
            for row in reader:
                race_id = str(row.get("race_id", "")).strip()
                if race_id:
                    records.append((race_id, date_text))
    frame = pd.DataFrame(records, columns=["race_id", "race_date"])
    frame["race_date"] = pd.to_datetime(frame["race_date"], format="%Y%m%d", errors="coerce")
    return frame.dropna().drop_duplicates("race_id")


def build_actual_date_map(secondary_root: Path, helper_2025_root: Path) -> tuple[pd.DataFrame, dict]:
    secondary = load_secondary_dates(secondary_root)
    helpers = pd.concat(
        [load_raceinfo_dates(year) for year in range(2021, 2025)],
        ignore_index=True,
    )
    helper_2025 = load_2025_helper_dates(helper_2025_root)

    overlap = secondary.merge(helpers, on="race_id", suffixes=("_secondary", "_helper"))
    overlap_mismatch = int(
        (overlap["race_date_secondary"] != overlap["race_date_helper"]).sum()
    )
    if overlap_mismatch:
        raise RuntimeError(f"actual-date helper mismatch in overlap: {overlap_mismatch}")

    date_map = pd.concat(
        [
            secondary.loc[secondary["race_date"].dt.year <= 2020],
            helpers,
            helper_2025,
        ],
        ignore_index=True,
    )
    conflict = (
        date_map.groupby("race_id")["race_date"].nunique(dropna=True).gt(1).sum()
    )
    if conflict:
        raise RuntimeError(f"conflicting dates for {conflict} race IDs")

    date_map = date_map.sort_values("race_date").drop_duplicates("race_id", keep="last")

    anchors = {
        "202205010101": pd.Timestamp("2022-01-29"),
        "202501010101": pd.Timestamp("2025-07-26"),
    }
    anchor_failures = []
    lookup = date_map.set_index("race_id")["race_date"]
    for race_id, expected in anchors.items():
        if race_id in lookup.index and lookup.loc[race_id] != expected:
            anchor_failures.append(
                f"{race_id}: got {lookup.loc[race_id]}, expected {expected}"
            )
    if anchor_failures:
        raise RuntimeError("official date-anchor mismatch: " + "; ".join(anchor_failures))

    diagnostics = {
        "secondary_rows": len(secondary),
        "helper_2021_2024_rows": len(helpers),
        "helper_2025_rows": len(helper_2025),
        "overlap_rows_checked": len(overlap),
        "overlap_mismatch": overlap_mismatch,
        "date_map_rows": len(date_map),
    }
    return date_map, diagnostics


def scan_malformed_jra_rows(path: Path, jra_ids: set[str]) -> tuple[int, int]:
    malformed_total = 0
    malformed_jra = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        expected = len(header)
        for row in reader:
            if len(row) != expected:
                malformed_total += 1
                if row and row[0] in jra_ids:
                    malformed_jra += 1
    return malformed_total, malformed_jra


def load_jra_results(path: Path, jra_ids: set[str]) -> pd.DataFrame:
    usecols = [
        "race_id",
        "rank",
        "frame",
        "number",
        "horse_id",
        "horse_name",
        "sex_age",
        "weight",
        "time",
        "passing",
        "last_3f",
    ]
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype="string",
        encoding="utf-8-sig",
        chunksize=200_000,
        on_bad_lines="skip",
        low_memory=False,
    ):
        selected = chunk.loc[chunk["race_id"].isin(jra_ids)].copy()
        if not selected.empty:
            frames.append(selected)
    if not frames:
        raise RuntimeError("no JRA result rows found")
    return pd.concat(frames, ignore_index=True)


def standardize(
    races: pd.DataFrame,
    results: pd.DataFrame,
    date_map: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    races = races.copy()
    races["race_id"] = races["race_id"].astype("string").str.strip()
    date_map = date_map.copy()
    date_map["race_id"] = date_map["race_id"].astype("string").str.strip()

    race = races.merge(date_map, on="race_id", how="left", validate="one_to_one")
    missing_date = int(race["race_date"].isna().sum())
    if missing_date:
        examples = race.loc[race["race_date"].isna(), "race_id"].head(20).tolist()
        raise RuntimeError(f"missing actual dates for {missing_date} JRA races: {examples}")

    results = results.copy()
    results["race_id"] = results["race_id"].astype("string").str.strip()
    results["horse_id"] = results["horse_id"].astype("string").str.strip()
    merged = results.merge(
        race[
            [
                "race_id",
                "race_date",
                "venue",
                "race_name",
                "course_type",
                "distance",
                "turn",
                "race_class",
            ]
        ],
        on="race_id",
        how="inner",
        validate="many_to_one",
    )

    rank_text = merged["rank"].fillna("").str.strip()
    nonstarter = rank_text.isin({"", "取", "除"})
    merged = merged.loc[~nonstarter].copy()

    merged["finish_position"] = merged["rank"].map(parse_rank)
    merged["race_time_seconds"] = merged["time"].map(parse_time_seconds)
    merged["early_position"] = merged["passing"].map(parse_first_position)
    merged["last_3f_num"] = pd.to_numeric(merged["last_3f"], errors="coerce")
    merged["assigned_weight_kg"] = pd.to_numeric(merged["weight"], errors="coerce")
    merged["horse_no"] = pd.to_numeric(merged["number"], errors="coerce")

    sex_age = merged["sex_age"].map(parse_sex_age)
    merged["sex"] = sex_age.map(lambda value: value[0])
    merged["age"] = sex_age.map(lambda value: value[1])

    merged["distance_m"] = pd.to_numeric(merged["distance"], errors="coerce")
    merged["surface"] = merged["course_type"].map({"芝": "turf", "ダ": "dirt"})
    merged["racecourse"] = merged["venue"].map(JRA_VENUES)
    merged["turn_direction"] = merged["turn"].map(TURN_MAP).fillna("unknown")
    merged["course_layout"] = "unknown"

    race_name = merged["race_name"].fillna("").astype(str)
    graded_pattern = r"(?:G[123ⅠⅡⅢ]|Ｇ[１２３ⅠⅡⅢ]|Jpn[123ⅠⅡⅢ])"
    merged["is_graded"] = race_name.str.contains(
        graded_pattern,
        regex=True,
        case=False,
    ).astype("int8")
    merged["is_open_plus"] = (
        merged["race_class"].fillna("").eq("オープン") | merged["is_graded"].eq(1)
    ).astype("int8")
    merged["handicap_indicator"] = np.nan

    merged["field_size"] = merged.groupby("race_id")["horse_id"].transform("size")

    winner_rows = merged.loc[merged["finish_position"].eq(1)]
    winner_last3f = winner_rows.groupby("race_id")["last_3f_num"].min()
    obstacle_ids = set(winner_last3f.loc[winner_last3f.lt(20.0)].index.astype(str))
    merged = merged.loc[
        merged["surface"].isin(["turf", "dirt"]) & ~merged["race_id"].isin(obstacle_ids)
    ].copy()

    required_not_null = [
        "race_id",
        "race_date",
        "horse_id",
        "horse_name",
        "horse_no",
        "field_size",
        "sex",
        "age",
        "assigned_weight_kg",
        "distance_m",
        "surface",
        "racecourse",
    ]
    before_drop = len(merged)
    merged = merged.dropna(subset=required_not_null).copy()
    invalid_core_rows = before_drop - len(merged)

    merged["horse_no"] = merged["horse_no"].astype(int)
    merged["field_size"] = merged["field_size"].astype(int)
    merged["age"] = merged["age"].astype(int)
    merged["distance_m"] = merged["distance_m"].astype(int)

    std = merged[
        [
            "race_id",
            "race_date",
            "horse_id",
            "horse_name",
            "horse_no",
            "field_size",
            "sex",
            "age",
            "assigned_weight_kg",
            "distance_m",
            "surface",
            "racecourse",
            "course_layout",
            "race_class",
            "handicap_indicator",
            "finish_position",
            "race_time_seconds",
            "early_position",
            "is_open_plus",
            "is_graded",
            "turn_direction",
        ]
    ].copy()

    duplicate = int(std.duplicated(["race_id", "horse_id"]).sum())
    if duplicate:
        raise RuntimeError(f"duplicate race_id × horse_id rows after standardization: {duplicate}")

    diagnostics = {
        "starter_rows_before_flat_filter": len(results) - int(nonstarter.sum()),
        "obstacle_races_excluded": len(obstacle_ids),
        "invalid_core_rows_excluded": invalid_core_rows,
        "standardized_rows": len(std),
        "standardized_races": int(std["race_id"].nunique()),
    }
    return std, diagnostics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, part in frame.groupby("temporal_split", sort=False):
        rows.append(
            {
                "split": split,
                "runner_rows": len(part),
                "races": part["race_id"].nunique(),
                "top3_prevalence": float(part["top3_label"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    helper_root = Path(os.environ["DATE_HELPER_2025_ROOT"])
    primary_root = download_dataset(PRIMARY, "data/historical_raw/primary")
    secondary_root = download_dataset(SECONDARY, "data/historical_raw/secondary")

    races_all = pd.read_csv(
        primary_root / "keiba_races.csv",
        dtype={"race_id": "string"},
        encoding="utf-8-sig",
        low_memory=False,
    )
    races = races_all.loc[races_all["venue"].isin(JRA_VENUES)].copy()
    races["race_id"] = races["race_id"].astype("string").str.strip()
    jra_ids = set(races["race_id"].dropna().astype(str))

    result_path = primary_root / "keiba_results.csv"
    malformed_total, malformed_jra = scan_malformed_jra_rows(result_path, jra_ids)
    if malformed_jra:
        raise RuntimeError(f"malformed-width JRA rows detected: {malformed_jra}")

    results = load_jra_results(result_path, jra_ids)
    date_map, date_diag = build_actual_date_map(secondary_root, helper_root)
    standardized, std_diag = standardize(races, results, date_map)

    panel = build_historical_panel(standardized)
    assert_strict_history(panel)

    panel["turn_direction"] = standardized.set_index(
        ["race_id", "horse_id"]
    )["turn_direction"].reindex(
        pd.MultiIndex.from_frame(panel[["race_id", "horse_id"]])
    ).to_numpy()

    cohort = select_phase_a_cohort(
        panel,
        surface="turf",
        min_field_size=8,
        min_prior_starts=3,
    )

    keep = [
        "race_id",
        "race_date",
        "horse_id",
        "horse_name",
        "horse_no",
        "field_size",
        "sex",
        "age",
        "assigned_weight_kg",
        "distance_m",
        "surface",
        "racecourse",
        "race_class",
        "turn_direction",
        "top3_label",
        "finish_position",
        "draw_pct",
        "career_starts",
        "career_top3",
        "turf_starts",
        "turf_top3",
        "same_distance_starts",
        "same_distance_top3",
        "same_course_starts",
        "same_course_top3",
        "days_since_prev",
        "distance_change_from_prev_m",
        "surface_changed_from_prev",
        "assigned_weight_delta_from_prev_kg",
        "recent3_finish_pct_mean",
        "recent3_top3_count",
        "recent3_open_plus_count",
        "recent3_graded_count",
        "recent3_relative_time_mean",
        "recent4_early_pos_pct_mean",
        "front_forward_share",
        "temporal_split",
    ]
    cohort = cohort[keep].copy()

    forbidden = find_forbidden_market_columns(cohort.columns)
    if forbidden:
        raise RuntimeError(f"market columns leaked into modeling panel: {forbidden}")

    duplicate = int(cohort.duplicated(["race_id", "horse_id"]).sum())
    strict_bad = int(
        (
            panel["prev_race_date"].notna()
            & (panel["prev_race_date"] >= panel["race_date"])
        ).sum()
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_path = OUT_DIR / "phase_a_all_turf_v1.parquet"
    cohort.to_parquet(all_path, index=False, compression="zstd")

    split_files: dict[str, dict] = {}
    split_names = {
        "train": "train_2016_2022.parquet",
        "validation": "validation_2023_2024.parquet",
        "test": "test_2025.parquet",
    }
    for split, filename in split_names.items():
        part = cohort.loc[cohort["temporal_split"].eq(split)].copy()
        path = OUT_DIR / filename
        part.to_parquet(path, index=False, compression="zstd")
        split_files[split] = {
            "file": filename,
            "rows": len(part),
            "races": int(part["race_id"].nunique()),
            "sha256": sha256_file(path),
        }

    all_sha = sha256_file(all_path)
    split_stats = split_summary(cohort)
    split_stats.to_csv(OUT_DIR / "split_summary.csv", index=False)

    standardized["year"] = pd.to_datetime(standardized["race_date"]).dt.year
    flat_counts = (
        standardized.groupby("year")
        .agg(flat_races=("race_id", "nunique"), starter_rows=("race_id", "size"))
        .reset_index()
    )
    cohort_year = cohort.assign(year=pd.to_datetime(cohort["race_date"]).dt.year)
    cohort_counts = (
        cohort_year.groupby("year")
        .agg(phase_a_turf_races=("race_id", "nunique"), phase_a_runner_rows=("race_id", "size"))
        .reset_index()
    )
    year_counts = flat_counts.merge(cohort_counts, on="year", how="left").fillna(0)
    YEAR_COUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    year_counts.to_csv(YEAR_COUNTS_PATH, index=False)

    benchmark_2016_2025 = int(
        standardized.loc[
            standardized["race_date"].dt.year.between(2016, 2025),
            "race_id",
        ].nunique()
    )
    benchmark_target = 33_290
    benchmark_match = benchmark_2016_2025 == benchmark_target

    missingness = cohort.isna().mean().sort_values(ascending=False)
    material_missing = missingness.loc[missingness.gt(0)]

    status = "PASS"
    blockers: list[str] = []
    if duplicate:
        blockers.append(f"duplicate race_id x horse_id rows: {duplicate}")
    if strict_bad:
        blockers.append(f"non-strict historical dates: {strict_bad}")
    if malformed_jra:
        blockers.append(f"malformed JRA source rows: {malformed_jra}")
    if not benchmark_match:
        blockers.append(
            f"2016-2025 flat race count {benchmark_2016_2025} != benchmark {benchmark_target}"
        )
    if blockers:
        status = "BLOCKED"

    manifest = {
        "dataset_version": "stage36_v1",
        "status": status,
        "primary": {
            "slug": PRIMARY,
            "license": "CC0: Public Domain",
            "kaggle_bundle_version": 1,
        },
        "secondary_date_map": {
            "slug": SECONDARY,
            "license": "CC BY 4.0",
            "kaggle_bundle_version": 1,
        },
        "date_helpers": {
            "2021_2024_commit": RACEINFO_COMMIT,
            "2025_commit": DATE_HELPER_COMMIT,
            "usage": "race_id-to-calendar-date facts only; raw helper rows not redistributed",
        },
        "all_turf": {
            "file": all_path.name,
            "rows": len(cohort),
            "races": int(cohort["race_id"].nunique()),
            "sha256": all_sha,
        },
        "splits": split_files,
        "qa": {
            "duplicate_race_horse": duplicate,
            "strict_date_violations": strict_bad,
            "malformed_source_rows_total": malformed_total,
            "malformed_jra_rows": malformed_jra,
            "flat_races_2016_2025": benchmark_2016_2025,
            "flat_race_benchmark_2016_2025": benchmark_target,
            "benchmark_match": benchmark_match,
            **date_diag,
            **std_diag,
        },
        "blockers": blockers,
    }
    (OUT_DIR / "panel_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    qa = [
        "# Stage 3.6 Historical Training Panel QA",
        "",
        f"Status: **{status}**",
        "",
        "## Sources",
        "",
        "- Primary historical race/results: Kaggle Japan Horse Racing Data 2010-2025, CC0.",
        "- 2010-2020 actual dates: Kaggle legacy JRA dataset, CC BY 4.0.",
        f"- 2021-2024 date helper pinned at commit {RACEINFO_COMMIT}.",
        f"- 2025 date helper pinned at commit {DATE_HELPER_COMMIT}.",
        "- Helper repositories are used only to reconstruct factual race_id-to-date mappings;",
        "  their row-level source data are not redistributed.",
        "",
        "## Source integrity",
        "",
        f"- malformed raw result rows (all racing): {malformed_total}",
        f"- malformed raw result rows belonging to JRA race IDs: {malformed_jra}",
        f"- date-map overlap rows cross-checked: {date_diag['overlap_rows_checked']}",
        f"- date-map overlap mismatches: {date_diag['overlap_mismatch']}",
        f"- date-map rows: {date_diag['date_map_rows']}",
        "",
        "## Standardized flat panel",
        "",
        f"- flat races: {std_diag['standardized_races']}",
        f"- starter rows: {std_diag['standardized_rows']}",
        f"- obstacle races excluded by documented winner-last3F<20 rule: "
        f"{std_diag['obstacle_races_excluded']}",
        f"- invalid core rows excluded: {std_diag['invalid_core_rows_excluded']}",
        f"- 2016-2025 flat races: {benchmark_2016_2025}",
        f"- independent benchmark: {benchmark_target}",
        f"- benchmark exact match: {benchmark_match}",
        "",
        "## Phase A cohort",
        "",
        "- surface: turf",
        "- field size: >= 8 starters",
        "- previous starts: >= 3",
        f"- runner rows: {len(cohort)}",
        f"- races: {cohort['race_id'].nunique()}",
        "",
        split_stats.to_markdown(index=False),
        "",
        "## Leakage / structural checks",
        "",
        f"- duplicate race_id x horse_id: {duplicate}",
        f"- previous date >= current date: {strict_bad}",
        f"- forbidden market columns in artifact: {len(forbidden)}",
        "- current-race odds, popularity and payout are absent.",
        "- rolling/cumulative horse features are shifted to strictly prior starts.",
        "",
        "## Missingness in Phase A artifact",
        "",
        "~~~text",
        material_missing.to_string() if not material_missing.empty else "No missing values.",
        "~~~",
        "",
        "## Blockers",
        "",
        *(["- none"] if not blockers else [f"- {item}" for item in blockers]),
        "",
        "Stage 4 model fitting is unblocked only when Status is PASS.",
    ]
    QA_PATH.write_text("\n".join(qa) + "\n", encoding="utf-8")

    fingerprint = [
        "# Historical Training Panel Fingerprint",
        "",
        f"Status: **{status}**",
        "",
        f"- file: {all_path.name}",
        f"- rows: {len(cohort)}",
        f"- races: {cohort['race_id'].nunique()}",
        f"- SHA-256: `{all_sha}`",
        "",
        "Split files:",
        "",
    ]
    for split, info in split_files.items():
        fingerprint.append(
            f"- {split}: {info['file']} — rows={info['rows']}, races={info['races']}, "
            f"SHA-256=`{info['sha256']}`"
        )
    FINGERPRINT_PATH.write_text("\n".join(fingerprint) + "\n", encoding="utf-8")

    if status != "PASS":
        raise SystemExit("Stage 3.6 QA BLOCKED: " + "; ".join(blockers))


if __name__ == "__main__":
    main()
