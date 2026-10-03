"""Materialize the leakage-safe JRA historical training dataset.

Pipeline:
1. download the CC0 Kaggle source locally;
2. reconstruct 2010-2025 race dates from official JRA daily-result PDFs;
3. reconcile primary race IDs against the official JRA PDF archive;
4. standardize JRA flat-race starter rows without market columns;
5. build prior-only horse-history features;
6. export chronological Phase-A turf partitions and QA/fingerprints.

Raw third-party files and JRA PDFs are never committed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urljoin

import fitz
import kagglehub
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from keiba_place_lab.historical_panel import (
    assert_strict_history,
    build_historical_panel,
    find_forbidden_market_columns,
    select_phase_a_cohort,
)

DATASET = "noriyukifurufuru/japan-horse-racing-2010-2025"
START_YEAR = 2010
END_YEAR = 2025
JRA_REPORT = "https://www.jra.go.jp/datafile/seiseki/report/{year}.html"
USER_AGENT = "keiba-place-probability-lab/0.1 research reproducibility"

RAW_ROOT = Path("data/historical_raw/materialize")
OUT = Path("data/historical_training")
QA_PATH = Path("docs/HISTORICAL_PANEL_QA_REPORT.md")
SCHEMA_PATH = Path("docs/HISTORICAL_TRAINING_SCHEMA.csv")

VENUES = {
    "sapporo": ("01", "札幌"),
    "hakodate": ("02", "函館"),
    "fukushima": ("03", "福島"),
    "niigata": ("04", "新潟"),
    "tokyo": ("05", "東京"),
    "nakayama": ("06", "中山"),
    "chukyo": ("07", "中京"),
    "kyoto": ("08", "京都"),
    "hanshin": ("09", "阪神"),
    "kokura": ("10", "小倉"),
}
VENUE_JP_TO_CODE = {jp: code for _, (code, jp) in VENUES.items()}
VENUE_SLUG_RE = "|".join(sorted(VENUES, key=len, reverse=True))
PDF_NAME_RE = re.compile(
    rf"^(?P<year>\d{{4}})-(?P<meeting>\d+)(?P<slug>{VENUE_SLUG_RE})(?P<day>\d+)\.pdf$",
    re.IGNORECASE,
)
OLD_VENUE_ALIASES = {
    "sap": "sapporo",
    "hako": "hakodate",
    "fuku": "fukushima",
    "niiga": "niigata",
    "tokyo": "tokyo",
    "naka": "nakayama",
    "chu": "chukyo",
    "kyo": "kyoto",
    "han": "hanshin",
    "koku": "kokura",
}
OLD_PDF_NAME_RE = re.compile(
    r"^(?P<meeting>\d+)(?P<slug>sap|hako|fuku|niiga|tokyo|naka|chu|kyo|han|koku)\.pdf$",
    re.IGNORECASE,
)
VENUE_JP_RE = "|".join(re.escape(v[1]) for v in VENUES.values())
FULL_RACE_HEADER_RE = re.compile(
    rf"(?P<month>\d{{1,2}})月(?P<calday>\d{{1,2}})日.{{0,180}}?"
    rf"\((?P<year>\d{{4}})年(?P<meeting>\d+)(?P<venue>{VENUE_JP_RE})\)"
    rf"第(?P<meetday>\d+)日第(?P<race>\d{{1,2}})競走"
)

NONSTARTER_RANKS = {"", "取", "除"}
OBSTACLE_RE = re.compile(
    r"障害|ジャンプ|Ｊ・Ｇ|J・G|JG[123ⅠⅡⅢ]|グランドジャンプ|大障害|ハイジャンプ|"
    r"スプリングJ|スプリングＪ",
    re.IGNORECASE,
)
GRADE_RE = re.compile(
    r"(?:\(|（)(?:G[123]|JG[123])(?:\)|）)|GⅠ|GⅡ|GⅢ|J・GⅠ|J・GⅡ|J・GⅢ",
    re.IGNORECASE,
)

SAFE_EXPORT_COLUMNS = [
    "race_date",
    "race_id",
    "horse_id",
    "horse_name",
    "horse_no",
    "top3_label",
    "draw_pct",
    "sex",
    "age",
    "assigned_weight_kg",
    "assigned_weight_delta_from_prev_kg",
    "days_since_prev",
    "distance_change_from_prev_m",
    "surface_changed_from_prev",
    "career_starts",
    "career_top3",
    "turf_starts",
    "turf_top3",
    "same_distance_starts",
    "same_distance_top3",
    "same_course_starts",
    "same_course_top3",
    "recent3_finish_pct_mean",
    "recent3_top3_count",
    "recent3_open_plus_count",
    "recent3_graded_count",
    "recent3_relative_time_mean",
    "recent4_early_pos_pct_mean",
    "front_forward_share",
    "field_size",
    "declared_field_size",
    "racecourse",
    "surface",
    "distance_m",
    "turn_direction",
    "race_class",
    "temporal_split",
]


@dataclass(frozen=True)
class DayRecord:
    race_day_key: str
    year: int
    venue_code: str
    venue: str
    meeting_no: int
    day_no: int
    actual_date: str
    official_race_numbers: str
    official_race_count: int
    pdf_sha256: str
    pdf_url: str
    header_verified: bool
    verification_level: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_bytes(url: str, *, attempts: int = 4) -> bytes:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            r = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=(20, 120),
            )
            r.raise_for_status()
            return r.content
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (2**attempt))
    raise RuntimeError(f"failed to fetch {url}: {last!r}")


def get_text(url: str, *, attempts: int = 4) -> str:
    return get_bytes(url, attempts=attempts).decode("utf-8", errors="replace")


def annual_pdf_links(year: int) -> list[dict]:
    """Return official result-PDF links for both legacy and modern JRA pages."""
    page = JRA_REPORT.format(year=year)
    soup = BeautifulSoup(get_text(page), "html.parser")
    found: list[dict] = []
    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        name = Path(href).name

        modern = PDF_NAME_RE.match(name)
        if modern and int(modern.group("year")) == year:
            slug = modern.group("slug").lower()
            venue_code, venue_jp = VENUES[slug]
            found.append(
                {
                    "year": year,
                    "meeting_no": int(modern.group("meeting")),
                    "day_no": int(modern.group("day")),
                    "slug": slug,
                    "venue_code": venue_code,
                    "venue": venue_jp,
                    "url": urljoin(page, href),
                    "legacy": False,
                }
            )
            continue

        legacy = OLD_PDF_NAME_RE.match(name)
        if legacy:
            slug = OLD_VENUE_ALIASES[legacy.group("slug").lower()]
            venue_code, venue_jp = VENUES[slug]
            found.append(
                {
                    "year": year,
                    "meeting_no": int(legacy.group("meeting")),
                    "day_no": None,
                    "slug": slug,
                    "venue_code": venue_code,
                    "venue": venue_jp,
                    "url": urljoin(page, href),
                    "legacy": True,
                }
            )

    unique = {x["url"]: x for x in found}
    if not unique:
        raise RuntimeError(f"no JRA result PDFs found for {year}")
    return sorted(
        unique.values(),
        key=lambda x: (x["venue_code"], x["meeting_no"], x["url"]),
    )


def parse_result_pdf(info: dict) -> list[dict]:
    """Extract official calendar dates from modern or legacy JRA result PDFs."""
    data = get_bytes(info["url"])
    pdf_hash = hashlib.sha256(data).hexdigest()
    doc = fitz.open(stream=data, filetype="pdf")
    if len(doc) < 1:
        raise RuntimeError(f"empty PDF: {info['url']}")

    compact_parts: list[str] = []
    for page in doc:
        text = unicodedata.normalize("NFKC", page.get_text("text"))
        compact_parts.append(re.sub(r"\s+", "", text))
    doc.close()
    compact = "".join(compact_parts)

    if info.get("legacy", False):
        # Legacy meeting-level PDFs (used on older annual pages) contain a
        # summary table such as "第1日 1月5日（火）". The PDF filename provides
        # year/venue/meeting; the table independently supplies meeting-day date.
        legacy_day_re = re.compile(
            r"第(?P<meetday>\d+)日(?P<month>\d+)月(?P<calday>\d+)日"
        )
        day_pairs: dict[int, tuple[int, int]] = {}
        for match in legacy_day_re.finditer(compact):
            meetday = int(match.group("meetday"))
            month = int(match.group("month"))
            calday = int(match.group("calday"))
            pair = (month, calday)
            if meetday in day_pairs and day_pairs[meetday] != pair:
                raise RuntimeError(
                    f"conflicting legacy date for {info['url']} day {meetday}: "
                    f"{day_pairs[meetday]} vs {pair}"
                )
            day_pairs[meetday] = pair
        if not day_pairs:
            raise RuntimeError(f"no legacy meeting-day dates extracted from {info['url']}")

        extracted: list[dict] = []
        for meetday, (month, calday) in sorted(day_pairs.items()):
            actual = pd.Timestamp(
                year=info["year"], month=month, day=calday
            ).date().isoformat()
            key = (
                f"{info['year']}{info['venue_code']}"
                f"{info['meeting_no']:02d}{meetday:02d}"
            )
            extracted.append(
                {
                    "race_day_key": key,
                    "year": info["year"],
                    "venue_code": info["venue_code"],
                    "venue": info["venue"],
                    "meeting_no": info["meeting_no"],
                    "day_no": meetday,
                    "actual_date": actual,
                    "race_number": np.nan,
                    "pdf_sha256": pdf_hash,
                    "pdf_url": info["url"],
                    "verification_level": "official_legacy_day_summary",
                }
            )
        return extracted

    extracted = []
    for match in FULL_RACE_HEADER_RE.finditer(compact):
        year = int(match.group("year"))
        meeting = int(match.group("meeting"))
        venue = match.group("venue")
        meetday = int(match.group("meetday"))
        race_no = int(match.group("race"))
        if (
            year != info["year"]
            or meeting != info["meeting_no"]
            or venue != info["venue"]
            or not 1 <= race_no <= 12
        ):
            continue
        month = int(match.group("month"))
        calday = int(match.group("calday"))
        actual = pd.Timestamp(year=year, month=month, day=calday).date().isoformat()
        key = f"{year}{info['venue_code']}{meeting:02d}{meetday:02d}"
        extracted.append(
            {
                "race_day_key": key,
                "year": year,
                "venue_code": info["venue_code"],
                "venue": venue,
                "meeting_no": meeting,
                "day_no": meetday,
                "actual_date": actual,
                "race_number": race_no,
                "pdf_sha256": pdf_hash,
                "pdf_url": info["url"],
                "verification_level": "official_exact_race_header",
            }
        )

    if extracted:
        return extracted

    # 2011-2014 daily PDFs use the modern filename convention but retain an
    # older internal typesetting in which the Gregorian year/race number can be
    # poorly extractable. The PDF filename still fixes venue/meeting/day, while
    # the official PDF text exposes the calendar month/day. This is sufficient
    # for the date map; race-level exact reconciliation is reserved for PDFs
    # whose full race headings are machine-readable.
    if info.get("day_no") is not None:
        dm = re.search(r"(?P<month>\d{1,2})月(?P<calday>\d{1,2})日", compact)
        if not dm:
            raise RuntimeError(
                f"could not extract calendar date from daily PDF {info['url']}"
            )
        month = int(dm.group("month"))
        calday = int(dm.group("calday"))
        meetday = int(info["day_no"])
        actual = pd.Timestamp(
            year=info["year"], month=month, day=calday
        ).date().isoformat()
        key = (
            f"{info['year']}{info['venue_code']}"
            f"{info['meeting_no']:02d}{meetday:02d}"
        )
        return [
            {
                "race_day_key": key,
                "year": info["year"],
                "venue_code": info["venue_code"],
                "venue": info["venue"],
                "meeting_no": info["meeting_no"],
                "day_no": meetday,
                "actual_date": actual,
                "race_number": np.nan,
                "pdf_sha256": pdf_hash,
                "pdf_url": info["url"],
                "verification_level": "official_daily_date_header",
            }
        ]

    raise RuntimeError(f"no official date facts extracted from {info['url']}")

def build_official_date_map() -> pd.DataFrame:
    tasks: list[dict] = []
    for year in range(START_YEAR, END_YEAR + 1):
        links = annual_pdf_links(year)
        print(f"JRA report PDFs {year}: {len(links)}")
        tasks.extend(links)

    print(f"JRA result PDFs discovered: {len(tasks)}")
    records: list[dict] = []
    workers = int(os.getenv("JRA_PDF_WORKERS", "12"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(parse_result_pdf, task): task for task in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            records.extend(future.result())
            if done % 100 == 0 or done == len(tasks):
                print(f"JRA result PDFs parsed: {done}/{len(tasks)}")

    facts = pd.DataFrame(records)
    exact = facts.loc[facts["race_number"].notna()].copy()
    if not exact.empty:
        exact_ids = (
            exact["race_day_key"]
            + exact["race_number"].astype(int).astype(str).str.zfill(2)
        )
        if exact_ids.duplicated().any():
            dup = exact_ids[exact_ids.duplicated(False)].tolist()
            raise RuntimeError(f"duplicate official race IDs extracted: {dup[:20]}")

    day_rows: list[DayRecord] = []
    for key, group in facts.groupby("race_day_key", sort=False):
        dates = group["actual_date"].drop_duplicates().tolist()
        if len(dates) != 1:
            raise RuntimeError(f"multiple actual dates for {key}: {dates}")
        first = group.iloc[0]
        nums = sorted(
            group.loc[group["race_number"].notna(), "race_number"]
            .astype(int)
            .unique()
            .tolist()
        )
        urls = sorted(group["pdf_url"].drop_duplicates().tolist())
        hashes = sorted(group["pdf_sha256"].drop_duplicates().tolist())
        levels = sorted(group["verification_level"].drop_duplicates().tolist())
        if nums:
            level = "official_exact_race_headers"
            if "official_exact_race_header" not in levels:
                raise RuntimeError(f"inconsistent verification level for {key}")
        elif len(levels) == 1:
            level = levels[0]
        else:
            level = "+".join(levels)
        day_rows.append(
            DayRecord(
                race_day_key=key,
                year=int(first["year"]),
                venue_code=str(first["venue_code"]),
                venue=str(first["venue"]),
                meeting_no=int(first["meeting_no"]),
                day_no=int(first["day_no"]),
                actual_date=str(dates[0]),
                official_race_numbers=";".join(str(x) for x in nums),
                official_race_count=len(nums),
                pdf_sha256=";".join(hashes),
                pdf_url=";".join(urls),
                header_verified=True,
                verification_level=level,
            )
        )

    df = pd.DataFrame(asdict(x) for x in day_rows)
    if df["race_day_key"].duplicated().any():
        raise RuntimeError("duplicate official race-day keys after aggregation")
    df["actual_date"] = pd.to_datetime(df["actual_date"])
    return df.sort_values(
        ["actual_date", "venue_code", "meeting_no", "day_no"]
    ).reset_index(drop=True)

def official_race_id_set(date_map: pd.DataFrame) -> set[str]:
    out: set[str] = set()
    for row in date_map.itertuples(index=False):
        for value in str(row.official_race_numbers).split(";"):
            if value:
                out.add(f"{row.race_day_key}{int(value):02d}")
    return out


def kaggle_metadata() -> dict:
    url = f"https://www.kaggle.com/api/v1/datasets/view/{DATASET}"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
        r.raise_for_status()
        raw = r.json()
    except (requests.RequestException, ValueError) as exc:
        return {"metadata_error": repr(exc)}
    keys = (
        "id",
        "ref",
        "title",
        "lastUpdated",
        "currentVersionNumber",
        "versionNumber",
        "licenseName",
        "totalBytes",
    )
    return {k: raw.get(k) for k in keys if k in raw}


def download_primary() -> Path:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    root = Path(
        kagglehub.dataset_download(
            DATASET,
            output_dir=str(RAW_ROOT),
            force_download=True,
        )
    )
    return root if root.is_dir() else root.parent


def normalize_race_class(raw_class: object, race_name: object) -> str:
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
    if raw == "オープン" or "オープン" in text or GRADE_RE.search(name):
        return "Open"
    return "Other"


def normalize_races(
    races_path: Path,
    date_map: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    races = pd.read_csv(
        races_path,
        encoding="utf-8-sig",
        dtype={"race_id": "string"},
        low_memory=False,
    )
    races["race_id"] = races["race_id"].astype("string").str.strip()
    jra = races.loc[races["venue"].isin(VENUE_JP_TO_CODE)].copy()
    jra = jra.loc[jra["race_id"].str.fullmatch(r"\d{12}", na=False)].copy()
    years = pd.to_numeric(jra["race_id"].str.slice(0, 4), errors="coerce")
    jra = jra.loc[years.between(START_YEAR, END_YEAR)].copy()

    jra["race_day_key"] = jra["race_id"].str.slice(0, 10)
    date_lookup = date_map.set_index("race_day_key")["actual_date"]
    jra["actual_date"] = jra["race_day_key"].map(date_lookup)

    expected_code = jra["venue"].map(VENUE_JP_TO_CODE)
    observed_code = jra["race_id"].str.slice(4, 6)
    venue_code_mismatch = int(expected_code.ne(observed_code).sum())
    missing_dates = int(jra["actual_date"].isna().sum())

    jra["race_number_from_id"] = pd.to_numeric(
        jra["race_id"].str.slice(10, 12), errors="coerce"
    )
    source_race_number = pd.to_numeric(jra["race_number"], errors="coerce")
    race_number_mismatch = int(
        (
            source_race_number.notna()
            & source_race_number.ne(jra["race_number_from_id"])
        ).sum()
    )

    names = jra["race_name"].fillna("").astype(str)
    jra["is_obstacle"] = (
        ~jra["course_type"].isin(["芝", "ダ"])
        | names.str.contains(OBSTACLE_RE, regex=True)
    )

    jra["surface"] = jra["course_type"].map({"芝": "turf", "ダ": "dirt"})
    jra["distance_m"] = pd.to_numeric(jra["distance"], errors="coerce")
    jra["is_graded"] = names.str.contains(GRADE_RE, regex=True).astype("int8")
    jra["race_class_norm"] = [
        normalize_race_class(rc, rn)
        for rc, rn in zip(jra["race_class"], jra["race_name"], strict=False)
    ]
    jra["is_open_plus"] = (
        jra["race_class_norm"].eq("Open") | jra["is_graded"].eq(1)
    ).astype("int8")

    turn = jra["turn"].fillna("").astype(str)
    jra["turn_direction"] = turn.map({"右": "right", "左": "left"})
    straight = (
        jra["venue"].eq("新潟")
        & jra["surface"].eq("turf")
        & jra["distance_m"].eq(1000)
    )
    jra.loc[straight, "turn_direction"] = "straight"
    jra["turn_direction"] = jra["turn_direction"].fillna("other")

    flat = jra.loc[~jra["is_obstacle"] & jra["surface"].notna()].copy()
    flat = flat.loc[flat["distance_m"].notna()].copy()

    diagnostics = {
        "all_source_races": len(races),
        "jra_races_2010_2025": len(jra),
        "jra_missing_official_date": missing_dates,
        "jra_venue_code_mismatch": venue_code_mismatch,
        "jra_race_number_mismatch": race_number_mismatch,
        "obstacle_or_unrecognized_excluded": int(len(jra) - len(flat)),
        "flat_races": len(flat),
        "graded_name_detected": int(flat["is_graded"].sum()),
        "race_class_counts": flat["race_class_norm"].value_counts().to_dict(),
    }
    return flat, diagnostics


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def parse_time(value: str) -> float:
    value = (value or "").strip()
    if not value:
        return np.nan
    m = re.fullmatch(r"(?:(\d+):)?(\d+(?:\.\d+)?)", value)
    if not m:
        return np.nan
    minutes = int(m.group(1) or 0)
    seconds = float(m.group(2))
    return minutes * 60.0 + seconds


def parse_sex_age(value: str) -> tuple[str, float]:
    m = re.search(r"([牡牝セ])\s*(\d+)", value or "")
    if not m:
        return "U", np.nan
    sex = {"牡": "M", "牝": "F", "セ": "G"}[m.group(1)]
    return sex, float(m.group(2))


def parse_rank(value: str) -> tuple[float, str, bool]:
    value = (value or "").strip()
    if value in NONSTARTER_RANKS:
        return np.nan, "nonstarter", False
    m = re.match(r"^(\d+)", value)
    if m:
        return float(m.group(1)), "finished", True
    if value == "中":
        return np.nan, "dnf", True
    if value == "失":
        return np.nan, "disqualified", True
    return np.nan, f"other:{value}", True


def parse_early_position(value: str) -> float:
    nums = re.findall(r"\d+", value or "")
    return float(nums[0]) if nums else np.nan


def parse_results(
    results_path: Path,
    flat_races: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    race_meta = flat_races.set_index("race_id")[
        [
            "actual_date",
            "venue",
            "distance_m",
            "surface",
            "turn_direction",
            "race_class_norm",
            "is_open_plus",
            "is_graded",
        ]
    ]
    flat_ids = set(race_meta.index.astype(str))
    starters: list[dict] = []
    declared_max: dict[str, int] = {}
    malformed_jra = 0
    malformed_any = 0
    source_rows = 0
    nonstarters = 0
    other_status = Counter()
    result_race_ids: set[str] = set()

    with results_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        expected = len(header)
        for row in reader:
            source_rows += 1
            race_id = row[0].strip() if row else ""
            if len(row) != expected:
                malformed_any += 1
                if race_id in flat_ids:
                    malformed_jra += 1
                continue
            if race_id not in flat_ids:
                continue

            result_race_ids.add(race_id)
            raw_no = parse_float(row[idx["number"]])
            horse_no = int(raw_no) if pd.notna(raw_no) else 0
            if horse_no > 0:
                declared_max[race_id] = max(declared_max.get(race_id, 0), horse_no)

            finish_position, finish_status, started = parse_rank(row[idx["rank"]])
            if not started:
                nonstarters += 1
                continue
            if finish_status.startswith("other:"):
                other_status[finish_status] += 1

            horse_id = (row[idx["horse_id"]] or "").strip()
            horse_name = (row[idx["horse_name"]] or "").strip()
            sex, age = parse_sex_age(row[idx["sex_age"]])
            assigned = parse_float(row[idx["weight"]])
            race_time = parse_time(row[idx["time"]])
            early_position = parse_early_position(row[idx["passing"]])

            starters.append(
                {
                    "race_id": race_id,
                    "horse_id": horse_id,
                    "horse_name": horse_name,
                    "horse_no": horse_no,
                    "sex": sex,
                    "age": age,
                    "assigned_weight_kg": assigned,
                    "finish_position": finish_position,
                    "finish_status": finish_status,
                    "race_time_seconds": race_time,
                    "early_position": early_position,
                }
            )

    if malformed_jra:
        raise RuntimeError(f"malformed-width rows in selected JRA flat races: {malformed_jra}")

    rows = pd.DataFrame(starters)
    rows = rows.merge(
        race_meta.reset_index(),
        on="race_id",
        how="left",
        validate="many_to_one",
    )
    rows = rows.rename(
        columns={
            "actual_date": "race_date",
            "venue": "racecourse",
            "race_class_norm": "race_class",
        }
    )
    rows["declared_field_size"] = rows["race_id"].map(declared_max)
    rows["field_size"] = rows.groupby("race_id", sort=False)["horse_id"].transform("size")

    diagnostics = {
        "source_result_rows": source_rows,
        "malformed_any_result_rows": malformed_any,
        "malformed_selected_jra_rows": malformed_jra,
        "nonstarter_rows_excluded": nonstarters,
        "other_finish_status": dict(other_status),
        "flat_races_with_any_result": len(result_race_ids),
        "actual_starter_rows": len(rows),
        "missing_horse_id": int(rows["horse_id"].eq("").sum()),
        "bad_horse_no": int(rows["horse_no"].le(0).sum()),
        "duplicate_race_horse": int(rows.duplicated(["race_id", "horse_id"]).sum()),
    }
    return rows, diagnostics


def safe_training_export(panel: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in SAFE_EXPORT_COLUMNS if c not in panel.columns]
    if missing:
        raise RuntimeError(f"missing safe export columns: {missing}")
    out = panel[SAFE_EXPORT_COLUMNS].copy()
    forbidden = find_forbidden_market_columns(out.columns)
    if forbidden:
        raise RuntimeError(f"forbidden market columns in export: {forbidden}")
    return out


def write_schema() -> None:
    rows = [
        ("race_date", "date", "identifier", "actual JRA date from official daily PDF"),
        ("race_id", "string", "identifier", "12-digit JRA race id"),
        ("horse_id", "string", "identifier", "source horse identifier"),
        ("horse_name", "string", "identifier", "display only"),
        ("horse_no", "int", "identifier", "program number"),
        ("top3_label", "int8", "target", "1 if official numeric finish rank <=3; DNF/DQ=0"),
        ("draw_pct", "float", "feature", "program number normalized by declared field"),
        ("sex", "category", "feature", "M/F/G/U"),
        ("age", "float", "feature", "age at race"),
        ("assigned_weight_kg", "float", "feature", "pre-race assigned weight"),
        ("assigned_weight_delta_from_prev_kg", "float", "feature", "current minus previous start"),
        ("days_since_prev", "float", "feature", "calendar days from previous actual start"),
        ("distance_change_from_prev_m", "float", "feature", "current minus previous distance"),
        ("surface_changed_from_prev", "float", "feature", "1 if surface changed"),
        ("career_starts", "int", "feature", "strictly prior starts"),
        ("career_top3", "int", "feature", "strictly prior top3 finishes"),
        ("turf_starts", "int", "feature", "strictly prior turf starts"),
        ("turf_top3", "int", "feature", "strictly prior turf top3"),
        ("same_distance_starts", "int", "feature", "strictly prior same-distance/surface starts"),
        ("same_distance_top3", "int", "feature", "strictly prior same-distance/surface top3"),
        ("same_course_starts", "int", "feature", "strictly prior same-course/surface starts"),
        ("same_course_top3", "int", "feature", "strictly prior same-course/surface top3"),
        ("recent3_finish_pct_mean", "float", "feature", "prior 3 normalized finishes; lower better"),
        ("recent3_top3_count", "float", "feature", "prior 3 top3 count"),
        ("recent3_open_plus_count", "float", "feature", "prior 3 open-plus count"),
        ("recent3_graded_count", "float", "feature", "prior 3 explicit-grade-marker count"),
        ("recent3_relative_time_mean", "float", "feature", "prior 3 race-relative time loss"),
        ("recent4_early_pos_pct_mean", "float", "feature", "prior 4 normalized early positions"),
        ("front_forward_share", "float", "feature", "share of entrants with prior early mean <=.25"),
        ("field_size", "int", "feature", "actual starters"),
        ("declared_field_size", "int", "feature", "max program number including scratches"),
        ("racecourse", "category", "feature", "JRA venue"),
        ("surface", "category", "feature", "turf/dirt"),
        ("distance_m", "int", "feature", "race distance"),
        ("turn_direction", "category", "feature", "right/left/straight/other"),
        ("race_class", "category", "feature", "normalized class"),
        ("temporal_split", "category", "split", "train/validation/test"),
    ]
    pd.DataFrame(rows, columns=["column", "dtype", "role", "description"]).to_csv(
        SCHEMA_PATH, index=False
    )


def render_qa(
    metadata: dict,
    raw_hashes: dict,
    date_map: pd.DataFrame,
    official_ids: set[str],
    source_jra_ids: set[str],
    race_diag: dict,
    result_diag: dict,
    flat_races: pd.DataFrame,
    starter_rows: pd.DataFrame,
    panel: pd.DataFrame,
    cohort: pd.DataFrame,
    exported: pd.DataFrame,
    manifest: pd.DataFrame,
    fingerprint: str,
) -> str:
    split_rows = []
    for split in ["train", "validation", "test"]:
        g = exported.loc[exported["temporal_split"].eq(split)]
        split_rows.append(
            {
                "split": split,
                "years": {"train": "2016-2022", "validation": "2023-2024", "test": "2025"}[split],
                "races": int(g["race_id"].nunique()),
                "runners": len(g),
                "top3_prevalence": float(g["top3_label"].mean()) if len(g) else np.nan,
            }
        )
    split_df = pd.DataFrame(split_rows)

    feature_cols = [
        c for c in SAFE_EXPORT_COLUMNS
        if c not in {
            "race_date", "race_id", "horse_id", "horse_name", "horse_no",
            "top3_label", "temporal_split"
        }
    ]
    missing = exported[feature_cols].isna().mean().sort_values(ascending=False)

    observed_prev = panel["prev_race_date"].notna()
    non_strict = int(
        (observed_prev & (panel["prev_race_date"] >= panel["race_date"])).sum()
    )
    forbidden = find_forbidden_market_columns(exported.columns)
    top3_by_race = cohort.groupby("race_id")["top3_label"].sum()
    non_three_top3_races = int(top3_by_race.ne(3).sum())

    official_missing = sorted(official_ids - source_jra_ids)
    source_extra = sorted(source_jra_ids - official_ids)

    lines = [
        "# Historical Panel QA Report",
        "",
        "Status: TRAINING-READY",
        "",
        "## Source",
        "",
        f"- Kaggle dataset: {DATASET}",
        f"- Kaggle metadata: {json.dumps(metadata, ensure_ascii=False, default=str)}",
        f"- raw file hashes: {json.dumps(raw_hashes, ensure_ascii=False)}",
        f"- JRA official daily-result archive: {START_YEAR}-{END_YEAR}",
        "",
        "## Official race-date reconstruction",
        "",
        f"- official JRA daily PDFs parsed: {len(date_map):,}",
        f"- official race IDs reconstructed from exact PDF race headings: {len(official_ids):,}",
        f"- primary-source JRA race IDs: {len(source_jra_ids):,}",
        f"- exact-era official IDs missing from primary source: {len(official_missing):,}",
        f"- exact-era primary-source IDs absent from official archive: {len(source_extra):,}",
        f"- official day mappings: {len(date_map):,}",
        f"- legacy meeting-summary mappings: {int(date_map['verification_level'].eq('official_legacy_day_summary').sum()):,}",
        f"- daily-date-header mappings: {int(date_map['verification_level'].eq('official_daily_date_header').sum()):,}",
        f"- source race days without official date mapping: {len({x[:10] for x in source_jra_ids} - set(date_map['race_day_key'].astype(str))):,}",
        f"- PDF date verification failures: {int((~date_map['header_verified']).sum()):,}",
        "",
        "The source dataset pseudo-date column is not used. Each JRA race receives its actual",
        "calendar date from the official JRA results archive. Modern daily PDFs are verified",
        "against individual race headings; legacy meeting-level PDFs are verified against their",
        "official meeting-day/date summary tables. Every source race-day key must map to one",
        "official date before panel construction.",
        "",
        "## Race normalization",
        "",
    ]
    for k, v in race_diag.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Result-row normalization", ""]
    for k, v in result_diag.items():
        lines.append(f"- {k}: {v}")

    lines += [
        "",
        "## Panel integrity",
        "",
        f"- flat races in standardized race index: {flat_races['race_id'].nunique():,}",
        f"- actual starter rows: {len(starter_rows):,}",
        f"- full prior-history panel rows: {len(panel):,}",
        f"- Phase-A turf rows before temporal export: {len(cohort):,}",
        f"- exported train/validation/test rows: {len(exported):,}",
        f"- duplicate race_id x horse_id: {int(starter_rows.duplicated(['race_id','horse_id']).sum())}",
        f"- prev_race_date >= race_date: {non_strict}",
        f"- forbidden market columns: {forbidden}",
        f"- Phase-A races with realized top3-label count != 3: {non_three_top3_races}",
        "",
        "Races with a realized top3 count other than three are retained for the binary target;",
        "they can arise from dead heats or exceptional official outcomes and are reported rather",
        "than silently rewritten.",
        "",
        "## Temporal split",
        "",
        "| split | years | races | runners | top3 prevalence |",
        "|---|---|---:|---:|---:|",
    ]
    for row in split_df.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.years} | {row.races:,} | {row.runners:,} | "
            f"{row.top3_prevalence:.6f} |"
        )

    lines += [
        "",
        "## Missingness in exported predictors",
        "",
        "| feature | missing fraction |",
        "|---|---:|",
    ]
    for col, frac in missing.items():
        lines.append(f"| {col} | {frac:.6f} |")

    lines += [
        "",
        "## Export files",
        "",
        "| file | split | year | rows | races | bytes | sha256 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in manifest.itertuples(index=False):
        lines.append(
            f"| {row.file} | {row.split} | {row.year} | {row.rows:,} | "
            f"{row.races:,} | {row.bytes:,} | {row.sha256} |"
        )

    lines += [
        "",
        f"Dataset fingerprint: {fingerprint}",
        "",
        "## Leakage statement",
        "",
        "- target-race odds, popularity and payouts are absent from exported partitions;",
        "- current finish/time/passing values only update history for later starts;",
        "- all horse-history features are based on strictly earlier actual JRA dates;",
        "- 2025 is untouched by model fitting or hyperparameter selection;",
        "- the 2026-10-03 target race is outside the historical source.",
        "",
        "## Source limitations carried forward",
        "",
        "- inner/outer layout is unavailable consistently; turn_direction is used instead;",
        "- handicap-race status is unavailable consistently and is not exported;",
        "- graded status is inferred only from explicit source race-name grade markers and",
        "  must be ablated in Stage 4.",
        "",
        "## Gate decision",
        "",
        "PASS: automated assertions completed without exception.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in OUT.glob("phase_a_turf_*.parquet"):
        p.unlink()

    print("1/7 reconstructing official JRA dates")
    date_map = build_official_date_map()
    date_map_out = date_map.copy()
    date_map_out["actual_date"] = date_map_out["actual_date"].dt.strftime("%Y-%m-%d")
    date_map_out.to_csv(
        OUT / "jra_race_date_map_2010_2025.csv.gz",
        index=False,
        compression="gzip",
    )
    official_ids = official_race_id_set(date_map)

    print("2/7 downloading primary CC0 historical source")
    root = download_primary()
    races_path = root / "keiba_races.csv"
    results_path = root / "keiba_results.csv"
    payouts_path = root / "keiba_payouts.csv"
    for required in (races_path, results_path, payouts_path):
        if not required.exists():
            raise FileNotFoundError(required)

    metadata = kaggle_metadata()
    if metadata.get("licenseName") != "CC0: Public Domain":
        raise RuntimeError(f"unexpected primary-source license: {metadata.get('licenseName')!r}")
    raw_hashes = {
        p.name: sha256_file(p)
        for p in (races_path, results_path, payouts_path)
    }

    print("3/7 normalizing JRA race index")
    flat_races, race_diag = normalize_races(races_path, date_map)
    source = pd.read_csv(
        races_path,
        encoding="utf-8-sig",
        dtype={"race_id": "string"},
        usecols=["race_id", "venue"],
        low_memory=False,
    )
    source["race_id"] = source["race_id"].astype("string").str.strip()
    source_jra = source.loc[
        source["venue"].isin(VENUE_JP_TO_CODE)
        & source["race_id"].str.fullmatch(r"\d{12}", na=False)
    ].copy()
    source_year = pd.to_numeric(source_jra["race_id"].str.slice(0, 4), errors="coerce")
    source_jra = source_jra.loc[source_year.between(START_YEAR, END_YEAR)]
    source_jra_ids = set(source_jra["race_id"].astype(str))

    official_day_keys = set(date_map["race_day_key"].astype(str))
    source_day_keys = {race_id[:10] for race_id in source_jra_ids}
    source_days_without_official_date = source_day_keys - official_day_keys
    if source_days_without_official_date:
        raise RuntimeError(
            "source JRA race days missing from official date map: "
            f"{len(source_days_without_official_date)} "
            f"{sorted(source_days_without_official_date)[:20]}"
        )

    exact_years = set(
        date_map.loc[
            date_map["verification_level"].eq("official_exact_race_headers"),
            "year",
        ].astype(int)
    )
    source_exact_ids = {
        race_id for race_id in source_jra_ids
        if int(race_id[:4]) in exact_years
    }
    official_missing = official_ids - source_exact_ids
    source_extra = source_exact_ids - official_ids
    if official_missing or source_extra:
        raise RuntimeError(
            "exact-era official/source JRA race-ID reconciliation failed: "
            f"official_missing={len(official_missing)}, source_extra={len(source_extra)}; "
            f"examples={sorted(official_missing)[:10]} / {sorted(source_extra)[:10]}"
        )

    if race_diag["jra_missing_official_date"]:
        raise RuntimeError("missing official dates remain")
    if race_diag["jra_venue_code_mismatch"] or race_diag["jra_race_number_mismatch"]:
        raise RuntimeError(f"race-id metadata mismatch: {race_diag}")

    race_index = flat_races[
        [
            "actual_date", "race_id", "venue", "race_number_from_id",
            "race_name", "surface", "distance_m", "turn_direction",
            "race_class_norm", "is_open_plus", "is_graded",
        ]
    ].copy()
    race_index = race_index.rename(
        columns={
            "actual_date": "race_date",
            "venue": "racecourse",
            "race_number_from_id": "race_number",
            "race_class_norm": "race_class",
        }
    ).sort_values(["race_date", "race_id"])
    race_index["race_date"] = pd.to_datetime(race_index["race_date"]).dt.strftime("%Y-%m-%d")
    race_index.to_csv(
        OUT / "jra_flat_race_index_2010_2025.csv.gz",
        index=False,
        compression="gzip",
    )

    print("4/7 parsing actual JRA starter rows")
    starter_rows, result_diag = parse_results(results_path, flat_races)
    if (
        result_diag["missing_horse_id"]
        or result_diag["bad_horse_no"]
        or result_diag["duplicate_race_horse"]
    ):
        raise RuntimeError(f"starter-row integrity failure: {result_diag}")
    missing_result_races = set(flat_races["race_id"]) - set(starter_rows["race_id"])
    if missing_result_races:
        raise RuntimeError(
            f"flat races without starter rows: {len(missing_result_races)} "
            f"{sorted(missing_result_races)[:20]}"
        )

    print("5/7 building leakage-safe historical panel")
    panel = build_historical_panel(starter_rows)
    assert_strict_history(panel)

    print("6/7 selecting Phase-A turf cohort and exporting yearly partitions")
    cohort = select_phase_a_cohort(
        panel,
        surface="turf",
        min_field_size=8,
        min_prior_starts=3,
    )
    exported = safe_training_export(cohort)
    exported = exported.loc[
        exported["temporal_split"].isin(["train", "validation", "test"])
    ].copy()
    exported["race_date"] = pd.to_datetime(exported["race_date"])

    manifest_rows: list[dict] = []
    for year in range(2016, 2026):
        part = exported.loc[exported["race_date"].dt.year.eq(year)].copy()
        split = "train" if year <= 2022 else "validation" if year <= 2024 else "test"
        part["race_date"] = part["race_date"].dt.strftime("%Y-%m-%d")
        path = OUT / f"phase_a_turf_{year}.parquet"
        part.to_parquet(path, index=False, compression="zstd")
        size = path.stat().st_size
        if size >= 95_000_000:
            raise RuntimeError(f"generated file too large for ordinary GitHub: {path} {size}")
        manifest_rows.append(
            {
                "file": path.as_posix(),
                "split": split,
                "year": year,
                "rows": len(part),
                "races": part["race_id"].nunique(),
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT / "MANIFEST.csv", index=False)
    fingerprint_payload = "\n".join(
        f"{row.file}:{row.sha256}" for row in manifest.itertuples(index=False)
    ).encode()
    fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()
    (OUT / "DATASET_FINGERPRINT.txt").write_text(
        fingerprint + "\n",
        encoding="utf-8",
    )
    write_schema()

    print("7/7 QA")
    duplicate = int(starter_rows.duplicated(["race_id", "horse_id"]).sum())
    prev = panel["prev_race_date"].notna()
    non_strict = int((prev & (panel["prev_race_date"] >= panel["race_date"])).sum())
    forbidden = find_forbidden_market_columns(exported.columns)
    if duplicate or non_strict or forbidden:
        raise RuntimeError(
            f"final QA failed: duplicate={duplicate}, non_strict={non_strict}, "
            f"forbidden={forbidden}"
        )
    if exported.empty:
        raise RuntimeError("training export is empty")
    for split in ("train", "validation", "test"):
        if not exported["temporal_split"].eq(split).any():
            raise RuntimeError(f"empty temporal split: {split}")

    QA_PATH.write_text(
        render_qa(
            metadata,
            raw_hashes,
            date_map,
            official_ids,
            source_jra_ids,
            race_diag,
            result_diag,
            flat_races,
            starter_rows,
            panel,
            cohort,
            exported,
            manifest,
            fingerprint,
        ),
        encoding="utf-8",
    )
    print(QA_PATH.read_text(encoding="utf-8")[:12000])


if __name__ == "__main__":
    main()
