"""Official JRA reconciliation and 2025 year-end supplement for Stage 3.6.

This module keeps two concerns separate:
1. race existence: reconcile incomplete PDF condition extraction against the
   official daily-result PDF itself by matching the number of race headers;
2. row-level supplementation: fill the four 2025 year-end JRA cards absent
   from Kaggle v1 from official JRADB result pages.

No odds/popularity/payout field is returned.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from urllib.parse import unquote, urljoin

import numpy as np
import pandas as pd
import pymupdf
import requests
from bs4 import BeautifulSoup

USER_AGENT = "keiba-place-probability-lab/0.1 research reproducibility"

# These four pages are stable, manually verified official JRA result-page seeds.
# Each page contains navigation links for its 12-race card.
SUPPLEMENT_SEEDS = {
    "2025060507": "https://www.jra.go.jp/JRADB/accessS.html?CNAME=pw01sde1006202505070120251227/A2",
    "2025090507": "https://www.jra.go.jp/JRADB/accessS.html?CNAME=pw01sde1009202505070120251227/80",
    "2025060508": "https://www.jra.go.jp/JRADB/accessS.html?CNAME=pw01sde1006202505080120251228/D2",
    "2025090508": "https://www.jra.go.jp/JRADB/accessS.html?CNAME=pw01sde1009202505080120251228/B0",
}

VENUE_CODE_TO_JP = {"06": "中山", "09": "阪神"}

# The daily PDFs below are known machine-text exceptions. Their 12-race cards
# were checked against the rendered official PDF / official continuation record.
OFFICIAL_RACE_COUNT_OVERRIDES = {
    "2014070401": 12,
    "2020040101": 12,
    "2020050205": 12,
    "2020060302": 12,
    "2020080305": 12,
}

RESULT_CNAME_RE = re.compile(
    r"pw01sde\d{2}"
    r"(?P<venue>\d{2})(?P<year>\d{4})(?P<meeting>\d{2})"
    r"(?P<day>\d{2})(?P<race>\d{2})(?P<date>\d{8})"
)
HORSE_CNAME_RE = re.compile(r"pw01dud\d{2}(?P<horse_id>\d{10})")
PDF_HEADER_RE = re.compile(
    r"(?P<serial>\d{5})[^\d]{0,24}"
    r"(?P<month>1[0-2]|[1-9])\s*月\s*"
    r"(?P<calday>3[01]|[12]\d|[1-9])\s*日"
)
COURSE_RE = re.compile(
    r"コース[:：]\s*(?P<distance>[\d,]+)メートル"
    r"[（(](?P<surface>芝|ダート)(?:[・･](?P<detail>[^）)]*))?[）)]"
)


def _get(url: str) -> requests.Response:
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=(20, 120),
    )
    response.raise_for_status()
    return response


def _normalize(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value)))


def _page_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _pdf_header_counts(url: str, actual_date: str) -> dict[str, int]:
    """Count unique official race-management headers for one daily PDF."""
    response = _get(url)
    with pymupdf.open(stream=response.content, filetype="pdf") as doc:
        raw = "\n".join(
            unicodedata.normalize("NFKC", page.get_text("text")) for page in doc
        )
    target = pd.Timestamp(actual_date)
    month = int(target.month)
    day = int(target.day)

    def count(text: str) -> int:
        serials: set[str] = set()
        for match in PDF_HEADER_RE.finditer(text):
            if int(match["month"]) == month and int(match["calday"]) == day:
                serials.add(match["serial"])
        return len(serials)

    compact = re.sub(r"\s+", "", "".join(ch for ch in raw if ch.isprintable()))
    return {
        "raw_header_count": count(raw),
        "compact_header_count": count(compact),
    }


def reconcile_official_inventory(
    days: pd.DataFrame,
    conditions: pd.DataFrame,
    source_ids: set[str],
    explicit_race_ids: set[str],
) -> tuple[set[str], pd.DataFrame]:
    """Return an official race inventory without mistaking parser gaps for missing races.

    Exact PDF race markers are used directly. If some markers are lost in machine
    text, the source day is admitted only when its realized race count is
    independently matched by the number of official daily-PDF race headers.
    """
    exact_ids = (
        set(conditions["race_id"].astype(str)) if not conditions.empty else set()
    )
    exact_ids.update(explicit_race_ids)
    inventory = set(exact_ids)

    source_by_day: dict[str, set[str]] = defaultdict(set)
    for race_id in source_ids:
        source_by_day[race_id[:10]].add(race_id)

    exact_by_day: dict[str, set[str]] = defaultdict(set)
    for race_id in exact_ids:
        exact_by_day[race_id[:10]].add(race_id)

    day_lookup = {
        str(row.race_day_key): row for row in days.itertuples(index=False)
    }
    preliminary_source_only = source_ids - inventory
    problem_days = sorted({race_id[:10] for race_id in preliminary_source_only})
    audit: list[dict] = []

    for day_key in problem_days:
        if day_key not in day_lookup:
            raise ValueError(f"Source race day absent from official day index: {day_key}")
        source_day = source_by_day[day_key]
        official_exact_day = exact_by_day.get(day_key, set())
        if not official_exact_day.issubset(source_day):
            raise ValueError(
                f"Official exact race markers disagree with source day {day_key}: "
                f"{sorted(official_exact_day - source_day)}"
            )

        row = day_lookup[day_key]
        source_numbers = sorted(int(rid[-2:]) for rid in source_day)
        if len(source_numbers) != len(set(source_numbers)):
            raise ValueError(f"Duplicate race numbers on source day {day_key}")
        if any(number < 1 or number > 12 for number in source_numbers):
            raise ValueError(f"Invalid race number on source day {day_key}")

        counts = _pdf_header_counts(
            str(row.official_source_url),
            str(row.actual_date),
        )
        expected = len(source_day)
        override = OFFICIAL_RACE_COUNT_OVERRIDES.get(day_key)
        candidates = {
            value for value in counts.values() if 1 <= int(value) <= 12
        }
        verified = expected in candidates
        method = "official_pdf_header_count"
        if not verified and override is not None and override == expected:
            verified = True
            method = "official_verified_count_override"
        if not verified:
            raise ValueError(
                f"Cannot independently verify realized race count for {day_key}: "
                f"source={expected}, counts={counts}, override={override}"
            )

        inventory.update(source_day)
        audit.append(
            {
                "race_day_key": day_key,
                "year": int(day_key[:4]),
                "actual_date": str(row.actual_date),
                "racecourse": str(row.racecourse),
                "source_race_count": expected,
                "exact_marker_count": len(official_exact_day),
                **counts,
                "count_override": override,
                "verification_method": method,
                "official_source_url": str(row.official_source_url),
            }
        )

    return inventory, pd.DataFrame(audit)


def _race_id_from_cname(value: str) -> str | None:
    match = RESULT_CNAME_RE.search(unquote(value))
    if not match:
        return None
    return (
        f"{match['year']}{match['venue']}{match['meeting']}"
        f"{match['day']}{match['race']}"
    )


def _discover_day_urls(seed_url: str, day_key: str) -> dict[str, str]:
    response = _get(seed_url)
    soup = BeautifulSoup(response.text, "html.parser")
    urls: dict[str, str] = {}

    current_id = _race_id_from_cname(response.url)
    if current_id and current_id.startswith(day_key):
        urls[current_id] = response.url

    for anchor in soup.find_all("a", href=True):
        href = urljoin(response.url, str(anchor["href"]))
        race_id = _race_id_from_cname(href)
        if race_id and race_id.startswith(day_key):
            urls[race_id] = href

    # Some JRADB navigation values can live in encoded attributes/scripts.
    if len(urls) < 12:
        for match in re.finditer(
            r"(?:/JRADB/)?accessS\.html\?CNAME="
            r"(?P<cname>pw01sde[^\"'<>\s&]+)",
            response.text,
        ):
            href = urljoin(
                response.url,
                "https://www.jra.go.jp/JRADB/accessS.html?CNAME="
                + match["cname"],
            )
            race_id = _race_id_from_cname(href)
            if race_id and race_id.startswith(day_key):
                urls[race_id] = href

    expected = {f"{day_key}{race_no:02d}" for race_no in range(1, 13)}
    if set(urls) != expected:
        raise ValueError(
            f"Official JRADB navigation did not expose all 12 races for {day_key}: "
            f"found={sorted(urls)}"
        )
    return dict(sorted(urls.items()))


def _normalize_header(value: str) -> str:
    return _normalize(value).replace("馬番", "馬番").replace("負担重量", "負担重量")


def _find_result_table(soup: BeautifulSoup):
    for table in soup.find_all("table"):
        text = _normalize(table.get_text(" ", strip=True))
        if all(token in text for token in ["着順", "馬番", "馬名", "性齢", "負担重量", "タイム"]):
            return table
    raise ValueError("Official JRADB result table not found")


def _parse_web_sex_age(value: str) -> tuple[str, float]:
    text = _normalize(value)
    match = re.search(r"(牡|牝|せん|セン|セ)(\d+)", text)
    if not match:
        raise ValueError(f"Cannot parse official sex/age: {value!r}")
    sex = {"牡": "M", "牝": "F", "せん": "G", "セン": "G", "セ": "G"}[match[1]]
    return sex, float(match[2])


def _parse_rank(value: str) -> tuple[float, str, bool]:
    text = _normalize(value)
    match = re.match(r"^(\d+)", text)
    if match:
        return float(match[1]), "finished", True
    if any(token in text for token in ["取消", "除外"]):
        return np.nan, "nonstarter", False
    if "中止" in text:
        return np.nan, "dnf", True
    if "失格" in text:
        return np.nan, "disqualified", True
    raise ValueError(f"Unhandled official placing status: {value!r}")


def _parse_time(value: str) -> float:
    text = _normalize(value)
    if not text:
        return np.nan
    match = re.fullmatch(r"(?:(\d+):)?(\d+(?:\.\d+)?)", text)
    if not match:
        return np.nan
    return int(match[1] or 0) * 60.0 + float(match[2])


def _parse_early_position(value: str) -> float:
    values = re.findall(r"\d+", value or "")
    return float(values[0]) if values else np.nan


def _race_class(text: str) -> str | None:
    normalized = _normalize(text)
    if "新馬" in normalized:
        return "Newcomer"
    if "未勝利" in normalized:
        return "Maiden"
    if "1勝クラス" in normalized:
        return "Class1"
    if "2勝クラス" in normalized:
        return "Class2"
    if "3勝クラス" in normalized:
        return "Class3"
    if "オープン" in normalized or re.search(r"G[ⅠⅡⅢ123]", normalized):
        return "Open"
    return None


def _parse_result_page(race_id: str, url: str) -> tuple[dict, list[dict], dict]:
    response = _get(url)
    content = response.content
    soup = BeautifulSoup(content, "html.parser")
    page_text = unicodedata.normalize("NFKC", soup.get_text(" ", strip=True))

    parsed_id = _race_id_from_cname(response.url)
    if parsed_id != race_id:
        raise ValueError(f"JRADB race-id redirect mismatch: {race_id} -> {parsed_id}")

    match = COURSE_RE.search(page_text)
    if not match:
        raise ValueError(f"Official course metadata not found for {race_id}")
    distance_m = int(match["distance"].replace(",", ""))
    surface = {"芝": "turf", "ダート": "dirt"}[match["surface"]]
    detail = match["detail"] or ""
    turn_direction = (
        "right" if "右" in detail else
        "left" if "左" in detail else
        "straight" if "直" in detail else
        "other"
    )
    course_layout = (
        "outer" if "外" in detail else
        "inner" if "内" in detail else
        "straight" if "直" in detail else
        np.nan
    )

    first_table = _find_result_table(soup)
    pre_table_text = page_text.split(_normalize(first_table.get_text(" ", strip=True))[:20], 1)[0]
    race_kind = "obstacle" if "障害" in pre_table_text else "flat"
    race_class = _race_class(pre_table_text)
    is_graded = int(bool(re.search(r"G[ⅠⅡⅢ123]", _normalize(pre_table_text))))
    is_open_plus = int(race_class == "Open" or is_graded == 1)
    handicap = int("ハンデ" in _normalize(pre_table_text))

    cells_rows: list[dict] = []
    declared_field_size = 0
    for tr in first_table.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if len(cells) < 10:
            continue
        texts = [cell.get_text(" ", strip=True) for cell in cells]
        if _normalize(texts[0]) in {"着順", ""}:
            continue

        horse_cell = cells[3] if len(cells) > 3 else None
        horse_link = None
        if horse_cell is not None:
            horse_link = horse_cell.find(
                "a",
                href=re.compile(r"accessU\.html\?CNAME=pw01dud"),
            )
        if horse_link is None:
            continue
        horse_match = HORSE_CNAME_RE.search(unquote(str(horse_link.get("href", ""))))
        if not horse_match:
            raise ValueError(f"Official horse id missing for {race_id}")

        try:
            horse_no = int(re.sub(r"\D", "", texts[2]))
        except ValueError as exc:
            raise ValueError(f"Invalid official horse number for {race_id}: {texts[2]!r}") from exc
        declared_field_size = max(declared_field_size, horse_no)

        finish_position, finish_status, started = _parse_rank(texts[0])
        sex, age = _parse_web_sex_age(texts[4])
        assigned_match = re.search(r"\d+(?:\.\d+)?", _normalize(texts[5]))
        if not assigned_match:
            raise ValueError(f"Invalid official assigned weight for {race_id}: {texts[5]!r}")

        cells_rows.append(
            {
                "race_id": race_id,
                "official_horse_id": horse_match["horse_id"],
                "horse_name": horse_link.get_text(" ", strip=True),
                "horse_no": horse_no,
                "sex": sex,
                "age": age,
                "assigned_weight_kg": float(assigned_match[0]),
                "finish_position": finish_position,
                "finish_status": finish_status,
                "started": started,
                "race_time_seconds": (
                    _parse_time(texts[7]) if finish_status == "finished" else np.nan
                ),
                "early_position": _parse_early_position(texts[9]),
            }
        )

    if not cells_rows:
        raise ValueError(f"No official result rows parsed for {race_id}")
    started_rows = [row for row in cells_rows if row["started"]]
    if not started_rows:
        raise ValueError(f"No starters parsed for {race_id}")

    year = int(race_id[:4])
    venue_code = race_id[4:6]
    meeting_number = int(race_id[6:8])
    meeting_day = int(race_id[8:10])
    race_number = int(race_id[10:12])
    date_match = RESULT_CNAME_RE.search(unquote(response.url))
    if not date_match:
        raise ValueError(f"Cannot recover date from result URL: {response.url}")
    actual_date = pd.Timestamp(date_match["date"]).date().isoformat()
    racecourse = VENUE_CODE_TO_JP.get(venue_code)
    if racecourse is None:
        raise ValueError(f"Unexpected supplemental venue code: {venue_code}")

    race = {
        "race_id": race_id,
        "year": year,
        "racecourse": racecourse,
        "meeting_number": meeting_number,
        "meeting_day": meeting_day,
        "race_number_from_id": race_number,
        "actual_date": actual_date,
        "surface": surface,
        "distance_m": distance_m,
        "turn_direction": turn_direction,
        "race_class_norm": race_class,
        "is_open_plus_final": is_open_plus,
        "is_graded_final": is_graded,
        "course_layout_final": course_layout,
        "handicap_indicator_final": handicap,
        "race_kind_final": race_kind,
        "official_source_url": response.url,
        "mapping_status": "official_jradb_supplement",
    }
    for row in started_rows:
        row.update(
            {
                "race_date": pd.Timestamp(actual_date),
                "racecourse": racecourse,
                "surface": surface,
                "distance_m": distance_m,
                "turn_direction": turn_direction,
                "race_class": race_class,
                "is_open_plus": is_open_plus,
                "is_graded": is_graded,
                "course_layout": course_layout,
                "handicap_indicator": handicap,
                "field_size": len(started_rows),
                "declared_field_size": declared_field_size,
            }
        )
        row.pop("started", None)

    manifest = {
        "race_id": race_id,
        "actual_date": actual_date,
        "racecourse": racecourse,
        "race_number": race_number,
        "race_kind": race_kind,
        "declared_field_size": declared_field_size,
        "starter_rows": len(started_rows),
        "official_source_url": response.url,
        "page_sha256": _page_sha256(content),
    }
    return race, started_rows, manifest


def fetch_official_2025_supplement(
    missing_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch the exact official JRADB races absent from Kaggle v1."""
    if not missing_ids:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    allowed_days = set(SUPPLEMENT_SEEDS)
    if any(race_id[:10] not in allowed_days for race_id in missing_ids):
        unexpected = sorted(
            race_id for race_id in missing_ids if race_id[:10] not in allowed_days
        )
        raise ValueError(
            f"Primary-source gaps outside frozen 2025 supplement scope: {unexpected[:20]}"
        )

    discovered: dict[str, str] = {}
    for day_key, seed in SUPPLEMENT_SEEDS.items():
        discovered.update(_discover_day_urls(seed, day_key))

    discovered_ids = set(discovered)
    if missing_ids != discovered_ids:
        raise ValueError(
            "Official 2025 supplement set changed: "
            f"requested={len(missing_ids)}, discovered={len(discovered_ids)}, "
            f"requested_only={sorted(missing_ids - discovered_ids)[:20]}, "
            f"discovered_only={sorted(discovered_ids - missing_ids)[:20]}"
        )

    races: list[dict] = []
    starters: list[dict] = []
    manifest: list[dict] = []
    for race_id in sorted(missing_ids):
        race, rows, audit = _parse_result_page(race_id, discovered[race_id])
        races.append(race)
        starters.extend(rows)
        manifest.append(audit)

    race_frame = pd.DataFrame(races)
    row_frame = pd.DataFrame(starters)
    manifest_frame = pd.DataFrame(manifest)
    if race_frame.race_id.duplicated().any():
        raise ValueError("Duplicate supplemental race ids")
    if row_frame.duplicated(["race_id", "official_horse_id"]).any():
        raise ValueError("Duplicate supplemental race/horse ids")
    if set(race_frame.race_id) != missing_ids:
        raise ValueError("Supplemental race set incomplete")
    return race_frame, row_frame, manifest_frame


def harmonize_supplement_horse_ids(
    supplement_rows: pd.DataFrame,
    source_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Map official year-end horses onto existing source IDs by exact normalized name."""
    if supplement_rows.empty:
        return supplement_rows.copy(), {
            "supplemental_name_mapped": 0,
            "supplemental_new_horses": 0,
            "supplemental_ambiguous_names": 0,
        }

    source_map: dict[str, set[str]] = defaultdict(set)
    for row in source_rows[["horse_name", "horse_id"]].itertuples(index=False):
        source_map[_normalize(row.horse_name)].add(str(row.horse_id))

    out = supplement_rows.copy()
    resolved: list[str] = []
    mapped = 0
    new = 0
    ambiguous = 0
    for row in out.itertuples(index=False):
        key = _normalize(row.horse_name)
        ids = source_map.get(key, set())
        if len(ids) == 1:
            resolved.append(next(iter(ids)))
            mapped += 1
        elif len(ids) == 0:
            resolved.append("jra:" + str(row.official_horse_id))
            new += 1
        else:
            ambiguous += 1
            official = str(row.official_horse_id)
            if official in ids:
                resolved.append(official)
            else:
                raise ValueError(
                    f"Ambiguous historical horse-name mapping for {row.horse_name}: "
                    f"{sorted(ids)} vs official {official}"
                )

    out["horse_id"] = resolved
    out = out.drop(columns=["official_horse_id"])
    return out, {
        "supplemental_name_mapped": mapped,
        "supplemental_new_horses": new,
        "supplemental_ambiguous_names": ambiguous,
    }


def supplemental_date_mapping(supplement_races: pd.DataFrame) -> pd.DataFrame:
    if supplement_races.empty:
        return pd.DataFrame()
    columns = [
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
    out = supplement_races[columns].rename(
        columns={"race_number_from_id": "race_number"}
    )
    return out.sort_values("race_id").reset_index(drop=True)
