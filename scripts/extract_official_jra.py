"""Extract official JRA race-day dates and best-effort race conditions.

Race-day dates are a hard requirement and are reconstructed from official JRA
annual-result PDFs. Race-level condition extraction is deliberately best-effort:
older PDFs can lose race-number glyphs in text extraction, so failure to recover
one condition row must never corrupt or invent a calendar date.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pymupdf
from materialize_historical_training_dataset import (
    VENUE_JP_RE,
    annual_pdf_links,
    get_bytes,
)

MODERN_MARKER = re.compile(
    rf"\((?P<year>\d{{4}})年(?P<meeting>\d+)(?P<venue>{VENUE_JP_RE})\)"
    rf"第(?P<meetday>\d+)日第(?P<race>\d{{1,2}})競走"
)
LEGACY_MARKER = re.compile(
    rf"\((?P<era>\d{{1,2}})(?P<venue>{VENUE_JP_RE})(?P<meeting>\d+)\)"
    rf"第(?P<meetday>\d+)日第(?P<race>\d{{1,2}})競走"
)
LEGACY_DAY = re.compile(r"第(?P<meetday>\d+)日(?P<month>\d{1,2})月(?P<calday>\d{1,2})日")
DATE_CANDIDATE = re.compile(
    r"(?=(?P<month>1[0-2]|[1-9])月(?P<calday>3[01]|[12]\d|[1-9])日)"
)

RACE_HEADER_DATE = re.compile(
    r"(?m)^[^\S\r\n]*\d{5}[^\S\r\n]+"
    r"(?P<month>1[0-2]|[1-9])\s*月\s*"
    r"(?P<calday>3[01]|[12]\d|[1-9])\s*日"
)

OFFICIAL_DATE_OVERRIDES = {
    "https://www.jra.go.jp/datafile/seiseki/report/2020/2020-1niigata1.pdf": "2020-05-09",
    "https://www.jra.go.jp/datafile/seiseki/report/2020/2020-2tokyo5.pdf": "2020-05-09",
    "https://www.jra.go.jp/datafile/seiseki/report/2020/2020-3kyoto5.pdf": "2020-05-09",
    # 3rd Nakayama meeting day 2 was split by snow: races 1-2 on Mar 29,
    # races 3-12 in continuation racing on Mar 31. The day-level map anchors
    # to the first official date; freeze_historical_panel applies race-level
    # official continuation overrides for races 3-12.
    "https://www.jra.go.jp/datafile/seiseki/report/2020/2020-3nakayama2.pdf": "2020-03-29",
}


def normalized_pdf_text(data: bytes) -> str:
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return "\n\f\n".join(
            unicodedata.normalize("NFKC", page.get_text("text")) for page in doc
        )


def compact_pdf_text(data: bytes) -> str:
    text = normalized_pdf_text(data)
    return re.sub(r"\s+", "", "".join(ch for ch in text if ch.isprintable()))


def _valid_date(year: int, month: int, day: int) -> str | None:
    try:
        return pd.Timestamp(year=year, month=month, day=day).date().isoformat()
    except ValueError:
        return None


def _valid_date_candidates(text: str, year: int) -> list[tuple[int, str]]:
    """Return valid date candidates while removing 11/12-month substring aliases.

    DATE_CANDIDATE intentionally overlaps so a true 1/31 can still be recovered
    from OCR/text such as "...211月31日" where an invalid 11/31 appears first.
    But for a valid "11月26日" the overlapping "1月26日" is an alias, not a
    second date. Likewise "12月26日" must not yield "2月26日".
    """
    raw: list[tuple[int, int, int, str]] = []
    for m in DATE_CANDIDATE.finditer(text):
        month = int(m["month"])
        day = int(m["calday"])
        date = _valid_date(year, month, day)
        if date is not None:
            raw.append((m.start(), month, day, date))
    out: list[tuple[int, str]] = []
    for pos, month, day, date in raw:
        shadowed = any(
            prev_pos == pos - 1 and prev_month in {11, 12} and prev_day == day
            for prev_pos, prev_month, prev_day, _ in raw
        )
        if not shadowed:
            out.append((pos, date))
    return out


def parse_race_days(
    info: dict, compact: str, raw_text: str | None = None
) -> list[dict]:
    """Recover official (meeting, day) -> calendar date facts."""
    year = int(info["year"])
    by_day: dict[int, str] = {}

    # Old meeting-level PDFs contain an official summary table with all meeting days.
    if info.get("legacy"):
        for m in LEGACY_DAY.finditer(compact):
            date = _valid_date(year, int(m["month"]), int(m["calday"]))
            if date is None:
                continue
            meetday = int(m["meetday"])
            old = by_day.get(meetday)
            if old is not None and old != date:
                raise ValueError(f"Conflicting official dates in {info['url']}: {old} vs {date}")
            by_day[meetday] = date
        if not by_day:
            raise ValueError(f"No official meeting-day summary dates: {info['url']}")
    else:
        # Real JRA result PDFs print a five-digit race-management code before the
        # calendar date on each race heading. Prefer that whitespace-preserving
        # header because compact text makes e.g. code-ending-1 + "1月5日"
        # indistinguishable from "11月5日".
        header_candidates: list[str] = []
        if raw_text is not None:
            for m in RACE_HEADER_DATE.finditer(raw_text):
                date = _valid_date(year, int(m["month"]), int(m["calday"]))
                if date is not None:
                    header_candidates.append(date)
        # In production extraction raw_text is always available. Do not fall
        # back to the compact string when a trustworthy header cannot be read:
        # concatenating a five-digit serial ending in 1/2 with 1月/2月 makes
        # January/November and February/December intrinsically ambiguous.
        # Synthetic/unit callers without raw_text may still exercise the compact
        # recovery logic below.
        override = OFFICIAL_DATE_OVERRIDES.get(info["url"])
        if raw_text is not None:
            unique = list(dict.fromkeys(header_candidates))
            if not unique:
                if override is None:
                    raise ValueError(
                        "No high-confidence five-digit JRA race-header date in "
                        f"official daily PDF: {info['url']}"
                    )
                unique = [override]
            if len(unique) > 1:
                if override is None:
                    raise ValueError(
                        f"Multiple official race-header dates in {info['url']}: {unique}"
                    )
                chosen = override
            else:
                chosen = unique[0]
        else:
            candidates = [date for _, date in _valid_date_candidates(compact, year)]
            unique = list(dict.fromkeys(candidates))
            if not unique:
                if override is None:
                    raise ValueError(
                        f"No valid calendar date in official daily PDF: {info['url']}"
                    )
                unique = [override]
            marker_days: list[str] = []
            for marker in [*MODERN_MARKER.finditer(compact), *LEGACY_MARKER.finditer(compact)]:
                prefix = compact[max(0, marker.start() - 240) : marker.start()]
                local = _valid_date_candidates(prefix, year)
                if local:
                    marker_days.append(local[-1][1])
            marker_unique = list(dict.fromkeys(marker_days))
            chosen = marker_unique[0] if len(marker_unique) == 1 else unique[0]
        meetday = int(info["day_no"])
        by_day[meetday] = chosen

    rows = []
    for meetday, date in sorted(by_day.items()):
        rows.append(
            {
                "race_day_key": (
                    f"{year}{info['venue_code']}{int(info['meeting_no']):02d}{meetday:02d}"
                ),
                "year": year,
                "racecourse": info["venue"],
                "meeting_number": int(info["meeting_no"]),
                "meeting_day": meetday,
                "actual_date": date,
                "official_source_url": info["url"],
                "mapping_status": (
                    "official_meeting_summary"
                    if info.get("legacy")
                    else "official_calendar_verified_override"
                    if info["url"] in OFFICIAL_DATE_OVERRIDES
                    else "official_daily_result_pdf"
                ),
            }
        )
    return rows


def parse_conditions(text: str) -> dict:
    """Extract only condition values explicitly present in official text."""
    s = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    course = re.search(r"\((芝|ダート)[・･]([^)]*)\)", s)
    head = s[:250]
    obstacle = bool(
        re.search(
            r"障害(?:2歳|3歳|4歳|4歳以上|未勝利|オープン)|J[・･]?G|ジャンプ|大障害",
            head,
        )
    )
    kind = "obstacle" if obstacle else "flat" if course else "unknown"
    surface = {"芝": "turf", "ダート": "dirt"}.get(course[1]) if course else None
    direction = None
    layout = None
    if course:
        detail = course[2]
        direction = (
            "right" if "右" in detail else "left" if "左" in detail else "straight" if "直" in detail else None
        )
        layout = (
            "outer" if "外" in detail else "inner" if "内" in detail else "straight" if "直" in detail else None
        )
    weight = re.search(r"負担重量[^本]{0,120}", s)
    handicap = None
    if weight:
        w = weight[0]
        handicap = 1 if "ハンデ" in w else 0 if any(t in w for t in ["馬齢", "定量", "別定"]) else None
    klass = None
    for tokens, label in [
        (["新馬"], "Newcomer"),
        (["未勝利"], "Maiden"),
        (["1勝クラス", "500万円以下", "500万下"], "Class1"),
        (["2勝クラス", "1000万円以下", "1000万下"], "Class2"),
        (["3勝クラス", "1600万円以下", "1600万下"], "Class3"),
        (["オープン"], "Open"),
    ]:
        if any(t in s for t in tokens):
            klass = label
            break
    grade = bool(re.search(r"\(G(?:III|II|I|1|2|3)\)", s))
    if grade:
        klass = "Open"
    dist = re.search(r"(?<!\d)(\d{4})(?!\d)", s)
    return {
        "official_distance_m": int(dist[1]) if dist else None,
        "race_kind": kind,
        "official_surface": surface,
        "course_layout": layout,
        "official_turn": direction,
        "handicap_indicator": handicap,
        "official_class": klass,
        "is_open_plus": None if klass is None else int(klass == "Open"),
        "is_graded": int(grade),
        "condition_excerpt": s[:700],
    }


def extract_pdf(info: dict, cache: Path) -> tuple[list[dict], list[dict]]:
    """Return required day facts plus optional race-condition facts."""
    pdf_cache = cache / (hashlib.sha256(info["url"].encode()).hexdigest() + ".pdf")
    if pdf_cache.exists():
        data = pdf_cache.read_bytes()
    else:
        time.sleep(0.25)
        data = get_bytes(info["url"])
        pdf_cache.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    raw_text = normalized_pdf_text(data)
    compact = re.sub(
        r"\s+", "", "".join(ch for ch in raw_text if ch.isprintable())
    )
    days = parse_race_days(info, compact, raw_text=raw_text)
    day_lookup = {int(r["meeting_day"]): r["actual_date"] for r in days}

    conditions: list[dict] = []
    matches = sorted(
        [*MODERN_MARKER.finditer(compact), *LEGACY_MARKER.finditer(compact)],
        key=lambda m: m.start(),
    )
    for i, m in enumerate(matches):
        year = int(info["year"])
        if (
            "year" in m.re.groupindex
            and m.groupdict().get("year")
            and int(m["year"]) != year
        ):
            continue
        if (
            "era" in m.re.groupindex
            and m.groupdict().get("era")
            and int(m["era"]) not in {year - 1988, year - 2018}
        ):
            continue
        if int(m["meeting"]) != int(info["meeting_no"]) or m["venue"] != info["venue"]:
            continue
        meetday = int(m["meetday"])
        race_no = int(m["race"])
        if not 1 <= race_no <= 12 or meetday not in day_lookup:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else min(len(compact), start + 5000)
        snippet = compact[start:end]
        row = {
            "race_id": (
                f"{year}{info['venue_code']}{int(info['meeting_no']):02d}{meetday:02d}{race_no:02d}"
            ),
            "year": year,
            "racecourse": info["venue"],
            "meeting_number": int(info["meeting_no"]),
            "meeting_day": meetday,
            "race_number": race_no,
            "actual_date": day_lookup[meetday],
            "official_source_url": info["url"],
            "pdf_sha256": digest,
            "mapping_status": "official_race_marker_verified",
        }
        row.update(parse_conditions(snippet))
        conditions.append(row)

    # Duplicate race markers in PDF text are harmless only when facts agree.
    if conditions:
        frame = pd.DataFrame(conditions)
        conflicts = frame.groupby("race_id").actual_date.nunique().gt(1)
        if conflicts.any():
            raise ValueError(f"Conflicting condition rows in {info['url']}")
        conditions = frame.drop_duplicates("race_id", keep="first").to_dict("records")
    return days, conditions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    cache = Path("data/historical_raw/official_fact_cache")
    cache.mkdir(parents=True, exist_ok=True)
    output = Path("data/derived")
    output.mkdir(parents=True, exist_ok=True)

    tasks = annual_pdf_links(args.year)
    day_rows: list[dict] = []
    condition_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(lambda t: extract_pdf(t, cache), tasks), 1):
            days, conditions = result
            day_rows.extend(days)
            condition_rows.extend(conditions)
            if i % 20 == 0 or i == len(tasks):
                print(f"{args.year}: {i}/{len(tasks)} PDFs", flush=True)

    days = pd.DataFrame(day_rows)
    if days.empty:
        raise ValueError(f"No official dates extracted for {args.year}")
    if days.groupby("race_day_key").actual_date.nunique().gt(1).any():
        raise ValueError("Conflicting official race-day dates")
    duplicate_course_dates = days.duplicated(
        ["actual_date", "racecourse"], keep=False
    )
    if duplicate_course_dates.any():
        bad = days.loc[
            duplicate_course_dates,
            ["race_day_key", "actual_date", "racecourse", "official_source_url"],
        ]
        raise ValueError(
            "Duplicate official course/date mapping: {}".format(
                bad.to_dict("records")[:20]
            )
        )
    days = (
        days.sort_values(["race_day_key", "official_source_url"])
        .drop_duplicates("race_day_key", keep="first")
        .reset_index(drop=True)
    )
    days.to_csv(output / f"jra_official_{args.year}.csv", index=False)

    conditions = pd.DataFrame(condition_rows)
    if conditions.empty:
        conditions = pd.DataFrame(
            columns=[
                "race_id", "year", "racecourse", "meeting_number", "meeting_day",
                "race_number", "actual_date", "official_source_url", "pdf_sha256",
                "mapping_status", "official_distance_m", "race_kind", "official_surface",
                "course_layout", "official_turn", "handicap_indicator", "official_class",
                "is_open_plus", "is_graded", "condition_excerpt",
            ]
        )
    if conditions.race_id.duplicated().any():
        raise ValueError("Duplicate official condition race ids")
    conditions.to_csv(output / f"jra_official_conditions_{args.year}.csv", index=False)

    print(
        {
            "official_days": len(days),
            "condition_races": len(conditions),
            "condition_race_kinds": conditions.race_kind.value_counts(dropna=False).to_dict()
            if "race_kind" in conditions
            else {},
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
