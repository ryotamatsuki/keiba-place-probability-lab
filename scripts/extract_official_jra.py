"""Extract official race dates and conditions; cache facts, never commit PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz
import pandas as pd
from materialize_historical_training_dataset import (
    FULL_RACE_HEADER_RE,
    VENUE_JP_TO_CODE,
    annual_pdf_links,
    get_bytes,
)


def parse_conditions(text: str) -> dict:
    """Only explicit official condition text produces values."""
    s = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    course = re.search(r"\((芝|ダート)[・･]([^)]*)\)", s)
    obstacle = "障害" in s or bool(re.search(r"J[・･]?G|ジャンプ|大障害", s))
    kind = "obstacle" if obstacle else "flat" if course else "unknown"
    surface = {"芝": "turf", "ダート": "dirt"}.get(course[1]) if course else None
    direction = None
    layout = None
    if course:
        detail = course[2]
        direction = (
            "right"
            if "右" in detail
            else "left"
            if "左" in detail
            else "straight"
            if "直" in detail
            else None
        )
        layout = (
            "outer"
            if "外" in detail
            else "inner"
            if "内" in detail
            else "straight"
            if "直" in detail
            else None
        )
    weight = re.search(r"負担重量[^本]*", s)
    handicap = None
    if weight:
        w = weight[0]
        handicap = (
            1 if "ハンデ" in w else 0 if any(t in w for t in ["馬齢", "定量", "別定"]) else None
        )
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
    return {
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


def extract(info: dict, cache: Path) -> list[dict]:
    key = hashlib.sha256(info["url"].encode()).hexdigest()
    fact = cache / f"{key}.json"
    if fact.exists():
        return json.loads(fact.read_text())
    time.sleep(0.3)
    data = get_bytes(info["url"])
    digest = hashlib.sha256(data).hexdigest()
    rows = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", page.get_text("text")))
            for m in FULL_RACE_HEADER_RE.finditer(compact):
                if (
                    int(m["year"]) != info["year"]
                    or int(m["meeting"]) != info["meeting_no"]
                    or m["venue"] != info["venue"]
                ):
                    raise ValueError(f"PDF/header disagreement: {info}")
                race_no = int(m["race"])
                date = pd.Timestamp(
                    year=int(m["year"]), month=int(m["month"]), day=int(m["calday"])
                )
                code = VENUE_JP_TO_CODE[m["venue"]]
                race_id = (
                    f"{m['year']}{code}{int(m['meeting']):02d}{int(m['meetday']):02d}{race_no:02d}"
                )
                start = m.end()
                end = compact.find("本賞", start)
                snippet = compact[start : end if end >= start else start + 500]
                row = {
                    "race_id": race_id,
                    "year": int(m["year"]),
                    "racecourse": m["venue"],
                    "meeting_number": int(m["meeting"]),
                    "meeting_day": int(m["meetday"]),
                    "race_number": race_no,
                    "actual_date": date.date().isoformat(),
                    "official_source_url": info["url"],
                    "pdf_sha256": digest,
                    "mapping_status": "official_header_verified",
                }
                row.update(parse_conditions(snippet))
                rows.append(row)
    if not rows:
        raise ValueError(f"No race headings: {info['url']}")
    fact.write_text(json.dumps(rows, ensure_ascii=False))
    return rows


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
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, records in enumerate(pool.map(lambda t: extract(t, cache), tasks), 1):
            rows.extend(records)
            if i % 20 == 0:
                print(f"{args.year}: {i}/{len(tasks)} PDFs", flush=True)
    df = pd.DataFrame(rows).sort_values("race_id").reset_index(drop=True)
    if df["race_id"].duplicated().any():
        raise ValueError("Duplicate official race headers")
    df.to_csv(output / f"jra_official_{args.year}.csv", index=False)
    print(df.groupby("race_kind").size().to_dict(), flush=True)
    print("condition null counts", df.isna().sum().to_dict(), flush=True)


if __name__ == "__main__":
    main()
