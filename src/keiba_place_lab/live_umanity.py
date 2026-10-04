"""Market-free published result/entry adapters for an alternative history source."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd
from bs4 import BeautifulSoup

from .live_public import VENUES, _sex_age


def source_id(race_id: str, race_date) -> str:
    return pd.Timestamp(race_date).strftime("%Y%m%d") + str(race_id)[4:]


def metadata(html: str, race_id: str, race_date):
    soup = BeautifulSoup(html, "html.parser")
    identity = soup.select_one('meta[property="og:url"]')
    rid = source_id(race_id, race_date)
    if identity is None or not re.search(rf"race_id={rid}(?:&|$)", identity.get("content", "")):
        raise ValueError("Alternative source race identity mismatch")
    heading = soup.select_one("h1")
    if heading is None:
        raise ValueError("Alternative source title absent")
    date = re.search(r"(\d{4})年(\d+)月(\d+)日", heading.get_text())
    if date is None or pd.Timestamp(f"{date[1]}-{date[2]}-{date[3]}") != pd.Timestamp(race_date):
        raise ValueError("Alternative source race date mismatch")
    off = soup.find(string=re.compile(r"\d{1,2}:\d{2}発走"))
    if off is None:
        raise ValueError("Alternative source race header absent")
    header = off.find_parent("div").parent
    text = unicodedata.normalize("NFKC", header.get_text(" ", strip=True))
    course = re.search(r"(芝|ダート)・(左|右|直(?:線)?)[^\d]*?(\d+)m", text)
    if course is None or "障害" in text:
        raise ValueError("Alternative source is not a complete flat race")
    race_class = next((name for token, name in [
        ("新馬", "Newcomer"), ("未勝利", "Maiden"), ("1勝", "Class1"),
        ("2勝", "Class2"), ("3勝", "Class3"), ("オープン", "Open"),
    ] if token in text), None)
    if race_class is None:
        raise ValueError("Unknown alternative source class")
    graded = bool(re.search(r"\bG(?:III|II|I|1|2|3)\b", text))
    return soup, {"race_id": str(race_id), "race_date": pd.Timestamp(race_date),
                  "racecourse": VENUES[str(race_id)[4:6]],
                  "surface": "turf" if course[1] == "芝" else "dirt",
                  "distance_m": int(course[3]),
                  "turn_direction": {"左": "left", "右": "right", "直": "straight", "直線": "straight"}[course[2]],
                  "race_class": race_class, "is_open_plus": int(race_class == "Open"),
                  "is_graded": int(graded)}


def identity(cell):
    link = cell.select_one('a.horsename[href*="horse_id="]')
    if link is None:
        raise ValueError("Alternative source horse identity absent")
    match = re.search(r"horse_id=(\d{10})(?:&|$)", link["href"])
    if not match:
        raise ValueError("Alternative source horse id invalid")
    return match[1], link.get_text(strip=True)


def parse_result(html: str, race_id: str, race_date, *, source_url: str, retrieved_at: str):
    from .live_history import _parse_race_time, _status_from_rank
    from .stage4_successor_phase3a import assert_full_field_context

    soup, meta = metadata(html, race_id, race_date)
    tables = [t for t in soup.select("table") if t.select_one("td.td_order")]
    if len(tables) != 1:
        raise ValueError("Alternative confirmed result table absent")
    rows = []
    for tr in tables[0].select("tr"):
        c = tr.find_all("td", recursive=False)
        if not c:
            continue
        if len(c) < 8:
            raise ValueError("Incomplete alternative result row")
        rank = c[0].get_text(strip=True).removesuffix("着")
        rank = {"消": "取消", "除": "除外", "止": "中止", "失": "失格"}.get(rank, rank)
        status, starter, position = _status_from_rank(rank)
        hid, name = identity(c[3])
        sex, age = _sex_age(c[3])
        weight = c[5].select_one("span.gray6")
        if weight is None or not re.fullmatch(r"[▲△☆★◇]*\d+(?:\.\d+)?", weight.get_text(strip=True)):
            raise ValueError("Alternative assigned weight absent")
        tm = c[6].find("div")
        seconds = _parse_race_time(tm.get_text(strip=True) if tm else "")
        if status == "finished" and pd.isna(seconds):
            raise ValueError("Alternative finished runner time absent")
        corner = c[7].select_one("span.text_passage_sq")
        early = float(corner.get_text()) if corner is not None else float("nan")
        rows.append({**meta, "horse_id": hid, "horse_name": name, "horse_no": int(c[1].get_text()),
                     "sex": sex, "age": age, "assigned_weight_kg": float(re.sub(r"[▲△☆★◇]", "", weight.get_text())),
                     "finish_status": status, "is_starter": starter, "finish_position": position,
                     "race_time_seconds": seconds, "early_position": early, "outcome_confirmed": True,
                     "result_source_url": source_url, "result_retrieved_at": retrieved_at})
    entries = pd.DataFrame(rows)
    if entries.empty or entries.horse_id.duplicated().any() or entries.horse_no.duplicated().any():
        raise ValueError("Empty/duplicate alternative result identities")
    starters = entries.loc[entries.is_starter].copy()
    if not starters.finish_position.eq(1).any():
        raise ValueError("Alternative winning result absent")
    entries["field_size"] = len(starters)
    entries["declared_field_size"] = int(entries.horse_no.max())
    starters = entries.loc[entries.is_starter].copy()
    assert_full_field_context(starters)
    return starters.reset_index(drop=True), entries.reset_index(drop=True)


def parse_entry(html: str, race_id: str, race_date, *, source_url: str, retrieved_at: str):
    soup, meta = metadata(html, race_id, race_date)
    table = soup.select_one("table#grace_table1")
    if table is None:
        raise ValueError("Alternative declared roster absent")
    rows = []
    for tr in table.select("tr"):
        c = tr.find_all("td", recursive=False)
        if not c:
            continue
        if len(c) < 5:
            raise ValueError("Incomplete alternative entry row")
        hid, name = identity(c[2])
        sex, age = _sex_age(c[2])
        status_text = c[0].get_text() + c[1].get_text()
        # Restrict status to this declaration, never a previous race in the row.
        status = "scratched" if "取消" in status_text else "excluded" if "除外" in status_text else "active"
        weight = c[4].select_one("div.bold.gray7")
        if weight is None:
            raise ValueError("Alternative roster weight absent")
        number = re.search(r"\d+", c[0].get_text())
        if number is None:
            raise ValueError("Alternative horse number absent")
        weight_text = weight.get_text(strip=True)
        if not re.fullmatch(r"[▲△☆★◇]*\d+(?:\.\d+)?", weight_text):
            raise ValueError("Alternative roster weight invalid")
        rows.append({**meta, "horse_id": hid, "horse_name": name, "horse_no": int(number[0]),
                     "sex": sex, "age": age, "assigned_weight_kg": float(re.sub(r"[▲△☆★◇]", "", weight_text)),
                     "entry_status": status, "is_starter": status == "active",
                     "entry_source_url": source_url, "entry_retrieved_at": retrieved_at})
    entries = pd.DataFrame(rows)
    if entries.empty or entries.horse_id.duplicated().any() or entries.horse_no.duplicated().any():
        raise ValueError("Empty/duplicate alternative declared identities")
    entries["declared_field_size"] = int(entries.horse_no.max())
    return entries
