"""Strict public Sports Navi adapters; market values never enter horse features."""
from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

TOKYO = ZoneInfo("Asia/Tokyo")
VENUES = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
          "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}


def canonical_id(provider_id: str) -> str:
    if not re.fullmatch(r"\d{10}", provider_id) or provider_id[2:4] not in VENUES:
        raise ValueError("Not a JRA race identity")
    return "20" + provider_id


def _identity(cell):
    link = cell.find("a", href=re.compile(r"/directory/horse/\d{10}/"))
    if link is None:
        raise ValueError("Horse identity absent")
    return re.search(r"horse/(\d{10})", link["href"])[1], link.get_text(strip=True)


def _sex_age(cell):
    m = re.search(r"(牡|牝|せん|セン|セ)(\d+)", cell.get_text())
    if m is None:
        raise ValueError("Sex/age absent")
    return {"牡": "M", "牝": "F", "せん": "G", "セン": "G", "セ": "G"}[m[1]], int(m[2])


def _weight(cell):
    text = cell.find("p").get_text(strip=True)
    match = re.fullmatch(r"[▲△☆★◇]*([0-9]+(?:\.[0-9]+)?)(?:kg)?", text)
    if match is None:
        raise ValueError(f"Invalid assigned weight: {text}")
    return float(match[1])


def metadata(html: str, provider_id: str):
    soup = BeautifulSoup(html, "html.parser")
    identity = soup.select_one('meta[property="og:url"]')
    if identity is None or not re.search(rf"/race/(?:[^/]+/)+{re.escape(provider_id)}(?:[/?#]|$)", identity.get("content", "")):
        raise ValueError("Page race identity mismatch")
    info = soup.select_one(".hr-predictRaceInfo")
    if info is None:
        raise ValueError("Race metadata absent")
    text = info.get_text(" ", strip=True)
    date = re.search(r"(\d{4})年(\d+)月(\d+)日", text)
    course = re.search(r"(芝|ダート)・(左|右|直線)[^\d]*?(\d+)m", text)
    off = re.search(r"(\d{1,2}):(\d{2})発走", text)
    if not date or not course or not off or "障害" in text:
        raise ValueError("Not a complete flat race")
    race_date = pd.Timestamp(f"{date[1]}-{int(date[2]):02}-{int(date[3]):02}")
    if race_date.year != int(canonical_id(provider_id)[:4]):
        raise ValueError("Race year mismatch")
    race_class = next((name for token, name in [("新馬", "Newcomer"), ("未勝利", "Maiden"),
                      ("1勝", "Class1"), ("2勝", "Class2"), ("3勝", "Class3"),
                      ("オープン", "Open")] if token in text), None)
    if race_class is None:
        raise ValueError("Unknown race class")
    title = info.select_one(".hr-predictRaceInfo__title").get_text(" ", strip=True)
    graded = bool(re.search(r"\bG(?:III|II|I)\b", title))
    return soup, {"race_id": canonical_id(provider_id), "race_date": race_date,
                  "racecourse": VENUES[provider_id[2:4]], "surface": {"芝": "turf", "ダート": "dirt"}[course[1]],
                  "distance_m": int(course[3]), "turn_direction": {"左": "left", "右": "right", "直線": "straight"}[course[2]],
                  "race_class": race_class, "is_open_plus": int(race_class == "Open"), "is_graded": int(graded),
                  "off_at": datetime(race_date.year, race_date.month, race_date.day, int(off[1]), int(off[2]), tzinfo=TOKYO).isoformat()}


def parse_roster(html: str, target: dict):
    soup, meta = metadata(html, target["provider_id"])
    for key in ("race_id", "surface", "distance_m", "racecourse", "turn_direction", "race_class", "off_at"):
        if meta[key] != target[key]:
            raise ValueError(f"Scheduled target changed: {key}")
    table = soup.select_one("#denma_list table")
    if table is None:
        raise ValueError("Roster absent")
    rows, market, declared = [], [], 0
    for tr in table.select("tr"):
        c = tr.find_all("td", recursive=False)
        if not c:
            continue
        number = int(c[1].get_text(strip=True)); declared = max(declared, number)
        if any(x in tr.get_text() for x in ("取消", "除外")):
            continue
        hid, name = _identity(c[2]); sex, age = _sex_age(c[2])
        r = {k: v for k, v in meta.items() if k not in ("off_at", "is_open_plus", "is_graded")}
        r.update(horse_id=hid, horse_name=name, horse_no=number, sex=sex, age=age,
                 assigned_weight_kg=_weight(c[3]))
        rows.append(r)
        odds = c[-1].find("span")
        market.append({"race_id": meta["race_id"], "horse_id": hid,
                       "win_odds": float(odds.get_text()) if odds else np.nan})
    if not rows:
        raise ValueError("Empty roster")
    roster = pd.DataFrame(rows).assign(field_size=len(rows), declared_field_size=declared)
    return roster, pd.DataFrame(market)


def parse_horse_inventory(html: str, cutoff: str):
    """All listed JRA flat races, excluding regional/foreign/obstacle races."""
    soup = BeautifulSoup(html, "html.parser")
    tables = [t for t in soup.select("table") if "通過順位" in t.get_text() and "日付" in t.get_text()]
    if len(tables) != 1:
        raise ValueError("Full horse history table absent")
    rows = []
    for tr in tables[0].select("tr"):
        c = tr.find_all("td", recursive=False)
        if not c:
            continue
        link = c[1].find("a", href=re.compile(r"/race/index/\d{10}"))
        if link is None or "障" in c[1].get_text():
            continue
        pid = re.search(r"index/(\d{10})", link["href"])[1]
        if pid[2:4] not in VENUES:
            continue
        date = pd.Timestamp(c[0].get_text(strip=True))
        if date > pd.Timestamp(cutoff):
            continue
        if any(x in c[2].get_text() for x in ("取消", "除外")):
            continue
        rows.append({"race_id": canonical_id(pid), "provider_id": pid, "race_date": date.isoformat()})
    if not rows or len({r['race_id'] for r in rows}) != len(rows):
        raise ValueError("Empty/duplicate horse history inventory")
    return rows


def parse_results(html: str, provider_id: str):
    soup, meta = metadata(html, provider_id)
    tables = [t for t in soup.select("table") if all(s in t.get_text() for s in ("着順", "馬名", "通過順位", "騎手名"))]
    if len(tables) != 1:
        raise ValueError("Confirmed result table absent")
    rows, declared = [], 0
    for tr in tables[0].select("tr"):
        c = tr.find_all("td", recursive=False)
        if not c:
            continue
        rank = c[0].get_text(strip=True); number = int(c[2].get_text(strip=True))
        declared = max(declared, number)
        if any(x in rank for x in ("取消", "除外")):
            continue
        if not rank.isdigit() and rank not in ("中止", "失格"):
            raise ValueError(f"Unconfirmed placing: {rank}")
        hid, name = _identity(c[3]); sex, age = _sex_age(c[3])
        tm = re.match(r"\s*(\d+):(\d+\.\d+)", c[4].get_text())
        early = re.match(r"\s*(\d+)", c[5].get_text())
        row = {k: v for k, v in meta.items() if k != "off_at"}
        row.update(horse_id=hid, horse_name=name, horse_no=number, sex=sex, age=age,
                   assigned_weight_kg=_weight(c[6]),
                   finish_position=float(rank) if rank.isdigit() else np.nan,
                   finish_status="finished" if rank.isdigit() else "dnf",
                   race_time_seconds=int(tm[1])*60+float(tm[2]) if tm else np.nan,
                   early_position=float(early[1]) if early else np.nan, outcome_confirmed=True)
        if rank.isdigit() and not tm:
            raise ValueError("Finished runner has no time")
        rows.append(row)
    if not rows or not any(r["finish_position"] == 1 for r in rows):
        raise ValueError("No confirmed winning result")
    result = pd.DataFrame(rows).assign(field_size=len(rows), declared_field_size=declared)
    if result.horse_id.duplicated().any() or result.horse_no.duplicated().any():
        raise ValueError("Duplicate result identities")
    return result


def parse_win_snapshot(html: str, target: dict):
    soup, meta = metadata(html, target["provider_id"])
    if meta["off_at"] != target["off_at"] or meta["race_id"] != target["race_id"]:
        raise ValueError("Market schedule/identity changed")
    updated = soup.select_one("time.hr-horseResult__update")
    if updated is None or not updated.get("datetime"):
        raise ValueError("Provider odds timestamp absent")
    asof = datetime.fromisoformat(updated["datetime"])
    if asof.tzinfo is None:
        asof = asof.replace(tzinfo=TOKYO)
    tables = [t for t in soup.select("table") if all(s in t.get_text() for s in ("馬名", "単勝", "複勝"))]
    if len(tables) != 1:
        raise ValueError("Win odds table absent")
    rows = []
    for tr in tables[0].select("tr"):
        c = tr.find_all("td", recursive=False)
        if not c:
            continue
        if any(x in tr.get_text() for x in ("取消", "除外")):
            continue
        hid, name = _identity(c[2])
        rows.append({"race_id": meta["race_id"], "horse_id": hid, "horse_name": name,
                     "horse_no": int(c[1].get_text(strip=True)), "win_odds": float(c[3].get_text(strip=True))})
    if not rows:
        raise ValueError("Empty win odds snapshot")
    return pd.DataFrame(rows), asof
