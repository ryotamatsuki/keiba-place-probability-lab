"""Fail-closed, cross-source verification of seven mislabeled frozen 2025 starters.

Only four pre-registered late-2025 races have malformed frozen JRA horse
identities, found by a 46-race exhaustive diagnostic. Keep the immutable
2010-2025 file and its SHA untouched; correct in-memory 2025 horse identity
and name only when Yahoo and Umanity independently agree on the same public
runners, numbers, ranks and published name.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from tennosho_f3_raw_audit import raw_result, umanity_raw_result

from keiba_place_lab import live_umanity
from keiba_place_lab.live_history import RESULT_URL, CachedFetcher
from keiba_place_lab.live_public import metadata



# race_id -> horse_no -> (frozen bad ID, actual verified ID, frozen bad name, rank)
CORRECTIONS = {
    "202506050706": {
        7: ("2011103929", "2023105213", "スペシャルギフト", 10),
        14: ("2016100062", "2023100693", "キタノローズ", 13),
    },
    "202506050806": {
        9: ("2008103901", "2023105382", "ブザービーター", 12),
        17: ("2016102619", "2023107088", "ソフィア", 11),
    },
    "202509050706": {
        6: ("2015101363", "2023105645", "サヴォアフェール", 2),
        9: ("2014105649", "2023106017", "ブラストビート", 8),
    },
    "202509050806": {
        6: ("2014105899", "2023102421", "ヴェネラブル", 12),
    },
}
AUDIT_RUN = 41175978988
TARGET = "202605041111"
CUTOFF = pd.Timestamp("2026-10-30")


def _id(value: object) -> str:
    value = str(value).removeprefix("jra:")
    if re.fullmatch(r"\d{10}", value) is None:
        raise ValueError(f"Invalid JRA result horse identity: {value!r}")
    return f"jra:{value}"


def _horse_names_from_yahoo(html: str, provider_id: str) -> dict[int, str]:
    soup, _ = metadata(html, provider_id)
    tables = [
        t for t in soup.select("table")
        if all(s in t.get_text() for s in ("着順", "馬名", "通過順位", "騎手名"))
    ]
    if len(tables) != 1:
        raise ValueError("No unique verifiable Yahoo result table")
    names = {}
    for tr in tables[0].select("tr"):
        cells = tr.find_all("td", recursive=False)
        if not cells:
            continue
        if len(cells) < 9:
            raise ValueError("Missing canonical Yahoo starter cells")
        no = int(cells[2].get_text(strip=True))
        a = cells[3].find("a", href=True)
        if a is None:
            raise ValueError("Canonical Yahoo horse-name anchor missing")
        name = a.get_text(strip=True)
        if not name or no in names:
            raise ValueError("Duplicate/blank canonical source horse-name")
        names[no] = name
    return names


def repair_frozen_2025_identities(
    original: pd.DataFrame, cache: Path,
) -> tuple[pd.DataFrame, list[dict]]:
    """Assert four exact races and known anomalous starters against two sources."""
    # This input was already content-SHA checked before calling.
    x = original.copy()
    x["horse_id"] = x.horse_id.map(_id)
    if x.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Frozen history has pre-existing race/horse duplicates")
    fetcher = CachedFetcher(cache, pause_seconds=.5, max_attempts=4)
    records = []
    for rid, replacements in CORRECTIONS.items():
        if rid == TARGET:
            raise ValueError("Target result must never enter frozen 2025 corrections")
        race = x.loc[x.race_id.astype(str).eq(rid)].copy()
        if race.empty or race.race_date.nunique() != 1:
            raise ValueError(f"Audited 2025 race absent or ambiguous: {rid}")
        date = pd.Timestamp(race.race_date.iloc[0])
        if not (pd.Timestamp("2025-12-27") <= date <= pd.Timestamp("2025-12-28")):
            raise ValueError(f"2025 correction date outside independently audited window: {rid}")
        if date > CUTOFF:
            raise ValueError("Future race in source repair")
        yahoo_url = RESULT_URL.format(provider_id=rid[2:])
        alt_url = f"https://umanity.jp/racing/result.php?race_id={live_umanity.source_id(rid,date)}"
        yahoo_html, yahoo_rec = fetcher.fetch(yahoo_url)
        alt_html, alt_rec = fetcher.fetch(alt_url)
        published_yahoo = raw_result(yahoo_html, rid[2:])
        published_alt = umanity_raw_result(alt_html, rid, date)
        y = published_yahoo.set_index("horse_no")
        u = published_alt.set_index("horse_no")
        frozen = race.set_index("horse_no")
        expected_numbers = set(frozen.index.astype(int))
        if len(frozen) != len(y) or len(frozen) != len(u) or (
            expected_numbers != set(y.index.astype(int))
            or expected_numbers != set(u.index.astype(int))
        ):
            raise ValueError(f"2025 source roster size or numbers disagree: {rid}")
        names = _horse_names_from_yahoo(yahoo_html, rid[2:])
        fixes = []
        for no in sorted(expected_numbers):
            original_row = frozen.loc[no]
            source_y = y.loc[no]
            source_u = u.loc[no]
            original_hid = _id(original_row.horse_id)
            source_hid = _id(source_y.horse_id)
            source_alt_hid = _id(source_u.horse_id)
            if source_hid != source_alt_hid:
                raise ValueError(f"Two independent sources disagree for {rid} number {no}")
            if str(source_y["rank"]) != str(source_u["rank"]):
                raise ValueError(f"Two published race ranks disagree {rid} number {no}")
            if no not in replacements:
                if original_hid != source_hid:
                    raise ValueError(f"Unregistered 2025 source disagreement {rid} number {no}")
                continue
            old_id, corrected_id, old_name, expected_rank = replacements[no]
            if (
                original_hid != _id(old_id)
                or source_hid != _id(corrected_id)
                or str(original_row.horse_name) != old_name
                or str(source_y["rank"]) != str(expected_rank)
                or pd.to_numeric(original_row.finish_position) != expected_rank
            ):
                raise ValueError(f"Unexpected audited original/source 2025 mismatch {rid} number {no}")
            real_name = names.get(no)
            if not real_name:
                raise ValueError(f"2025 corrected horse name unconfirmed: {rid} number {no}")
            mask = x.race_id.astype(str).eq(rid) & x.horse_no.eq(no)
            if mask.sum() != 1:
                raise ValueError(f"Ambiguous 2025 corrected source row {rid} number {no}")
            x.loc[mask, "horse_id"] = source_hid
            x.loc[mask, "horse_name"] = real_name
            fixes.append({
                "race_id": rid, "horse_no": no, "old_frozen_id": original_hid,
                "old_frozen_name": old_name, "published_id": source_hid,
                "published_name": real_name, "verified_rank": expected_rank,
            })
        if len(fixes) != len(replacements):
            raise ValueError(f"2025 source repairs missed some preregistered entries: {rid}")
        records.append({
            "race_id": rid,
            "race_date": date.date().isoformat(),
            "yahoo_source": yahoo_rec.url,
            "yahoo_sha256": yahoo_rec.sha256,
            "umanity_source": alt_rec.url,
            "umanity_sha256": alt_rec.sha256,
            "frozen_source_untouched": True,
            "historical_identity_diagnostic_run": AUDIT_RUN,
            "verified_corrections": fixes,
        })
        print(json.dumps({"audited_2025_identity_repair": rid,
                          "entries": len(fixes)}, ensure_ascii=False), flush=True)
    if x.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Duplicate 2025 identity after independently verified correction")
    return x, records
