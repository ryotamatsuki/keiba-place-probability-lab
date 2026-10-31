"""Collect *past* full-field 2025/26 Yahoo raw F3 evidence, fail closed.

No target-race result, closing odds or label is fetched. Source rows stay on
the runner; only independently verifiable source hashes and derived F3 may be
committed after all history checks pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from keiba_place_lab.live_history import CachedFetcher, RESULT_URL, _status_from_rank
from keiba_place_lab.live_public import _identity, metadata

TARGET = "202605041111"
SOURCE_CUTOFF = pd.Timestamp("2026-10-30")


def raw_result(html: str, provider_id: str) -> pd.DataFrame:
    soup, meta = metadata(html, provider_id)
    if meta["race_id"] == TARGET or meta["race_date"] > SOURCE_CUTOFF:
        raise ValueError("Target or future race appears in prior result source")
    tables = [
        t for t in soup.select("table")
        if all(s in t.get_text() for s in ("着順", "馬名", "通過順位", "騎手名"))
    ]
    if len(tables) != 1:
        raise ValueError("Single published full result table absent")
    info = soup.select_one(".hr-predictRaceInfo")
    condition_match = re.search(
        r"馬場\s*[:：]\s*(稍重|不良|良|重)",
        info.get_text(" ", strip=True) if info else "",
    )
    if condition_match is None:
        raise ValueError("Published race track condition absent")
    rows = []
    for tr in tables[0].select("tr"):
        c = tr.find_all("td", recursive=False)
        if not c:
            continue
        if len(c) < 9:
            raise ValueError("Published historical result has fewer than 9 cells")
        status, is_starter, _ = _status_from_rank(c[0].get_text(strip=True))
        if not is_starter:
            continue
        hid, _ = _identity(c[3])
        passage_text = c[5].get_text(" ", strip=True)
        passing = re.match(r"^\s*(\d+(?:-\d+){0,3})(?:\s|$)", passage_text)
        times = re.findall(r"(?<![\d.])([2-6]\d\.\d)(?!\d)", passage_text)
        mass = re.search(r"([3-7]\d{2})\s*\(\s*([+-]?\d+)\s*\)", c[3].get_text(" ", strip=True))
        jockey = c[6].find("a")
        trainer = c[8].find("a")
        rows.append({
            "race_id": meta["race_id"],
            "horse_id": hid,
            "horse_no": int(c[2].get_text(strip=True)),
            "race_date": meta["race_date"].date().isoformat(),
            "rank": c[0].get_text(strip=True),
            "last_3f": times[-1] if times else None,
            "passing": passing[1] if passing and meta["turn_direction"] != "straight" else None,
            "horse_weight": f"{mass[1]}({mass[2]})" if mass else None,
            "jockey_name": jockey.get_text(strip=True) if jockey else None,
            "trainer": trainer.get_text(strip=True) if trainer else None,
            "track_condition": condition_match[1],
            "finish_status": status,
        })
    if not rows:
        raise ValueError("No previous confirmed runners")
    x = pd.DataFrame(rows)
    if x.duplicated(["race_id", "horse_id"]).any() or x.horse_no.duplicated().any():
        raise ValueError("Duplicate runner or number in published prior results")
    if x.jockey_name.notna().sum() < len(x) * .95 or x.trainer.notna().sum() < len(x) * .95:
        raise ValueError("Raw jockey/trainer parsing insufficient for complete F3")
    finished = x.finish_status.eq("finished")
    if finished.any() and x.loc[finished, "last_3f"].notna().mean() < .9:
        raise ValueError("Raw final-3F coverage inadequate: refuse synthetic F3")
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path("/tmp/tennosho-f3-raw-cache"))
    p.add_argument("--race-ids", nargs="*", help="Optional limited two-race parsing smoke test")
    p.add_argument("--max-workers", type=int, default=6)
    args = p.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("Immutable raw capture paths already exist")
    history = pd.read_parquet(args.history)
    history["race_id"] = history.race_id.astype(str)
    history["horse_id"] = history.horse_id.astype(str)
    history["race_date"] = pd.to_datetime(history.race_date)
    if history.race_date.max() > SOURCE_CUTOFF:
        raise ValueError("Input has target/future date")
    if history.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Duplicated official historical starter identity")
    if args.race_ids:
        ids = set(args.race_ids)
        if not ids.issubset(set(history.race_id)):
            raise ValueError("Requested smoke race absent in audited official history")
        history = history.loc[history.race_id.isin(ids)].copy()
    race_ids = sorted(set(history.race_id))
    fetcher = CachedFetcher(args.cache, pause_seconds=.5)
    def one(rid: str):
        url = RESULT_URL.format(provider_id=rid[2:])
        html, record = fetcher.fetch(url)
        parsed = raw_result(html, rid[2:])
        expected = history.loc[history.race_id.eq(rid), ["horse_id", "horse_no"]].copy()
        actual = parsed[["horse_id", "horse_no"]].copy()
        if len(actual) != len(expected) or not actual.sort_values(
            ["horse_id", "horse_no"]).reset_index(drop=True).equals(
                expected.sort_values(["horse_id", "horse_no"]).reset_index(drop=True)
        ):
            raise ValueError(f"Official 2026 roster disagrees with source raw results: {rid}")
        return parsed, record
    parts, records, errors = [], [], {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        future = {pool.submit(one, rid): rid for rid in race_ids}
        for done in as_completed(future):
            rid = future[done]
            try:
                frame, record = done.result()
                parts.append(frame)
                records.append({"race_id": rid, "source_url": record.url, "source_sha256": record.sha256,
                                "captured_at_utc": record.retrieved_at})
            except Exception as exc:
                errors[rid] = str(exc)
            count = len(parts) + len(errors)
            if count % 200 == 0 or count == len(race_ids):
                print(json.dumps({"done": count, "total": len(race_ids), "failures": len(errors)}), flush=True)
    if errors:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps({
            "status": "INCOMPLETE_RAW_SOURCE_NOT_F3_READY",
            "attempted": len(race_ids), "failures": errors,
            "successful": len(parts), "finished_at_utc": datetime.now(UTC).isoformat(),
        }, ensure_ascii=False, indent=2) + "\n")
        raise RuntimeError(f"Raw source incomplete for {len(errors)} JRA races; no F3 vector created")
    df = pd.concat(parts).sort_values(["race_date", "race_id", "horse_no"]).reset_index(drop=True)
    if len(df) != len(history) or df.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Raw source missing complete historical runner rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, lineterminator="\n")
    records.sort(key=lambda r: r["race_id"])
    audit = {
        "status": "COMPLETE_RAW_PRIOR_RESULT_SOURCE",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "cutoff": str(SOURCE_CUTOFF.date()),
        "race_count": len(race_ids), "rows": len(df),
        "source_csv_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "history_input_sha256": hashlib.sha256(args.history.read_bytes()).hexdigest(),
        "source_records": records,
        "no_target_race_fetch": True,
    }
    args.manifest.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": audit["status"], "races": len(race_ids), "rows": len(df),
        "sha256": audit["source_csv_sha256"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
