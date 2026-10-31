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

from keiba_place_lab import live_umanity
from keiba_place_lab.live_history import RESULT_URL, CachedFetcher, _status_from_rank
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


def umanity_raw_result(html: str, race_id: str, race_date: pd.Timestamp) -> pd.DataFrame:
    """Verified alternate result source for Yahoo 500s, using the same raw schema.

    Never impute unobserved 3F, mass, running order, agent or going. The
    independent full-field identity check in main() still applies.
    """
    soup, meta = live_umanity.metadata(html, race_id, race_date)
    if meta["race_id"] == TARGET or meta["race_date"] > SOURCE_CUTOFF:
        raise ValueError("Target or future race in alternate result source")
    tables = [t for t in soup.select("table") if t.select_one("td.td_order")]
    if len(tables) != 1:
        raise ValueError("Alternate published full result table absent")
    header = soup.get_text(" ", strip=True).split("結果払戻", maxsplit=1)[0]
    condition_match = re.search(r"馬場\s*(稍重|不良|良|重)", header)
    if not condition_match:
        raise ValueError("Alternate published race going missing")
    going = condition_match[1]
    rows = []
    for tr in tables[0].select("tr"):
        c = tr.find_all("td", recursive=False)
        if not c:
            continue
        if len(c) < 11:
            raise ValueError("Incomplete alternate historical raw result")
        rank = c[0].get_text(" ", strip=True).replace(" ", "").removesuffix("着")
        rank = {"消": "取消", "除": "除外", "止": "中止", "失": "失格"}.get(rank, rank)
        status, starter, _ = _status_from_rank(rank)
        if not starter:
            continue
        hid, _ = live_umanity.identity(c[3])
        horse_no = int(c[1].get_text(strip=True))
        mass = re.search(
            r"(?<!\d)([3-7]\d{2})\s*([+-]?\d+|±0)(?!\d)",
            c[3].get_text(" ", strip=True),
        )
        # Umanity shows passage squares followed by "後3F" in cell 7.
        passage = c[7].get_text(" ", strip=True)
        times = re.findall(r"(?<![\d.])([2-6]\d\.\d)(?!\d)", passage)
        ahead = passage[:passage.rfind(times[-1])].strip() if times else passage
        blocks = re.findall(r"\b\d{1,2}\b", ahead)
        if len(blocks) > 4:
            raise ValueError("Alternate passage has >4 corners")
        jockey = c[5].find("a")
        trainer = c[10].find("a")
        jockey_name = (
            jockey.get_text(strip=True) if jockey else
            re.sub(r"\s*[\d.]+$", "", c[5].get_text(" ", strip=True)).strip()
        )
        trainer_name = (
            trainer.get_text(strip=True) if trainer else
            c[10].get_text(" ", strip=True).split(" ")[0]
        )
        rows.append({
            "race_id": race_id,
            "horse_id": hid,
            "horse_no": horse_no,
            "race_date": pd.Timestamp(race_date).date().isoformat(),
            "rank": rank,
            "last_3f": times[-1] if times else None,
            "passing": "-".join(blocks) if blocks and meta["turn_direction"] != "straight" else None,
            "horse_weight": f"{mass[1]}({mass[2].replace('±', '')})" if mass else None,
            "jockey_name": jockey_name or None,
            "trainer": trainer_name or None,
            "track_condition": going,
            "finish_status": status,
        })
    if not rows:
        raise ValueError("No independently identified alternate prior runners")
    x = pd.DataFrame(rows)
    if x.duplicated(["race_id", "horse_id"]).any() or x.horse_no.duplicated().any():
        raise ValueError("Duplicate alternate runner identity")
    if x.jockey_name.notna().mean() < .95 or x.trainer.notna().mean() < .95:
        raise ValueError("Alternate jockey/trainer incomplete")
    finished = x.finish_status.eq("finished")
    if finished.any() and x.loc[finished, "last_3f"].notna().mean() < .9:
        raise ValueError("Alternate published 3F incomplete")
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path("/tmp/tennosho-f3-raw-cache"))
    p.add_argument("--race-ids", nargs="*", help="Optional limited two-race parsing smoke test")
    p.add_argument("--max-workers", type=int, default=6)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
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
    if not 1 <= args.shard_count <= 64 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid raw source shard count/index")
    if args.shard_count > 1:
        race_ids = race_ids[args.shard_index::args.shard_count]
        history = history.loc[history.race_id.isin(race_ids)].copy()
    if not race_ids:
        raise ValueError("Selected raw source shard is empty")
    fetcher = CachedFetcher(args.cache, pause_seconds=.5)
    alt_fetcher = CachedFetcher(args.cache, pause_seconds=1.)
    def has_cache(url: str) -> bool:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return (args.cache / f"{key}.html").is_file() and (
            args.cache / f"{key}.json").is_file()
    def one(rid: str):
        cohort = history.loc[history.race_id.eq(rid)].copy()
        expected = cohort[["horse_id", "horse_no"]].copy()
        date = pd.Timestamp(cohort.race_date.iloc[0])
        published_source = str(cohort.result_source_url.iloc[0]) if "result_source_url" in cohort else ""
        selected_alt = published_source.startswith("https://umanity.jp/")
        if selected_alt:
            source_rid = live_umanity.source_id(rid, date)
            alt_url = f"https://umanity.jp/racing/result.php?race_id={source_rid}"
            html, record = alt_fetcher.fetch(alt_url)
            parsed = umanity_raw_result(html, rid, date)
        else:
            url = RESULT_URL.format(provider_id=rid[2:])
            if not has_cache(url):
                # The standardized result was already audited. When Yahoo
                # cannot be recovered from the prepared HTML cache, avoid six
                # redundant Yahoo 500 retries before a separately identity-
                # verified alternate. This never synthesizes missing fields.
                sid = live_umanity.source_id(rid, date)
                alternate_url = f"https://umanity.jp/racing/result.php?race_id={sid}"
                html, record = alt_fetcher.fetch(alternate_url)
                parsed = umanity_raw_result(html, rid, date)
            else:
                try:
                    html, record = fetcher.fetch(url)
                    parsed = raw_result(html, rid[2:])
                except (ValueError, RuntimeError, OSError) as exc:
                    source_rid = live_umanity.source_id(rid, date)
                    alt_url = f"https://umanity.jp/racing/result.php?race_id={source_rid}"
                    print(json.dumps({"alternate_source": rid, "reason": str(exc)[:150]}), flush=True)
                    html, record = alt_fetcher.fetch(alt_url)
                    parsed = umanity_raw_result(html, rid, date)
                except Exception as exc:  # noqa: BLE001 - HTTP status and timeout only
                    source_rid = live_umanity.source_id(rid, date)
                    alt_url = f"https://umanity.jp/racing/result.php?race_id={source_rid}"
                    print(json.dumps({"alternate_source": rid, "reason": str(exc)[:150]}), flush=True)
                    html, record = alt_fetcher.fetch(alt_url)
                    parsed = umanity_raw_result(html, rid, date)
        parsed["source_provider"] = "umanity" if record.url.startswith(
            "https://umanity.jp/"
        ) else "yahoo"
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
            except Exception as exc:  # noqa: BLE001 - record every independent source failure
                errors[rid] = str(exc)
                print(json.dumps({"raw_source_race_failed": rid, "reason": str(exc)[:600]},
                                 ensure_ascii=False), flush=True)
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
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "source_records": records,
        "no_target_race_fetch": True,
    }
    args.manifest.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": audit["status"], "races": len(race_ids), "rows": len(df),
        "sha256": audit["source_csv_sha256"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
