"""Independent full-field prior JRA flat race inventory for 2026 Tennosho.

This stage discovers dates and race IDs only. It MUST NOT claim reconstructed
F3 values, even if all 17 horse inventories match the older snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from keiba_place_lab.live_public import parse_horse_inventory, parse_roster

TARGET_RACE = "202605041111"
CUTOFF = "2026-10-30"
URL = "https://sports.yahoo.co.jp/keiba/directory/horse/{horse_id}/"


def get(url: str) -> tuple[str, dict]:
    response = None
    for attempt in range(4):
        try:
            response = requests.get(
                url,
                timeout=(10, 35),
                headers={"User-Agent": "keiba-place-lab/0.1 research history audit"},
            )
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(1 + 2 * attempt)
    assert response is not None
    return response.text, {
        "url": url,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "html_sha256": hashlib.sha256(response.content).hexdigest(),
        "status": response.status_code,
    }


def collect(config: dict, baseline: pd.DataFrame, cutoff: str) -> tuple[list[dict], dict]:
    html, roster_source = get(config["roster_url"])
    roster, _ = parse_roster(html, config)
    expected = config["horse_no_to_name"]
    if (
        len(roster) != 17
        or roster.horse_id.duplicated().any()
        or {str(r.horse_no): str(r.horse_name) for r in roster.itertuples(index=False)}
        != expected
    ):
        raise ValueError("Roster has changed; cannot attest original 17 horses")
    baseline = baseline.copy()
    baseline["race_date"] = pd.to_datetime(baseline.race_date)
    baseline["horse_id"] = baseline.horse_id.astype(str)
    baseline["race_id"] = baseline.race_id.astype(str)
    if baseline.race_date.max() > pd.Timestamp(cutoff):
        raise ValueError("Baseline contains rows later than cutoff")
    records = []
    def one(item):
        no, hid, name = item
        body, source = get(URL.format(horse_id=hid))
        inv = parse_horse_inventory(body, cutoff)
        prior = baseline.loc[baseline.horse_id.eq(hid), "race_id"]
        recorded = set(prior)
        published = {r["race_id"] for r in inv}
        if not recorded.issubset(published):
            raise ValueError(f"Stored 2026 JRA starts not in independent profile: {hid} {sorted(recorded-published)}")
        dates = [str(pd.Timestamp(r["race_date"]).date()) for r in inv]
        if len(inv) < 3 or any(pd.Timestamp(d) >= pd.Timestamp("2026-11-01") for d in dates):
            raise ValueError("Target horse historical inventory insufficient or future-dated")
        return {
            "horse_no": no, "horse_id": hid, "horse_name": name,
            "expected": len(inv),
            "historical_races": inv,
            "known_2026_before_or_on_oct3": len(recorded),
            "not_in_oct3_2026_baseline": sorted(published-recorded),
            "latest_jra_flat_date": max(dates),
            "inventory_source": source,
        }
    with ThreadPoolExecutor(max_workers=5) as pool:
        items = [
            (int(row.horse_no), str(row.horse_id), str(row.horse_name))
            for row in roster.itertuples(index=False)
        ]
        records.extend(pool.map(one, items))
    records.sort(key=lambda item: item["horse_no"])
    return records, roster_source


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--historical", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cutoff", default=CUTOFF)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Do not overwrite immutable history source inventory")
    if pd.Timestamp(a.cutoff) >= pd.Timestamp("2026-11-01"):
        raise ValueError("Target or future date not allowed")
    cfg = json.loads(a.config.read_text())
    data = pd.read_parquet(a.historical)
    if not pd.to_datetime(data.race_date).dt.year.eq(2026).all():
        raise ValueError("Baseline must be the registered 2026 current snapshot")
    records, roster_source = collect(cfg, data, a.cutoff)
    aggregate = {
        "status": "INVENTORY_ONLY_F3_NOT_YET_RECONSTRUCTED",
        "race_id": TARGET_RACE,
        "cutoff": a.cutoff,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "horse_count": len(records),
        "roster_source": roster_source,
        "historical_source_file": str(a.historical),
        "historical_source_sha256": hashlib.sha256(a.historical.read_bytes()).hexdigest(),
        "horses": records,
        "no_current_race_results_used": True,
    }
    if len(records) != 17:
        raise ValueError("Expected exactly 17 horses")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({
        "status": aggregate["status"], "cutoff": a.cutoff, "races": len({
            row["race_id"] for horse in records for row in horse["historical_races"]
        }),
        "missing_since_old_snapshot": [
            {"horse_no": horse["horse_no"], "horse_name": horse["horse_name"],
             "latest": horse["latest_jra_flat_date"],
             "additional_past_races": len([
                 race for race in horse["historical_races"]
                 if pd.Timestamp(race["race_date"]) > pd.Timestamp("2026-10-03")
             ])}
            for horse in records
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
