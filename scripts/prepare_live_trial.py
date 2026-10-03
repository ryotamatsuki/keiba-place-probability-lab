"""Collect exhaustive target-horse histories and full past-race context, then predict."""
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
from bs4 import BeautifulSoup

from keiba_place_lab.live_public import parse_horse_inventory, parse_results, parse_roster
from keiba_place_lab.stage4_successor_phase3a import assert_full_field_context


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(url, cache):
    path = cache / (hashlib.sha256(url.encode()).hexdigest() + ".html")
    side = path.with_suffix(".json")
    if not path.exists():
        response = requests.get(url, timeout=(15, 45), headers={"User-Agent": "keiba-place-lab research/0.1"})
        response.raise_for_status()
        path.write_text(response.text)
        side.write_text(json.dumps({"url": url, "retrieved_at": datetime.now(UTC).isoformat(),
                                   "sha256": sha(path)}, indent=2))
        time.sleep(.5)
    return path.read_text(), json.loads(side.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/live_trial_20261004.json")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="analysis/live_trial_20261004")
    parser.add_argument("--cache", default="data/raw/live_trial_20261004")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out, cache = Path(args.output), Path(args.cache)
    out.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_parquet(args.source)
    provenance, rosters = [], []
    for target in cfg["targets"]:
        html, audit = fetch(f"https://sports.yahoo.co.jp/keiba/race/denma/{target['provider_id']}", cache)
        roster, _ = parse_roster(html, target)
        roster.to_csv(out / f"{target['race_id']}_roster.csv", index=False)
        rosters.append(roster); provenance.append(audit)
    horses = sorted(set(pd.concat(rosters).horse_id.astype(str)))
    print(f"Targets: {len(rosters)}, horses: {len(horses)}", flush=True)

    def get_horse(hid):
        html, audit = fetch(f"https://sports.yahoo.co.jp/keiba/directory/horse/{hid}/", cache)
        return hid, parse_horse_inventory(html, cfg["history_cutoff"]), audit

    inventory = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for hid, records, audit in pool.map(get_horse, horses):
            inventory[hid] = records; provenance.append(audit)
            print(f"Horse {hid}: {len(records)} past JRA flat races", flush=True)
    (out / "horse_inventory.json").write_text(json.dumps(inventory, indent=2))
    expected = {r["race_id"]: r["provider_id"] for rs in inventory.values() for r in rs}
    for hid in horses:
        known = set(baseline.loc[baseline.horse_id.astype(str).eq(hid), "race_id"].astype(str))
        listed = {r["race_id"] for r in inventory[hid]}
        if not known.issubset(listed):
            raise ValueError(f"Public inventory omits baseline starts for {hid}: {known-listed}")
    baseline_ids = set(baseline.race_id.astype(str))
    missing = {rid: pid for rid, pid in expected.items() if rid not in baseline_ids}
    # Include the complete most recent JRA flat cards, independently of target horse form.
    # Obstacle cards are rejected by metadata and must be explicitly excluded below.
    for venue in ("05", "08"):
        for number in range(1, 13):
            pid = f"26{venue}0401{number:02}"
            missing["20" + pid] = pid
    print(f"Fetching {len(missing)} complete past races", flush=True)

    def get_race(item):
        rid, pid = item
        html, audit = fetch(f"https://sports.yahoo.co.jp/keiba/race/result/{pid}", cache)
        try:
            result = parse_results(html, pid)
        except ValueError:
            info = BeautifulSoup(html, "html.parser").select_one(".hr-predictRaceInfo")
            if rid not in expected and info is not None and "障害" in info.get_text():
                return None, audit
            raise
        return result, audit

    supplements = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for frame, audit in pool.map(get_race, sorted(missing.items())):
            provenance.append(audit)
            if frame is not None:
                supplements.append(frame)
                print(f"Race {frame.race_id.iloc[0]}: {len(frame)} starters", flush=True)
    selected_baseline = baseline[baseline.race_id.astype(str).isin(expected)].copy()
    selected_baseline["outcome_confirmed"] = True
    history = pd.concat([selected_baseline, *supplements], ignore_index=True)
    history["race_date"] = pd.to_datetime(history.race_date)
    history = history.sort_values(["race_date", "race_id", "horse_no"])
    assert_full_field_context(history)
    if history.race_date.max() != pd.Timestamp(cfg["history_cutoff"]):
        raise ValueError("Latest completed cards not confirmed")
    coverage = []
    for hid in horses:
        expected_horse = {r["race_id"] for r in inventory[hid]}
        actual = set(history.loc[history.horse_id.astype(str).eq(hid), "race_id"].astype(str))
        if actual != expected_horse:
            raise ValueError(f"Incomplete target history for {hid}: missing={expected_horse-actual}, extra={actual-expected_horse}")
        coverage.append({"horse_id": hid, "expected": len(expected_horse), "actual": len(actual)})
    history.to_parquet(out / "target_history.parquet", index=False)
    (out / "history_manifest.json").write_text(json.dumps({
        "coverage_scope": "exhaustive listed JRA flat histories of all target horses plus full starters of every prior race and latest completed JRA cards",
        "global_2026_feed": False, "source_sha256": sha(args.source),
        "history_sha256": sha(out / "target_history.parquet"), "cutoff": cfg["history_cutoff"],
        "rows": len(history), "races": history.race_id.nunique(), "horse_coverage": coverage,
        "sources": provenance, "no_future_rows": bool(history.race_date.le(pd.Timestamp(cfg["history_cutoff"])).all()),
    }, indent=2, ensure_ascii=False))
    print(f"Complete scoped history: {len(history)} rows, {history.race_id.nunique()} races", flush=True)


if __name__ == "__main__":
    main()
