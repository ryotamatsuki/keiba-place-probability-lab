"""Fetch a fresh active roster and timestamped odds; preview or create a real timed lock."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from keiba_place_lab.live_public import parse_roster, parse_win_snapshot
from keiba_place_lab.market import harville_top_k_probabilities, normalized_win_probabilities


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url, path):
    response = requests.get(url, timeout=(5, 12), headers={"User-Agent": "keiba-place-lab research/0.1", "Cache-Control": "no-cache"})
    response.raise_for_status()
    path.write_text(response.text)
    return response.text, {"url": url, "retrieved_at": datetime.now(UTC).isoformat(),
                           "html_sha256": digest(path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["preview", "lock"])
    p.add_argument("--race-id", required=True)
    p.add_argument("--config", type=Path, default=Path("config/live_trial_20261004.json"))
    p.add_argument("--prepared", type=Path, default=Path("analysis/live_trial_20261004"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    target = next(t for t in json.loads(args.config.read_text())["targets"] if t["race_id"] == args.race_id)
    off = datetime.fromisoformat(target["off_at"])
    if datetime.now(UTC) >= off:
        raise ValueError("Race already off: never backfill a live snapshot")
    if args.mode == "lock":
        seconds = (off - datetime.now(UTC)).total_seconds()
        if seconds > 660 or seconds < 540:
            raise ValueError("Invoke lock within eleven to nine minutes before off")
        while (off - datetime.now(UTC)).total_seconds() > 600:
            time.sleep(1)
    history = args.prepared / "target_history.parquet"
    audit = json.loads((args.prepared / "history_manifest.json").read_text())
    if digest(history) != audit["history_sha256"] or audit["no_future_rows"] is not True:
        raise ValueError("Prepared history integrity failure")
    args.output.mkdir(parents=True, exist_ok=False)
    raw = Path("data/raw/live_trial_20261004") / args.output.name
    raw.mkdir(parents=True, exist_ok=False)
    html, roster_source = fetch(f"https://sports.yahoo.co.jp/keiba/race/denma/{target['provider_id']}", raw / "roster.html")
    roster, _ = parse_roster(html, target)
    known = {x["horse_id"] for x in audit["horse_coverage"] if x["expected"] == x["actual"]}
    if not set(roster.horse_id).issubset(known):
        raise ValueError("Changed roster contains a horse without audited complete history")
    roster_path = args.output / "roster.csv"
    roster.to_csv(roster_path, index=False)
    predictions = args.output / "shadow.csv"
    subprocess.run([sys.executable, "scripts/run_stage4_scope_v3.py", "predict", "--roster", str(roster_path),
                    "--history", str(history), "--output", str(predictions)], check=True)
    html, odds_source = fetch(f"https://sports.yahoo.co.jp/keiba/race/odds/tfw/{target['provider_id']}", raw / "odds.html")
    odds, asof = parse_win_snapshot(html, target)
    age = (datetime.now(UTC) - asof).total_seconds()
    if not 0 <= age <= 300:
        raise ValueError(f"Provider market timestamp stale/future: {age:.1f} seconds")
    keys = ["race_id", "horse_id", "horse_no", "horse_name"]
    merged = roster[keys].merge(odds, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("Roster and current market identities disagree; recapture before lock")
    win, overround = normalized_win_probabilities(merged.win_odds.to_numpy())
    market = merged[["race_id", "horse_id"]].assign(p_market=harville_top_k_probabilities(win))
    market_path = args.output / "market.csv"; market.to_csv(market_path, index=False)
    odds.to_csv(args.output / "win_odds.csv", index=False)
    meta = {"canonical_stage5_version": "stage5_v2_market_only", "snapshot_at": asof.isoformat(),
            "snapshot_sha256": digest(market_path), "conversion": "normalized win odds + Harville top3",
            "overround": overround, "sources": {"roster": roster_source, "odds": odds_source},
            "config_sha256": digest(args.config), "history_audit_sha256": digest(args.prepared / "history_manifest.json"),
            "mode": args.mode, "formal_lock": args.mode == "lock"}
    market_manifest = args.output / "market.manifest.json"
    market_manifest.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    if args.mode == "lock":
        subprocess.run([sys.executable, "scripts/run_scope_prospective.py", "lock", "--predictions", str(predictions),
                        "--prediction-manifest", str(predictions.with_suffix(".manifest.json")),
                        "--market", str(market_path), "--market-manifest", str(market_manifest),
                        "--off-at", target["off_at"], "--output", f"prospective/locks/{target['race_id']}"], check=True)
    print(json.dumps({"race_id": target["race_id"], "mode": args.mode, "starters": len(roster),
                      "snapshot_at": asof.isoformat(), "output": str(args.output)}))


if __name__ == "__main__":
    main()
