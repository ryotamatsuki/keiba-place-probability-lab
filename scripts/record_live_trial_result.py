"""Reconcile provider results with an explicit official JRA result URL, without scores."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
import stage36_official_supplement as official

from keiba_place_lab.live_public import VENUES, parse_results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--race-id", required=True)
    p.add_argument("--official-url", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not args.official_url.startswith("https://www.jra.go.jp/JRADB/accessS.html?"):
        raise ValueError("Use a verified official JRA result URL")
    config = json.loads(Path("config/live_trial_20261004.json").read_text())
    target = next(t for t in config["targets"] if t["race_id"] == args.race_id)
    if datetime.now(UTC) <= datetime.fromisoformat(target["off_at"]):
        raise ValueError("Race not yet off")
    official.VENUE_CODE_TO_JP.update(VENUES)
    _, rows, audit = official._parse_result_page(args.race_id, args.official_url)
    verified = pd.DataFrame(rows).rename(columns={"official_horse_id": "horse_id"})
    provider_url = f"https://sports.yahoo.co.jp/keiba/race/result/{target['provider_id']}"
    response = requests.get(provider_url, timeout=(10, 30)); response.raise_for_status()
    provider = parse_results(response.text, target["provider_id"])
    a = verified.set_index("horse_id").finish_position.sort_index()
    b = provider.set_index("horse_id").finish_position.sort_index()
    if not a.index.equals(b.index) or not a.fillna(-1).eq(b.fillna(-1)).all():
        raise ValueError("Official/provider results disagree; leave outcomes pending")
    lock_dir = Path(f"prospective/locks/{args.race_id}")
    events = []
    if lock_dir.exists():
        locked = pd.read_csv(lock_dir / "predictions.csv", dtype={"horse_id": "string"})
        if set(locked.horse_id) != set(verified.horse_id):
            events.append({"race_id": args.race_id, "event_at": datetime.now(UTC).isoformat(),
                           "reason": "late_cancellation"})
    args.output.mkdir(parents=True, exist_ok=False)
    verified[["race_id", "horse_id"]].assign(top3_label=verified.finish_position.between(1, 3).astype(int)).to_csv(args.output / "official_outcomes.csv", index=False)
    pd.DataFrame(events, columns=["race_id", "event_at", "reason"]).to_csv(args.output / "events.csv", index=False)
    (args.output / "provenance.json").write_text(json.dumps({"official": audit, "provider_url": provider_url,
        "verified_at": datetime.now(UTC).isoformat(), "reconciliation": "all starter IDs and finishing ranks matched",
        "event_time_semantics": "first_observed_at",
        "interim_scores_computed": False, "stage5_canonical": "market-only"}, indent=2, ensure_ascii=False))
    print(json.dumps({"race_id": args.race_id, "official_starters": len(verified), "exclusions": len(events)}))


if __name__ == "__main__":
    main()
