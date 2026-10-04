"""Create the live-history inventory from independently extracted official JRA reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from keiba_place_lab.live_history import EVENTS_PATH, apply_confirmed_events, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--through", required=True)
    parser.add_argument("--output", type=Path, default=Path("analysis/live_history_2026/official_race_ledger.csv"))
    args = parser.parse_args()
    through = pd.Timestamp(args.through)
    facts = pd.read_csv(args.conditions, dtype={"race_id": "string"})
    facts["actual_date"] = pd.to_datetime(facts.actual_date)
    facts = facts.loc[facts.actual_date.le(through)].copy()
    if facts.empty or facts.race_id.duplicated().any() or not facts.actual_date.dt.year.eq(through.year).all():
        raise ValueError("Invalid official report inventory")
    if not facts.race_kind.isin(["flat", "obstacle"]).all():
        raise ValueError("Unresolved flat/obstacle classification in official inventory")
    ledger = facts.rename(columns={"actual_date": "race_date", "race_number": "race_no"})
    ledger["provider_id"] = ledger.race_id.str[2:]
    ledger["meeting_day_id"] = ledger.race_id.str[2:10]
    ledger["target_flat"] = ledger.race_kind.eq("flat")
    ledger["schedule_source_url"] = ledger.official_source_url
    ledger["schedule_row_text"] = ledger.condition_excerpt.str.split("本賞", n=1).str[0]
    ledger = ledger.drop(columns=["condition_excerpt"])
    events = json.loads(EVENTS_PATH.read_text())
    # Reports may omit abandoned races entirely: retain their audited inventory entries.
    extra = []
    for event in events:
        date = pd.Timestamp(event["race_date"])
        if date > through:
            continue
        for rid in event["race_ids"]:
            if rid not in set(ledger.race_id.astype(str)):
                extra.append({"race_id": rid, "provider_id": rid[2:], "meeting_day_id": rid[2:10],
                              "race_no": int(rid[-2:]), "race_date": date,
                              "target_flat": event.get("race_kind", "flat") == "flat",
                              "schedule_source_url": event["source_url"], "schedule_row_text": event["reason"]})
    if extra:
        ledger = pd.concat([ledger, pd.DataFrame(extra)], ignore_index=True)
    ledger = apply_confirmed_events(ledger, events)
    ledger = ledger.sort_values(["race_date", "race_id"]).reset_index(drop=True)
    for mid, group in ledger.groupby("meeting_day_id"):
        if sorted(group.race_no.astype(int)) != list(range(1, 13)):
            raise ValueError(f"Official meeting-day race inventory incomplete: {mid}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(args.output, index=False)
    summary = {"through": through.date().isoformat(), "rows": len(ledger),
               "meeting_days": int(ledger.meeting_day_id.nunique()),
               "calendar_days": int(ledger.race_date.nunique()),
               "conditions_sha256": sha256_file(args.conditions), "ledger_sha256": sha256_file(args.output),
               "latest_race_date": ledger.race_date.max().date().isoformat()}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
