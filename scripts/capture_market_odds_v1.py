"""Import licensed market snapshots without inventing provider timestamps.

Examples:
  python scripts/capture_market_odds_v1.py ingest --source-csv my_export.csv \\
    --output-root data/raw/market_odds_v1
  python scripts/capture_market_odds_v1.py select --ledger data/raw/market_odds_v1/race.csv \\
    --roster roster.csv --minutes-before 10 --output data/raw/market_odds_v1/lock.csv

No scraping, credential management or synthetic final-odds timestamps here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from keiba_place_lab.probability_improvement_v1 import (
    SNAPSHOT_COLUMNS,
    select_snapshot_at,
    validate_snapshot_rows,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ingest(source: Path, output_root: Path) -> Path:
    data = validate_snapshot_rows(pd.read_csv(source, dtype={"race_id": str, "horse_id": str}))
    if data["race_id"].nunique() != 1:
        raise ValueError("Ingest one race per immutable ledger")
    race_id = str(data.race_id.iloc[0])
    if not race_id.replace("_", "").isalnum():
        raise ValueError("Unsafe race_id")
    out = output_root / f"{race_id}.csv"
    if out.exists():
        existing = validate_snapshot_rows(
            pd.read_csv(out, dtype={"race_id": str, "horse_id": str})
        )
        old_keys = set(map(tuple, existing[
            ["race_id", "horse_id", "bet_type", "snapshot_id"]].to_numpy()))
        new_keys = set(map(tuple, data[
            ["race_id", "horse_id", "bet_type", "snapshot_id"]].to_numpy()))
        if old_keys & new_keys:
            raise ValueError("Append-only ledger rejects repeated snapshots")
        data = pd.concat([existing, data], ignore_index=True)
        validate_snapshot_rows(data)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(".new")
    data.to_csv(temp, index=False, columns=list(SNAPSHOT_COLUMNS),
                date_format="%Y-%m-%dT%H:%M:%S%z")
    temp.replace(out)
    manifest = {"schema": "market_odds_v1", "race_id": race_id,
                "source_file_sha256": sha(source), "ledger_sha256": sha(out),
                "rows": len(data), "snapshot_ids": sorted(data.snapshot_id.unique().tolist()),
                "provider_license": "operator-attested: verify before reuse",
                "data_not_redistributed": True}
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2)+"\n")
    return out


def select(ledger: Path, roster: Path, minutes_before: int, output: Path) -> None:
    data = pd.read_csv(ledger, dtype={"race_id": str, "horse_id": str})
    active = pd.read_csv(roster, dtype={"race_id": str, "horse_id": str})
    if active.race_id.nunique() != 1 or set(active.race_id) != set(data.race_id):
        raise ValueError("Roster and market race ID disagree")
    if active.horse_id.duplicated().any():
        raise ValueError("Duplicate active roster horse ID")
    chosen = select_snapshot_at(
        data, minutes_before=minutes_before,
        expected_horse_ids=set(active.horse_id.astype(str)),
    )
    if output.exists():
        raise FileExistsError("Immutable checkpoint output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(output, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
    result = {"schema": "market_odds_v1", "checkpoint_minutes_before": minutes_before,
              "selection_source_sha256": sha(ledger), "snapshot_sha256": sha(output),
              "snapshot_ids": sorted(chosen.snapshot_id.unique().tolist()),
              "validation": "captured/provider <= checkpoint, age <= 5/10 minutes"}
    output.with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    i = sub.add_parser("ingest")
    i.add_argument("--source-csv", type=Path, required=True)
    i.add_argument("--output-root", type=Path, default=Path("data/raw/market_odds_v1"))
    s = sub.add_parser("select")
    s.add_argument("--ledger", type=Path, required=True)
    s.add_argument("--roster", type=Path, required=True)
    s.add_argument("--minutes-before", type=int, choices=(60, 30, 10), required=True)
    s.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.command == "ingest":
        print(ingest(a.source_csv, a.output_root))
    else:
        select(a.ledger, a.roster, a.minutes_before, a.output)


if __name__ == "__main__":
    main()
