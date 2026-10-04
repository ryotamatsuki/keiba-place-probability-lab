"""Build monthly verified checkpoints through a specified live-history cutoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from keiba_place_lab.live_history import update_live_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", required=True)
    parser.add_argument("--root", type=Path, default=Path("data/live_history/2026"))
    parser.add_argument("--cache", type=Path, default=Path("data/raw/live_history_2026"))
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--report", type=Path, default=Path("analysis/live_history_2026/checkpoints.json"))
    args = parser.parse_args()
    through = pd.Timestamp(args.through).normalize()
    pointer = args.root / "current.json"
    if pointer.exists():
        start = pd.Timestamp(json.loads(pointer.read_text())["complete_through"]) + pd.Timedelta(days=1)
    else:
        start = pd.Timestamp(year=through.year, month=1, day=1)
    if start.year != through.year and start <= through:
        raise ValueError("Current snapshot belongs to another year")
    checkpoints = json.loads(args.report.read_text()) if args.report.exists() else []
    while start <= through:
        cutoff = min(start + pd.offsets.MonthEnd(0), through)
        manifest = update_live_history(
            year=through.year, through=cutoff, root=args.root, cache_dir=args.cache,
            recheck_days=14, max_workers=args.max_workers,
        )
        summary = {key: manifest[key] for key in (
            "complete_through", "latest_race_date", "expected_races", "confirmed_races",
            "missing_race_ids", "expected_race_days", "confirmed_race_days", "rows",
            "entry_rows", "snapshot_id", "history_sha256", "abandoned_race_ids",
        )}
        checkpoints.append(summary)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(checkpoints, ensure_ascii=False, indent=2) + "\n")
        print("MONTH_COMPLETE " + json.dumps(summary, ensure_ascii=False), flush=True)
        start = cutoff + pd.Timedelta(days=1)


if __name__ == "__main__":
    main()
