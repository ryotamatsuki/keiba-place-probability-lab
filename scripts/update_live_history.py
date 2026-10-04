"""Build or reconcile an immutable year-to-date JRA flat live-history snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from keiba_place_lab.live_history import update_live_history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--through", required=True, help="Inclusive YYYY-MM-DD cutoff")
    parser.add_argument("--root", type=Path, default=Path("data/live_history/2026"))
    parser.add_argument("--cache", type=Path, default=Path("data/raw/live_history_2026"))
    parser.add_argument("--recheck-days", type=int, default=14)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--source", choices=("yahoo", "umanity"), default="yahoo")
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()
    manifest = update_live_history(
        year=args.year,
        through=pd.Timestamp(args.through),
        root=args.root,
        cache_dir=args.cache,
        recheck_days=args.recheck_days,
        max_workers=args.max_workers,
        source=args.source,
        ledger_path=args.ledger,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
