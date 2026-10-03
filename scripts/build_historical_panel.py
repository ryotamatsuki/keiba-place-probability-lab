"""Build a local historical feature panel from standardized race rows.

Raw and processed historical row-level data are intentionally git-ignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from keiba_place_lab.historical_panel import (
    assert_strict_history,
    build_historical_panel,
    select_phase_a_cohort,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/historical_raw/standardized_race_rows.csv.gz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/historical_processed/phase_a_panel.csv.gz"),
    )
    parser.add_argument("--surface", default="turf")
    parser.add_argument("--min-prior-starts", type=int, default=3)
    args = parser.parse_args()

    rows = pd.read_csv(args.input)
    panel = build_historical_panel(rows)
    assert_strict_history(panel)
    cohort = select_phase_a_cohort(
        panel,
        surface=args.surface,
        min_prior_starts=args.min_prior_starts,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output, index=False, compression="gzip")

    race_count = cohort["race_id"].nunique()
    print(f"rows={len(cohort)} races={race_count}")
    print(cohort["temporal_split"].value_counts(dropna=False).sort_index().to_string())


if __name__ == "__main__":
    main()
