"""Optional local acquisition helper for the Stage 3.6 Kaggle candidates.

This script downloads data only to git-ignored local directories. Before building
or redistributing any derived panel, verify and record the license shown by Kaggle
for the exact dataset version downloaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PRIMARY = "noriyukifurufuru/japan-horse-racing-2010-2025"
SECONDARY = "takamotoki/jra-horse-racing-dataset"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["primary", "secondary"],
        default="primary",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/historical_raw"),
    )
    args = parser.parse_args()

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit(
            "Install the optional historical dependency first: "
            "pip install -e '.[historical]'"
        ) from exc

    slug = PRIMARY if args.dataset == "primary" else SECONDARY
    destination = args.output_root / slug.replace("/", "__")
    destination.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(slug, path=str(destination), unzip=True)

    marker = destination / "LICENSE_VERIFICATION_REQUIRED.txt"
    marker.write_text(
        "Before processing or redistributing this dataset, verify the exact "
        "license and version on the Kaggle dataset page and record it in the "
        "project source manifest.\n",
        encoding="utf-8",
    )

    print(f"downloaded={slug}")
    print(f"path={destination}")
    print("license_verification=REQUIRED")


if __name__ == "__main__":
    main()
