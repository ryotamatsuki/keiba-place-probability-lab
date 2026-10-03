"""Probe the legacy JRA Kaggle dataset for actual race-date mapping."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.request import Request, urlopen

import kagglehub

DATASET = "takamotoki/jra-horse-racing-dataset"
OUT = Path("docs/HISTORICAL_SECONDARY_SOURCE_PROBE.md")


def metadata() -> dict:
    url = f"https://www.kaggle.com/api/v1/datasets/view/{DATASET}"
    req = Request(url, headers={"User-Agent": "keiba-place-probability-lab/0.1"})
    with urlopen(req, timeout=60) as r:
        return json.load(r)


def main() -> None:
    root = Path(
        kagglehub.dataset_download(
            DATASET,
            output_dir="data/historical_raw/secondary_probe",
            force_download=True,
        )
    )
    if not root.is_dir():
        root = root.parent

    meta = metadata()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    lines = [
        "# Historical Secondary Source Probe",
        "",
        f"Dataset: {DATASET}",
        "",
        "## Metadata",
        "",
        "~~~json",
        json.dumps(
            {k: meta.get(k) for k in [
                "title", "ref", "lastUpdated", "licenseName", "totalBytes", "versionNumber"
            ] if k in meta},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        "~~~",
        "",
        "## Files",
        "",
        "| path | bytes | first columns |",
        "|---|---:|---|",
    ]
    for p in files:
        rel = p.relative_to(root).as_posix()
        cols = ""
        if p.suffix.lower() == ".csv":
            try:
                with p.open("r", encoding="utf-8-sig", newline="") as f:
                    row = next(csv.reader(f))
                cols = " | ".join(row[:20]).replace("|", "\\|")
            except Exception:  # noqa: BLE001
                try:
                    with p.open("r", encoding="cp932", newline="") as f:
                        row = next(csv.reader(f))
                    cols = " | ".join(row[:20]).replace("|", "\\|")
                except Exception as exc:  # noqa: BLE001
                    cols = repr(exc).replace("|", "\\|")
        lines.append(f"| {rel} | {p.stat().st_size} | {cols} |")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
