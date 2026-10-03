"""Probe the primary public historical dataset in an ephemeral CI runner.

This script records metadata, filenames, schemas and coarse row counts only.
It does not commit raw row-level data.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

DATASET = "noriyukifurufuru/japan-horse-racing-2010-2025"
OUT_MD = Path("docs/HISTORICAL_SOURCE_PROBE.md")
OUT_CSV = Path("docs/HISTORICAL_SOURCE_FILES.csv")


def fetch_kaggle_metadata() -> dict:
    url = f"https://www.kaggle.com/api/v1/datasets/view/{DATASET}"
    req = Request(url, headers={"User-Agent": "keiba-place-probability-lab/0.1"})
    try:
        with urlopen(req, timeout=60) as r:
            return json.load(r)
    except Exception as exc:  # noqa: BLE001
        return {"_metadata_error": repr(exc)}


def detect_csv(path: Path) -> tuple[str, str]:
    encodings = ("utf-8-sig", "utf-8", "cp932", "shift_jis")
    seps = (",", "\t")
    last_exc: Exception | None = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, nrows=5, low_memory=False)
                if len(df.columns) >= 2:
                    return enc, sep
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
    raise RuntimeError(f"could not parse {path}: {last_exc!r}")


def inspect_file(path: Path, root: Path) -> dict:
    rel = path.relative_to(root).as_posix()
    item = {
        "path": rel,
        "bytes": path.stat().st_size,
        "kind": path.suffix.lower().lstrip("."),
        "encoding": "",
        "separator": "",
        "columns": "",
        "sample_rows_read": 0,
        "row_count": "",
        "error": "",
    }
    try:
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv", ".txt"}:
            enc, sep = detect_csv(path)
            sample = pd.read_csv(path, encoding=enc, sep=sep, nrows=5, low_memory=False)
            item["encoding"] = enc
            item["separator"] = "TAB" if sep == "\t" else sep
            item["columns"] = " | ".join(map(str, sample.columns))
            item["sample_rows_read"] = len(sample)
            with open(path, "rb") as fh:
                item["row_count"] = max(sum(1 for _ in fh) - 1, 0)
        elif suffix in {".parquet", ".pq"}:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            item["row_count"] = pf.metadata.num_rows
            item["columns"] = " | ".join(pf.schema.names)
        elif suffix == ".json":
            item["columns"] = "(json metadata)"
    except Exception as exc:  # noqa: BLE001
        item["error"] = repr(exc)
    return item


def main() -> None:
    import kagglehub

    cache_root = Path("data/historical_raw/probe")
    cache_root.mkdir(parents=True, exist_ok=True)

    downloaded = Path(
        kagglehub.dataset_download(
            DATASET,
            output_dir=str(cache_root),
            force_download=True,
        )
    )
    root = downloaded if downloaded.is_dir() else downloaded.parent

    metadata = fetch_kaggle_metadata()
    files = [p for p in root.rglob("*") if p.is_file()]
    inspected = [inspect_file(p, root) for p in sorted(files)]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = list(inspected[0].keys()) if inspected else [
        "path", "bytes", "kind", "encoding", "separator", "columns",
        "sample_rows_read", "row_count", "error"
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(inspected)

    selected_meta = {
        k: metadata.get(k)
        for k in (
            "title",
            "subtitle",
            "id",
            "ref",
            "lastUpdated",
            "versionNumber",
            "licenseName",
            "totalBytes",
            "usabilityRating",
        )
        if k in metadata
    }
    if "_metadata_error" in metadata:
        selected_meta["_metadata_error"] = metadata["_metadata_error"]

    total_bytes = sum(p.stat().st_size for p in files)
    report = [
        "# Historical Source Probe — primary Kaggle candidate",
        "",
        f"Dataset: {DATASET}",
        "",
        "## Kaggle metadata",
        "",
        "~~~json",
        json.dumps(selected_meta, ensure_ascii=False, indent=2, default=str),
        "~~~",
        "",
        "## Download result",
        "",
        f"- files: {len(files)}",
        f"- total downloaded bytes: {total_bytes}",
        "- local CI path: ephemeral / not committed",
        "",
        "## File schemas",
        "",
        "| file | bytes | rows | encoding | columns | error |",
        "|---|---:|---:|---|---|---|",
    ]
    for x in inspected:
        cols = str(x["columns"]).replace("|", "\\|")
        err = str(x["error"]).replace("|", "\\|")
        report.append(
            f"| {x['path']} | {x['bytes']} | {x['row_count']} | "
            f"{x['encoding']} | {cols} | {err} |"
        )

    report.extend(
        [
            "",
            "## Safety",
            "",
            "- Raw files were downloaded only into the ephemeral Actions runner.",
            "- No raw row-level source file is committed by this probe.",
            "- The exact license field above must be verified before any row-level derived data is published.",
        ]
    )
    OUT_MD.write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
