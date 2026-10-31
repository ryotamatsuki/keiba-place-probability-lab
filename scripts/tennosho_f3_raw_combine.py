"""Reassemble deterministic checkpointed 2026 source shards with complete identity audit.

Accept only matching independent, source-hashed, full-field shards. Rebuild the
single unsharded manifest expected by the frozen F3 prior-feature generator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble(
    history_path: Path, shard_root: Path, count: int,
    output: Path, manifest_path: Path,
) -> dict:
    if not 1 <= count <= 64:
        raise ValueError("Invalid checkpoint count")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Refuse to overwrite immutable assembled F3 source")
    history = pd.read_parquet(history_path)
    history["race_id"] = history.race_id.astype(str)
    history["horse_id"] = history.horse_id.astype(str)
    history["horse_no"] = pd.to_numeric(history.horse_no, errors="raise").astype(int)
    if pd.to_datetime(history.race_date).max() > pd.Timestamp("2026-10-30"):
        raise ValueError("History includes future target race")
    if history.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("2026 history identity duplicated")

    expected_ids = sorted(set(history.race_id))
    expected_sources = expected_ids
    history_sha = sha(history_path)
    frames, records, per_shard = [], [], []
    for idx in range(count):
        folder = shard_root / f"shard-{idx}"
        csv_path = folder / "2026_raw.csv"
        meta_path = folder / "2026_raw.manifest.json"
        if not csv_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"Missing complete independently saved shard {idx}")
        meta = json.loads(meta_path.read_text())
        if not (
            meta.get("status") == "COMPLETE_RAW_PRIOR_RESULT_SOURCE"
            and meta.get("shard_index") == idx
            and meta.get("shard_count") == count
            and meta.get("history_input_sha256") == history_sha
            and meta.get("source_csv_sha256") == sha(csv_path)
            and meta.get("cutoff") == "2026-10-30"
            and meta.get("no_target_race_fetch") is True
        ):
            raise ValueError(f"Shard {idx} provenance or content hash invalid")
        expected_subset = expected_sources[idx::count]
        source_records = meta.get("source_records", [])
        if len(source_records) != len(expected_subset) or sorted(
            str(r["race_id"]) for r in source_records
        ) != expected_subset:
            raise ValueError(f"Shard {idx} lacks exact source race records")
        for record in source_records:
            if len(str(record["source_sha256"])) != 64 or not str(
                record["source_url"]
            ).startswith(("https://sports.yahoo.co.jp/", "https://umanity.jp/")):
                raise ValueError(f"Missing independent hash / domain in shard {idx}")
        frame = pd.read_csv(csv_path, dtype={"race_id": str, "horse_id": str})
        if (
            set(frame.race_id) != set(expected_subset)
            or len(frame) != meta.get("rows")
            or len(expected_subset) != meta.get("race_count")
            or frame.duplicated(["race_id", "horse_id"]).any()
            or "source_provider" not in frame
        ):
            raise ValueError(f"Shard {idx} result count / identity invalid")
        frames.append(frame)
        records.extend(source_records)
        per_shard.append({
            "shard_index": idx, "races": len(expected_subset),
            "rows": len(frame), "csv_sha256": meta["source_csv_sha256"],
            "manifest_sha256": sha(meta_path),
        })

    if len(records) != len(expected_ids) or sorted(
        str(r["race_id"]) for r in records
    ) != expected_ids:
        raise ValueError("Duplicate or missing source race provenance")
    df = pd.concat(frames, ignore_index=True)
    df["horse_no"] = pd.to_numeric(df.horse_no, errors="raise").astype(int)
    if df.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Duplicate raw starter across checkpoints")
    exp = history[["race_id", "horse_id", "horse_no"]].sort_values(
        ["race_id", "horse_id"], kind="stable"
    ).reset_index(drop=True)
    actual = df[["race_id", "horse_id", "horse_no"]].sort_values(
        ["race_id", "horse_id"], kind="stable"
    ).reset_index(drop=True)
    if len(df) != len(history) or not exp.equals(actual):
        raise ValueError("Combined source fails exact every-starter match")
    if "202605041111" in set(df.race_id):
        raise ValueError("Forbidden target result leakage")
    if not df.source_provider.isin(("yahoo", "umanity")).all():
        raise ValueError("Source provider provenance absent")

    df = df.sort_values(["race_date", "race_id", "horse_no"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, lineterminator="\n")
    records.sort(key=lambda r: str(r["race_id"]))
    audit = {
        "status": "COMPLETE_RAW_PRIOR_RESULT_SOURCE",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "cutoff": "2026-10-30",
        "race_count": len(expected_ids),
        "rows": len(df),
        "source_csv_sha256": sha(output),
        "history_input_sha256": history_sha,
        "source_records": records,
        "no_target_race_fetch": True,
        "reconstruction": "independent_sha256_verified_sharded_2026_prior_results_v1",
        "verified_shards": per_shard,
    }
    manifest_path.write_text(json.dumps(
        audit, ensure_ascii=False, indent=2
    ) + "\n")
    return audit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--shards", type=Path, required=True)
    p.add_argument("--shard-count", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()
    audit = assemble(
        args.history, args.shards, args.shard_count, args.output, args.manifest
    )
    print(json.dumps({
        "status": audit["status"], "races": audit["race_count"],
        "rows": audit["rows"], "sha256": audit["source_csv_sha256"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
