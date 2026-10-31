"""Checkpointed F3 evidence must be complete and cryptographically reconciled."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tennosho_f3_raw_combine import assemble


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_shards(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    history = pd.DataFrame({
        "race_id": ["202601010101", "202601010101", "202601010102",
                    "202601010102", "202601010103", "202601010103"],
        "horse_id": ["h1", "h2", "h3", "h4", "h5", "h6"],
        "horse_no": [1, 2, 1, 2, 1, 2],
        "race_date": ["2026-01-01"] * 6,
    })
    original = tmp_path / "history.parquet"
    history.to_parquet(original, index=False)
    shard_root = tmp_path / "shards"
    races = sorted(history.race_id.unique())
    for idx in range(2):
        dest = shard_root / f"shard-{idx}"
        dest.mkdir(parents=True)
        ids = races[idx::2]
        frame = history[history.race_id.isin(ids)].copy()
        frame["source_provider"] = "yahoo"
        csv = dest / "2026_raw.csv"
        frame.to_csv(csv, index=False, lineterminator="\n")
        meta = {
            "status": "COMPLETE_RAW_PRIOR_RESULT_SOURCE",
            "shard_index": idx,
            "shard_count": 2,
            "history_input_sha256": checksum(original),
            "source_csv_sha256": checksum(csv),
            "cutoff": "2026-10-30",
            "no_target_race_fetch": True,
            "rows": len(frame),
            "race_count": len(ids),
            "source_records": [
                {"race_id": rid,
                 "source_url": "https://sports.yahoo.co.jp/keiba/race/result/2601010101",
                 "source_sha256": "a" * 64, "captured_at_utc": "2026-10-31T00:00:00Z"}
                for rid in ids
            ],
        }
        (dest / "2026_raw.manifest.json").write_text(json.dumps(meta))
    return original, shard_root, tmp_path / "out.csv", tmp_path / "out.manifest.json"


def test_complete_checkpoints_rebuild_exact_all_starters(tmp_path: Path):
    source, shards, output, manifest = fixture_shards(tmp_path)
    result = assemble(source, shards, 2, output, manifest)
    assert result["status"] == "COMPLETE_RAW_PRIOR_RESULT_SOURCE"
    assert result["race_count"] == 3
    assert result["rows"] == 6
    assert result["source_csv_sha256"] == checksum(output)
    assert len(result["verified_shards"]) == 2


def test_partial_checkpoint_is_rejected(tmp_path: Path):
    source, shards, output, manifest = fixture_shards(tmp_path)
    (shards / "shard-1" / "2026_raw.csv").unlink()
    with pytest.raises(FileNotFoundError, match="Missing complete"):
        assemble(source, shards, 2, output, manifest)


def test_tampered_checkpoint_is_rejected(tmp_path: Path):
    source, shards, output, manifest = fixture_shards(tmp_path)
    path = shards / "shard-1" / "2026_raw.csv"
    path.write_text(path.read_text().replace("h2", "forged_horse"))
    with pytest.raises(ValueError, match="provenance or content hash"):
        assemble(source, shards, 2, output, manifest)


def test_wrong_source_history_is_rejected(tmp_path: Path):
    source, shards, output, manifest = fixture_shards(tmp_path)
    old = pd.read_parquet(source)
    old.loc[0, "horse_id"] = "wrong"
    old.to_parquet(source, index=False)
    with pytest.raises(ValueError, match="provenance or content hash"):
        assemble(source, shards, 2, output, manifest)
