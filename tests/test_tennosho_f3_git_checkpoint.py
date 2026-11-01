"""Frozen, source-hashed partial live-history checkpoint load tests."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tennosho_f3_prior_builder import _load_f3_git_checkpoint


def history_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "live"
    sid = "jra_flat_live_history_v1-20261030-test123"
    folder = root / "snapshots" / sid
    folder.mkdir(parents=True)
    source = folder / "jra_flat_history.parquet"
    pd.DataFrame({
        "race_id": ["202601010101", "202601010101"],
        "horse_id": ["h1", "h2"],
        "race_date": ["2026-01-01", "2026-01-01"],
    }).to_parquet(source, index=False)
    (root / "current.json").write_text(json.dumps({
        "schema_version": "jra_flat_live_history_v1",
        "snapshot_id": sid,
        "complete_through": "2026-10-30",
        "history_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }))
    return root, source


def test_git_checkpoint_loads_only_sha_verified_existing_history(tmp_path: Path):
    root, _ = history_root(tmp_path)
    current, frame = _load_f3_git_checkpoint(root)
    assert current["complete_through"] == "2026-10-30"
    assert frame.horse_id.tolist() == ["h1", "h2"]


def test_git_checkpoint_hash_tamper_refused(tmp_path: Path):
    root, source = history_root(tmp_path)
    source.write_bytes(source.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        _load_f3_git_checkpoint(root)


def test_git_checkpoint_target_result_refused(tmp_path: Path):
    root, source = history_root(tmp_path)
    bad = pd.read_parquet(source)
    bad.loc[0, "race_id"] = "202605041111"
    bad.loc[0, "race_date"] = "2026-11-01"
    bad.to_parquet(source, index=False)
    pointer = root / "current.json"
    meta = json.loads(pointer.read_text())
    meta["history_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    pointer.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Future, target"):
        _load_f3_git_checkpoint(root)
