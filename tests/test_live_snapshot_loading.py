import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_stage4_scope_v3 import _load_live_snapshot

from keiba_place_lab.live_history import sha256_file


def test_prediction_rejects_legacy_time_parser_snapshot(tmp_path):
    snapshot = tmp_path / "snapshots" / "old"
    snapshot.mkdir(parents=True)
    pd.DataFrame({"race_id": ["202606010101"]}).to_parquet(snapshot / "jra_flat_history.parquet")
    (snapshot / "manifest.json").write_text(json.dumps({"snapshot_id": "old"}))
    (tmp_path / "current.json").write_text(json.dumps({"snapshot_id": "old"}))
    with pytest.raises(ValueError, match="corrected time-parser reparse"):
        _load_live_snapshot(tmp_path)


def test_prediction_rejects_changed_corrected_snapshot(tmp_path):
    snapshot = tmp_path / "snapshots" / "corrected"
    snapshot.mkdir(parents=True)
    path = snapshot / "jra_flat_history.parquet"
    pd.DataFrame({"race_id": ["202606010101"]}).to_parquet(path)
    manifest = {"snapshot_id": "corrected", "schema_version": "v1",
                "source_parser_contract": "time_margin_separated_v1", "history_sha256": sha256_file(path)}
    (snapshot / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "current.json").write_text(json.dumps(manifest))
    assert len(_load_live_snapshot(tmp_path)[0]) == 1
    pd.DataFrame({"race_id": ["changed"]}).to_parquet(path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        _load_live_snapshot(tmp_path)
