"""The 2010-2025 frozen JRA prefix must survive all 2026 joins."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from keiba_place_lab import information_expansion_v1 as raw_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tennosho_f3_prior_builder import _jra_external_id, _jra_internal_id


def test_normalize_2025_frozen_and_2026_public_horse_to_same_identity():
    frozen = "jra:2023105344"
    public = "2023105344"
    assert _jra_internal_id(frozen) == _jra_internal_id(public)
    assert _jra_external_id(frozen) == public
    assert _jra_external_id(public) == public


@pytest.mark.parametrize("value", ["", "not-a-horse", "jra:123", "202605041111", None])
def test_invalid_jra_identity_refused(value):
    with pytest.raises(ValueError, match="Invalid verified JRA horse identity"):
        _jra_internal_id(value)


def test_audited_kaggle_raw_ids_match_frozen_prefixed_identity(monkeypatch):
    race_id = "202506050701"
    starters = pd.DataFrame({
        "race_id": [race_id, race_id],
        "horse_id": ["jra:2023101860", "jra:2023100283"],
        "horse_no": [10, 14],
        "top3_label": [True, True],
        "field_size": [15, 15],
    })
    kaggle = pd.DataFrame({
        "race_id": [race_id, race_id],
        "horse_id": ["2023101860", "2023100283"],
        "rank": ["1", "2"],
        "number": ["10", "14"],
        "jockey_name": ["A", "B"],
        "trainer": ["C", "D"],
        "last_3f": ["36.0", "36.4"],
        "passing": ["5-4", "3-2"],
        "horse_weight": ["500(+2)", "480(-2)"],
    })
    race = pd.DataFrame({"race_id": [race_id], "track_condition": ["良"]})

    def fake_read(path, _cols, _ids):
        return kaggle.copy() if path.name == "results.csv" else race.copy()

    monkeypatch.setattr(raw_module, "_read_csv_select", fake_read)
    enriched, audit = raw_module.raw_observations(
        starters, Path("results.csv"), Path("races.csv"),
        horse_id_normalizer=_jra_internal_id,
    )
    assert audit["matched_raw_runner_rows"] == 2
    assert enriched.raw_last3f.notna().all()
    assert enriched.horse_id.tolist() == starters.horse_id.tolist()
