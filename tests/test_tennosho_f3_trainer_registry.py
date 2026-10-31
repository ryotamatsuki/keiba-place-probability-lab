"""Abridged 2026 trainer names are reconciled only through dated registries."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tennosho_f3_prior_builder import (
    _VERIFIED_2026_TRAINERS,
    _audited_trainer_identity,
    _canonicalize_extra_agents,
)


@pytest.mark.parametrize("alias", sorted(_VERIFIED_2026_TRAINERS))
def test_registry_alias_has_documented_full_name_and_cutoff(alias: str):
    full, active_since, url = _VERIFIED_2026_TRAINERS[alias]
    name, record = _audited_trainer_identity(alias, "2026-10-30")
    assert name == full
    assert record["publication"] == url
    assert record["active_from"] == active_since
    assert url.startswith("https://")


@pytest.mark.parametrize("alias", ["三樹祐", "古賀大", "平岩大"])
def test_new_trainer_cannot_be_used_before_registered_opening(alias: str):
    with pytest.raises(ValueError, match="start/cutoff"):
        _audited_trainer_identity(alias, "2026-01-01")


def test_registry_resolves_unmatched_current_trainer_with_provenance():
    original = pd.DataFrame({
        "race_id": ["202601010101"],
        "horse_id": ["jra:2023100001"],
        "jockey_name": ["既存騎手"],
        "trainer": ["他の調教師"],
    })
    extra = pd.DataFrame({
        "race_id": ["202601010102"],
        "horse_id": ["jra:2023100002"],
        "jockey_name": ["既存騎手"],
        "trainer": ["三樹祐"],
        "race_date": ["2026-10-29"],
        "source_provider": ["umanity"],
    })
    normalized, audit = _canonicalize_extra_agents(original, extra)
    assert normalized.trainer.iloc[0] == "三樹祐輔"
    assert len(audit["trainer"]["independently_published_trainer_identities"]) == 1


def test_trainer_alias_without_registry_still_rejected():
    original = pd.DataFrame({
        "race_id": ["202601010101"],
        "horse_id": ["jra:2023100001"],
        "jockey_name": ["騎手A"],
        "trainer": ["調教師B"],
    })
    extra = pd.DataFrame({
        "race_id": ["202601010102"],
        "horse_id": ["jra:2023100002"],
        "jockey_name": ["騎手A"],
        "trainer": ["未知略称"],
        "race_date": ["2026-10-29"],
        "source_provider": ["umanity"],
    })
    with pytest.raises(ValueError, match="Ambiguous or unknown abbreviated trainer"):
        _canonicalize_extra_agents(original, extra)
