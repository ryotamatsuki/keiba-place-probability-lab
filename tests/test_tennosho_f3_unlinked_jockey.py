"""Independent source-verified rides, not name similarity, rescue a missing profile ID."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tennosho_f3_prior_builder import _verify_unlinked_umanity_jockey_rides


def _two_published_rides():
    rows = pd.DataFrame([
        {"race_id": "202601020103", "horse_id": "jra:2023100001",
         "race_date": "2026-08-22", "horse_no": 9, "jockey_name": "Ｓナレ"},
        {"race_id": "202601020212", "horse_id": "jra:2023100002",
         "race_date": "2026-08-23", "horse_no": 7, "jockey_name": "Ｓナレ"},
    ])
    original = pd.DataFrame([
        {"race_id": rows.race_id.iloc[0], "horse_id": rows.horse_id.iloc[0],
         "horse_name": "モンロワイヤル"},
        {"race_id": rows.race_id.iloc[1], "horse_id": rows.horse_id.iloc[1],
         "horse_name": "ベネスピラ"},
    ])
    return rows, original


def test_unlinked_jockey_requires_two_specific_dated_horses():
    rides, base = _two_published_rides()
    audit = _verify_unlinked_umanity_jockey_rides("Ｓナレ", rides, base)
    assert audit["verified_full_name"] == "Ｓ．ナレドゥ"
    assert audit["matched_historical_ride_count"] == 2
    assert audit["profile_link_unusable"] is True
    assert audit["published_identity_url"].startswith("https://www.jra.go.jp/")


def test_unlinked_jockey_refuses_unknown_horse_in_correct_race():
    rides, base = _two_published_rides()
    base.loc[1, "horse_name"] = "別の馬"
    with pytest.raises(ValueError, match="published ride mismatch"):
        _verify_unlinked_umanity_jockey_rides("Ｓナレ", rides, base)


def test_unlinked_jockey_refuses_one_day_only():
    rides, base = _two_published_rides()
    with pytest.raises(ValueError, match="Insufficient dated"):
        _verify_unlinked_umanity_jockey_rides("Ｓナレ", rides.iloc[:1], base)
