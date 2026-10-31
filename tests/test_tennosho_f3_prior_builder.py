"""F3 postrace source alignment: future Elo must include verified 2026 results."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tennosho_f3_prior_builder import _canonicalize_extra_agents, _extra_raw_observations


def _sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    race = "202605040211"
    horse_ids = ["2020100680", "2022104416", "2020103075"]
    extra = pd.DataFrame({
        "race_id": [race] * 3,
        "horse_id": horse_ids,
        "horse_no": [1, 2, 3],
        "rank": ["1", "2", "中止"],
        "last_3f": ["33.2", "34.0", None],
        "passing": ["1-1", "2-2", None],
        "horse_weight": ["500(+2)", "460(-4)", None],
        "track_condition": ["良"] * 3,
        "jockey_name": ["騎手A", "騎手B", "騎手C"],
        "trainer": ["調教師A", "調教師B", "調教師C"],
    })
    raw = pd.DataFrame({
        "race_id": [race] * 3, "horse_id": horse_ids, "horse_no": [1, 2, 3],
        "field_size": [3] * 3, "finish_position": [1, 2, np.nan],
        "raw_finish_position": [np.nan] * 3,
    })
    for name in (
        "raw_last3f", "raw_corner_first", "raw_corner_last", "raw_early_pct",
        "raw_passing_gain", "raw_mass", "raw_mass_change", "raw_wet",
    ):
        raw[name] = np.nan
    return raw, extra


def test_new_2026_ranks_update_elo_and_dnf_remains_missing():
    raw, extra = _sources()
    combined, audit = _extra_raw_observations(raw, extra)
    assert combined.raw_finish_position.iloc[:2].tolist() == [1, 2]
    assert np.isnan(combined.raw_finish_position.iloc[2])
    assert combined.raw_last3f.tolist()[:2] == [33.2, 34.0]
    assert combined.raw_wet.eq(0).all()
    assert combined.raw_last3f_rel.iloc[0] == pytest.approx(1.0)
    assert audit["extra_source_rows"] == 3


def test_new_2026_rank_must_match_confirmed_official_result():
    raw, extra = _sources()
    extra.loc[1, "rank"] = "3"
    with pytest.raises(ValueError, match="finishing ranks disagree"):
        _extra_raw_observations(raw, extra)


def test_umanity_abbreviations_resolve_to_real_2026_yahoo_full_names():
    history = pd.DataFrame({
        "jockey_name": ["岩田 康誠", "横山 典弘", "C.ルメール", "幸 英明"],
        "trainer": ["奥村 武", "橋口 慎介", "昆 貢", "森 一誠"],
    })
    published = pd.DataFrame({
        "source_provider": ["yahoo", "umanity", "umanity", "umanity"],
        "jockey_name": ["岩田 康誠", "岩田康", "Ｃルメ", "幸英明"],
        "trainer": ["奥村 武", "奥村武", "橋口慎", "森一誠"],
    })
    normalized, audit = _canonicalize_extra_agents(history, published)
    assert normalized.jockey_name.tolist() == [
        "岩田 康誠", "岩田 康誠", "C.ルメール", "幸 英明",
    ]
    assert normalized.trainer.tolist() == [
        "奥村 武", "奥村 武", "橋口 慎介", "森 一誠",
    ]
    assert audit["jockey_name"]["alternate_distinct_aliases_resolved"] == 3


def test_abbreviated_agents_are_rejected_when_homonyms_ambiguous():
    history = pd.DataFrame({
        "jockey_name": ["佐々木 大輔", "佐々木 竹見"],
        "trainer": ["橋口 慎介", "昆 貢"],
    })
    shortened = pd.DataFrame({
        "source_provider": ["umanity"],
        "jockey_name": ["佐々木"],
        "trainer": ["橋口慎"],
    })
    with pytest.raises(ValueError, match="Ambiguous or unknown abbreviated jockey_name"):
        _canonicalize_extra_agents(history, shortened)
