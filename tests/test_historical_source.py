import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_official_jra import LEGACY_HEADER, MODERN_HEADER, parse_conditions
from materialize_historical_training_dataset import parse_rank
from test_historical_panel import _synthetic_rows

from keiba_place_lab.historical_panel import build_historical_panel


def test_month_cannot_absorb_official_serial():
    modern = MODERN_HEADER.search("000031月5日晴良(2025年1中山)第1日第3競走")
    legacy = LEGACY_HEADER.search("000111月5日晴良(22中山1)第1日第11競走")
    assert modern["month"] == "1"
    assert legacy["month"] == "1"


def test_official_obstacle_even_when_surface_is_turf():
    c = parse_conditions("2,880サラブレッド系障害4歳以上(芝・右外→内)未勝利;負担重量は、定量")
    assert c["race_kind"] == "obstacle"
    assert c["official_distance_m"] == 2880


def test_layout_and_handicap_unknown_stay_missing():
    c = parse_conditions("1,200 3歳未勝利(芝・右)負担重量は、馬齢重量")
    assert c["course_layout"] is None
    assert c["handicap_indicator"] == 0
    assert parse_conditions("1,200(芝・右外)ハンデ")["handicap_indicator"] is None


def test_official_weight_clause_handicap():
    assert (
        parse_conditions("1,200(芝・右内)オープン;負担重量は、ハンデキャップ")["handicap_indicator"]
        == 1
    )


@pytest.mark.parametrize(
    "rank,started,status",
    [
        ("取", False, "nonstarter"),
        ("除", False, "nonstarter"),
        ("中", True, "dnf"),
        ("失", True, "disqualified"),
    ],
)
def test_rank_status(rank, started, status):
    value, result, flag = parse_rank(rank)
    assert np.isnan(value)
    assert flag == started
    assert result == status


def test_demoted_and_rerun_ranks():
    assert parse_rank("2(降)") == (2.0, "finished", True)
    assert parse_rank("12(再)") == (12.0, "finished", True)


def test_target_outcome_permutation_does_not_change_target_predictors():
    rows = _synthetic_rows()
    original = build_historical_panel(rows)
    mask = rows.race_id.eq("r2")
    rows.loc[mask, "finish_position"] = rows.loc[mask, "finish_position"].to_numpy()[::-1]
    rows.loc[mask, "race_time_seconds"] = 999
    rows.loc[mask, "early_position"] = 1
    changed = build_historical_panel(rows)
    predictors = [
        "career_starts",
        "career_top3",
        "recent3_top3_count",
        "recent3_finish_pct_mean",
        "recent3_relative_time_mean",
        "recent4_early_pos_pct_mean",
        "front_forward_share",
    ]
    pd.testing.assert_frame_equal(original.loc[mask, predictors], changed.loc[mask, predictors])
