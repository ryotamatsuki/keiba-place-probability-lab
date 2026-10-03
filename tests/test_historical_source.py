import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_official_jra import parse_conditions, parse_race_days
from materialize_historical_training_dataset import parse_rank
from test_historical_panel import _synthetic_rows

from keiba_place_lab.historical_panel import build_historical_panel


def test_daily_date_parser_rejects_serial_contaminated_invalid_date():
    info = {
        "year": 2014,
        "venue_code": "05",
        "venue": "東京",
        "meeting_no": 1,
        "day_no": 2,
        "url": "https://example.invalid/2014-1tokyo2.pdf",
        "legacy": False,
    }
    rows = parse_race_days(
        info,
        "0030211月31日晴良(26東京1)第2日第9競走",
    )
    assert rows[0]["actual_date"] == "2014-01-31"


def test_daily_date_parser_uses_five_digit_header_serial_for_january():
    info = {
        "year": 2011,
        "venue_code": "08",
        "venue": "京都",
        "meeting_no": 1,
        "day_no": 1,
        "url": "https://example.invalid/2011-1kyoto1.pdf",
        "legacy": False,
    }
    raw = "05001 1月5日 晴 良 (23京都1) 第1日 第1競走"
    rows = parse_race_days(
        info,
        "050011月5日晴良(23京都1)第1日第1競走",
        raw_text=raw,
    )
    assert rows[0]["actual_date"] == "2011-01-05"


def test_daily_date_parser_uses_five_digit_header_serial_for_november():
    info = {
        "year": 2016,
        "venue_code": "08",
        "venue": "京都",
        "meeting_no": 5,
        "day_no": 1,
        "url": "https://example.invalid/2016-5kyoto1.pdf",
        "legacy": False,
    }
    raw = "30001 11月5日 晴 良 (28京都5) 第1日 第1競走"
    rows = parse_race_days(
        info,
        "3000111月5日晴良(28京都5)第1日第1競走",
        raw_text=raw,
    )
    assert rows[0]["actual_date"] == "2016-11-05"


def test_daily_date_parser_prefers_two_digit_november_over_overlap_alias():
    info = {
        "year": 2017,
        "venue_code": "08",
        "venue": "京都",
        "meeting_no": 5,
        "day_no": 8,
        "url": "https://example.invalid/2017-5kyoto8.pdf",
        "legacy": False,
    }
    rows = parse_race_days(
        info,
        "08084 11月26日晴良(29京都5)第8日第4競走",
    )
    assert rows[0]["actual_date"] == "2017-11-26"


def test_daily_date_parser_prefers_two_digit_december_over_overlap_alias():
    info = {
        "year": 2010,
        "venue_code": "09",
        "venue": "阪神",
        "meeting_no": 5,
        "day_no": 8,
        "url": "https://example.invalid/2010-5hanshin8.pdf",
        "legacy": False,
    }
    rows = parse_race_days(
        info,
        "09084 12月26日晴良(22阪神5)第8日第12競走",
    )
    assert rows[0]["actual_date"] == "2010-12-26"


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
