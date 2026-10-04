import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_official_jra import parse_conditions, parse_race_days
from materialize_historical_training_dataset import parse_rank
from stage36_official_supplement import _parse_rank as parse_official_rank
from stage36_official_supplement import (
    _race_id_from_cname,
    harmonize_supplement_horse_ids,
)
from test_historical_panel import _synthetic_rows

from keiba_place_lab.historical_panel import build_historical_panel


def test_runner_name_jump_does_not_change_flat_race_kind():
    conditions = parse_conditions(
        "2000 3歳未勝利 発走12時15分 (芝・右) 負担重量は馬齢重量 "
        "本賞5900000円 711ミッキージャンプ牡3栗57"
    )
    assert conditions["race_kind"] == "flat"
    assert conditions["official_distance_m"] == 2000
    assert parse_conditions("12003歳未勝利(ダート・右)本賞4815000円")["official_distance_m"] == 1200


def test_official_result_cname_roundtrips_race_id():
    url = (
        "https://www.jra.go.jp/JRADB/accessS.html?"
        "CNAME=pw01sde1006202505070120251227/A2"
    )
    assert _race_id_from_cname(url) == "202506050701"


@pytest.mark.parametrize(
    ("raw", "expected_status", "expected_started", "expected_pos"),
    [
        ("1", "finished", True, 1.0),
        ("10", "finished", True, 10.0),
        ("中止", "dnf", True, np.nan),
        ("失格", "disqualified", True, np.nan),
        ("取消", "nonstarter", False, np.nan),
        ("除外", "nonstarter", False, np.nan),
    ],
)
def test_official_supplement_rank_statuses(
    raw, expected_status, expected_started, expected_pos
):
    pos, status, started = parse_official_rank(raw)
    assert status == expected_status
    assert started is expected_started
    if np.isnan(expected_pos):
        assert np.isnan(pos)
    else:
        assert pos == expected_pos


def test_official_supplement_horse_ids_reuse_primary_history():
    supplement = pd.DataFrame(
        [
            {
                "race_id": "202506050701",
                "official_horse_id": "2023101860",
                "horse_name": "テストホース",
            },
            {
                "race_id": "202506050701",
                "official_horse_id": "2024999999",
                "horse_name": "新馬名",
            },
        ]
    )
    primary = pd.DataFrame(
        [{"horse_name": "テストホース", "horse_id": "source-horse-1"}]
    )
    out, diag = harmonize_supplement_horse_ids(supplement, primary)
    assert out.horse_id.tolist() == ["source-horse-1", "jra:2024999999"]
    assert diag["supplemental_name_mapped"] == 1
    assert diag["supplemental_new_horses"] == 1
    assert diag["supplemental_ambiguous_names"] == 0


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
    raw = "第1回 京都競馬 第1日\n05001 1月 5日 晴 良 (23京都1) 第1日 第1競走"
    rows = parse_race_days(
        info,
        "050011月5日晴良(23京都1)第1日第1競走",
        raw_text=raw,
    )
    assert rows[0]["actual_date"] == "2011-01-05"


def test_daily_date_parser_uses_fixed_width_serial_for_compact_raw_header():
    info = {
        "year": 2011,
        "venue_code": "08",
        "venue": "京都",
        "meeting_no": 1,
        "day_no": 1,
        "url": "https://example.invalid/2011-1kyoto1.pdf",
        "legacy": False,
    }
    for raw in [
        "010011月 5日 晴良 (23京都1) 第1日 第1競走",
        "01001\x011月 5日 晴良 (23京都1) 第1日 第1競走",
    ]:
        rows = parse_race_days(
            info,
            "010011月5日晴良(23京都1)第1日第1競走",
            raw_text=raw,
        )
        assert rows[0]["actual_date"] == "2011-01-05"


def test_daily_date_parser_recovers_compact_fixed_width_header():
    info = {
        "year": 2014,
        "venue_code": "07",
        "venue": "中京",
        "meeting_no": 4,
        "day_no": 1,
        "url": "https://example.invalid/2014-4chukyo1.pdf",
        "legacy": False,
    }
    rows = parse_race_days(
        info,
        "9993500112月6日曇良(26中京4)第1日第1競走",
        raw_text="machine text without a usable header",
    )
    assert rows[0]["actual_date"] == "2014-12-06"


def test_daily_date_parser_recovers_header_after_concatenated_leading_digit():
    info = {
        "year": 2014,
        "venue_code": "07",
        "venue": "中京",
        "meeting_no": 4,
        "day_no": 1,
        "url": "https://example.invalid/2014-4chukyo1.pdf",
        "legacy": False,
    }
    rows = parse_race_days(
        info,
        "935001・12月6日曇良(26中京4)第1日第1競走",
        raw_text="machine text without a usable header",
    )
    assert rows[0]["actual_date"] == "2014-12-06"


def test_daily_date_parser_does_not_treat_management_suffix_as_race_number():
    info = {
        "year": 2011,
        "venue_code": "01",
        "venue": "札幌",
        "meeting_no": 1,
        "day_no": 2,
        "url": "https://example.invalid/2011-1sapporo2.pdf",
        "legacy": False,
    }
    rows = parse_race_days(
        info,
        "250138月14日曇良(23札幌1)第2日第1競走",
        raw_text="machine text without a usable header",
    )
    assert rows[0]["actual_date"] == "2011-08-14"


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
    raw = "第5回 京都競馬 第1日 30001 11月 5日 晴 良 (28京都5) 第1日 第1競走"
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
