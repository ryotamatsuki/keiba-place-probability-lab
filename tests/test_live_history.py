from __future__ import annotations

import pandas as pd
import pytest

from keiba_place_lab.live_history import (
    _status_from_rank,
    parse_monthly_schedule,
    parse_race_list,
    verify_full_field,
)


def test_monthly_schedule_and_race_list_enumeration():
    monthly = """
    <html><body><h1>2026年10月 日程・結果</h1>
      <a href="/keiba/race/list/26050401">東京</a>
      <a href="/keiba/race/list/26080401">京都</a>
    </body></html>
    """
    assert parse_monthly_schedule(monthly, year=2026, month=10) == ["26050401", "26080401"]

    race_list = """
    <html><body><h2>2026年10月3日（土）</h2><table>
      <tr><td><a href="/keiba/race/result/2605040101">1R</a></td><td>2歳未勝利 芝1600m</td></tr>
      <tr><td><a href="/keiba/race/result/2605040104">4R</a></td><td>障害3歳上オープン 障害・芝→ダート</td></tr>
      <tr><td><a href="/keiba/race/result/2605040112">12R</a></td><td>3歳以上2勝 芝1800m</td></tr>
    </table></body></html>
    """
    out = parse_race_list(race_list, meeting_day_id="26050401")
    assert out.provider_id.tolist() == ["2605040101", "2605040104", "2605040112"]
    assert out.target_flat.tolist() == [True, False, True]
    assert out.race_date.dt.strftime("%Y-%m-%d").unique().tolist() == ["2026-10-03"]


def test_result_statuses_are_distinct():
    assert _status_from_rank("1") == ("finished", True, 1.0)
    status, starter, pos = _status_from_rank("中止")
    assert (status, starter) == ("dnf", True)
    assert pd.isna(pos)
    assert _status_from_rank("失格")[:2] == ("disqualified", True)
    assert _status_from_rank("取消")[:2] == ("scratched", False)
    assert _status_from_rank("除外")[:2] == ("excluded", False)


def test_full_field_verification_distinguishes_nonstarters():
    result = pd.DataFrame(
        {
            "horse_no": [1, 2, 3, 4],
            "horse_id": ["a", "b", "c", "d"],
            "horse_name": ["A", "B", "C", "D"],
            "finish_status": ["finished", "dnf", "scratched", "excluded"],
            "is_starter": [True, True, False, False],
            "field_size": [2, 2, 2, 2],
            "declared_field_size": [4, 4, 4, 4],
        }
    )
    roster = pd.DataFrame(
        {
            "horse_no": [1, 2, 3, 4],
            "horse_id": ["a", "b", "c", "d"],
            "horse_name": ["A", "B", "C", "D"],
            "entry_status": ["active", "active", "scratched", "excluded"],
            "is_starter": [True, True, False, False],
            "declared_field_size": [4, 4, 4, 4],
        }
    )
    qa = verify_full_field(result, roster)
    assert qa == {
        "declared_entries": 4,
        "actual_starters": 2,
        "declared_field_size": 4,
        "scratched": 1,
        "excluded": 1,
        "dnf": 1,
        "disqualified": 0,
    }


def test_full_field_verification_rejects_missing_entry():
    result = pd.DataFrame(
        {
            "horse_no": [1, 2],
            "horse_id": ["a", "b"],
            "horse_name": ["A", "B"],
            "finish_status": ["finished", "finished"],
            "is_starter": [True, True],
            "field_size": [2, 2],
            "declared_field_size": [2, 2],
        }
    )
    roster = pd.DataFrame(
        {
            "horse_no": [1],
            "horse_id": ["a"],
            "horse_name": ["A"],
            "entry_status": ["active"],
            "is_starter": [True],
            "declared_field_size": [2],
        }
    )
    with pytest.raises(ValueError, match="absent from declared roster"):
        verify_full_field(result, roster)


def test_full_field_verification_allows_result_to_omit_scratched_horse():
    result = pd.DataFrame(
        {
            "horse_no": [1, 2],
            "horse_id": ["a", "b"],
            "horse_name": ["A", "B"],
            "finish_status": ["finished", "dnf"],
            "is_starter": [True, True],
            "field_size": [2, 2],
            "declared_field_size": [2, 2],
        }
    )
    roster = pd.DataFrame(
        {
            "horse_no": [1, 2, 3],
            "horse_id": ["a", "b", "c"],
            "horse_name": ["A", "B", "C"],
            "entry_status": ["active", "active", "scratched"],
            "is_starter": [True, True, False],
            "declared_field_size": [3, 3, 3],
        }
    )
    qa = verify_full_field(result, roster)
    assert qa["declared_field_size"] == 3
    assert qa["actual_starters"] == 2
    assert qa["scratched"] == 1
