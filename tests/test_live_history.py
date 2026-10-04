from __future__ import annotations

import pandas as pd
import pytest

from keiba_place_lab.live_history import (
    CachedFetcher,
    _parse_race_time,
    _status_from_rank,
    apply_confirmed_events,
    expected_results,
    load_frozen_history,
    parse_monthly_schedule,
    parse_race_list,
    sha256_file,
    update_live_history,
    verify_full_field,
)


def test_unknown_legacy_base_cannot_be_marked_confirmed(tmp_path):
    path = tmp_path / "unknown.parquet"
    pd.DataFrame({"race_date": ["2025-01-01"]}).to_parquet(path)
    with pytest.raises(ValueError, match="Unrecognized frozen"):
        load_frozen_history(path)


def test_cached_capture_tampering_cannot_be_used(tmp_path):
    import hashlib
    import json
    url = "https://example.com/result"
    key = hashlib.sha256(url.encode()).hexdigest()
    html = tmp_path / f"{key}.html"
    html.write_text("original")
    (tmp_path / f"{key}.json").write_text(json.dumps({
        "url": url, "sha256": sha256_file(html), "retrieved_at": "2026-10-04T00:00:00Z",
        "cache_path": str(html)}))
    fetcher = CachedFetcher(tmp_path)
    assert fetcher.fetch(url)[0] == "original"
    html.write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        fetcher.fetch(url)


def test_independent_inventory_cannot_certify_a_later_cutoff(tmp_path):
    import json
    ledger = tmp_path / "ledger.csv"
    ledger.write_text("race_id,race_date,target_flat\n202608040101,2026-10-03,True\n")
    ledger.with_suffix(".manifest.json").write_text(json.dumps({
        "through": "2026-10-03", "ledger_sha256": sha256_file(ledger)}))
    with pytest.raises(ValueError, match="does not cover requested cutoff"):
        update_live_history(year=2026, through=pd.Timestamp("2026-10-04"), root=tmp_path / "db",
                            cache_dir=tmp_path / "cache", ledger_path=ledger)


def test_abandoned_race_is_audited_but_does_not_require_a_result():
    ledger = pd.DataFrame({
        "race_id": ["202605010307", "202605010308", "202605010309"],
        "race_date": pd.to_datetime(["2026-02-07"] * 3),
        "target_flat": [True, True, False],
    })
    event = {"race_date": "2026-02-07", "race_ids": ["202605010308"],
             "status": "abandoned", "source_url": "https://jra.jp/news/202602/020707.html",
             "reason": "積雪により取りやめ"}
    got = apply_confirmed_events(ledger, [event])
    assert len(got) == 3
    assert expected_results(got).race_id.tolist() == ["202605010307"]
    assert got.loc[1, "status_source_url"] == event["source_url"]
    assert expected_results(apply_confirmed_events(ledger, [])).race_id.tolist() == [
        "202605010307", "202605010308"
    ]
    with pytest.raises(ValueError, match="date mismatch"):
        apply_confirmed_events(ledger, [{**event, "race_date": "2026-02-08"}])
    with pytest.raises(ValueError, match="unverified"):
        apply_confirmed_events(ledger, [{**event, "source_url": "https://example.com"}])


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


def test_archived_entry_may_leave_late_scratch_marked_active():
    result = pd.DataFrame(
        {
            "horse_no": [1, 2, 3],
            "horse_id": ["a", "b", "c"],
            "horse_name": ["A", "B", "C"],
            "finish_status": ["finished", "dnf", "scratched"],
            "is_starter": [True, True, False],
            "field_size": [2, 2, 2],
            "declared_field_size": [3, 3, 3],
        }
    )
    roster = pd.DataFrame(
        {
            "horse_no": [1, 2, 3],
            "horse_id": ["a", "b", "c"],
            "horse_name": ["A", "B", "C"],
            "entry_status": ["active", "active", "active"],
            "is_starter": [True, True, True],
            "declared_field_size": [3, 3, 3],
        }
    )
    qa = verify_full_field(result, roster)
    assert qa["actual_starters"] == 2
    assert qa["declared_entries"] == 3


def test_parse_race_time_accepts_sub_minute_and_minute_formats():
    assert _parse_race_time("58.9") == 58.9
    assert _parse_race_time("1:00.1") == 60.1
    assert _parse_race_time("2:34.5") == 154.5
    assert _parse_race_time("1:56.4 -") == 116.4
    assert _parse_race_time("1:56.4-") == 116.4
    assert _parse_race_time("58.9 -") == 58.9
    assert _parse_race_time("58.9-") == 58.9
    assert pd.isna(_parse_race_time(""))
