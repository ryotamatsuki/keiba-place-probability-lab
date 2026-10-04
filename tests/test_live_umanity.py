import pandas as pd
import pytest

from keiba_place_lab.live_history import verify_full_field
from keiba_place_lab.live_umanity import parse_entry, parse_result


def page(kind, table):
    return f'''<meta property="og:url" content="https://umanity.jp/racing/{kind}.php?race_id=2026021510010811">
    <h1>テスト 2026年2月15日 小倉11R</h1>
    <div><div>テスト GⅢ</div><div>9:55発走｜芝・右 1200m</div>
    <div>４歳以上オープン</div></div>{table}'''


def result_row(rank, number, hid, time, corner):
    return f'''<tr><td class="td_order">{rank}</td><td>{number}</td><td></td>
    <td><a class="horsename" href="/horse?horse_id={hid}">馬{number}</a>牡5</td>
    <td></td><td><span class="gray6">57.0</span></td>
    <td><div>{time}</div></td><td>{corner}</td></tr>'''


def entry_row(number, hid, status=""):
    return f'''<tr><td>{number}</td><td>{status}</td>
    <td><a class="horsename" href="/horse?horse_id={hid}">馬{number}</a>牡5</td>
    <td></td><td><div class="bold gray7">57.0</div></td>
    <td>過去の競走中止・取消は現在の出走資格と無関係</td></tr>'''


def test_alternative_source_preserves_nonstarter_and_dnf_and_ignores_market():
    kwargs = {"race_id": "202610010811", "race_date": "2026-02-15", "source_url": "fixture", "retrieved_at": "test"}
    result = page("result", "<table>" +
                  result_row("1着", 1, "2021100001", "58.9", '<span class="text_passage_sq">2</span>') +
                  result_row("中止", 2, "2021100002", "", "") +
                  result_row("取消", 3, "2021100003", "", "") + "</table>")
    roster = page("card", '<table id="grace_table1">' +
                  entry_row(1, "2021100001") + entry_row(2, "2021100002") +
                  entry_row(3, "2021100003", "取消") + "</table>")
    starters, entries = parse_result(result, **kwargs)
    declared = parse_entry(roster, **kwargs)
    qa = verify_full_field(entries, declared)
    assert qa["actual_starters"] == 2 and qa["scratched"] == 1 and qa["dnf"] == 1
    assert starters.race_time_seconds.iloc[0] == 58.9
    assert starters.early_position.iloc[0] == 2
    assert pd.isna(starters.finish_position.iloc[1])
    assert starters.is_graded.eq(1).all()
    assert not any("odds" in c or "u_index" in c for c in starters)
    with pytest.raises(ValueError, match="identity mismatch"):
        parse_result(result.replace("2026021510010811", "2026021510010812"), **kwargs)
