from datetime import datetime

import pytest

from keiba_place_lab.live_public import (
    canonical_id,
    parse_horse_inventory,
    parse_results,
    parse_win_snapshot,
)

META = '''<meta property="og:url" content="https://sports.yahoo.co.jp/keiba/race/result/2605030211">
<section class="hr-predictRaceInfo">11R 2026年6月7日（日）3回東京2日
15:40発走 <h3 class="hr-predictRaceInfo__title">安田記念 GI</h3>
芝・左 1600m 3歳以上 オープン</section>'''
TARGET = {"provider_id": "2605030211", "race_id": "202605030211", "off_at": "2026-06-07T15:40:00+09:00"}


def horse(hid="2021105724"):
    return f'<a href="/keiba/directory/horse/{hid}/">馬</a><p>牡5</p>'


def test_public_market_uses_provider_clock_and_full_horse_id():
    html = META + '<time class="hr-horseResult__update" datetime="2026-06-07T15:30:06"></time>'
    html += '<table><tr><th>馬名 単勝 複勝</th></tr><tr><td>1</td><td>4</td><td>'
    html += horse() + '</td><td>10.0</td><td>2.0 - 3.0</td></tr></table>'
    html = html.replace('/race/result/', '/race/odds/tfw/')
    frame, asof = parse_win_snapshot(html, TARGET)
    assert asof == datetime.fromisoformat("2026-06-07T15:30:06+09:00")
    assert frame.horse_id.tolist() == ["2021105724"]
    assert frame.win_odds.tolist() == [10.0]
    with pytest.raises(ValueError, match="timestamp"):
        parse_win_snapshot(html.replace('datetime=', 'invalid='), TARGET)
    with pytest.raises(ValueError, match="identity"):
        parse_win_snapshot(html.replace('2605030211', '2605040211'), TARGET)


def test_full_results_exclude_nonstarter_and_preserve_confirmed_dnf():
    html = META + '<table><tr><th>着順 馬名 通過順位 騎手名</th></tr>'
    for rank, no, hid, tm in [("1", 1, "2021105724", "1:32.1"), ("中止", 2, "2021105725", ""), ("取消", 3, "2021105726", "")]:
        html += f'<tr><td>{rank}</td><td>1</td><td>{no}</td><td>{horse(hid)}</td>'
        html += f'<td>{tm}<p>-</p></td><td>02-02<p>33.9</p></td><td>騎手<p>▲52.0</p></td><td>市場</td><td>調教師</td></tr>'
    frame = parse_results(html + '</table>', "2605030211")
    assert len(frame) == 2 and frame.field_size.eq(2).all()
    assert frame.declared_field_size.eq(3).all()
    assert frame.outcome_confirmed.all() and frame.finish_position.isna().sum() == 1
    assert "win_odds" not in frame and "市場" not in frame.columns
    assert frame.race_time_seconds.iloc[0] == 92.1
    assert frame.early_position.iloc[0] == 2
    assert frame.assigned_weight_kg.eq(52).all()


def test_inventory_keeps_other_distance_but_excludes_future_and_nonstarter():
    html = '<table><tr><th>日付 通過順位</th></tr>'
    for date, rid, rank in [("2026/06/07", "2605030211", "1"), ("2026/10/04", "2605040211", "1"), ("2025/11/30", "2505050812", "取消")]:
        html += f'<tr><td>{date}</td><td><a href="/keiba/race/index/{rid}">芝2400m</a></td><td>{rank}</td></tr>'
    assert len(parse_horse_inventory(html + '</table>', "2026-10-03")) == 1
    with pytest.raises(ValueError):
        canonical_id("2600040211")
