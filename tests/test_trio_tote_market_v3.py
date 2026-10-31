"""Real 3-renpuku market integration rejects missing and incoherent snapshots."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from keiba_place_lab.trio_tote_market_v3 import (
    fit_market_pool,
    geometric_joint_blend,
    toto_market_distribution,
    validate_archived_tote_snapshot,
)


def quotes():
    triplets = list(combinations(range(1, 6), 3))
    tickets = pd.DataFrame(triplets, columns=["horse_1","horse_2","horse_3"])
    probability = tickets.copy()
    probability["probability"] = .1
    offered = tickets.copy()
    offered["odds"] = np.arange(11, 21, dtype=float)
    offered["race_id"] = "202605041111"
    offered["source_updated_at"] = "2026-11-01T15:29:00+09:00"
    offered["capture_at"] = "2026-11-01T15:30:00+09:00"
    return probability, offered


def test_real_quotes_are_source_matched_before_normalizing():
    p,q = quotes()
    frame,implied = validate_archived_tote_snapshot(
        p,q,expected_horse_numbers=set(range(1,6)),
        expected_race_id="202605041111",
    )
    assert len(frame)==len(implied)==10
    assert implied.sum()==pytest.approx(1)
    with pytest.raises(ValueError):
        validate_archived_tote_snapshot(
            p,q.iloc[1:],expected_horse_numbers=set(range(1,6)),
            expected_race_id="202605041111",
        )


def test_real_market_geometric_pool_has_expected_limits():
    f=np.array([.1,.2,.3,.4])
    m=toto_market_distribution(np.array([10,12,15,20]))
    np.testing.assert_allclose(geometric_joint_blend(f,m,market_weight=0),f)
    np.testing.assert_allclose(geometric_joint_blend(f,m,market_weight=1),m)
    assert geometric_joint_blend(f,m,market_weight=.5).sum()==pytest.approx(1)


def test_never_train_without_historic_real_ticket_quotes():
    with pytest.raises(ValueError,match="Insufficient"):
        fit_market_pool([],fit_end_year=2023)
    p=np.full(10,.1)
    odds=np.full(10,15.0)
    rows=[{
        "race_year":2024,"quote_year":2024,
        "f3_joint":p,"real_trio_odds":odds,"actual_index":0,
    } for _ in range(100)]
    with pytest.raises(ValueError,match="leak"):
        fit_market_pool(rows,fit_end_year=2023)
