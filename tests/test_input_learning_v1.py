"""Full-field market and strictly-prior relative information regression."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba_place_lab.input_learning_v1 import (
    MARKET_FIELD,
    MARKET_SPLITS,
    add_calibrated_market_splits,
    attach_information,
    full_starter_market_context,
    full_starter_relative,
)
from keiba_place_lab.stage4_successor_phase3a import SOURCE_RULES


def fixtures():
    rows=[]
    for race_id in ("r1","r2"):
        for no in (1,2,3,4):
            r={"race_id":race_id,"horse_id":f"{race_id}_h{no}",
               "horse_no":no,"field_size":4}
            for source,_,_ in SOURCE_RULES:
                r[source]=0.2*no
            rows.append(r)
    field=pd.DataFrame(rows)
    market=field[["race_id","horse_no"]].copy()
    odds=np.tile([0.5,0.25,0.15,0.1],2)
    market["market_p_win"]=odds
    market["market_p_top3"]=np.tile([0.9,0.8,0.7,0.6],2)
    market["market_field_size"]=4
    eligible=field.loc[field.horse_no.isin([1,2,3])].copy()
    return eligible,field,market


def test_relative_uses_ineligible_fourth_starter():
    eligible,full,market=fixtures()
    enriched,coverage=full_starter_relative(eligible,full)
    assert len(enriched)==6 and len(coverage)==5
    # Horse 1 faces horses 2,3,4, not only eligible 2,3.
    expected=0.2-(0.4+0.6+0.8)/3
    assert np.isclose(
        enriched.loc[enriched.horse_no.eq(1),
                     "rel_career_top3_vs_others"].iloc[0], expected)
    # Row missing current outcome, so later target outcome cannot influence features.
    assert not any("label" in k for k in enriched.columns)


def test_market_rank_concentration_uses_all_starters():
    _,_,market=fixtures()
    engineered=full_starter_market_context(market)
    assert len(engineered)==8
    a=engineered.loc[engineered.race_id.eq("r1")].sort_values("horse_no")
    assert np.allclose(a.market_win_rank_pct,[0,1/3,2/3,1])
    assert np.allclose(a.market_favorite_win_share,0.5)
    assert np.allclose(a.market_top2_win_gap,0.25)
    assert np.allclose(a.market_win_hhi,0.5**2+.25**2+.15**2+.1**2)
    assert set(MARKET_FIELD).issubset(a.columns)


def test_incomplete_market_or_relative_context_fail_closed():
    eligible,full,market=fixtures()
    with pytest.raises(ValueError,match="complete|missing|Full-field"):
        full_starter_relative(eligible,full.iloc[:-1])
    with pytest.raises(ValueError,match="Full-field"):
        full_starter_market_context(market.iloc[:-1])
    with pytest.raises(ValueError,match="Duplicate"):
        full_starter_market_context(pd.concat([market,market.iloc[:1]]))
    _,coverage=attach_information(eligible,full,market)
    assert len(coverage)==5


def test_target_outcome_changes_do_not_affect_feature_values():
    eligible,full,market=fixtures()
    full["top3_label"]=[1,1,1,0]*2
    first,_=full_starter_relative(eligible,full)
    full["top3_label"]=[0,0,0,1]*2
    second,_=full_starter_relative(eligible,full)
    pd.testing.assert_frame_equal(first,second)


def test_logit_split_from_strictly_prior_calibrated_market():
    a=pd.DataFrame({"p_market_oof":[0.1,0.5,0.9]})
    x=add_calibrated_market_splits(a)
    np.testing.assert_allclose(x["market_calibrated_top3_p"],[0.1,0.5,0.9])
    np.testing.assert_allclose(x["market_calibrated_top3_logit"],
                               np.log([0.1,0.5,0.9])-np.log1p(-np.array([0.1,0.5,0.9])))
    assert set(MARKET_SPLITS).issubset(x)
    with pytest.raises(ValueError,match="not fitted"):
        add_calibrated_market_splits(pd.DataFrame({"p_market_wrong":[0.1]}))
