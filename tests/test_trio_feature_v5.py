"""V5 counterfactual same-day leakage guards and full-starter integrity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba_place_lab.trio_feature_v5 import (
    FEATURE_BLOCKS, audit_asof_invariance, engineer_v5_asof,
)


def synthetic():
    rows=[]
    for day in ("2023-01-01","2023-01-08","2023-01-15"):
        ix=int(day[-2:])
        for h in range(1,5):
            rows.append({
                "race_id":f"test-{day}","race_date":day,
                "horse_id":f"horse-{h}","horse_no":h,"field_size":4,
                "top3_label":int(h!=4),"jockey_name":f"j-{h%2}",
                "trainer":f"tr-{h%3}","raw_last3f_rel":h*(.2+ix*.01),
                "raw_finish_position":h,"raw_passing_gain":float(h)*.1,
                "history_early_pct3":(.15,.35,.66,.8)[h-1],
                "market_win_probability":(.4,.3,.2,.1)[h-1],
            })
    return pd.DataFrame(rows)


def test_current_day_labels_are_not_used_even_for_other_starters():
    a=synthetic()
    b=a.copy()
    mask=b.race_date.eq("2023-01-08")
    b.loc[mask,"top3_label"]=[0,1,0,1]
    fa=engineer_v5_asof(a)
    fb=engineer_v5_asof(b)
    # Same-day features must be IDENTICAL despite same-day outcome changes.
    assert audit_asof_invariance(
        fa.loc[mask].copy(),fb.loc[mask].copy()
    )
    # But next week safely reflects prior observations:
    after=a.race_date.eq("2023-01-15")
    assert not np.allclose(
        fa.loc[after,"v5_jockey_prior_place"],
        fb.loc[after,"v5_jockey_prior_place"],
    )


def test_replacing_today_raw_results_cannot_change_today_dynamics():
    a=synthetic()
    b=a.copy()
    change=b.race_date.eq("2023-01-15")
    b.loc[change,"raw_last3f_rel"]=[-100,-2,88,1]
    b.loc[change,"raw_finish_position"]=[4,3,2,1]
    b.loc[change,"raw_passing_gain"]=[20,20,20,20]
    fa=engineer_v5_asof(a)
    fb=engineer_v5_asof(b)
    assert audit_asof_invariance(fa.loc[change],fb.loc[change])


def test_missing_starter_causes_fail_closed():
    x=synthetic().iloc[:-1]
    with pytest.raises(ValueError,match="ALL actual starters"):
        engineer_v5_asof(x)


def test_all_preregistered_features_exist():
    z=engineer_v5_asof(synthetic())
    assert set(FEATURE_BLOCKS["all"]).issubset(z.columns)
    assert z.groupby("race_id").v5_front_count.nunique().max()==1
    assert z.groupby("race_id").v5_front_market_share.nunique().max()==1


def test_asof_feature_rejects_duplicate_starters():
    z=synthetic()
    with pytest.raises(ValueError,match="Duplicate starter"):
        engineer_v5_asof(pd.concat([z,z.iloc[[0]]],ignore_index=True))
