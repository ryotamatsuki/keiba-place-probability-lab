"""RealMLP CPU pilot prior-only full-race cohort and imputation invariants."""
from __future__ import annotations

import sys
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
runtime=run_path(str(Path(__file__).resolve().parents[1]/
                     "scripts/run_realmlp_cpu_pilot_v1.py"))
cohorts=runtime["cohorts"]
prepare_realmlp=runtime["prepare_realmlp"]


def fake():
    items=[]
    for year,n in ((2022,1510),(2023,260)):
        for index in range(n):
            items.append({"race_date":pd.Timestamp(f"{year}-01-01")+
                          pd.Timedelta(days=index%360),
                          "race_id":f"{year}_{index:05}",
                          "horse_id":f"{year}_h{index}","horse_no":1,
                          "draw_pct":float(index%7)/7,
                          "surface":"turf" if index%2==0 else "dirt"})
    return pd.DataFrame(items)


def test_historical_pilot_selection_by_race_not_outcomes():
    train,test=cohorts(fake())
    assert train.race_id.nunique()==1500
    assert test.race_id.nunique()==250
    assert train.race_date.max()<test.race_date.min()
    x=fake()
    x["top3_label"]=np.arange(len(x))%2
    a,b=cohorts(x)
    x["top3_label"]=1-x.top3_label
    c,d=cohorts(x)
    assert a.race_id.tolist()==c.race_id.tolist()
    assert b.race_id.tolist()==d.race_id.tolist()


def test_train_only_missing_numeric_and_future_category():
    tr=pd.DataFrame({"draw_pct":[1.,np.nan,3.,4.,5.,6.],
                     "surface":["turf","turf","dirt","dirt","turf","dirt"]})
    te=pd.DataFrame({"draw_pct":[np.nan,30.],
                     "surface":["turf","unknown_future_category"]})
    x,z=prepare_realmlp(tr,te,["draw_pct"],["surface"])
    assert np.isfinite(x).all() and np.isfinite(z).all()
    assert x.shape[1]==z.shape[1]
