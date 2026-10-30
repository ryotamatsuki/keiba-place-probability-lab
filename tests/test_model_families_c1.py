"""C1 strict market offset and chronological family interface tests."""
from __future__ import annotations

from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest

SCRIPT=run_path(str(Path(__file__).resolve().parents[1]/
                    "scripts/run_model_families_c1.py"))
_margin=SCRIPT["_market_margin"]
_categorical_input=SCRIPT["_categorical_input"]
_race_weight=SCRIPT["_race_weight"]
fit_predict_lightgbm=SCRIPT["fit_predict_lightgbm"]
fit_predict_catboost=SCRIPT["fit_predict_catboost"]


def small_data():
    rng=np.random.default_rng(73)
    rows=90
    frame=pd.DataFrame({
        "race_date":pd.Timestamp("2022-12-31"),
        "race_id":[f"r{i//5}" for i in range(rows)],
        "horse_id":[f"h{i}" for i in range(rows)],
        "horse_no":np.tile([1,2,3,4,5],rows//5),
        "p_market_oof":rng.uniform(0.15,0.65,rows),
        "top3_label":rng.integers(0,2,size=rows),
        "draw_pct":rng.uniform(0,1,rows),
        "surface":np.where(np.arange(rows)%2==0,"turf","dirt"),
    })
    return frame


def test_no_future_market_and_equal_race_weights():
    frame=small_data()
    w=_race_weight(frame)
    assert np.isclose(w.mean(),1)
    assert np.isfinite(_margin(frame)).all()
    frame.loc[0,"p_market_oof"]=np.nan
    with pytest.raises(ValueError,match="chronological"):
        _margin(frame)


def test_training_only_feature_format_and_missing():
    f=small_data()
    f["draw_pct"]=f.draw_pct.astype("Float64")
    f.loc[0,"draw_pct"]=pd.NA
    formatted=_categorical_input(f,["draw_pct"],["surface"])
    assert formatted.draw_pct.dtype==float
    assert np.isnan(formatted.draw_pct.iloc[0])
    assert formatted.surface.dtype==object


def test_lightgbm_real_offset_api_if_installed():
    pytest.importorskip("lightgbm")
    training=small_data()
    target=training.iloc[:5].copy()
    target.race_date=pd.Timestamp("2023-01-05")
    p=fit_predict_lightgbm(training,target,["draw_pct"],["surface"])
    assert p.shape==(5,)
    assert np.all(np.isfinite(p)&(p>0)&(p<1))
    with pytest.raises(ValueError,match="label leak"):
        fit_predict_lightgbm(training,training,["draw_pct"],["surface"])


def test_catboost_real_baseline_api_if_installed():
    pytest.importorskip("catboost")
    training=small_data()
    target=training.iloc[:5].copy()
    target.race_date=pd.Timestamp("2023-01-05")
    p=fit_predict_catboost(training,target,["draw_pct"],["surface"])
    assert p.shape==(5,)
    assert np.all(np.isfinite(p)&(p>0)&(p<1))
    with pytest.raises(ValueError,match="label leak"):
        fit_predict_catboost(training,training,["draw_pct"],["surface"])
