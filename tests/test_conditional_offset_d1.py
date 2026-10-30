"""Conditional E5 replacement cannot use 2024 outcomes in training."""
from __future__ import annotations

from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest

runtime=run_path(str(Path(__file__).resolve().parents[1]/
                     "scripts/run_conditional_offset_d1.py"))
fit_conditional=runtime["fit_conditional"]
conditional_probability=runtime["conditional_probability"]


def oof():
    rng=np.random.default_rng(77)
    n=80
    return pd.DataFrame({
        "race_date":pd.Timestamp("2023-05-01"),
        "race_id":np.repeat(np.arange(n//5).astype(str),5),
        "top3_label":np.tile([1,0,0,1,0],n//5),
        "p_M0_market":rng.uniform(.15,.65,n),
        "p_X2_nonlinear_strength":rng.uniform(.15,.65,n),
        "distance_m":np.tile([1200,1400,1600,1800,2000],n//5),
        "surface":np.where(np.arange(n)%2==0,"turf","dirt"),
    })


def test_zero_correction_recovers_market_and_unit_recovers_x2():
    f=oof()
    for name in ("D1","D2"):
        zeros=np.zeros(2 if name=="D1" else 4)
        np.testing.assert_allclose(conditional_probability(f,zeros,name),
                                   f.p_M0_market,atol=1e-12)
        zeros[0]=1.0
        np.testing.assert_allclose(conditional_probability(f,zeros,name),
                                   f.p_X2_nonlinear_strength,atol=1e-12)


def test_optimizer_uses_only_prior_calendar_year():
    f=oof()
    for name in ("D1","D2"):
        r=fit_conditional(f,name)
        assert r["candidate"]==name
        assert r["train_rows"]==len(f)
    leaked=f.copy()
    leaked.loc[0,"race_date"]=pd.Timestamp("2024-01-01")
    with pytest.raises(ValueError,match="2023"):
        fit_conditional(leaked,"D1")


def test_conditional_does_not_depend_on_same_year_results():
    frame=oof()
    params=np.array([1.0,0.1,0.05,0.04])
    before=conditional_probability(frame,params,"D2")
    frame.loc[:,"top3_label"]=1-frame.top3_label
    after=conditional_probability(frame,params,"D2")
    np.testing.assert_array_equal(before,after)
