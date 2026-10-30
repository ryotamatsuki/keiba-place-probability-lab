"""Strict same-cohort and fixed 2023-only residual stacking."""
from __future__ import annotations

from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest

script=run_path(str(Path(__file__).resolve().parents[1]/
                    "scripts/run_diverse_ensemble_d2.py"))
combined_probability=script["combined_probability"]
fit_2023=script["fit_2023"]
corrections=script["corrections"]


def sample():
    rng=np.random.default_rng(171)
    n=100
    m=rng.uniform(.15,.75,n)
    x=np.clip(m+rng.normal(0,.03,n),.02,.98)
    l=np.clip(m+rng.normal(0,.04,n),.02,.98)
    c=np.clip(m+rng.normal(0,.04,n),.02,.98)
    return pd.DataFrame({
        "race_date":pd.Timestamp("2023-01-01"),
        "race_id":np.repeat(np.arange(n//5).astype(str),5),
        "top3_label":np.tile([1,1,0,0,0],n//5),
        "p_M0_market":m,"p_X2_nonlinear_strength":x,
        "p_LightGBM":l,"p_CatBoost":c,
    })


def test_zero_market_only_x2_identity():
    df=sample()
    np.testing.assert_allclose(combined_probability(df,np.zeros(3)),
                               df.p_M0_market,atol=1e-12)
    np.testing.assert_allclose(combined_probability(df,np.array([1.,0,0])),
                               df.p_X2_nonlinear_strength,atol=1e-12)
    mm,cor=corrections(df)
    assert len(mm)==len(df) and cor.shape==(len(df),3)


def test_2023_only_ensemble_fit():
    df=sample()
    result=fit_2023(df)
    p=np.asarray(result["weights"],float)
    assert p.shape==(3,) and (p>=-1e-9).all() and p.sum()<=2+1e-9
    df.loc[0,"race_date"]=pd.Timestamp("2024-01-01")
    with pytest.raises(ValueError,match="2023"):
        fit_2023(df)


def test_missing_or_nonfinite_probabilities_fail():
    df=sample()
    df.loc[0,"p_CatBoost"]=np.nan
    with pytest.raises(ValueError,match="CatBoost"):
        corrections(df)
    with pytest.raises(ValueError,match="three"):
        combined_probability(sample(),np.array([1.,0]))
