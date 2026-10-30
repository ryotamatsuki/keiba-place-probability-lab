"""Fit learner-specific scaling on 2023 only and recover market identities."""
from __future__ import annotations

import sys
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
learn_alpha=run_path(str(Path(__file__).resolve().parents[1]/
                        "scripts/run_model_calibration_c2.py"))["learn_alpha"]


def sample():
    rng=np.random.default_rng(196)
    n=100
    return pd.DataFrame({
        "race_date":pd.Timestamp("2023-07-01"),
        "race_id":np.repeat(np.arange(n//5).astype(str),5),
        "top3_label":np.tile([1,1,0,0,0],n//5),
    }),rng.uniform(.1,.7,n),rng.uniform(.1,.7,n)


def test_only_2023_meta_calibration_and_bounds():
    f,m,p=sample()
    scalar=learn_alpha(m,p,f)
    assert 0<=scalar<=1.5
    changed=f.copy()
    changed.loc[0,"race_date"]=pd.Timestamp("2024-01-01")
    with pytest.raises(ValueError,match="2023"):
        learn_alpha(m,p,changed)


def test_alpha_rejects_nan_and_unmatched_rows():
    f,m,p=sample()
    p[0]=np.nan
    with pytest.raises(ValueError,match="Missing valid"):
        learn_alpha(m,p,f)
    with pytest.raises(ValueError,match="Missing valid"):
        learn_alpha(m,np.ones(10),f)
