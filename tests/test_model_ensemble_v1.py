"""Chronology and probability validity tests for ensemble V1."""
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest

SCRIPT = run_path(str(Path(__file__).resolve().parents[1] /
                      "scripts/run_model_ensemble_v1.py"))
ensemble_predictions=SCRIPT["ensemble_predictions"]
learn_alpha=SCRIPT["learn_alpha"]


def synthetic_2023():
    rng=np.random.default_rng(31)
    n=80
    return pd.DataFrame({
        "race_date":pd.Timestamp("2023-01-01"),
        "race_id":np.repeat([f"race{i}" for i in range(20)],4),
        "top3_label":np.tile([1,0,1,0],20),
        "p_M0_market":rng.uniform(0.10,0.75,n),
        "p_X2_nonlinear_strength":rng.uniform(0.10,0.75,n),
    })


def test_probability_pool_reproduces_component_identity():
    m=np.array([0.12,0.55,0.82])
    x0=np.array([0.25,0.48,0.74])
    x1=np.array([0.22,0.51,0.77])
    x2=np.array([0.19,0.53,0.76])
    out=ensemble_predictions({"M0":m,"X0":x0,"X1":x1,"X2":x2},alpha=1)
    assert list(out)==["M0","X2","E1","E2","E3","E4","E5"]
    np.testing.assert_allclose(out["E1"],0.5*(x0+x2))
    np.testing.assert_allclose(out["E3"],0.5*(x1+x2))
    np.testing.assert_allclose(out["E4"],0.1*m+0.9*x2)
    np.testing.assert_allclose(out["E5"],x2,atol=1e-12)
    null=ensemble_predictions({"M0":m,"X0":x0,"X1":x1,"X2":x2},alpha=0)
    np.testing.assert_allclose(null["E5"],m,atol=1e-12)
    for p in out.values():
        assert np.all((0<=p)&(p<=1))


def test_logit_pool_not_necessarily_probability_pool():
    m=np.array([0.2,0.7])
    x=np.array([0.7,0.9])
    p=ensemble_predictions({"M0":m,"X0":m,"X1":m,"X2":x},alpha=0.5)
    assert not np.allclose(p["E1"],p["E2"])


def test_2023_only_fit_no_2024_labels():
    training=synthetic_2023()
    alpha=learn_alpha(training)
    assert 0<=alpha<=1.5
    changed=training.copy()
    changed.loc[0,"race_date"]=pd.Timestamp("2024-01-01")
    with pytest.raises(ValueError,match="2023"):
        learn_alpha(changed)


def test_explicit_same_row_arrays_and_invalid_probability_fail():
    a={"M0":np.array([0.3,0.4]),"X0":np.array([0.35,0.5]),
       "X1":np.array([0.31,0.49]),"X2":np.array([0.4])}
    with pytest.raises(ValueError,match="Unaligned"):
        ensemble_predictions(a,1)
    a["X2"]=np.array([0.4,float("nan")])
    with pytest.raises(ValueError,match="Invalid component"):
        ensemble_predictions(a,1)
