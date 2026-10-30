"""Finalized family comparison requires actual same-key past-only OOF."""
from __future__ import annotations

import sys
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
MOD = run_path(str(SCRIPTS / "finish_round_sweep_1600_v1.py"))
optional_ensemble_gate = MOD["optional_ensemble_gate"]
learned_weights = MOD["learned_weights"]
mix_logits = MOD["mix_logits"]
market_match = MOD["market_match"]
aligned = MOD["aligned"]
read_frame = MOD["read_frame"]


def synthetic_ooof() -> pd.DataFrame:
    rng = np.random.default_rng(321)
    races = 25
    n = races * 8
    market = rng.uniform(.10,.65,n)
    xgb = np.clip(market+rng.normal(0,.02,n),.01,.99)
    lgb = np.clip(market+rng.normal(0,.025,n),.01,.99)
    cat = np.clip(market+rng.normal(0,.025,n),.01,.99)
    return pd.DataFrame({
        "race_date": pd.Timestamp("2023-06-15"),
        "race_id": np.repeat([f"r{i:04d}" for i in range(races)],8),
        "horse_id": [f"h{i:04d}" for i in range(n)],
        "horse_no": np.tile(np.arange(1,9),races),
        "top3_label": np.tile([1,1,1,0,0,0,0,0],races),
        "p_M0": market,
        "p_xgb_raw": xgb, "p_lgb_raw": lgb,"p_cat_raw": cat,
    })


def test_explicit_2023_only_gate_and_fit():
    train = synthetic_ooof()
    result = optional_ensemble_gate(train)
    assert set(result["individual_trigger"]) == {"lgb","cat"}
    assert result["run_three_preregistered_ensembles"] == any(
        result["individual_trigger"].values(),
    )
    fit = learned_weights(train)
    w = np.asarray(fit["weights_xgb_lgb_cat"])
    assert np.isfinite(w).all()
    assert np.all((w >= -1e-8)&(w <= 1.5+1e-8))
    assert w.sum() <= 2.00000001
    original = mix_logits(train, np.array([1.0,0.0,0.0]))
    np.testing.assert_allclose(original,train.p_xgb_raw,atol=1e-12)
    np.testing.assert_allclose(
        mix_logits(train,np.zeros(3)),train.p_M0,atol=1e-12,
    )
    future=train.copy()
    future.race_date = pd.Timestamp("2024-01-01")
    with pytest.raises(ValueError,match="2023"):
        optional_ensemble_gate(future)
    with pytest.raises(ValueError,match="2023"):
        learned_weights(future)


def test_no_target_label_depends_on_model_probabilities():
    source=synthetic_ooof()
    p=source.p_xgb_raw.copy()
    source.top3_label=1-source.top3_label
    np.testing.assert_array_equal(source.p_xgb_raw,p)


def test_identical_runner_truth_market_and_serialization(tmp_path):
    data=synthetic_ooof()
    file=tmp_path/"example.csv"
    data.to_csv(file,index=False)
    read=read_frame(file)
    original=data.sort_values(
        ["race_date","race_id","horse_id","horse_no"],
        kind="stable",
    ).reset_index(drop=True)
    aligned(original,read,"serialize")
    market_match(original.p_M0.to_numpy(float),read.p_M0.to_numpy(float),"M0")
    bad=read.copy()
    bad.loc[0,"horse_id"]="wrong"
    with pytest.raises(ValueError,match="rows"):
        aligned(original,bad,"wrong")
    bad=read.copy()
    bad.loc[0,"top3_label"]=0
    with pytest.raises(ValueError,match="labels"):
        aligned(original,bad,"bad label")
    with pytest.raises(ValueError,match="market"):
        market_match(np.array([0.5]),np.array([0.6]),"market")
