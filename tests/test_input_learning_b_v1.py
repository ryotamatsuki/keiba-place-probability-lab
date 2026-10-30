"""Ensure nested 2021–2023 selection never sees an outer-year label."""
from __future__ import annotations

from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest

runtime=run_path(str(Path(__file__).resolve().parents[1]/
                     "scripts/run_input_learning_b_v1.py"))
select_rounds=runtime["select_rounds"]
alpha_fit_from_prior_oof=runtime["alpha_fit_from_prior_oof"]
CHECKPOINTS=runtime["CHECKPOINTS"]


def scores():
    data=[]
    for year in (2021,2022,2023):
        for rounds in CHECKPOINTS:
            data.append({"year":year,"rounds":rounds,"races":15,
                         "race_macro_brier":.14+(abs(rounds-120)/100000)+(
                             -.02 if year==2023 and rounds==400 else 0)})
    return pd.DataFrame(data)


def test_inner_selection_excludes_current_or_future_outer_year():
    orig=scores()
    assert select_rounds(orig,2023)==120
    assert select_rounds(orig,2024)==400
    changed=orig.copy()
    changed.loc[changed.year.eq(2023),"race_macro_brier"]=100.0
    assert select_rounds(changed,2023)==120
    with pytest.raises(ValueError):
        select_rounds(orig,2025)


def oof():
    rng=np.random.default_rng(333)
    n=20
    rows=[]
    for y in (2021,2022,2023):
        for i in range(n):
            p=rng.uniform(.1,.75)
            row={"race_date":pd.Timestamp(f"{y}-07-01"),"race_id":f"{y}_{i//4}",
                 "top3_label":i%3==0,"p_market_oof":p}
            for j in CHECKPOINTS:
                row["p_rounds_"+str(j)]=np.clip(
                    p+rng.normal(0,.03),.02,.98)
            rows.append(row)
    return pd.DataFrame(rows)


def test_alpha_2023_never_uses_2023_outcomes():
    data=oof()
    prior=alpha_fit_from_prior_oof(data,2023,120)
    changed=data.copy()
    changed.loc[changed.race_date.dt.year.eq(2023),"top3_label"]=False
    assert alpha_fit_from_prior_oof(changed,2023,120)==prior
    assert 0<=prior<=1.5


def test_learning_curve_requires_inner_eval_dates():
    m=oof()
    with pytest.raises(KeyError):
        alpha_fit_from_prior_oof(m,2023,100)
