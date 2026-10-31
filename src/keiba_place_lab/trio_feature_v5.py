"""Pre-event historical feature blocks for 3-renpuku residual learning.

Research only. Labels from the CURRENT race are never used in that race's
features. Agents update only at date boundaries, not between same-day races.
Current horse raw results enter only after groupby(horse_id).shift(1+).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AGENT_FEATURES=(
    "v5_jockey_prior_place","v5_trainer_prior_place",
    "v5_jockey_log_starts","v5_trainer_log_starts",
    "v5_jockey_relative_place","v5_trainer_relative_place",
)
DYNAMICS_FEATURES=(
    "v5_last3f_trend","v5_last3f_volatility",
    "v5_finish_trend","v5_finish_volatility",
    "v5_passing_trend","v5_last3f_observed","v5_finish_observed",
)
PACE_FEATURES=(
    "v5_front_share","v5_front_market_share","v5_front_count",
    "v5_early_field_std","v5_early_relative","v5_early_valid_share",
)
FEATURE_BLOCKS={"agents":AGENT_FEATURES,"dynamics":DYNAMICS_FEATURES,
                "pace":PACE_FEATURES,
                "all":(*AGENT_FEATURES,*DYNAMICS_FEATURES,*PACE_FEATURES),
                "offset_only":()}


def _numeric(x:pd.Series)->pd.Series:
    return pd.to_numeric(x,errors="coerce").astype(float)


def _asof_agent(x:pd.DataFrame,column:str,prefix:str)->pd.DataFrame:
    """Return smoothed same-day-before rating for current (not previous) agent."""
    valid=x[column].notna()
    tmp=x.loc[valid,[column,"race_date","top3_label"]].copy()
    day=tmp.groupby([column,"race_date"],sort=False,dropna=True).agg(
        starts=("top3_label","size"),successes=("top3_label","sum")
    ).reset_index().sort_values([column,"race_date"],kind="stable")
    group=day.groupby(column,sort=False)
    day["before_starts"]=group.starts.cumsum()-day.starts
    day["before_wins"]=group.successes.cumsum()-day.successes
    day[prefix+"_prior_place"]=(day.before_wins+4.)/(day.before_starts+20.)
    day[prefix+"_log_starts"]=np.log1p(day.before_starts)
    joined=x[[column,"race_date"]].merge(
        day[[column,"race_date",prefix+"_prior_place",prefix+"_log_starts"]],
        on=[column,"race_date"],how="left",sort=False,validate="many_to_one"
    )
    return joined[[prefix+"_prior_place",prefix+"_log_starts"]]


def engineer_v5_asof(frame:pd.DataFrame, *, validate_races=None)->pd.DataFrame:
    """Use complete historical panel; audit ALL starters of evaluated races.

    Historical background races are allowed outside the validated F3 scope,
    but every target race must have its entire original field available.
    """
    req={"race_id","horse_id","race_date","horse_no","field_size",
         "top3_label","jockey_name","trainer",
         "raw_last3f_rel","raw_finish_position","raw_passing_gain",
         "history_early_pct3","market_win_probability"}
    missing=req-set(frame)
    if missing:
        raise ValueError(f"Missing source for as-of features: {sorted(missing)}")
    x=frame.sort_values(
        ["race_date","race_id","horse_no"],kind="stable"
    ).reset_index(drop=True).copy()
    x["race_date"]=pd.to_datetime(x.race_date)
    if x.duplicated(["race_id","horse_no"]).any():
        raise ValueError("Duplicate starter numbers")
    if x.duplicated(["horse_id","race_date"]).any():
        raise ValueError("Horse cannot race twice on a single date")
    for who,prefix in (("jockey_name","v5_jockey"),("trainer","v5_trainer")):
        agent=_asof_agent(x,who,prefix)
        for col in agent:
            x[col]=agent[col].to_numpy()
        # Uses entire PRE-event field for comparison; no label or current result.
        peer=x.groupby("race_id")[prefix+"_prior_place"].transform("mean")
        x[prefix+"_relative_place"]=x[prefix+"_prior_place"]-peer

    # Historical runner results: only 1,2,3 past starts.
    # These raw columns must not be used in the unshifted current row.
    finish=(_numeric(x.raw_finish_position)-1.)/(
        _numeric(x.field_size)-1.).where(_numeric(x.field_size)>1)
    finish=finish.where(finish.between(0,1))
    source={
        "last3f":_numeric(x.raw_last3f_rel),
        "finish":finish,
        "passing":_numeric(x.raw_passing_gain),
    }
    horse=x.groupby("horse_id",sort=False)
    for key,values in source.items():
        x["_v5_source_"+key]=values
        prior=[horse["_v5_source_"+key].shift(k) for k in (1,2,3)]
        arr=pd.concat(prior,axis=1).to_numpy(float)
        observed=np.isfinite(arr).sum(axis=1)
        recent=arr[:,0]
        previous=arr[:,1]
        trend=recent-previous
        # A volatility proxy computed only from previous performances.
        with np.errstate(invalid="ignore",divide="ignore"):
            vol=np.nanstd(arr,axis=1,ddof=0)
        vol[observed<2]=np.nan
        x["v5_"+key+"_trend"]=np.where(
            np.isfinite(recent)&np.isfinite(previous),trend,np.nan
        )
        if key in ("last3f","finish"):
            x["v5_"+key+"_volatility"]=vol
            x["v5_"+key+"_observed"]=observed.astype(float)

    # Field pace composition from ONLY past early-running records.
    e=_numeric(x.history_early_pct3)
    front=e.le(.3)&e.notna()
    m=_numeric(x.market_win_probability)
    if (m.dropna()<=0).any() or (m.dropna()>=1).any():
        raise ValueError("Market share invalid; cannot synthesize T-10 odds")
    x["_v5_front"]=front.astype(float)
    x["_v5_front_share_market"]=np.where(front,m,0.)
    n=x.groupby("race_id").race_id.transform("size").to_numpy(float)
    expected=_numeric(x.field_size).to_numpy(float)
    # The full historical background contains races outside the F3 eligibility
    # scope. Never demand completeness there; but NEVER relax it for evaluated
    # races, which must supply exact true starter field composition.
    checking=x.race_id.isin(validate_races).to_numpy() if validate_races is not None else np.ones(len(x),dtype=bool)
    if validate_races is not None and set(validate_races)-set(x.race_id):
        raise ValueError("Missing target races from full historical starter context")
    mismatched=checking & ((n!=expected) | ~np.isfinite(expected))
    if mismatched.any():
        examples=x.loc[mismatched,["race_id","field_size"]].copy()
        examples["actual_context_rows"]=n[mismatched].astype(int)
        examples=examples.drop_duplicates("race_id").head(10).to_dict("records")
        raise ValueError("Field pace context requires ALL actual starters: "+repr(examples))
    group=x.groupby("race_id",sort=False)
    x["v5_front_share"]=group["_v5_front"].transform("mean")
    x["v5_front_count"]=group["_v5_front"].transform("sum")
    x["v5_front_market_share"]=group["_v5_front_share_market"].transform("sum")
    x["v5_early_field_std"]=group.history_early_pct3.transform("std")
    x["v5_early_relative"]=e-group.history_early_pct3.transform("mean")
    x["v5_early_valid_share"]=e.notna().groupby(x.race_id).transform("mean")
    if set(FEATURE_BLOCKS["all"])-set(x):
        raise AssertionError("Some preregistered feature not constructed")
    return x.drop(columns=[n for n in x if n.startswith("_v5_")])


def audit_asof_invariance(a:pd.DataFrame,b:pd.DataFrame)->bool:
    """For the SAME race-day, any change to that day's outcomes must not alter features."""
    keys=["race_id","horse_id","horse_no"]
    left=a.sort_values(keys).reset_index(drop=True)
    right=b.sort_values(keys).reset_index(drop=True)
    if not left[keys].equals(right[keys]):
        raise ValueError("Changed race identity invalidates leak check")
    for col in FEATURE_BLOCKS["all"]:
        if not np.allclose(left[col].to_numpy(float),
                           right[col].to_numpy(float),equal_nan=True):
            return False
    return True
