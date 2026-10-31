"""Fair 2025 same-family market-vs-F3 trio comparator and top-K uncertainty.

No NEW model fitted; all 2025 scores are descriptive because already examined.
Same conditional-Bernoulli maximum-entropy transformation for both marginal
probability sets. Historical market is final odds, not T-10.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_trio_f3_v1 import paired_day_ci

from keiba_place_lab.trio_bridge_v1 import invert_subset_marginals, project_exactly_three
from keiba_place_lab.trio_residual_v2 import score_probabilities
from scipy.special import logsumexp


def score_market_maxent(runner:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for rid,g in runner.groupby("race_id",sort=False):
        g=g.sort_values("horse_no").reset_index(drop=True)
        nums=g.horse_no.to_numpy(int)
        q=project_exactly_three(g.p_market_frozen_cal.to_numpy(float))
        w=invert_subset_marginals(q)
        from itertools import combinations
        choices=np.asarray(list(combinations(range(len(g)),3)))
        raw=w[choices].sum(axis=1)
        prob=np.exp(raw-logsumexp(raw))
        actual=tuple(sorted(g.loc[g.top3_label.eq(1),"horse_no"].tolist()))
        outcome=np.flatnonzero(np.all(np.sort(nums[choices],axis=1)==actual,axis=1))
        if len(actual)!=3 or len(outcome)!=1:
            raise ValueError("Incomplete 2025 final trio outcome")
        values=score_probabilities(prob,int(outcome[0]))
        rows.append({
            "race_date":str(pd.Timestamp(g.race_date.iloc[0]).date()),
            "race_id":rid, "field_size":len(g), **{
                "market_maxent_"+k:v for k,v in values.items()
            },
        })
    return pd.DataFrame(rows)


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--runners",required=True,type=Path)
    p.add_argument("--existing-v1",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Comparison is immutable")
    runners=pd.read_csv(args.runners,dtype={"race_id":str,"horse_id":str})
    old=pd.read_csv(args.existing_v1,dtype={"race_id":str})
    market=score_market_maxent(runners)
    if market.race_id.nunique()!=1253 or len(old)!=1253:
        raise ValueError("Not on exact frozen F3 2025 1253-race sample")
    target_cols=[
        "race_id","race_date","v1_raw_joint_logloss",
        "v1_raw_joint_brier","v1_raw_top10_hit","v1_raw_top10_mass",
        "market_pl_joint_logloss","market_pl_joint_brier",
        "market_pl_top10_hit","market_pl_top10_mass",
    ]
    merged=market.merge(old[target_cols],on=["race_id","race_date"],
                        validate="one_to_one",indicator=True)
    if len(merged)!=len(market) or not merged._merge.eq("both").all():
        raise ValueError("2025 full-race population mismatch")
    merged.drop(columns="_merge",inplace=True)
    names=("market_pl","market_maxent","v1_raw")
    columns=("joint_logloss","joint_brier","top10_hit","top10_mass")
    summary={
        name:{
            col:float(merged[name+"_"+col].mean()) for col in columns
        } for name in names
    }
    paired={}
    for first,second in (
        ("market_pl","market_maxent"),
        ("market_maxent","v1_raw"),
        ("market_pl","v1_raw"),
    ):
        for m in ("joint_logloss","joint_brier","top10_hit"):
            # For logloss/Brier, baseline-candidate positive means candidate
            # wins. For top10_hit, reverse interpretation: prefer negative.
            paired[first+"_minus_"+second+"_"+m]=paired_day_ci(
                merged,first+"_"+m,second+"_"+m,draws=2500
            )
    merged=merged.sort_values(["race_date","race_id"])
    args.out.mkdir(parents=True,exist_ok=True)
    merged.to_csv(args.out/"2025_same_family_joint_scores.csv",index=False)
    report={
        "status":"DESCRIPTIVE_2025_ALREADY_EXAMINED_NOT_FRESH_CONFIRMATION",
        "races":int(len(merged)),
        "individual_market_input":"frozen_pre2025_calibrated Harville top3",
        "same_exact_3_subset_maxentropy_used_for_market_and_f3":True,
        "market_prices":"historic final WIN odds, not actual trio odds and not T10",
        "summary":summary,"date_cluster_bootstrap":paired,
        "primary_check":"market_maxent_minus_v1_raw_joint_logloss",
        "no_fitted_parameters_on_2025":True,
        "trio_market_prices_observed":False,
    }
    (args.out/"2025_fair_market_comparison.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n"
    )
    print("2025_FAIR_MARKET_MAXENT="+json.dumps(report,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
