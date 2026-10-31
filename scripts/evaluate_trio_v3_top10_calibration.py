"""Ex-post 2024 trio top-K coverage significance and calibration.

Uses unchanged model scores from matched 800/400 2023->2024 experiment.
No model training or selection on test year.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from research_trio_f3_v1 import paired_day_ci

CANDIDATES=("market_pl","market_maxent","v1","v2_matched","v3_context")


def evaluate(source:Path, output:Path)->dict:
    d=pd.read_csv(source,dtype={"race_id":str})
    if len(d)!=1336 or d.race_id.nunique()!=1336:
        raise ValueError("Historical strict 2024 800/400 cohort differs from source")
    if d.race_date.str.slice(0,4).ne("2024").any():
        raise ValueError("Calibration cannot include 2023 training scores")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Never overwrite immutable research calibration")
    records=[]
    summary={}
    for name in CANDIDATES:
        mass=d[name+"_top10_mass"].astype(float)
        hit=d[name+"_top10_hit"].astype(int)
        if not mass.between(0,1).all() or not hit.isin([0,1]).all():
            raise ValueError("Invalid top10 model probability or actual outcome")
        df=pd.DataFrame({
            "predicted_top10_mass":mass,
            "hit_top10":hit,
            "probability_bin":pd.qcut(mass,10,duplicates="drop"),
        })
        grp=df.groupby("probability_bin",observed=True)
        bucket=grp.agg(
            n=("hit_top10","size"),
            predicted=("predicted_top10_mass","mean"),
            empirical=("hit_top10","mean"),
        ).reset_index()
        bucket["candidate"]=name
        bucket["absolute_calibration_error"]=(bucket.empirical-bucket.predicted).abs()
        ece=float((bucket.n*bucket.absolute_calibration_error).sum()/len(df))
        records.append(bucket)
        summary[name]={
            "observed_top10_coverage":float(hit.mean()),
            "predicted_top10_mass":float(mass.mean()),
            "top10_mean_calibration_gap":float(hit.mean()-mass.mean()),
            "top10_ece_10_bins":ece,
            "top10_binary_brier":float(((mass-hit)**2).mean()),
        }
    paired={}
    for name in ("market_maxent","v2_matched","v3_context"):
        # negative means the V1 comparator has higher top10 hit coverage.
        v=paired_day_ci(d,name+"_top10_hit","v1_top10_hit",draws=2500)
        paired[name+"_minus_v1_top10_hit"]=v
    for name in ("v2_matched","v3_context"):
        v=paired_day_ci(d,"v1_top10_hit",name+"_top10_hit",draws=2500)
        paired["v1_minus_"+name+"_top10_hit"]=v
    output.mkdir(parents=True,exist_ok=True)
    pd.concat(records,ignore_index=True).to_csv(
        output/"2024_top10_calibration_deciles.csv",index=False,
    )
    obj={
        "status":"2024_OOT_DEVELOPMENT_TOP10_COVERAGE_AND_CALIBRATION",
        "races":len(d),"already_observed_2024_not_fresh_test":True,
        "predicted_mass_is_probability_true_trio_is_among_top10":True,
        "metrics":summary,"paired_date_bootstrap":paired,
        "source_data":str(source),
        "no_predictions_retrained":True,
    }
    (output/"2024_top10_metrics.json").write_text(
        json.dumps(obj,indent=2,ensure_ascii=False)+"\n"
    )
    print("TRIO_V3_2024_TOP10_DIAGNOSTICS="+json.dumps(obj),flush=True)
    return obj


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    evaluate(a.source,a.out)


if __name__=="__main__":
    main()
