"""Postprocess EXACT frozen F3 2025 independent holdout without any fit/retraining.

Works exclusively from source-verified 2025 scored predictions persisted by
validate_frozen_f3_2025.py. All 3-renpuku joint scores come from same immutable
ensemble; historical final market odds are not asserted available at T-10.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from research_trio_f3_v1 import paired_day_ci

MODELS={
    "market":"p_market_frozen_cal",
    "xgb400":"p_XGB400",
    "xgb800":"p_XGB800",
    "cat400":"p_CatBoost400",
    "f3_frozen_xc75_25":"p_XC_75_25_logit",
}


def summarize(runners:Path, trio:Path, audit:Path, output:Path)->dict:
    d=pd.read_csv(runners,dtype={"race_id":str,"horse_id":str})
    t=pd.read_csv(trio,dtype={"race_id":str})
    original=json.loads(audit.read_text())
    if not original["frozen_2026_preview_parity"]["exact_frozen_model_parity_max_abs"]:
        raise ValueError("No strict 2026 parity attestation")
    if original["2025_trio_model"]["valid_2025_full_races"]!=t.race_id.nunique():
        raise ValueError("Full-race 2025 sample count changed")
    if d.race_id.nunique()!=t.race_id.nunique() or d.duplicated(
        ["race_id","horse_no"]
    ).any():
        raise ValueError("Missing original 2025 prediction rows")
    d["race_date"]=pd.to_datetime(d.race_date)
    if not d.race_date.dt.year.eq(2025).all():
        raise ValueError("Future labels in independent 2025 test")
    actual=d.groupby("race_id").top3_label.sum()
    if not actual.eq(3).all():
        raise ValueError("All independently checked 2025 races need exactly 3 placers")
    metric={}
    for key,col in MODELS.items():
        p=pd.to_numeric(d[col],errors="raise").to_numpy(float)
        y=d.top3_label.to_numpy(float)
        if np.any((p<=0)|(p>=1)) or not np.isfinite(p).all():
            raise ValueError("Invalid frozen individual probability")
        d[key+"_brier"]=(p-y)**2
        d[key+"_logloss"]=-(y*np.log(p)+(1-y)*np.log1p(-p))
    groups={
        "race_date":"first",
        **{name+"_"+metric:"mean" for name in MODELS for metric in ("brier","logloss")},
    }
    by_race=d.groupby("race_id",sort=False).agg(groups).reset_index()
    by_race["race_date"]=by_race.race_date.dt.strftime("%Y-%m-%d")
    output.mkdir(parents=True,exist_ok=True)
    by_race.to_csv(output/"2025_racewise_top3_scores.csv",index=False)
    for name in MODELS:
        metric[name]={
            "race_macro_brier":float(by_race[name+"_brier"].mean()),
            "race_macro_logloss":float(by_race[name+"_logloss"].mean()),
            "runner_macro_brier":float(d[name+"_brier"].mean()),
            "runner_macro_logloss":float(d[name+"_logloss"].mean()),
        }
    f3="f3_frozen_xc75_25"
    paired={}
    for baseline in ("market","xgb800","cat400","xgb400"):
        for loss in ("brier","logloss"):
            paired[baseline+"_minus_frozen_"+loss]=paired_day_ci(
                by_race,baseline+"_"+loss,f3+"_"+loss,draws=2500
            )
    # Keep pre-declared calibration check simple and reproducible.
    d["pred_bin"]=pd.qcut(d[MODELS[f3]],q=10,labels=False,duplicates="drop")
    reliability=d.groupby("pred_bin",observed=True).agg(
        n=("race_id","size"),
        mean_p=(MODELS[f3],"mean"),
        empirical_top3=("top3_label","mean")
    ).reset_index()
    reliability.to_csv(output/"2025_frozen_f3_reliability.csv",index=False)
    # Distinguish the original holdout from a prospective T-minus-10 market.
    report={
        "status":"INDEPENDENT_2025_FROZEN_2024_END_800_400_HOLDOUT",
        "2018_2024_training_only":True,
        "2025_test_races":int(t.race_id.nunique()),
        "2025_test_runners":len(d),
        "2025_test_days":int(d.race_date.nunique()),
        "full_field_3_placers_each_race":True,
        "original_frozen_model_sha256":original["source_model_sha"],
        "exact_2026_forward_preview_parity":original["frozen_2026_preview_parity"],
        "market_data":"historical final win odds, NOT historical T-10",
        "model_metrics":metric,
        "f3_paired_date_cluster_95ci":paired,
        "trio_models":original["2025_trio_model"],
        "frozen_f3_unchanged":True,
        "model_retraining_count":0,
        "new_calibration_count":0,
        "live_betting_promotion":False,
        "limits":[
            "No archived pre-race 10-minute market snapshots for 2025.",
            "No historical all-ticket 3-renpuku prices or payouts were observed.",
            "Evaluation subset requires every starter has at least three prior JRA starts.",
            "Only races with exact original full source and three unique podium ranks included.",
        ]
    }
    (output/"2025_frozen_holdout_summary.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n"
    )
    print("FROZEN_F3_2025_PROPER_SCORES="+json.dumps({
        "races":report["2025_test_races"],
        "runners":len(d),
        "individual":metric,
        "trio":original["2025_trio_model"]["methods"],
        "trio_paired":original["2025_trio_model"]["paired_day_ci"],
    },ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    for name in ("runners","trio","audit","output"):
        p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args()
    summarize(a.runners,a.trio,a.audit,a.output)


if __name__=="__main__":
    main()
