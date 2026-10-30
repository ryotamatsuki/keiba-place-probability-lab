"""Matched-size CPU RealMLP vs 120-tree offset-XGB F3, feasibility only.

RealMLP is a direct classifier with market logit as input, unlike XGB's
explicit offset. Identical train/test and F3 information, but different loss
weighting; no formal best-model claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_information_expansion_v1 import _build_pipeline, _fit_xgb, _sort_keyed
from run_input_learning_a_v1 import (
    check_reference,
    feature_registry,
    prepare_dataset,
)
from sklearn.impute import SimpleImputer

from keiba_place_lab.model_selection import (
    candidate_score,
    paired_delta,
    select_with_incumbent_one_se,
)

SEED=20261031
TRAIN_RACES=1500
TEST_RACES=250
KEYS=["race_date","race_id","horse_id","horse_no"]


def cohorts(frame:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    train_pool=frame.loc[frame.race_date.dt.year.between(2018,2022)]
    eval_pool=frame.loc[frame.race_date.dt.year.eq(2023)]
    def take(df:pd.DataFrame,n:int,*,latest:bool)->pd.DataFrame:
        races=df[["race_date","race_id"]].drop_duplicates().sort_values(
            ["race_date","race_id"],kind="stable")
        if len(races)<n:
            raise ValueError("Insufficient full races for fixed pilot cohort")
        ids=set(races.tail(n).race_id if latest else races.head(n).race_id)
        out=_sort_keyed(df.loc[df.race_id.isin(ids)].copy())
        if out.race_id.nunique()!=n:
            raise ValueError("Race-only pilot selection broke cohort")
        return out
    tr=take(train_pool,TRAIN_RACES,latest=True)
    te=take(eval_pool,TEST_RACES,latest=False)
    if tr.race_date.max()>=te.race_date.min():
        raise ValueError("Pilot train/eval dates overlap")
    return tr,te


def prepare_realmlp(train:pd.DataFrame,test:pd.DataFrame,
                    numeric:list[str],categorical:list[str]
                    )->tuple[np.ndarray,np.ndarray]:
    # Preserve exactly XGB F3 categories and train-fit one-hot categories,
    # then train-fit RealMLP-specific median numeric imputation.
    a=train[numeric+categorical].copy()
    b=test[numeric+categorical].copy()
    for col in numeric:
        a[col]=pd.to_numeric(a[col],errors="coerce").to_numpy(
            dtype=float,na_value=np.nan)
        b[col]=pd.to_numeric(b[col],errors="coerce").to_numpy(
            dtype=float,na_value=np.nan)
    for col in categorical:
        a[col]=a[col].astype("object").where(a[col].notna(),None)
        b[col]=b[col].astype("object").where(b[col].notna(),None)
    prep=_build_pipeline(numeric,categorical).named_steps["preprocess"]
    x=prep.fit_transform(a)
    z=prep.transform(b)
    impute=SimpleImputer(strategy="median",keep_empty_features=True)
    x=impute.fit_transform(x).astype(np.float32)
    z=impute.transform(z).astype(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError("RealMLP numerical matrix still has missing values")
    return x,z


def run(frame:pd.DataFrame,reference:pd.DataFrame,out:Path)->dict:
    from pytabkit import RealMLP_TD_Classifier

    train,test=cohorts(frame)
    numeric,categorical=feature_registry()["F3_market_field"]
    x,z=prepare_realmlp(train,test,numeric,categorical)
    learner=RealMLP_TD_Classifier(
        device="cpu",random_state=SEED,n_cv=1,n_refit=0,
        n_epochs=32,hidden_sizes=[128,128],batch_size=1024,
        use_ls=False,val_metric_name="brier",n_threads=2,
        use_early_stopping=False)
    print(f"RealMLP pilot CPU training={len(train)}, test={len(test)}",flush=True)
    learner.fit(x,train.top3_label.to_numpy(int))
    p_mlp=np.asarray(learner.predict_proba(z)[:,1],dtype=float)
    if not np.isfinite(p_mlp).all() or np.any((p_mlp<=0)|(p_mlp>=1)):
        raise ValueError("RealMLP failed finite probability contract")
    p_xgb=_fit_xgb(train,test,numeric,categorical)
    ref=reference.loc[
        reference.race_date.dt.year.eq(2023)
    ].sort_values(KEYS,kind="stable").reset_index(drop=True)
    select=ref.merge(test[KEYS],on=KEYS,how="inner",
                     validate="one_to_one")
    if len(select)!=len(test):
        raise ValueError("Pilot target cohort differs from frozen PR32")
    paired=test[KEYS].merge(ref,on=KEYS,how="left",
                             validate="one_to_one",sort=False)
    if len(paired)!=len(test) or paired.p_X2_nonlinear_strength.isna().any():
        raise ValueError("Original X2 2023 OOF missing pilot runners")
    if not test.top3_label.astype(int).equals(paired.top3_label.astype(int)):
        raise ValueError("Pilot historical outcomes do not match PR32")
    preds={
        "M0":test.p_market_oof.to_numpy(float),
        "X2_full_history":paired.p_X2_nonlinear_strength.to_numpy(float),
        "XGB_1500_races":p_xgb,
        "RealMLP_1500_races":p_mlp,
    }
    if not np.allclose(preds["M0"],paired.p_M0_market,atol=1e-10,rtol=0):
        raise ValueError("Pilot M0 baseline differs from historical OOF")
    out.mkdir(parents=True,exist_ok=True)
    scores=[candidate_score(test,p,name=name).__dict__
            for name,p in preds.items()]
    pd.DataFrame(scores).to_csv(out/"RealMLP_pilot_scores.csv",index=False)
    p=test[KEYS+["top3_label","surface","distance_m","race_class"]].copy()
    differences=[]
    for name,pred in preds.items():
        p["p_"+name]=pred
        delta=paired_delta(test,p_xgb,pred,
                           candidate_name="XGB_1500_races",reference_name=name)
        differences.append({"candidate":name,
                            "XGB_minus_candidate_brier":delta.mean_brier_delta,
                            "date_cluster_se":delta.se_brier_delta})
    p.to_csv(out/"RealMLP_pilot_predictions.csv",index=False)
    pd.DataFrame(differences).to_csv(out/"RealMLP_pilot_paired.csv",index=False)
    result={
        "status":"CPU_FEASIBILITY_ONLY_NOT_FULL_SCOPE_SELECTION",
        "training_rows":len(train),"training_races":train.race_id.nunique(),
        "evaluation_rows":len(test),"evaluation_races":test.race_id.nunique(),
        "training_last_date":str(train.race_date.max().date()),
        "test_first_date":str(test.race_date.min().date()),
        "realmlp_epochs":32,
        "network":[128,128],
        "source_F3_numeric":len(numeric),
        "source_F3_categorical":categorical,
        "different_objective_weights":"RealMLP direct classifier; matched XGB offset race weighting",
        "scores":scores,
        "no_model_selection":True,
        "not_t_minus_10":True}
    (out/"RealMLP_pilot_manifest.json").write_text(json.dumps(
        result,indent=2,ensure_ascii=False)+"\n")
    return result


def main():
    p=argparse.ArgumentParser()
    for a in ("historical-dir","results-csv","races-csv",
              "reference-oof","output-dir"):
        p.add_argument("--"+a,type=Path,required=True)
    args=p.parse_args()
    frame,_=prepare_dataset(
        args.historical_dir,args.results_csv,args.races_csv)
    ref=check_reference(frame,args.reference_oof)
    print(json.dumps(run(frame,ref,args.output_dir),indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
