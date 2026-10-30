"""Frozen research C1 CPU learners on identical prior-only F3 and market.

LightGBM and CatBoost training receive the market logit as an offset.
No target-year fitting, no current result features, no live promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_information_expansion_v1 import KEYS, _sort_keyed
from run_input_learning_a_v1 import (
    check_reference,
    feature_registry,
    prepare_dataset,
    row_fingerprint,
)
from scipy.special import expit, logit

from keiba_place_lab.model_selection import (
    candidate_score,
    paired_delta,
    select_with_incumbent_one_se,
)

LIGHTGBM_PARAMS={
    "objective":"binary",
    "metric":"binary_logloss",
    "boost_from_average":False,
    "learning_rate":0.03,
    "num_leaves":15,
    "max_depth":4,
    "min_data_in_leaf":80,
    "lambda_l1":2.0,
    "lambda_l2":20.0,
    "feature_fraction":0.8,
    "bagging_fraction":0.8,
    "bagging_freq":1,
    "verbosity":-1,
    "num_threads":2,
    "seed":20261031,
    "deterministic":True,
    "force_col_wise":True,
}
CATBOOST_PARAMS={
    "iterations":120,
    "depth":4,
    "learning_rate":0.03,
    "l2_leaf_reg":20,
    "random_strength":0.5,
    "random_seed":20261031,
    "thread_count":2,
    "loss_function":"Logloss",
    "verbose":False,
    "allow_writing_files":False,
    "has_time":True,
}
CANDIDATES=("LightGBM_C1","CatBoost_C1")


def _market_margin(frame:pd.DataFrame)->np.ndarray:
    p=frame.p_market_oof.to_numpy(float)
    if not np.isfinite(p).all() or np.any((p<=0)|(p>=1)):
        raise ValueError("Missing chronological market-OOF probabilities")
    return logit(p)


def _categorical_input(frame:pd.DataFrame,
                       numeric:list[str],categorical:list[str])->pd.DataFrame:
    x=frame[numeric+categorical].copy()
    for col in numeric:
        x[col]=pd.to_numeric(x[col],errors="coerce").to_numpy(
            dtype=float,na_value=np.nan)
    for col in categorical:
        x[col]=x[col].astype("string").fillna("MISSING").astype(str)
    return x


def _race_weight(train:pd.DataFrame)->np.ndarray:
    counts=train.groupby("race_id").race_id.transform("size").to_numpy(float)
    w=1.0/counts
    w*=len(w)/w.sum()
    return w


def fit_predict_lightgbm(train:pd.DataFrame,target:pd.DataFrame,
                         numeric:list[str],categorical:list[str])->np.ndarray:
    import lightgbm as lgb
    from run_information_expansion_v1 import _build_pipeline

    if train.race_date.max()>=target.race_date.min():
        raise ValueError("LightGBM target-year label leak")
    # Exactly the PR32 XGBoost preprocessing, fit on training rows only.
    prep=_build_pipeline(numeric,categorical).named_steps["preprocess"]
    a=_categorical_input(train,numeric,categorical)
    b=_categorical_input(target,numeric,categorical)
    X=prep.fit_transform(a)
    Z=prep.transform(b)
    d=lgb.Dataset(
        X,label=train.top3_label.to_numpy(int),
        weight=_race_weight(train),init_score=_market_margin(train),
        free_raw_data=False)
    model=lgb.train(LIGHTGBM_PARAMS,train_set=d,num_boost_round=120)
    raw=np.asarray(model.predict(Z,raw_score=True,num_threads=2),float)
    # lgb predict(raw_score=True) returns tree-only score, not init_score.
    p=expit(raw+_market_margin(target))
    if np.any(~np.isfinite(p)) or np.any((p<=0)|(p>=1)):
        raise ValueError("Invalid LightGBM offset prediction")
    return p


def fit_predict_catboost(train:pd.DataFrame,target:pd.DataFrame,
                         numeric:list[str],categorical:list[str])->np.ndarray:
    from catboost import CatBoostClassifier, Pool

    if train.race_date.max()>=target.race_date.min():
        raise ValueError("CatBoost target-year label leak")
    tr=train.sort_values(["race_date","race_id","horse_no"],kind="stable")
    te=target.sort_values(["race_date","race_id","horse_no"],kind="stable")
    # Caller eval frame is sorted identically; fail before data alignment.
    if not te[KEYS].reset_index(drop=True).equals(
        target[KEYS].reset_index(drop=True)
    ):
        raise ValueError("Target must be sorted in strict chronological order")
    X=_categorical_input(tr,numeric,categorical)
    Z=_categorical_input(te,numeric,categorical)
    m_train=_market_margin(tr)
    m_test=_market_margin(te)
    fitpool=Pool(
        X,label=tr.top3_label.to_numpy(int),
        cat_features=categorical,
        baseline=m_train,weight=_race_weight(tr))
    learner=CatBoostClassifier(**CATBOOST_PARAMS)
    learner.fit(fitpool)
    # The CatBoost Python API is explicit: predict() on a Pool with baseline
    # should return raw model value + pool baseline; test both paths rather
    # than silently adding twice or silently dropping the offset.
    input_with=Pool(Z,cat_features=categorical,baseline=m_test)
    input_without=Pool(Z,cat_features=categorical)
    raw_with=np.asarray(learner.predict(input_with,
                                        prediction_type="RawFormulaVal"),float).reshape(-1)
    raw_without=np.asarray(learner.predict(input_without,
                                           prediction_type="RawFormulaVal"),float).reshape(-1)
    diff=raw_with-raw_without
    if np.allclose(diff,m_test,atol=1e-5,rtol=1e-5):
        raw=raw_with
    elif np.allclose(diff,0,atol=1e-7,rtol=0):
        raw=raw_without+m_test
    else:
        raise ValueError("CatBoost baseline additivity is not verified; stop scoring")
    p=expit(raw)
    if np.any(~np.isfinite(p)) or np.any((p<=0)|(p>=1)):
        raise ValueError("Invalid CatBoost offset probability")
    return p


def predict_outer(frame:pd.DataFrame,reference:pd.DataFrame)->pd.DataFrame:
    numeric,categorical=feature_registry()["F3_market_field"]
    source=_sort_keyed(frame.loc[frame.race_date.dt.year.isin([2023,2024])])
    rows=source[KEYS+["top3_label","surface","distance_m","race_class"]].copy()
    rows["p_M0"]=source.p_market_oof.to_numpy(float)
    for name in CANDIDATES:
        rows["p_"+name]=np.nan
    for year in (2023,2024):
        train=frame.loc[frame.race_date.dt.year.between(2018,year-1)].copy()
        test=_sort_keyed(source.loc[source.race_date.dt.year.eq(year)])
        if train.empty or test.empty:
            raise ValueError("Missing strict earlier-year training fold")
        mask=rows.race_date.dt.year.eq(year)
        if not rows.loc[mask,KEYS].reset_index(drop=True).equals(
            test[KEYS].reset_index(drop=True)
        ):
            raise ValueError("Fold row order changed")
        models={
            "LightGBM_C1":fit_predict_lightgbm,
            "CatBoost_C1":fit_predict_catboost,
        }
        for name,trainer in models.items():
            print(f"C1 {name} outer {year} training rows={len(train)}",flush=True)
            pred=trainer(train,test,numeric,categorical)
            rows.loc[mask,"p_"+name]=pred
    if rows[["p_"+x for x in CANDIDATES]].isna().any().any():
        raise ValueError("C1 predictions missing")
    if not rows[KEYS].equals(reference[KEYS]):
        raise ValueError("C1 does not match frozen PR32 evaluation cohort")
    if not np.allclose(rows.p_M0,reference.p_M0_market,atol=1e-10,rtol=0):
        raise ValueError("C1 market baseline drift")
    return rows


def evaluate(rows:pd.DataFrame,reference:pd.DataFrame,
             e5:Path,out:Path)->dict:
    out.mkdir(parents=True,exist_ok=True)
    rows.to_csv(out/"C_outer_oof_predictions.csv",index=False)
    p={"M0":rows.p_M0.to_numpy(float),
       "X2":reference.p_X2_nonlinear_strength.to_numpy(float)}
    p.update({name:rows["p_"+name].to_numpy(float) for name in CANDIDATES})
    full,score=select_with_incumbent_one_se(rows,p,incumbent="X2")
    score.to_csv(out/"C_all_years_scores.csv",index=False)
    e5d=pd.read_csv(e5,dtype={"race_id":str,"horse_id":str})
    e5d.race_date=pd.to_datetime(e5d.race_date)
    e5d=_sort_keyed(e5d)
    subset=_sort_keyed(rows.loc[rows.race_date.dt.year.eq(2024)])
    if not subset[KEYS].equals(e5d[KEYS]):
        raise ValueError("C1 E5 same-2024-row comparison failed")
    if not np.allclose(subset.p_M0,e5d.p_M0,atol=1e-10,rtol=0):
        raise ValueError("C1 E5 historical market mismatch")
    candidates={"M0":e5d.p_M0.to_numpy(float),
                "X2":e5d.p_X2.to_numpy(float),
                "E5":e5d.p_E5.to_numpy(float)}
    candidates.update({name:subset["p_"+name].to_numpy(float)
                       for name in CANDIDATES})
    verdict,table=select_with_incumbent_one_se(
        subset,candidates,incumbent="E5")
    table.to_csv(out/"C_2024_E5_comparison.csv",index=False)
    paired=[]
    for name,pred in candidates.items():
        d=paired_delta(subset,candidates["E5"],pred,
                       candidate_name="E5",reference_name=name)
        paired.append({"candidate":name,"e5_minus_candidate":d.mean_brier_delta,
                       "date_cluster_se":d.se_brier_delta,
                       "passes_one_se":bool(d.mean_brier_delta>d.se_brier_delta),
                       **candidate_score(subset,pred,name=name).__dict__})
    pd.DataFrame(paired).to_csv(out/"C_2024_paired.csv",index=False)
    return {"historical_2023_2024":{
                "winner":full.winner,"incumbent_retained":bool(full.incumbent_retained)},
            "2024_vs_E5":{
                "winner":verdict.winner,"incumbent_retained":bool(verdict.incumbent_retained)},
            "2024_fingerprint":row_fingerprint(subset)}


def main():
    p=argparse.ArgumentParser()
    for arg in ("historical-dir","results-csv","races-csv",
                "reference-oof","e5-2024","output-dir"):
        p.add_argument("--"+arg,type=Path,required=True)
    args=p.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    frame,audit=prepare_dataset(
        args.historical_dir,args.results_csv,args.races_csv)
    ref=check_reference(frame,args.reference_oof)
    rows=predict_outer(frame,ref)
    result=evaluate(rows,ref,args.e5_2024,out)
    manifest={
        "status":"HISTORICAL_FINAL_WIN_ODDS_DEVELOPMENT_ONLY",
        "pre_registration":"docs/MODEL_FAMILIES_C1_SPEC.md",
        "candidate_params":{"LightGBM":LIGHTGBM_PARAMS,"CatBoost":CATBOOST_PARAMS},
        "source_sha256":hashlib.sha256(args.reference_oof.read_bytes()).hexdigest(),
        "source_audit":audit,
        "result":result,
        "real_T_minus_10_test":False,
        "production_unchanged":True,
    }
    (out/"C_manifest.json").write_text(json.dumps(
        manifest,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
