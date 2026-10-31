"""Preregistered three-depth 400-tree XGBoost F3-input exploration on genuine OOF.

Train cutoff: 2023 folds end 2022; selected previous-year family refit through
2023 for 2024; no results from 2024 or 2025 in model/preprocessing/tuning.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from freeze_tennosho_f3_20261101 import numeric_categorical
from research_trio_f3_v1 import paired_day_ci
from run_information_expansion_v1 import _build_pipeline
from run_input_learning_a_v1 import feature_registry, prepare_dataset
from run_model_families_c1 import _race_weight
from scipy.special import logit

from keiba_place_lab.trio_pool_v4 import v1_joint
from keiba_place_lab.trio_residual_v2 import score_probabilities

SPECS={
    "prior_400":(2,60,20.),
    "depth3":(3,5,10.),
    "depth4":(4,10,20.),
}


def learn_predict(train:pd.DataFrame,test:pd.DataFrame,spec:tuple
                  )->tuple[np.ndarray,float]:
    started=time.perf_counter()
    numeric,cat=feature_registry()["F3_market_field"]
    train=train.sort_values(["race_date","race_id","horse_id","horse_no"]).reset_index(drop=True)
    mat=numeric_categorical(train,numeric,cat)
    candidates=numeric_categorical(test,numeric,cat)
    pipe=_build_pipeline(numeric,cat)
    depth,child,reg_lambda=spec
    pipe.named_steps["model"].set_params(
        n_estimators=400,max_depth=depth,min_child_weight=child,reg_lambda=reg_lambda,
        reg_alpha=2.0,learning_rate=.03,n_jobs=2,
    )
    x=train.p_market_oof.to_numpy(float)
    z=test.p_market_oof.to_numpy(float)
    if np.any((x<=0)|(x>=1)) or np.any((z<=0)|(z>=1)):
        raise ValueError("No as-of-market OOF for XGB comparison")
    pipe.fit(mat,train.top3_label.to_numpy(int),
             model__sample_weight=_race_weight(train),model__base_margin=logit(x))
    features=pipe.named_steps["preprocess"].transform(candidates)
    prob=pipe.named_steps["model"].predict_proba(
        features,base_margin=logit(z)
    )[:,1]
    if len(prob)!=len(test) or not np.isfinite(prob).all():
        raise ValueError("XGB candidate missing some chronological rows")
    return prob,time.perf_counter()-started


def score_complete(test:pd.DataFrame,ref:pd.DataFrame,prediction:np.ndarray,
                   *,name:str)->tuple[pd.DataFrame,pd.DataFrame]:
    rows=test[["race_id","race_date","horse_no","field_size","top3_label"]].copy()
    rows["prediction"]=prediction
    keys=set(ref.race_id.astype(str))
    if len(keys)!=len(ref) or not keys.issubset(set(rows.race_id)):
        raise ValueError("Candidate missing PR40 full race")
    records=[]
    all_tickets=[]
    for rid,g in rows.loc[rows.race_id.isin(keys)].groupby("race_id",sort=True):
        g=g.sort_values("horse_no").reset_index(drop=True)
        h=g.horse_no.to_numpy(int)
        if len(g)!=int(g.field_size.iloc[0]) or int(g.top3_label.sum())!=3:
            raise ValueError("Cannot change complete race cohort by candidate")
        triples,prob=v1_joint(g.prediction.to_numpy(float),h)
        winners=tuple(sorted(h[g.top3_label.to_numpy(int).astype(bool)]))
        actual=np.flatnonzero(np.all(triples==winners,axis=1))
        if len(actual)!=1:
            raise ValueError("2024 actual trio missing")
        s=score_probabilities(prob,int(actual[0]))
        records.append({
            "race_id":rid,"race_date":str(g.race_date.iloc[0].date()),
            **{name+"_"+k:v for k,v in s.items()}
        })
        for (h1,h2,h3),p in zip(triples,prob,strict=True):
            all_tickets.append({
                "race_id":rid,"horse_1":int(h1),"horse_2":int(h2),
                "horse_3":int(h3),name:float(p),
            })
    if len(records)!=len(ref):
        raise ValueError("Model silently reduced eligible race population")
    return pd.DataFrame(records),pd.DataFrame(all_tickets)


def main():
    p=argparse.ArgumentParser()
    for k in ("historical-dir","results","races","pr40-2023","pr40-2024",
              "reference-oof","out"):
        p.add_argument("--"+k,required=True,type=Path)
    a=p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError("Immutable preregistered depth study")
    began=time.perf_counter()
    source,audit=prepare_dataset(a.historical_dir,a.results,a.races)
    ref23=pd.read_csv(a.pr40_2023,dtype={"race_id":str})
    ref24=pd.read_csv(a.pr40_2024,dtype={"race_id":str})
    old=pd.read_csv(a.reference_oof,dtype={"race_id":str,"horse_id":str})
    if len(ref23)!=1355 or len(ref24)!=1336:
        raise ValueError("Missing identical prior study reference cohorts")
    res23=source.loc[source.race_date.dt.year.eq(2023)].copy()
    f23=source.loc[source.race_date.dt.year.between(2018,2022)].copy()
    if f23.race_date.max()>=res23.race_date.min():
        raise ValueError("2023 development model leaks 2023 labels")
    scores=[]
    for name,spec in SPECS.items():
        prediction,seconds=learn_predict(f23,res23,spec)
        result,_=score_complete(res23,ref23,prediction,name=name)
        score=float(result[name+"_joint_logloss"].mean())
        scores.append({
            "candidate":name,"parameters":spec,
            "2023_joint_logloss":score,"2023_fit_seconds":seconds,
        })
        print("XGB_V4_2023_PRECHOICE "+json.dumps(scores[-1]),flush=True)
    # Only 2023 loss selects the sole candidate allowed into 2024
    selected=min(scores,key=lambda x:(x["2023_joint_logloss"],x["candidate"]))
    f24=source.loc[source.race_date.dt.year.between(2018,2023)].copy()
    res24=source.loc[source.race_date.dt.year.eq(2024)].copy()
    pred24,sec24=learn_predict(f24,res24,SPECS[selected["candidate"]])
    race24,ticket24=score_complete(
        res24,ref24,pred24,name=selected["candidate"]
    )
    ref=ref24.set_index("race_id")
    baseline=old.loc[pd.to_datetime(old.race_date).dt.year.eq(2024)].copy()
    baseline=baseline[["race_id","horse_id","horse_no","p_matched_f3"]]
    exact=res24[["race_id","horse_id","horse_no","race_date","field_size",
                 "top3_label"]].copy()
    exact=exact.merge(baseline,on=["race_id","horse_id","horse_no"],
                      how="inner",validate="one_to_one")
    if len(exact)!=len(res24):
        raise ValueError("Original PR40 F3 marginal predictions not complete")
    incumbent,_=score_complete(exact,ref24,exact.p_matched_f3.to_numpy(float),
                                name="incumbent")
    final=race24.merge(incumbent,on=["race_id","race_date"],validate="one_to_one")
    error=max(abs(row.incumbent_joint_logloss-
                  float(ref.loc[row.race_id,"v1_joint_logloss"]))
              for row in final.itertuples(index=False))
    if error>5e-6:
        raise ValueError("Selected depth study modified original baseline")
    winner=selected["candidate"]
    result={
        "status":"2024_DEVELOPMENT_PRIOR_ONLY_XGB_400_DEPTH_EXPLORATION",
        "candidates_frozen_before_execution":SPECS,
        "training_years":{"selection_fit":"2018-2022",
                          "selection":"2023","evaluation_fit":"2018-2023",
                          "evaluation":"2024"},
        "2023_select_scores":scores,
        "best_2023_configuration":selected,
        "2024_selected_joint_nll":float(final[winner+"_joint_logloss"].mean()),
        "2024_incumbent_joint_nll":float(final.incumbent_joint_logloss.mean()),
        "2024_paired_cluster_bootstrap":paired_day_ci(
            final,"incumbent_joint_logloss",winner+"_joint_logloss",draws=2500
        ),
        "2024_top10_paired":paired_day_ci(
            final,"incumbent_top10_hit",winner+"_top10_hit",draws=2500),
        "2024_joint_brier_delta":float(
            final.incumbent_joint_brier.mean()-
            final[winner+"_joint_brier"].mean()
        ),
        "evaluation_races":len(final),
        "eval_train_seconds":sec24,
        "total_runtime_seconds":time.perf_counter()-began,
        "frozen_F3_weight_changes":0,
        "source":audit["source_sha256"],
    }
    a.out.mkdir(parents=True,exist_ok=True)
    final.to_csv(a.out/"2024_selected_depth_racewise.csv",index=False)
    ticket24.to_csv(a.out/"2024_selected_depth_tickets.csv.gz",
                    compression="gzip",index=False)
    (a.out/"depth_experiment.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False)+"\n"
    )
    print("XGB_DEPTH_V4_SELECTED="+json.dumps({
        "candidate":winner,"score":result["2024_selected_joint_nll"],
        "paired":result["2024_paired_cluster_bootstrap"]
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
