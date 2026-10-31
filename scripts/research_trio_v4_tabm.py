"""Official TabM 0.0.3 as a genuine F3-input horse-level model, chronological.

Train independent member BCE on prior-only F3 numeric/categorical input.
Each member learns additive residual to frozen-method logit(market OOF).
Use mean of MEMBER probabilities at prediction, then exact-3 V1 joint.
Never call this a direct-set-loss learner: TabM here optimizes horse BCE.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tabm
import torch
from scipy.special import logit
from sklearn.metrics import log_loss

from research_trio_f3_v1 import paired_day_ci
from run_input_learning_a_v1 import feature_registry, prepare_dataset

from keiba_place_lab.trio_pool_v4 import v1_joint
from keiba_place_lab.trio_residual_v2 import score_probabilities

SEED=20261031
EPOCHS=12
BATCH=512


class PreviousYearFeatures:
    def fit(self,train:pd.DataFrame,numerics:list[str],categories:list[str]):
        self.numeric=list(numerics)
        self.categories=list(categories)
        n=train[self.numeric].apply(pd.to_numeric,errors="coerce").to_numpy(float)
        n[~np.isfinite(n)]=np.nan
        self.median=np.nanmedian(n,axis=0)
        self.median=np.nan_to_num(self.median,nan=0.)
        filled=np.where(np.isfinite(n),n,self.median)
        self.mean=np.mean(filled,axis=0)
        self.std=np.maximum(np.std(filled,axis=0),1e-3)
        self.vocab={}
        for name in self.categories:
            vals=train[name].astype("string").fillna("MISSING").astype(str)
            self.vocab[name]={value:i+1 for i,value in enumerate(sorted(vals.unique()))}
        return self

    def transform(self,d:pd.DataFrame)->tuple[torch.Tensor,torch.Tensor]:
        x=d[self.numeric].apply(pd.to_numeric,errors="coerce").to_numpy(float)
        x[~np.isfinite(x)]=np.nan
        x=np.where(np.isfinite(x),x,self.median)
        x=np.clip((x-self.mean)/self.std,-8,8).astype(np.float32)
        cats=np.column_stack([
            d[k].astype("string").fillna("MISSING").astype(str).map(
                self.vocab[k]).fillna(0).astype(int).to_numpy()
            for k in self.categories
        ])
        return torch.from_numpy(x),torch.tensor(cats,dtype=torch.long)

    def cardinalities(self)->list[int]:
        return [len(self.vocab[k])+1 for k in self.categories]


def data_tensors(
    df:pd.DataFrame,prior:PreviousYearFeatures
)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    num,cat=prior.transform(df)
    y=torch.from_numpy(df.top3_label.to_numpy(np.float32))
    offset=torch.from_numpy(logit(df.p_market_oof.to_numpy(float)).astype(np.float32))
    n=df.groupby("race_id").race_id.transform("size").to_numpy(float)
    w=(1./n).astype(np.float32)
    w*=len(w)/w.sum()
    return num,cat,y,offset,torch.from_numpy(w)


def learn_tabm(train:pd.DataFrame,eval_frame:pd.DataFrame,
               *,epochs:int,validation:pd.DataFrame|None=None
               )->tuple[np.ndarray,dict]:
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    numerics,categories=feature_registry()["F3_market_field"]
    prior=PreviousYearFeatures().fit(train,numerics,categories)
    x,c,y,off,w=data_tensors(train,prior)
    model=tabm.TabM.make(
        n_num_features=len(numerics),cat_cardinalities=prior.cardinalities(),
        d_out=1,k=4,n_blocks=2,d_block=64,dropout=.10,
    )
    optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    xt,ct,yt,ot,_wt=data_tensors(eval_frame,prior)
    rng=np.random.default_rng(SEED)
    best_epoch, best_loss=1,float("inf")
    history=[]
    if validation is not None:
        xv,cv,yv,ov,_wv=data_tensors(validation,prior)
    begin=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train()
        ids=rng.permutation(len(x))
        for batch in np.array_split(ids,max(1,int(np.ceil(len(ids)/BATCH)))):
            index=torch.from_numpy(batch)
            z=model(x[index],c[index]).squeeze(-1)
            truth=y[index,None].expand_as(z)
            offsets=off[index,None].expand_as(z)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(
                z+offsets,truth,reduction="none"
            )
            loss=(loss.mean(dim=1)*w[index]).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optim.step()
        model.eval()
        with torch.no_grad():
            if validation is not None:
                chunks=[]
                for a in range(0,len(xv),2048):
                    logits=model(xv[a:a+2048],cv[a:a+2048]).squeeze(-1)
                    chunks.append(torch.sigmoid(logits+ov[a:a+2048,None]).mean(1))
                pp=torch.cat(chunks).numpy()
                score=float(log_loss(yv.numpy(),np.clip(pp,1e-7,1-1e-7)))
            else:
                score=float("nan")
        history.append({"epoch":epoch,"validation_binary_logloss":score})
        if validation is not None and score<best_loss:
            best_loss,best_epoch=score,epoch
        print("TABM_EPOCH "+json.dumps({
            "trained":len(train),"epoch":epoch,
            "validation_binary_nll":score,
        }),flush=True)
    model.eval()
    with torch.no_grad():
        chunks=[]
        for a in range(0,len(xt),2048):
            logits=model(xt[a:a+2048],ct[a:a+2048]).squeeze(-1)
            pp=torch.sigmoid(logits+ot[a:a+2048,None]).mean(1)
            chunks.append(pp)
        result=torch.cat(chunks).numpy()
    if len(result)!=len(eval_frame) or not np.isfinite(result).all():
        raise ValueError("Official TabM missing some held-out F3 rows")
    return np.clip(result,1e-7,1-1e-7),{
        "epochs":epochs,"best_validation_epoch":best_epoch,
        "last_validation_loss":best_loss if validation is not None else None,
        "training_rows":len(train),"prediction_rows":len(eval_frame),
        "numeric_features":len(numerics),"categorical_features":len(categories),
        "num_model_members":4,"train_seconds":time.perf_counter()-begin,
        "preprocessing_fit_max_date":str(pd.to_datetime(train.race_date).max().date()),
        "training_history":history,
    }


def score_cohort(frame:pd.DataFrame,ref:pd.DataFrame,year:int
                 )->tuple[pd.DataFrame,pd.DataFrame]:
    records=[]
    full=[]
    keep=set(ref.race_id.astype(str))
    subset=frame.loc[frame.race_id.isin(keep)].copy()
    if len(keep)!=len(ref) or set(subset.race_id)!=keep:
        raise ValueError("TabM cohort must be PR40 full-field race IDs")
    for rid,g in subset.groupby("race_id",sort=True):
        g=g.sort_values("horse_no").reset_index(drop=True)
        horses=g.horse_no.to_numpy(int)
        if (len(g)!=int(g.field_size.iloc[0]) or
                int(g.top3_label.sum())!=3 or
                pd.to_datetime(g.race_date.iloc[0]).year!=year):
            raise ValueError("TabM prediction lacks complete identical race")
        idx,prob=v1_joint(g.p_tabm.to_numpy(float),horses)
        ids,base=v1_joint(g.p_matched_f3.to_numpy(float),horses)
        if not np.array_equal(idx,ids):
            raise ValueError("TabM and F3 ticket keys differ")
        actual=tuple(sorted(horses[g.top3_label.to_numpy(int).astype(bool)]))
        truth=np.flatnonzero(np.all(idx==actual,axis=1))
        if len(truth)!=1:
            raise ValueError("TabM true trio not found")
        metrics=score_probabilities(prob,int(truth[0]))
        row={"race_id":rid,"race_date":str(g.race_date.iloc[0].date()),
             "field_size":len(g)}
        row.update({"tabm_"+k:v for k,v in metrics.items()})
        row.update({"incumbent_"+k:v for k,v in
                    score_probabilities(base,int(truth[0])).items()})
        records.append(row)
        for (h1,h2,h3),p,q in zip(idx,prob,base,strict=True):
            full.append({"race_id":rid,"horse_1":int(h1),"horse_2":int(h2),
                         "horse_3":int(h3),"tabm":float(p),"incumbent":float(q)})
    return pd.DataFrame(records),pd.DataFrame(full)


def main():
    p=argparse.ArgumentParser()
    for k in ("historical-dir","results","races","reference-oof",
              "pr40-2023","pr40-2024","out"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError("TabM evaluation cannot overwrite any existing research")
    start=time.perf_counter()
    data,audit=prepare_dataset(a.historical_dir,a.results,a.races)
    oof=pd.read_csv(a.reference_oof,dtype={"race_id":str,"horse_id":str})
    refs={}
    for year in (2023,2024):
        refs[year]=pd.read_csv(getattr(a,f"pr40_{year}"),dtype={"race_id":str})
    results={}
    predictions=[]
    all_scores={}
    for year in (2023,2024):
        target=data.loc[data.race_date.dt.year.eq(year)].copy()
        if target.empty or target.p_market_oof.isna().any():
            raise ValueError("TabM prediction year features missing")
        train=data.loc[data.race_date.dt.year.between(2018,year-1)].copy()
        # Select ONLY number of epochs using training years: final prior
        # calendar year is validation; no external target-year tuning.
        validation=train.loc[train.race_date.dt.year.eq(year-1)].copy()
        history=train.loc[train.race_date.dt.year.lt(year-1)].copy()
        if len(history)<20000 or len(validation)<20000:
            raise ValueError("Insufficient past-only TabM validation")
        _,tuner=learn_tabm(history,validation,epochs=EPOCHS,
                           validation=validation)
        best=tuner["best_validation_epoch"]
        pred,model_info=learn_tabm(train,target,epochs=best)
        target["p_tabm"]=pred
        joincols=["race_id","horse_id","horse_no","p_tabm"]
        vals=target[joincols].merge(oof.loc[
            pd.to_datetime(oof.race_date).dt.year.eq(year)
        ].drop(columns="p_tabm",errors="ignore"),
                                  on=["race_id","horse_id","horse_no"],
                                  how="inner",validate="one_to_one")
        ref_keys=target[["race_id","horse_id","horse_no"]]
        if len(vals)!=len(ref_keys) or vals.duplicated(
                ["race_id","horse_id","horse_no"]).any():
            raise ValueError("Full matched 2023/24 TabM OOF rows missing")
        vals["race_date"]=pd.to_datetime(vals.race_date)
        race_scores,tickets=score_cohort(vals,refs[year],year)
        ref_score=refs[year].set_index("race_id")
        maximum=max(abs(float(x.incumbent_joint_logloss)-
                        float(ref_score.loc[x.race_id,"v1_joint_logloss"]))
                    for x in race_scores.itertuples(index=False)
                    if "v1_joint_logloss" in ref_score.columns)
        if year==2024 and maximum>5e-6:
            raise ValueError("TabM comparator altered existing 2024 F3+V1")
        predictions.append(vals[["race_id","race_date","horse_id","horse_no","p_tabm"]])
        results[year]={"tuning":tuner,"fit":model_info,"scores":{
            "tabm_joint_logloss":float(race_scores.tabm_joint_logloss.mean()),
            "incumbent_joint_logloss":float(race_scores.incumbent_joint_logloss.mean()),
            "top10_hit":float(race_scores.tabm_top10_hit.mean()),
        }}
        all_scores[year]=(race_scores,tickets)
        print("TABM_OOF_YEAR="+json.dumps({
            "year":year,"full_races":len(race_scores),
            "score":results[year]["scores"],
        }),flush=True)
    s24,t24=all_scores[2024]
    result={
        "status":"TABM_OFFICIAL_0.0.3_BCE_THEN_EXACT3_V1_2024_DEVELOPMENT",
        "f3_input_only":True,"member_loss":"mean member BCE, not BCE(mean output)",
        "individual_prediction":"mean of member sigmoid(logit+market_offset)",
        "source_package":"yandex-research/tabm Apache-2.0",
        "tabm_version":tabm.__version__,"torch_version":torch.__version__,
        "python":platform.python_version(),"seed":SEED,"cpu_threads":2,
        "train_info":results,
        "2024_incumbent_minus_tabm_ci":paired_day_ci(
            s24,"incumbent_joint_logloss","tabm_joint_logloss",draws=2500),
        "2024_top10_ci":paired_day_ci(
            s24,"incumbent_top10_hit","tabm_top10_hit",draws=2500),
        "runtime_seconds":time.perf_counter()-start,
        "source_original":audit["source_sha256"],
        "no_model_promotion":True,
        "2024_not_pristine_holdout":True,
    }
    a.out.mkdir(parents=True,exist_ok=True)
    s24.to_csv(a.out/"2024_tabm_racewise.csv",index=False)
    t24.to_csv(a.out/"2024_tabm_all_tickets.csv.gz",
               index=False,compression="gzip")
    all_scores[2023][0].to_csv(a.out/"2023_tabm_oof_racewise.csv",index=False)
    all_scores[2023][1].to_csv(a.out/"2023_tabm_oof_all_tickets.csv.gz",
                             index=False,compression="gzip")
    pd.concat(predictions).to_csv(a.out/"2023_2024_tabm_horse_oof.csv",index=False)
    (a.out/"tabm_experiment.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("TABM_RESULT="+json.dumps({
        "score":results,"validation":result["2024_incumbent_minus_tabm_ci"]
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
