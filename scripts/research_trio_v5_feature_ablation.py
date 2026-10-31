"""Empirical V5 feature-block ablation on strict PR40 2023->2024 F3 OOF.

Train only 2023 EARLY labels, select regularization/block on 2023 LATE
race-macro 3-subset NLL, refit all 2023, score exact 2024 1336 race cohort.
Not XGB800+Cat400 retraining. Not an untouched independent holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from research_trio_f3_v1 import paired_day_ci
from run_input_learning_a_v1 import prepare_dataset

from keiba_place_lab.trio_feature_v5 import FEATURE_BLOCKS, engineer_v5_asof
from keiba_place_lab.trio_pool_v4 import v1_joint
from keiba_place_lab.trio_residual_v2 import score_probabilities

SEED=20261031
C_VALUES=(.01,.1,1.)
ALL_CANDIDATES=("offset_only","agents","dynamics","pace","all")


class OffsetResidual:
    """Unchanged F3 OOF marginal + regularized prior-year-only logistic residual."""

    def __init__(self, block:str, c:float):
        if block not in FEATURE_BLOCKS or c not in C_VALUES:
            raise ValueError("Unregistered feature block or regularization strength")
        self.block=block
        self.c=float(c)

    def _matrix(self,x:pd.DataFrame)->np.ndarray:
        cols=FEATURE_BLOCKS[self.block]
        if not cols:
            return np.ones((len(x),1),dtype=float)
        raw=x[list(cols)].apply(pd.to_numeric,errors="coerce").to_numpy(float)
        raw[~np.isfinite(raw)]=np.nan
        if not hasattr(self,"medians"):
            if np.isnan(raw).all():
                raise ValueError("All feature values missing")
            self.medians=np.nanmedian(raw,axis=0)
            self.medians=np.nan_to_num(self.medians,nan=0)
            z=np.where(np.isfinite(raw),raw,self.medians)
            self.center=z.mean(axis=0)
            self.scale=np.maximum(z.std(axis=0),1e-5)
        z=np.where(np.isfinite(raw),raw,self.medians)
        standardized=np.clip((z-self.center)/self.scale,-7.,7.)
        missing=(~np.isfinite(raw)).astype(float)
        return np.column_stack([np.ones(len(x)),standardized,missing])

    def fit(self,df:pd.DataFrame):
        X=self._matrix(df)
        f3=df.p_matched_f3.to_numpy(float)
        y=df.top3_label.to_numpy(float)
        if not np.all((f3>0)&(f3<1)) or not np.isin(y,[0,1]).all():
            raise ValueError("Invalid OOF F3 marginal or top3 outcomes")
        offs=logit(f3)
        # Every race contributes unit weight, no arbitrary field-size bias.
        wt=1./df.groupby("race_id").race_id.transform("size").to_numpy(float)
        wt=wt/wt.sum()
        reg=100/(self.c*len(df))
        def loss_and_grad(beta):
            linear=offs+X@beta
            losses=np.logaddexp(0,linear)-y*linear
            p=expit(linear)
            value=float(wt@losses+.5*reg*np.dot(beta[1:],beta[1:]))
            grad=X.T@(wt*(p-y))
            grad[1:]+=reg*beta[1:]
            return value,grad
        opt=minimize(loss_and_grad,np.zeros(X.shape[1]),jac=True,
                     method="L-BFGS-B",options={"maxiter":300,"ftol":1e-11})
        if not np.isfinite(opt.fun) or not np.isfinite(opt.x).all():
            raise ValueError("OOF residual optimization failed")
        self.beta=opt.x
        self.fit_info={"n_features":X.shape[1],"C":self.c,
                       "objective":float(opt.fun),"iterations":int(opt.nit),
                       "success":bool(opt.success),"coefficients":opt.x.tolist()}
        return self

    def predict(self,df:pd.DataFrame)->np.ndarray:
        if not hasattr(self,"beta"):
            raise ValueError("Residual has not been fitted")
        X=self._matrix(df)
        p=expit(logit(df.p_matched_f3.to_numpy(float))+X@self.beta)
        return np.clip(p,1e-9,1.-1e-9)


def cohort(data:pd.DataFrame,ref:pd.DataFrame,year:int)->pd.DataFrame:
    ids=set(ref.race_id.astype(str))
    if len(ref)!=len(ids):
        raise ValueError("Duplicate race IDs in original PR40")
    d=data.loc[data.race_id.isin(ids)].copy()
    if set(d.race_id)!=ids:
        raise ValueError("Missing frozen PR40 race cohort")
    if d.duplicated(["race_id","horse_no"]).any():
        raise ValueError("Repeated starter number")
    counts=d.groupby("race_id").race_id.transform("size").to_numpy(int)
    if not np.array_equal(counts,d.field_size.to_numpy(int)):
        raise ValueError("Full-field coverage impossible for some PR40 race")
    labels=d.groupby("race_id").top3_label.transform("sum")
    if not labels.eq(3).all():
        raise ValueError("Every race must have exactly 3 placers")
    if not pd.to_datetime(d.race_date).dt.year.eq(year).all():
        raise ValueError("Year drift in reference races")
    return d.sort_values(["race_date","race_id","horse_no"]).reset_index(drop=True)


def joint_race_score(g:pd.DataFrame,probabilities:np.ndarray)->tuple[dict,np.ndarray,np.ndarray]:
    h=g.horse_no.to_numpy(int)
    combinations,p=v1_joint(probabilities,h)
    actual=tuple(sorted(g.loc[g.top3_label.eq(1),"horse_no"].astype(int)))
    found=np.flatnonzero(np.all(combinations==actual,axis=1))
    if len(found)!=1:
        raise ValueError("Actual unordered winning triple not uniquely found")
    return score_probabilities(p,int(found[0])),combinations,p


def score_cohort(df:pd.DataFrame,
                 predictions:dict[str,np.ndarray],
                 *,write_tickets:bool=False)->tuple[pd.DataFrame,list[dict]]:
    if any(len(v)!=len(df) for v in predictions.values()):
        raise ValueError("At least one model has incomplete runner predictions")
    rows=[]
    tickets=[]
    for rid,g in df.groupby("race_id",sort=True):
        idx=g.index.to_numpy()
        row={"race_id":str(rid),"race_date":str(pd.Timestamp(
            g.race_date.iloc[0]).date()),"field_size":len(g)}
        coords=None
        for name,pred in predictions.items():
            metrics,trios,joint=joint_race_score(g,pred[idx])
            if coords is None:
                coords=trios
                local={name:joint}
            else:
                if not np.array_equal(coords,trios):
                    raise ValueError("Ticket coordinates differ within race")
                local[name]=joint
            row.update({name+"_"+k:v for k,v in metrics.items()})
        rows.append(row)
        if write_tickets:
            winners=tuple(sorted(g.loc[g.top3_label.eq(1),"horse_no"].astype(int)))
            tickets.extend({
                "race_id":str(rid),"horse_1":int(a),"horse_2":int(b),
                "horse_3":int(c),"actual":int((a,b,c)==winners),
                **{name:float(p[i]) for name,p in local.items()}
            } for i,(a,b,c) in enumerate(coords))
    return pd.DataFrame(rows),tickets


def main():
    parser=argparse.ArgumentParser()
    for key in ("historical-dir","results","races","matched-oof",
                "pr40-2023","pr40-2024","out"):
        parser.add_argument("--"+key,required=True,type=Path)
    a=parser.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError("No overwriting versioned experiment output")
    started=time.perf_counter()
    source,audit,full_history,full_market=prepare_dataset(
        a.historical_dir,a.results,a.races,include_full_context=True
    )
    print("V5_ELIGIBLE_PANEL_READY "+str(len(source)),flush=True)
    # The original F3 source is filtered by career_starts>=3. Consequently
    # it CANNOT provide a full starter race for 3-renpuku pace construction.
    # Compute all as-of history on the unfiltered 2010–2024 raw panel, then
    # use whole-field final market shares for races in the eligible panel.
    wanted=set(source.race_id.astype(str))
    if set(full_market.race_id.astype(str)) != wanted:
        raise ValueError("Full-field market races differ from eligible source")
    runner_keys=["race_id","horse_id","horse_no"]
    context=full_history.merge(
        full_market[["race_id","horse_no","market_p_win"]].rename(
            columns={"market_p_win":"market_win_probability"}
        ),on=["race_id","horse_no"],how="left",sort=False,
        validate="one_to_one"
    )
    absent=context.race_id.isin(wanted) & context.market_win_probability.isna()
    if absent.any():
        examples=context.loc[absent,runner_keys].head(10).to_dict("records")
        raise ValueError("No full-field market share for known starter: "+repr(examples))
    # Only the evaluation-scope race IDs need all-field completeness; the
    # 2010–2015 and excluded historical races remain available for lagged
    # jockey/trainer/horse updates, irrespective of their market coverage.
    panel=engineer_v5_asof(context,validate_races=wanted)
    v5_cols=list(FEATURE_BLOCKS["all"])
    source=source.merge(panel[runner_keys+v5_cols],on=runner_keys,
                        how="left",sort=False,validate="one_to_one",
                        indicator=True)
    if not source["_merge"].eq("both").all():
        examples=source.loc[source["_merge"].ne("both"),runner_keys].head(10)
        raise ValueError("Failed to attach full-context V5 starter features: "+
                         repr(examples.to_dict("records")))
    source.drop(columns="_merge",inplace=True)
    print("V5_FULL_FIELD_CONTEXT_READY "+str(len(context))+
          " rows; "+str(len(wanted))+" eligible races",flush=True)
    oof=pd.read_csv(a.matched_oof,dtype={"race_id":str,"horse_id":str})
    cols=["race_id","horse_id","horse_no","p_matched_f3"]
    if len(oof)!=len(oof.drop_duplicates(cols[:3])):
        raise ValueError("Duplicate frozen F3 OOF rows")
    data=panel.merge(oof[cols],on=cols[:3],how="inner",
                     sort=False,validate="one_to_one")
    data["race_id"]=data.race_id.astype(str)
    refs={
        2023:pd.read_csv(a.pr40_2023,dtype={"race_id":str}),
        2024:pd.read_csv(a.pr40_2024,dtype={"race_id":str}),
    }
    if len(refs[2023])!=1355 or len(refs[2024])!=1336:
        raise ValueError("Wrong original PR40 cohort sizes")
    d23=cohort(data,refs[2023],2023)
    d24=cohort(data,refs[2024],2024)
    days=np.sort(pd.to_datetime(d23.race_date.unique()).to_numpy())
    cutoff=pd.Timestamp(days[int(.75*len(days))])
    early=d23[d23.race_date.lt(cutoff)].copy()
    late=d23[d23.race_date.ge(cutoff)].copy()
    if early.race_id.nunique()<500 or late.race_id.nunique()<200:
        raise ValueError("Insufficient chronological 2023 inner train/validation")
    if early.race_date.max()>=late.race_date.min():
        raise ValueError("Temporal validation cutoffs leaked")

    candidates=[("offset_only",1.0)]
    candidates.extend((b,c) for b in ("agents","dynamics","pace","all")
                      for c in C_VALUES)
    selection=[]
    for block,c in candidates:
        fit=OffsetResidual(block,c).fit(early)
        late_pred=fit.predict(late)
        scored,_=score_cohort(late,{"f3":late.p_matched_f3.to_numpy(float),
                                       "candidate":late_pred})
        selection.append({
            "block":block,"C":c,"inner_validation_joint_logloss":
            float(scored.candidate_joint_logloss.mean()),
            "inner_F3_joint_logloss":float(scored.f3_joint_logloss.mean()),
            "inner_train_races":early.race_id.nunique(),
            "inner_valid_races":late.race_id.nunique(),
            "fit":fit.fit_info,
        })
        print("V5_INNER_SELECTION "+json.dumps({
            "block":block,"C":c,
            "joint_nll":selection[-1]["inner_validation_joint_logloss"]
        }),flush=True)
    chosen=min(selection,key=lambda z:(z["inner_validation_joint_logloss"],
                                       z["block"]!="offset_only",
                                       z["block"],z["C"]))
    print("V5_LOCKED_CHOICE_FROM_2023 "+json.dumps({
        "block":chosen["block"],"C":chosen["C"]
    }),flush=True)

    predictions={"f3_v1":d24.p_matched_f3.to_numpy(float)}
    fitted={}
    # Same 2024 test set; do not select among these using test labels.
    for block in ALL_CANDIDATES:
        c=chosen["C"] if block==chosen["block"] else (
            min([z for z in selection if z["block"]==block],
                key=lambda z:z["inner_validation_joint_logloss"])["C"]
            if block!="offset_only" else 1.0
        )
        model=OffsetResidual(block,float(c)).fit(d23)
        predictions[block]=model.predict(d24)
        fitted[block]={"C":c,"training_year":2023,**model.fit_info}
    scored,tickets=score_cohort(d24,predictions,write_tickets=True)
    reference=refs[2024].set_index("race_id")
    match=max(abs(row.f3_v1_joint_logloss-
                  float(reference.loc[row.race_id,"v1_joint_logloss"]))
              for row in scored.itertuples(index=False))
    if match>5e-6:
        raise ValueError(f"Frozen F3+V1 original racewise parity broken: {match}")
    summary={}
    paired={}
    for model in predictions:
        summary[model]={
            metric:float(scored[model+"_"+metric].mean())
            for metric in ("joint_logloss","joint_brier","top1_hit",
                           "top5_hit","top10_hit","top1_mass","top10_mass")
        }
        if model!="f3_v1":
            paired[model]={
                "joint_nll":paired_day_ci(
                    scored,"f3_v1_joint_logloss",model+"_joint_logloss",
                    draws=2500
                ),
                "top10_hit":paired_day_ci(
                    scored,"f3_v1_top10_hit",model+"_top10_hit",
                    draws=2500
                ),
            }
    report={
        "status":"2024_KNOWN_DEVELOPMENT_NOT_NEW_HOLDOUT",
        "purpose":"feature_block_ablation_then_joint_maxent",
        "base_f3":"PR40 matched 800+400 yearly OOF; unchanged coefficients",
        "not_800_400_retraining":True,
        "strict_2023_2024_races":[d23.race_id.nunique(),d24.race_id.nunique()],
        "training_until":str(d23.race_date.max().date()),
        "inner_2023_cutoff":str(cutoff.date()),
        "inner_scores":selection,
        "pre_2024_locked_choice":chosen,
        "2023_full_fit_models":fitted,
        "2024_scores":summary,
        "2024_paired_day_bootstrap":paired,
        "2024_baseline_parity_max_abs":match,
        "asof_day_only_agents":True,
        "asof_horse_only_history":True,
        "full_field_pace":True,
        "training_source_sha256":audit["source_sha256"],
        "other_source_sha256":{
            "matched_oof":hashlib.sha256(a.matched_oof.read_bytes()).hexdigest(),
            "pr40_2024":hashlib.sha256(a.pr40_2024.read_bytes()).hexdigest(),
        },
        "seed":SEED,"cpu_only":True,
        "elapsed_seconds":time.perf_counter()-started,
        "no_production_changes":True,
        "no_real_3renpuku_odds":True,
    }
    a.out.mkdir(parents=True,exist_ok=True)
    scored.to_csv(a.out/"2024_v5_features_racewise.csv",index=False)
    pd.DataFrame(tickets).to_csv(a.out/"2024_v5_features_full_trio.csv.gz",
                                 index=False,compression="gzip")
    d24[["race_id","race_date","horse_no","horse_id"]].assign(
        **predictions
    ).to_csv(a.out/"2024_v5_features_runner_probabilities.csv",index=False)
    (a.out/"v5_features_experiment.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n"
    )
    print("V5_FEATURES_REALDATA="+json.dumps({
        "cohort":report["strict_2023_2024_races"],
        "chosen":(chosen["block"],chosen["C"]),"summary":summary,
        "paired":paired,
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
