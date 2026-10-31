"""Preselected V4 rank/TabM/F3 joint ensembles; all component years leak-free.

Weights from late-2023 rank meta-holdout ONLY: rank model was fitted on earlier
2023 dates, TabM's 2023 predictions trained on data ending 2022, F3 is OOF.
2024 is already-seen development; no new claims of pristine validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from research_trio_f3_v1 import paired_day_ci

from keiba_place_lab.trio_pool_v4 import checked_pool
from keiba_place_lab.trio_residual_v2 import score_probabilities

KEYS=["race_id","horse_1","horse_2","horse_3"]
CANDIDATES=("f3_plus_rank","f3_plus_tabm","f3_plus_rank_plus_tabm")


def aligned(rank:Path,tabm:Path,incumbent:Path|None=None)->pd.DataFrame:
    rk=pd.read_csv(rank,dtype={"race_id":str})
    tb=pd.read_csv(tabm,dtype={"race_id":str})
    for frame in (rk,tb):
        if frame.duplicated(KEYS).any():
            raise ValueError("Duplicate 3-renpuku ticket identifiers")
    if not set(KEYS+["rank","incumbent"]).issubset(rk):
        raise ValueError("Missing rank holdout scores")
    if not set(KEYS+["tabm","incumbent"]).issubset(tb):
        raise ValueError("Missing trained TabM genuine OOF scores")
    merged=rk.merge(tb,on=KEYS,how="inner",validate="one_to_one",
                    suffixes=("","_tabm"))
    if len(merged)!=len(rk) or len(merged)!=len(tb):
        raise ValueError("Rank and TabM 2023/24 ticket space differs")
    if not np.allclose(merged.incumbent,merged.incumbent_tabm,atol=1e-7):
        raise ValueError("Separate TabM/rank implementation changed old F3+V1")
    merged.drop(columns="incumbent_tabm",inplace=True)
    if incumbent is not None:
        old=pd.read_csv(incumbent,dtype={"race_id":str})
        subset=old[KEYS+["incumbent","actual"]]
        merged=merged.merge(subset,on=KEYS,how="inner",validate="one_to_one",
                            suffixes=("","_fixed"))
        if len(merged)!=len(old):
            raise ValueError("Three-model ensemble silently dropped PR40 tickets")
        if not np.allclose(merged.incumbent,merged.incumbent_fixed,atol=1e-7):
            raise ValueError("Incumbent is not original PR40 exact 2024 F3")
        merged["actual"]=merged["actual_fixed"]
        merged.drop(columns=["incumbent_fixed","actual_fixed"],inplace=True)
    else:
        if "actual" not in merged:
            raise ValueError("Chronological rank validation lacks actual ordered label")
    return merged


def weights_on_2023(train:pd.DataFrame)->dict:
    if train.race_id.nunique()<200:
        raise ValueError("No sufficient late 2023 independent rank validation")
    training=[]
    for _,g in train.groupby("race_id",sort=True):
        if len(g)<4 or int(g.actual.sum())!=1:
            raise ValueError("Invalid 2023 rank meta OOF race label")
        true=int(np.flatnonzero(g.actual.to_numpy(int))[0])
        training.append((g.incumbent.to_numpy(float),g["rank"].to_numpy(float),
                         g.tabm.to_numpy(float),true))
    fixed={}
    def score(w:tuple[float,float,float])->float:
        return float(np.mean([
            -np.log(checked_pool([a,b,c],np.asarray(w))[idx])
            for a,b,c,idx in training
        ]))
    for name in CANDIDATES:
        choices=[]
        if name=="f3_plus_rank":
            for x in np.round(np.arange(0,1.00001,.05),10):
                choices.append((score((1-x,x,0)),(1-x,x,0)))
        elif name=="f3_plus_tabm":
            for x in np.round(np.arange(0,1.00001,.05),10):
                choices.append((score((1-x,0,x)),(1-x,0,x)))
        else:
            for i in range(11):
                for j in range(11-i):
                    k=10-i-j
                    choices.append((score((i/10,j/10,k/10)),
                                    (i/10,j/10,k/10)))
        val,w=min(choices,key=lambda z:(z[0],-z[1][0],-z[1][1]))
        fixed[name]={"weights":list(w),"2023_late_joint_nll":val,
                     "candidates":len(choices)}
    return fixed


def evaluate(train:pd.DataFrame,test:pd.DataFrame,ref:pd.DataFrame,
             output:Path)->dict:
    if len(ref)!=1336 or test.race_id.nunique()!=1336 or train.race_id.nunique()<200:
        raise ValueError("Must use exact PR40 2024 cohort and late 2023 training")
    fit=weights_on_2023(train)
    rows=[]
    forecasts=[]
    marginal_rows=[]
    changed={"rank":0,"tabm":0,"f3_plus_rank":0,
             "f3_plus_tabm":0,"f3_plus_rank_plus_tabm":0}
    for rid,g in test.groupby("race_id",sort=True):
        g=g.sort_values(KEYS[1:]).reset_index(drop=True)
        triples=g[KEYS[1:]].to_numpy(int)
        if len(g)!=len(set(map(tuple,triples))) or int(g.actual.sum())!=1:
            raise ValueError("Nonunique winning trio in 2024")
        actual=int(np.flatnonzero(g.actual.to_numpy(int))[0])
        base=g.incumbent.to_numpy(float)
        rank=g["rank"].to_numpy(float)
        tabm=g.tabm.to_numpy(float)
        comps={"incumbent":base,"rank":rank,"tabm":tabm}
        for name in CANDIDATES:
            comps[name]=checked_pool([base,rank,tabm],fit[name]["weights"])
        row={"race_id":rid}
        for name,p in comps.items():
            m=score_probabilities(p,actual)
            row.update({name+"_"+k:v for k,v in m.items()})
            if name in changed:
                changed[name]+=int(np.argmax(p)!=np.argmax(base))
        rows.append(row)
        unique=np.unique(triples)
        for horse in unique:
            mask=np.any(triples==horse,axis=1)
            marg={"race_id":rid,"horse_no":int(horse)}
            for name,p in comps.items():
                marg[name]=float(p[mask].sum())
            marginal_rows.append(marg)
        forecasts.extend({
            "race_id":rid,"horse_1":int(h1),"horse_2":int(h2),"horse_3":int(h3),
            "actual":int(actual==i),
            **{name:float(p[i]) for name,p in comps.items()}
        } for i,(h1,h2,h3) in enumerate(triples))
    scored=pd.DataFrame(rows)
    if set(scored.race_id)!=set(ref.race_id.astype(str)):
        raise ValueError("Only common PR40 races may be compared")
    compare=ref.set_index("race_id")
    shift=max(abs(row.incumbent_joint_logloss-
                  float(compare.loc[row.race_id,"v1_joint_logloss"]))
              for row in scored.itertuples(index=False))
    if shift>5e-6:
        raise ValueError("Third structure changed the frozen incumbent")
    scores={}
    deltas={}
    for name in ("incumbent","rank","tabm",*CANDIDATES):
        scores[name]={
            k:float(scored[name+"_"+k].mean()) for k in (
                "joint_logloss","joint_brier","top1_hit","top5_hit",
                "top10_hit","top1_mass","top10_mass")
        }
        if name!="incumbent":
            deltas[name]={
                "joint_nll":paired_day_ci(scored.join(ref.set_index("race_id")[
                    "race_date"],on="race_id"),
                    "incumbent_joint_logloss",name+"_joint_logloss",draws=2500),
                "top10_hit":paired_day_ci(scored.join(ref.set_index("race_id")[
                    "race_date"],on="race_id"),
                    "incumbent_top10_hit",name+"_top10_hit",draws=2500)
            }
    m=pd.DataFrame(marginal_rows)
    cor=m[["incumbent","rank","tabm"]].corr().to_dict()
    output.mkdir(parents=True,exist_ok=True)
    scored.to_csv(output/"2024_multistructure_racewise.csv",index=False)
    pd.DataFrame(forecasts).to_csv(
        output/"2024_multistructure_all_tickets.csv.gz",
        index=False,compression="gzip")
    m.to_csv(output/"2024_multistructure_marginals.csv.gz",
             index=False,compression="gzip")
    result={
        "status":"2024_PREVIOUSLY_VIEWED_DEVELOPMENT_ONLY",
        "mixture_training_year":"late_2023_rank_model_strict_inner_holdout",
        "training_races":train.race_id.nunique(),"test_races":len(scored),
        "learned_weights":fit,"models":scores,"paired_dates":deltas,
        "horsewise_marginal_correlations":cor,
        "changed_top1_ticket_counts":changed,
        "no_model_selected_on_2024":True,
        "no_true_trio_odds":True,"original_f3_unchanged":True,
    }
    (output/"ensemble_experiment.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser()
    for k in ("rank-late-2023","tabm-2023","rank-2024","tabm-2024",
              "incumbent-2024","pr40-2024","out"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError("Immutable multistructure evaluation must not overwrite")
    train=aligned(a.rank_late_2023,a.tabm_2023)
    test=aligned(a.rank_2024,a.tabm_2024,a.incumbent_2024)
    ref=pd.read_csv(a.pr40_2024,dtype={"race_id":str})
    result=evaluate(train,test,ref,a.out)
    print("V4_MULTISTRUCTURE="+json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
