"""Chronological ordered top-three PL/GPL on same 2023/24 PR40 OOF cohort.

Rank labels from original Kaggle V1 JRA results, audited against exactly three
OOF placers per race. 2023 first 75% days fit, last 25% days select; refit 2023,
evaluate 2024 as development only. Archived FINAL win odds, not T-10 prices.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research_trio_f3_v1 import paired_day_ci

from keiba_place_lab.trio_covariate_rank_v4 import (
    fit_rank_model,
    ordered_top3_probability,
    preknown_covariates,
    unordered_top3_distribution,
)
from keiba_place_lab.trio_pool_v4 import v1_joint
from keiba_place_lab.trio_residual_v2 import score_probabilities


def audit_rank_source(file:Path,race_ids:set[str])->dict:
    by={}
    with file.open(encoding="utf-8-sig",newline="") as handle:
        rd=csv.DictReader(handle)
        if not {"race_id","number","rank"}.issubset(rd.fieldnames or []):
            raise ValueError("Original rank source is not Kaggle V1")
        for raw in rd:
            rid=str(raw["race_id"]).strip()
            if rid not in race_ids:
                continue
            position=str(raw.get("rank","")).strip()
            if position not in ("1","2","3"):
                continue
            num=int(float(raw["number"]))
            by.setdefault(rid,{})
            if int(position) in by[rid]:
                raise ValueError("Duplicate rank in original source")
            by[rid][int(position)]=num
    if set(by)!=race_ids or any(set(v)!={1,2,3} for v in by.values()):
        raise ValueError("Cannot strictly recover original 1,2,3 finishing order")
    return by


def prepare(oof:pd.DataFrame,reference:pd.DataFrame,rank:dict)->list[dict]:
    keep=set(reference.race_id.astype(str))
    if len(keep)!=len(reference):
        raise ValueError("Repeated race id in original PR40 reference")
    sub=oof.loc[oof.race_id.isin(keep)]
    if set(sub.race_id)!=keep:
        raise ValueError("Rank-covariate model missing some PR40 complete races")
    result=[]
    for rid,g in sub.groupby("race_id",sort=True):
        g=g.sort_values("horse_no").reset_index(drop=True)
        horse=g.horse_no.to_numpy(int)
        if (len(g)!=int(g.field_size.iloc[0]) or g.top3_label.sum()!=3 or
                len(set(horse))!=len(horse)):
            raise ValueError("Rank model requires all complete starter records")
        actual=[rank[rid][k] for k in (1,2,3)]
        if set(actual)!=set(horse[g.top3_label.to_numpy(int).astype(bool)]):
            raise ValueError("Original 1–3 order mismatches PR40 OOF result labels")
        offset={int(h):i for i,h in enumerate(horse)}
        x=preknown_covariates(
            g.p_matched_xgb800.to_numpy(float),
            g.p_matched_cat400.to_numpy(float),
            g.p_market_oof.to_numpy(float),
        )
        coordinates,v1=v1_joint(g.p_matched_f3.to_numpy(float),horse)
        expected=tuple(sorted(actual))
        true=np.flatnonzero(np.all(coordinates==expected,axis=1))
        if len(true)!=1:
            raise ValueError("Actual unordered trio missing")
        result.append({
            "race_id":rid,"race_date":str(pd.Timestamp(g.race_date.iloc[0]).date()),
            "horses":horse,"x":x,
            "order":tuple(offset[v] for v in actual),
            "actual_idx":int(true[0]),"triplets":coordinates,"incumbent":v1,
        })
    return result


def joint_for(race:dict,model:dict)->np.ndarray:
    trios,p=unordered_top3_distribution(
        race["x"],np.array(model["theta"]),model["kind"]
    )
    if not np.array_equal(race["horses"][trios],race["triplets"]):
        raise ValueError("Distinct ordered PL trio coordinates")
    return p


def mean_nll(races:list[dict],model:dict)->float:
    return float(np.mean([
        -np.log(joint_for(r,model)[r["actual_idx"]]) for r in races
    ]))


def main():
    parser=argparse.ArgumentParser()
    for n in ("oof","pr40-2023","pr40-2024","ranks","out"):
        parser.add_argument("--"+n,required=True,type=Path)
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Immutable rank model evidence cannot be overwritten")
    start=time.perf_counter()
    oof=pd.read_csv(args.oof,dtype={"race_id":str,"horse_id":str})
    r23=pd.read_csv(args.pr40_2023,dtype={"race_id":str})
    r24=pd.read_csv(args.pr40_2024,dtype={"race_id":str})
    if len(r23)!=1355 or len(r24)!=1336:
        raise ValueError("Strict PR40 cohort changed")
    mapping=audit_rank_source(args.ranks,set(r23.race_id)|set(r24.race_id))
    tr=prepare(oof,r23,mapping)
    te=prepare(oof,r24,mapping)
    cutoff=pd.Timestamp(sorted(set(r["race_date"] for r in tr))[
        int(.75*len(set(r["race_date"] for r in tr)))
    ])
    early=[r for r in tr if pd.Timestamp(r["race_date"])<cutoff]
    late=[r for r in tr if pd.Timestamp(r["race_date"])>=cutoff]
    if len(early)<500 or len(late)<200 or max(r["race_date"] for r in early)>=min(
            r["race_date"] for r in late):
        raise ValueError("2023 split is not chronological")
    specs=[("pl",.03),("pl",.10),("fixed_gpl",.03),("fixed_gpl",.10),
           ("learned_gpl",.10)]
    candidates=[]
    for kind,lam in specs:
        f=fit_rank_model(early,kind,lam)
        candidates.append({"kind":kind,"lambda":lam,
                           "late_2023_joint_nll":mean_nll(late,f),
                           "first_window_fit":f})
        print("2023_META_RANK_VALIDATION "+json.dumps({
            "kind":kind,"lambda":lam,
            "late_joint_nll":candidates[-1]["late_2023_joint_nll"],
        }),flush=True)
    best=min(candidates,key=lambda q:(q["late_2023_joint_nll"],
                                      len(q["first_window_fit"]["theta"]),q["kind"]))
    final_fit=fit_rank_model(tr,best["kind"],best["lambda"])
    # The late-2023 actual rankings may train mixture weights only using
    # the EARLY-FITTED model, not the model refitted using its own outcome.
    late_model=best["first_window_fit"]
    calibration=[]
    for r in late:
        p=joint_for(r,late_model)
        calibration.append({
            "race_id":r["race_id"],"race_date":r["race_date"],
            "actual_idx":r["actual_idx"],"rank_joint":p,
            "incumbent":r["incumbent"],
        })
    rows=[]
    tickets=[]
    for r in te:
        p=joint_for(r,final_fit)
        metrics=score_probabilities(p,r["actual_idx"])
        rows.append({
            "race_id":r["race_id"],"race_date":r["race_date"],
            **{"rank_"+key:value for key,value in metrics.items()},
            **{"incumbent_"+key:value for key,value in
               score_probabilities(r["incumbent"],r["actual_idx"]).items()},
        })
        for (h1,h2,h3),u,v in zip(r["triplets"],p,r["incumbent"],strict=True):
            tickets.append({
                "race_id":r["race_id"],
                "horse_1":int(h1),"horse_2":int(h2),"horse_3":int(h3),
                "rank":float(u),"incumbent":float(v),
            })
    df=pd.DataFrame(rows)
    ref=r24.set_index("race_id")
    diff=max(abs(float(row.incumbent_joint_logloss)-
                 float(ref.loc[row.race_id,"v1_joint_logloss"]))
             for row in df.itertuples(index=False))
    if diff>5e-6:
        raise ValueError("2024 rank model changed incumbent probability")
    sig=paired_day_ci(df,"incumbent_joint_logloss","rank_joint_logloss",draws=2500)
    hit=paired_day_ci(df,"incumbent_top10_hit","rank_top10_hit",draws=2500)
    # Save only authentic late-2023 mix fitting scores, not refitted in-sample.
    calib=[]
    for r in calibration:
        calib.append({
            "race_id":r["race_id"],"race_date":r["race_date"],
            "rank_loss":float(-np.log(r["rank_joint"][r["actual_idx"]])),
            "incumbent_loss":float(-np.log(r["incumbent"][r["actual_idx"]])),
        })
    args.out.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.out/"2024_rank_racewise.csv",index=False)
    pd.DataFrame(tickets).to_csv(
        args.out/"2024_rank_trio_probabilities.csv.gz",
        index=False,compression="gzip",
    )
    pd.DataFrame(calib).to_csv(
        args.out/"late_2023_rank_metavalidation.csv",index=False
    )
    result={
        "status":"2024_DEVELOPMENT_RANK_META_NONINDEPENDENT_F3_BASE",
        "2023_first_train":len(early),"2023_late_select":len(late),
        "inner_date_cutoff":str(cutoff.date()),
        "2023_total_rank_training":len(tr),
        "2024_test_races":len(te),
        "candidate_validation":candidates,
        "selected_model":final_fit,
        "2024_rank_joint_loss":float(df.rank_joint_logloss.mean()),
        "2024_incumbent_loss":float(df.incumbent_joint_logloss.mean()),
        "2024_rank_joint_brier":float(df.rank_joint_brier.mean()),
        "2024_rank_top10_hit":float(df.rank_top10_hit.mean()),
        "2024_rank_top10_mass":float(df.rank_top10_mass.mean()),
        "2024_date_bootstrap":sig,
        "2024_top10_bootstrap":hit,
        "original_rank_source_sha256":hashlib.sha256(args.ranks.read_bytes()).hexdigest(),
        "no_2024_train_labels":True,
        "no_2025_labels":True,
        "ordered_target_instead_of_top3_BCE":True,
        "rank_strength_inputs_are_preknown_F3_OOF_meta_features":True,
        "seconds":time.perf_counter()-start,
    }
    (args.out/"rank_model_experiment.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n"
    )
    print("V4_RANK_2024="+json.dumps({
        "selected":final_fit,"date_ci":sig,"top10":hit,
        "nll":result["2024_rank_joint_loss"],"races":len(te)
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
