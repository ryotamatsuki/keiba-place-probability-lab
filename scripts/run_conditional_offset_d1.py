"""Frozen conditional offset D1/D2 on original PR32 OOF, fit 2023 ONLY."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from keiba_place_lab.model_selection import (
    candidate_score,
    paired_delta,
    race_loss_table,
    select_with_incumbent_one_se,
)
from keiba_place_lab.nonmarket import calibration_table
from keiba_place_lab.stage4_successor import calibration_intercept_slope

KEYS=["race_date","race_id","horse_id","horse_no"]
FIXED_REF_FINGERPRINT="cb984b2553647d423e05742d0625988da3129dab3183f79b6fb84c74776a582f"
BOUNDS=((0.0,1.5),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5))
LAMBDA_COMMON=0.0001
LAMBDA_DEVIATION=0.001


def fingerprint(frame:pd.DataFrame)->str:
    rows=frame[KEYS].sort_values(KEYS).copy()
    rows.race_date=pd.to_datetime(rows.race_date).dt.strftime("%Y-%m-%d")
    return hashlib.sha256(rows.to_csv(index=False,lineterminator="\n").encode()).hexdigest()


def _strict_logits(frame:pd.DataFrame)->tuple[np.ndarray,np.ndarray]:
    m=frame.p_M0_market.to_numpy(float)
    x=frame.p_X2_nonlinear_strength.to_numpy(float)
    if not np.isfinite(m).all() or not np.isfinite(x).all():
        raise ValueError("Non-finite market/X2 probabilities")
    if np.any((m<=0)|(m>=1)) or np.any((x<=0)|(x>=1)):
        raise ValueError("Expected interior market/X2 probabilities")
    return logit(m),logit(x)-logit(m)


def _conditional_matrix(frame:pd.DataFrame,candidate:str)->np.ndarray:
    if candidate not in ("D1","D2"):
        raise ValueError("Candidate not preregistered")
    surf=frame.surface.astype(str)
    if not surf.isin(("turf","dirt")).all():
        raise ValueError("Unknown current race surface")
    distance=frame.distance_m.to_numpy(float)
    market=frame.p_M0_market.to_numpy(float)
    if not np.isfinite(distance).all() or not np.isfinite(market).all():
        raise ValueError("Missing pre-race context")
    columns=[
        np.ones(len(frame),dtype=float),
        surf.eq("turf").to_numpy(float)-0.5,
    ]
    if candidate=="D2":
        columns.extend([(distance-1600.0)/1000.0,market-0.3])
    return np.column_stack(columns)


def conditional_probability(frame:pd.DataFrame,params:np.ndarray,
                            candidate:str)->np.ndarray:
    design=_conditional_matrix(frame,candidate)
    p=np.asarray(params,dtype=float)
    if p.ndim!=1 or len(p)!=design.shape[1] or not np.isfinite(p).all():
        raise ValueError("Wrong coefficient count or nonfinite")
    offset,f=_strict_logits(frame)
    alpha=np.clip(design@p,0,1.5)
    result=expit(offset+alpha*f)
    if not np.isfinite(result).all() or np.any((result<=0)|(result>=1)):
        raise ValueError("Invalid conditional probability")
    return result


def fit_conditional(train:pd.DataFrame,candidate:str)->dict:
    years=pd.to_datetime(train.race_date).dt.year.unique()
    if list(years)!=[2023]:
        raise ValueError("Only 2023 may fit conditional weights for 2024")
    matrix=_conditional_matrix(train,candidate)
    init=np.zeros(matrix.shape[1],dtype=float)
    init[0]=1.0
    bounds=BOUNDS[:len(init)]
    def objective(w:np.ndarray)->float:
        predicted=conditional_probability(train,w,candidate)
        metric=candidate_score(train,predicted,name=candidate).race_macro_brier
        return float(metric+
                     LAMBDA_COMMON*(w[0]-1.0)**2+
                     LAMBDA_DEVIATION*np.sum(w[1:]**2))
    fit=minimize(objective,init,method="L-BFGS-B",bounds=bounds,
                 options={"maxiter":250,"ftol":1e-12})
    if not fit.success or not np.isfinite(fit.fun):
        raise ValueError("Prior-year conditional optimization failed")
    return {"candidate":candidate,
            "params":fit.x.tolist(),
            "train_rows":len(train),
            "train_races":int(train.race_id.nunique()),
            "train_max_date":str(train.race_date.max().date()),
            "training_objective":float(fit.fun)}


def run(source:Path,e5path:Path,out:Path)->dict:
    data=pd.read_csv(source,dtype={"race_id":str,"horse_id":str})
    data.race_date=pd.to_datetime(data.race_date)
    data=data.sort_values(KEYS,kind="stable").reset_index(drop=True)
    if data.duplicated(KEYS).any() or fingerprint(data)!=FIXED_REF_FINGERPRINT:
        raise ValueError("Source original PR32 OOF fingerprint mismatch")
    if set(data.race_date.dt.year)!={2023,2024}:
        raise ValueError("Unexpected outer years")
    train=data.loc[data.race_date.dt.year.eq(2023)].copy()
    test=data.loc[data.race_date.dt.year.eq(2024)].copy().reset_index(drop=True)
    e5=pd.read_csv(e5path,dtype={"race_id":str,"horse_id":str})
    e5.race_date=pd.to_datetime(e5.race_date)
    e5=e5.sort_values(KEYS,kind="stable").reset_index(drop=True)
    if not test[KEYS].equals(e5[KEYS]):
        raise ValueError("E5 same-row market comparison invalid")
    if not np.allclose(test.p_M0_market,e5.p_M0,rtol=0,atol=1e-10):
        raise ValueError("E5-market predictions differ")
    if not np.allclose(test.p_X2_nonlinear_strength,e5.p_X2,
                       rtol=0,atol=1e-10):
        raise ValueError("E5-X2 predictions differ")
    choices={key:fit_conditional(train,key) for key in ("D1","D2")}
    predictions={
        "M0":test.p_M0_market.to_numpy(float),
        "X2":test.p_X2_nonlinear_strength.to_numpy(float),
        "E5":e5.p_E5.to_numpy(float),
    }
    for key,choice in choices.items():
        predictions[key]=conditional_probability(
            test,np.asarray(choice["params"],dtype=float),key)
    out.mkdir(parents=True,exist_ok=True)
    eval_frame=test[KEYS+["top3_label","surface","distance_m","race_class"]].copy()
    for key,p in predictions.items():
        eval_frame["p_"+key]=p
    eval_frame.to_csv(out/"D_2024_oof_predictions.csv",index=False)
    verdict,scores=select_with_incumbent_one_se(
        test,predictions,incumbent="E5")
    scores.to_csv(out/"D_2024_scores.csv",index=False)
    losses=[]
    paired=[]
    diagnostic=[]
    reliability=[]
    subgroup=[]
    for name,p in predictions.items():
        delta=paired_delta(test,predictions["E5"],p,
                           candidate_name="E5",reference_name=name)
        paired.append({"candidate":name,"e5_minus_candidate":delta.mean_brier_delta,
                       "date_cluster_se":delta.se_brier_delta,
                       "passes_one_se":bool(delta.mean_brier_delta>delta.se_brier_delta)})
        loss=race_loss_table(test,p)
        loss["candidate"]=name
        losses.append(loss)
        rel=calibration_table(test,p)
        rel["candidate"]=name
        reliability.append(rel)
        intercept,slope=calibration_intercept_slope(
            test.top3_label.to_numpy(int),p)
        diagnostic.append({"candidate":name,"intercept":intercept,"slope":slope})
        groups={
            "surface":test.surface.astype(str),
            "distance_band":pd.cut(test.distance_m,[999,1400,2000,2600],
                labels=["1000_1400","1401_2000","2001_2600"]).astype(str),
            "race_class":test.race_class.fillna("MISSING").astype(str),
        }
        for group,values in groups.items():
            for value in sorted(values.unique()):
                selector=values.eq(value).to_numpy()
                res=candidate_score(test.loc[selector],p[selector],name=name)
                subgroup.append({"candidate":name,"group":group,"value":value,
                                 **res.__dict__})
    pd.DataFrame(paired).to_csv(out/"D_2024_paired.csv",index=False)
    pd.DataFrame(diagnostic).to_csv(out/"D_2024_calibration.csv",index=False)
    pd.DataFrame(subgroup).to_csv(out/"D_2024_subgroups.csv",index=False)
    pd.concat(losses).to_csv(out/"D_2024_race_losses.csv",index=False)
    pd.concat(reliability).to_csv(out/"D_2024_reliability.csv",index=False)
    manifest={
        "status":"KNOWN_2024_FINAL_ODDS_RESEARCH_ONLY",
        "pre_registration_sha":"ff98d95ab92b6f39efb31cf71bfa205dfde0fbf8",
        "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "e5_source_sha256":hashlib.sha256(e5path.read_bytes()).hexdigest(),
        "train_only_2023":choices,
        "2024_fingerprint":fingerprint(test),
        "2024_rows":len(test),"2024_races":int(test.race_id.nunique()),
        "selection":{"winner":verdict.winner,
                     "incumbent_retained":bool(verdict.incumbent_retained)},
        "no_production_adoption":True}
    (out/"D_manifest.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference-oof",type=Path,required=True)
    p.add_argument("--e5-2024",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(a.reference_oof,a.e5_2024,a.output_dir),
                     indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
