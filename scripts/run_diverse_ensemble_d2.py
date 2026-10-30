"""Pre-frozen D2 diversity audit: 2023-only low-dimensional residual stacking.

Requires actual tested and strictly outer-OOF C1 LightGBM/CatBoost predictions;
no synthetic fallback and no target-year model fitting. FINAL odds only.
"""
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
EXPECTED_ALL_FINGERPRINT="cb984b2553647d423e05742d0625988da3129dab3183f79b6fb84c74776a582f"
LAMBDA=0.0005
MODEL_NAMES=("X2","LightGBM","CatBoost")


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(frame:pd.DataFrame)->str:
    x=frame[KEYS].copy()
    x.race_date=pd.to_datetime(x.race_date).dt.strftime("%Y-%m-%d")
    return hashlib.sha256(x.to_csv(index=False,lineterminator="\n").encode()).hexdigest()


def read_checked(path:Path)->pd.DataFrame:
    frame=pd.read_csv(path,dtype={"race_id":str,"horse_id":str})
    frame.race_date=pd.to_datetime(frame.race_date)
    frame.horse_no=frame.horse_no.astype(int)
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate model OOF keys")
    return frame.sort_values(KEYS,kind="stable").reset_index(drop=True)


def common_outer(reference:Path,c1:Path)->pd.DataFrame:
    r=read_checked(reference)
    c=read_checked(c1)
    if fingerprint(r)!=EXPECTED_ALL_FINGERPRINT:
        raise ValueError("Original PR32 OOF fingerprint changed")
    if not r[KEYS].equals(c[KEYS]):
        raise ValueError("C1 and PR32 models are not on exact same rows")
    if not r.top3_label.astype(int).equals(c.top3_label.astype(int)):
        raise ValueError("C1 and PR32 model labels differ")
    p0=r.p_M0_market.to_numpy(float)
    if not np.allclose(p0,c.p_M0,rtol=0,atol=1e-10):
        raise ValueError("C1 altered original calibrated market")
    merged=r[KEYS+["top3_label","surface","distance_m","race_class",
                    "p_M0_market","p_X2_nonlinear_strength"]].copy()
    merged["p_LightGBM"]=c.p_LightGBM_C1.to_numpy(float)
    merged["p_CatBoost"]=c.p_CatBoost_C1.to_numpy(float)
    return merged


def corrections(frame:pd.DataFrame)->tuple[np.ndarray,np.ndarray]:
    m=frame.p_M0_market.to_numpy(float)
    paths={
        "X2":frame.p_X2_nonlinear_strength.to_numpy(float),
        "LightGBM":frame.p_LightGBM.to_numpy(float),
        "CatBoost":frame.p_CatBoost.to_numpy(float),
    }
    for name,p in [("M0",m),*paths.items()]:
        if not np.isfinite(p).all() or np.any((p<=0)|(p>=1)):
            raise ValueError(f"Invalid {name} probabilities")
    mm=logit(m)
    mat=np.column_stack([logit(paths[k])-mm for k in MODEL_NAMES])
    return mm,mat


def combined_probability(frame:pd.DataFrame,weights:np.ndarray)->np.ndarray:
    w=np.asarray(weights,dtype=float)
    if w.shape!=(3,) or np.any(~np.isfinite(w)):
        raise ValueError("Expect exactly three finite correction weights")
    m,mat=corrections(frame)
    p=expit(m+mat@w)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Invalid diverse ensemble")
    return p


def fit_2023(train:pd.DataFrame)->dict:
    if set(train.race_date.dt.year)!={2023}:
        raise ValueError("OOF meta fit must use 2023 labels only")
    init=np.array([1.0,0.0,0.0],float)
    def objective(w):
        return float(candidate_score(train,combined_probability(train,w),
             name="D3_train").race_macro_brier+
             LAMBDA*np.square(w-init).sum())
    fit=minimize(objective,x0=init,method="SLSQP",
                 bounds=[(0,1.5)]*3,
                 constraints=[{"type":"ineq","fun":lambda w:2.0-np.sum(w)}],
                 options={"maxiter":300,"ftol":1e-12})
    if not fit.success or np.any(fit.x<-1e-9) or fit.x.sum()>2.0000001:
        raise ValueError("Nonnegative 2023 OOF optimizer did not converge")
    return {"weights":fit.x.tolist(),"train_rows":len(train),
            "train_races":int(train.race_id.nunique()),
            "train_max_date":str(train.race_date.max().date()),
            "train_regularized_objective":float(fit.fun)}


def run(reference:Path,c1:Path,e5_path:Path,out:Path)->dict:
    frame=common_outer(reference,c1)
    train=frame.loc[frame.race_date.dt.year.eq(2023)].copy()
    test=frame.loc[frame.race_date.dt.year.eq(2024)].copy().reset_index(drop=True)
    e5=read_checked(e5_path)
    if not test[KEYS].equals(e5[KEYS]):
        raise ValueError("2024 PR33 E5 cohort does not match C1")
    if not np.allclose(test.p_M0_market,e5.p_M0,rtol=0,atol=1e-10):
        raise ValueError("Frozen E5 market mismatch")
    if not np.allclose(test.p_X2_nonlinear_strength,e5.p_X2,rtol=0,atol=1e-10):
        raise ValueError("Frozen E5 X2 mismatch")
    learned=fit_2023(train)
    choices={
        "D0":[0.5,0.5,0],
        "D1":[0.5,0,0.5],
        "D2":[1/3,1/3,1/3],
        "D3":learned["weights"],
    }
    p={
        "M0":e5.p_M0.to_numpy(float),
        "X2":e5.p_X2.to_numpy(float),
        "E5":e5.p_E5.to_numpy(float),
        "LightGBM":test.p_LightGBM.to_numpy(float),
        "CatBoost":test.p_CatBoost.to_numpy(float),
    }
    p.update({name:combined_probability(test,np.asarray(w,float))
              for name,w in choices.items()})
    out.mkdir(parents=True,exist_ok=True)
    prediction=test[KEYS+["top3_label","surface","distance_m","race_class"]].copy()
    for name,value in p.items():
        prediction["p_"+name]=value
    prediction.to_csv(out/"D2_2024_oof_predictions.csv",index=False)
    decision,score=select_with_incumbent_one_se(
        test,p,incumbent="E5")
    score.to_csv(out/"D2_2024_scores.csv",index=False)
    rows=[]
    losses=[]
    reliability=[]
    diagnostics=[]
    for name,pred in p.items():
        delta=paired_delta(test,p["E5"],pred,
                           candidate_name="E5",reference_name=name)
        rows.append({"candidate":name,
                     "e5_minus_candidate":delta.mean_brier_delta,
                     "date_cluster_se":delta.se_brier_delta,
                     "passes_one_se":bool(delta.mean_brier_delta>delta.se_brier_delta)})
        loss=race_loss_table(test,pred)
        loss["candidate"]=name
        losses.append(loss)
        rel=calibration_table(test,pred)
        rel["candidate"]=name
        reliability.append(rel)
        ci,cs=calibration_intercept_slope(test.top3_label.to_numpy(int),pred)
        diagnostics.append({"candidate":name,"calibration_intercept":ci,
                            "calibration_slope":cs})
    pd.DataFrame(rows).to_csv(out/"D2_2024_paired.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(out/"D2_2024_calibration.csv",index=False)
    pd.concat(reliability).to_csv(out/"D2_2024_reliability.csv",index=False)
    pd.concat(losses).to_csv(out/"D2_2024_race_losses.csv",index=False)
    # Residual correlations and per-race complementarity diagnostics are
    # descriptive; not used to select model weights or modify registry.
    _,mat=corrections(test)
    pd.DataFrame(mat,columns=MODEL_NAMES).corr().to_csv(
        out/"D2_2024_residual_correlations.csv")
    losses=pd.concat(losses)
    baseline=losses.loc[losses.candidate.eq("E5"),
                        ["race_id","block","brier"]].rename(columns={
                            "brier":"E5_brier"})
    differences=losses.merge(baseline,on=["race_id","block"],validate="many_to_one")
    differences["E5_minus_candidate"]=differences.E5_brier-differences.brier
    differences.to_csv(out/"D2_2024_race_differences.csv",index=False)
    manifest={
        "status":"HISTORICAL_FINAL_ODDS_2024_DEVELOPMENT_ONLY",
        "pre_registration_sha":"9881f3c02f5fae07e4523dc6ba6f83ec6cee469c",
        "component_source_sha256":{"PR32":sha(reference),"C1":sha(c1),"E5":sha(e5_path)},
        "2023_fit":learned,
        "2024_races":int(test.race_id.nunique()),
        "2024_rows":len(test),
        "2024_fingerprint":fingerprint(test),
        "frozen_candidate_weights":choices,
        "verdict":{"winner":decision.winner,
                   "incumbent_E5_retained":bool(decision.incumbent_retained)},
        "production_unchanged":True}
    (out/"D2_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference-oof",type=Path,required=True)
    p.add_argument("--c1-oof",type=Path,required=True)
    p.add_argument("--e5-2024",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(run(args.reference_oof,args.c1_oof,args.e5_2024,
                         args.output_dir),indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
