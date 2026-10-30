"""Preregistered strictly-chronological PR32 model ensembles (2023 -> 2024).

Uses already-frozen X0/X1/X2 OOF; NEVER refits a component on its own
training labels, never uses 2024 outcomes to fit weights, and never promotes
a production model. This is historical FINAL-odds research, NOT T−10.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from keiba_place_lab.model_selection import (
    candidate_score,
    paired_delta,
    race_loss_table,
    select_with_incumbent_one_se,
)
from keiba_place_lab.nonmarket import calibration_table
from keiba_place_lab.stage4_successor import calibration_intercept_slope

KEYS = ("race_date", "race_id", "horse_id", "horse_no")
SOURCE_COLS = {
    "M0": "p_M0_market",
    "X0": "p_X0_nonlinear_old",
    "X1": "p_X1_nonlinear_history",
    "X2": "p_X2_nonlinear_strength",
}
FROZEN = ("M0","X2","E1","E2","E3","E4","E5")
SOURCE_FINGERPRINT_2023_2024 = (
    "cb984b2553647d423e05742d0625988da3129dab3183f79b6fb84c74776a582f"
)
FIXED_E5_PENALTY = 0.0001


def file_sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for buffer in iter(lambda:f.read(1024*1024),b""):
            h.update(buffer)
    return h.hexdigest()


def row_fingerprint(frame: pd.DataFrame) -> str:
    ordered=frame[list(KEYS)].sort_values(list(KEYS),kind="stable").copy()
    ordered["race_date"]=pd.to_datetime(ordered.race_date).dt.strftime("%Y-%m-%d")
    return hashlib.sha256(ordered.to_csv(index=False,lineterminator="\n").encode()).hexdigest()


def _safe_logit(p: np.ndarray) -> np.ndarray:
    p=np.asarray(p,dtype=float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Invalid component probabilities")
    return logit(np.clip(p,1e-12,1-1e-12))


def ensemble_predictions(probs: dict[str,np.ndarray], alpha: float
                         ) -> dict[str,np.ndarray]:
    required=set(SOURCE_COLS)
    if set(probs)!=required:
        raise ValueError("All market/old/history/Elo OOF predictions mandatory")
    if not np.isfinite(alpha) or not 0<=alpha<=1.5:
        raise ValueError("Frozen shrinkage alpha out of bounds")
    m,x0,x1,x2=(np.asarray(probs[name],float)
                 for name in ("M0","X0","X1","X2"))
    for name,arr in probs.items():
        if len(arr)!=len(m):
            raise ValueError(f"Unaligned prediction rows: {name}")
        _safe_logit(arr)
    out={
        "M0":m,
        "X2":x2,
        "E1":0.5*x0+0.5*x2,
        "E2":expit(0.5*_safe_logit(x0)+0.5*_safe_logit(x2)),
        "E3":0.5*x1+0.5*x2,
        "E4":0.1*m+0.9*x2,
        "E5":expit(_safe_logit(m)+alpha*(_safe_logit(x2)-_safe_logit(m))),
    }
    for value in out.values():
        if not np.isfinite(value).all() or np.any((value<0)|(value>1)):
            raise ValueError("Invalid ensemble probabilities")
    return out


def learn_alpha(train: pd.DataFrame) -> float:
    # This function MUST be passed 2023 only, never labels from 2024.
    years=pd.to_datetime(train.race_date).dt.year.unique()
    if len(years)!=1 or years[0]!=2023:
        raise ValueError("E5 alpha fitting requires 2023 OOF exclusively")
    market=_safe_logit(train[SOURCE_COLS["M0"]].to_numpy(float))
    residual=_safe_logit(train[SOURCE_COLS["X2"]].to_numpy(float))-market

    def score(alpha: float) -> float:
        p=expit(market+alpha*residual)
        return candidate_score(train,p,name="E5_2023_fit").race_macro_brier+(
            FIXED_E5_PENALTY*(alpha-1.0)**2
        )
    fitted=minimize_scalar(score,bounds=(0,1.5),method="bounded",
                           options={"xatol":1e-7,"maxiter":200})
    if not fitted.success:
        raise ValueError("Strict 2023 alpha optimization failed")
    # If the minimum lies on the boundary do not trust bounded optimizer's
    # interior-only search, compare both endpoints as part of the fixed fit.
    points=[(score(0.0),0.0),(score(1.5),1.5),
            (score(float(fitted.x)),float(fitted.x))]
    return min(points,key=lambda v:(v[0],abs(v[1]-1.0)))[1]


def evaluate(csv: Path,out: Path) -> dict:
    frame=pd.read_csv(csv,dtype={"race_id":str,"horse_id":str})
    frame.race_date=pd.to_datetime(frame.race_date)
    if frame.duplicated(list(KEYS)).any():
        raise ValueError("Duplicate source OOF keys")
    if not set(SOURCE_COLS.values()).issubset(frame.columns):
        raise ValueError("Missing a required original OOF model")
    if not set(frame.race_date.dt.year)=={2023,2024}:
        raise ValueError("Source OOF years differ from frozen specification")
    if row_fingerprint(frame)!=SOURCE_FINGERPRINT_2023_2024:
        raise ValueError("PR32 source OOF row fingerprint changed")
    training=frame.loc[frame.race_date.dt.year.eq(2023)].copy()
    evaluation=frame.loc[frame.race_date.dt.year.eq(2024)].copy()
    alpha=learn_alpha(training)
    if training.race_date.max()>=evaluation.race_date.min():
        raise ValueError("OOF meta-training overlaps evaluation")
    evaluation=evaluation.sort_values(list(KEYS),kind="stable").reset_index(drop=True)
    source={name:evaluation[col].to_numpy(float)
            for name,col in SOURCE_COLS.items()}
    preds=ensemble_predictions(source,alpha)
    scores={name:candidate_score(evaluation,p,name=name)
            for name,p in preds.items()}
    market_selection,market_table=select_with_incumbent_one_se(
        evaluation,preds,incumbent="M0")
    x2_selection,x2_table=select_with_incumbent_one_se(
        evaluation,preds,incumbent="X2")
    out.mkdir(parents=True,exist_ok=True)
    market_table.to_csv(out/"joint_market_gate.csv",index=False)
    x2_table.to_csv(out/"joint_x2_gate.csv",index=False)
    paired=[]
    cal=[]
    race=[]
    reliabilities=[]
    pred_frame=evaluation[list(KEYS)+[
        "top3_label","surface","distance_m","race_class"]].copy()
    for name,p in preds.items():
        pred_frame["p_"+name]=p
        d0=paired_delta(evaluation,preds["M0"],p,
                        candidate_name="M0",reference_name=name)
        d2=paired_delta(evaluation,preds["X2"],p,
                        candidate_name="X2",reference_name=name)
        paired.append({
            "candidate":name,
            "market_minus_candidate_brier":d0.mean_brier_delta,
            "market_paired_se":d0.se_brier_delta,
            "x2_minus_candidate_brier":d2.mean_brier_delta,
            "x2_paired_se":d2.se_brier_delta,
            "beats_x2_one_se":bool(d2.mean_brier_delta>d2.se_brier_delta),
        })
        ci,cs=calibration_intercept_slope(
            evaluation.top3_label.to_numpy(int),p)
        cal.append({"candidate":name,"calibration_intercept":ci,
                    "calibration_slope":cs})
        losses=race_loss_table(evaluation,p)
        losses["candidate"]=name
        race.append(losses)
        rel=calibration_table(evaluation,p)
        rel["candidate"]=name
        reliabilities.append(rel)
    pred_frame.to_csv(out/"2024_predictions.csv",index=False)
    pd.DataFrame(paired).to_csv(out/"paired_vs_incumbents.csv",index=False)
    pd.DataFrame(cal).to_csv(out/"calibration.csv",index=False)
    pd.concat(reliabilities).to_csv(out/"reliability.csv",index=False)
    race=pd.concat(race,ignore_index=True)
    race.to_csv(out/"race_losses.csv",index=False)
    baseline=race.loc[race.candidate.eq("X2"),
                      ["race_id","block","brier","log_loss"]].rename(
        columns={"brier":"x2_brier","log_loss":"x2_logloss"})
    deltas=race.merge(baseline,on=["race_id","block"],validate="many_to_one")
    deltas["x2_minus_candidate_brier"]=deltas.x2_brier-deltas.brier
    deltas.to_csv(out/"race_differences.csv",index=False)
    manifest={
        "status":"HISTORICAL_FINAL_ODDS_2024_DEVELOPMENT_AUDIT_ONLY",
        "source_oof_sha256":file_sha(csv),
        "source_2023_2024_fingerprint":SOURCE_FINGERPRINT_2023_2024,
        "evaluation_2024_fingerprint":row_fingerprint(evaluation),
        "alpha_fitted_only_on_2023":alpha,
        "alpha_penalty":FIXED_E5_PENALTY,
        "meta_fit_max_date":str(training.race_date.max().date()),
        "evaluation_min_date":str(evaluation.race_date.min().date()),
        "evaluation_rows":len(evaluation),
        "evaluation_races":int(evaluation.race_id.nunique()),
        "candidate_names":list(FROZEN),
        "market_gate":{"winner":market_selection.winner,
                       "incumbent_retained":bool(market_selection.incumbent_retained)},
        "X2_gate":{"winner":x2_selection.winner,
                   "incumbent_retained":bool(x2_selection.incumbent_retained)},
        "scores":{name:score.__dict__ for name,score in scores.items()},
        "not_a_new_unseen_test":True,
        "not_a_t_minus_10_market_test":True,
        "no_live_model_change":True,
    }
    (out/"manifest.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--oof",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(evaluate(args.oof,args.output_dir),indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
