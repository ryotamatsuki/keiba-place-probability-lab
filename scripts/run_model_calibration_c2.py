"""C2: fit each C1 model's market logit offset strength on 2023 OOF only."""
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

from run_diverse_ensemble_d2 import (
    KEYS,
    common_outer,
    read_checked,
)

PENALTY=0.0001
MODELS=("LightGBM","CatBoost")


def learn_alpha(market:np.ndarray,model:np.ndarray,
                frame:pd.DataFrame)->float:
    if set(frame.race_date.dt.year)!={2023}:
        raise ValueError("Model alpha must fit ONLY 2023 OOF")
    a=np.asarray(market,float)
    b=np.asarray(model,float)
    if len(a)!=len(frame) or len(b)!=len(frame) or (
        np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or
        np.any((a<=0)|(a>=1)) or np.any((b<=0)|(b>=1))
    ):
        raise ValueError("Missing valid paired probabilities")
    m=logit(a)
    correction=logit(b)-m
    def loss(alpha):
        p=expit(m+alpha*correction)
        return candidate_score(frame,p,name="alpha").race_macro_brier+(
            PENALTY*(alpha-1)**2)
    result=minimize_scalar(loss,bounds=(0,1.5),method="bounded",
                           options={"maxiter":200,"xatol":1e-7})
    if not result.success:
        raise ValueError("2023-only alpha minimization did not converge")
    return float(min((loss(v),v) for v in (0,1.5,float(result.x)))[1])


def run(reference:Path,c1:Path,e5_path:Path,out:Path)->dict:
    all_rows=common_outer(reference,c1)
    train=all_rows.loc[all_rows.race_date.dt.year.eq(2023)].copy()
    test=all_rows.loc[all_rows.race_date.dt.year.eq(2024)].copy().reset_index(drop=True)
    baseline=read_checked(e5_path)
    if not test[KEYS].equals(baseline[KEYS]):
        raise ValueError("C2 E5 cohort mismatch")
    if not np.allclose(test.p_M0_market,baseline.p_M0,
                       atol=1e-10,rtol=0):
        raise ValueError("C2 market input mismatch")
    train_market=train.p_M0_market.to_numpy(float)
    test_market=test.p_M0_market.to_numpy(float)
    cal={}
    predictions={
        "M0":baseline.p_M0.to_numpy(float),
        "X2":baseline.p_X2.to_numpy(float),
        "E5":baseline.p_E5.to_numpy(float),
    }
    for name in MODELS:
        raw=train["p_"+name].to_numpy(float)
        scalar=learn_alpha(train_market,raw,train)
        cal[name]=scalar
        test_raw=test["p_"+name].to_numpy(float)
        predictions[name]=test_raw
        predictions[name+"_scaled"]=expit(
            logit(test_market)+scalar*(logit(test_raw)-logit(test_market))
        )
    out.mkdir(parents=True,exist_ok=True)
    out_frame=test[KEYS+["top3_label","surface","distance_m","race_class"]].copy()
    for key,p in predictions.items():
        out_frame["p_"+key]=p
    out_frame.to_csv(out/"C2_2024_oof_predictions.csv",index=False)
    decision,scores=select_with_incumbent_one_se(
        test,predictions,incumbent="E5")
    scores.to_csv(out/"C2_2024_scores.csv",index=False)
    paired=[]
    losses=[]
    for name,p in predictions.items():
        e=paired_delta(test,predictions["E5"],p,
                       candidate_name="E5",reference_name=name)
        record={"candidate":name,"E5_minus_candidate_brier":e.mean_brier_delta,
                "E5_paired_date_se":e.se_brier_delta,
                "one_se_pass":bool(e.mean_brier_delta>e.se_brier_delta)}
        if name.endswith("_scaled"):
            raw=predictions[name.removesuffix("_scaled")]
            d=paired_delta(test,raw,p,
                           candidate_name=name.removesuffix("_scaled"),
                           reference_name=name)
            record.update({
                "raw_minus_scaled_brier":d.mean_brier_delta,
                "raw_paired_date_se":d.se_brier_delta})
        paired.append(record)
        loss=race_loss_table(test,p)
        loss["candidate"]=name
        losses.append(loss)
    pd.DataFrame(paired).to_csv(out/"C2_2024_paired.csv",index=False)
    pd.concat(losses).to_csv(out/"C2_2024_race_losses.csv",index=False)
    manifest={
        "status":"HISTORICAL_FINAL_ODDS_KNOWN_2024_ONLY",
        "pre_registration_sha":"5eaf5748d7920db871d957a07e0d6578fe51250a",
        "alpha_only_2023":cal,
        "rows":len(test),"races":int(test.race_id.nunique()),
        "score_selection":{"winner":decision.winner,
            "E5_incumbent_retained":bool(decision.incumbent_retained)},
        "source_sha256":{
            "C1":hashlib.sha256(c1.read_bytes()).hexdigest(),
            "X2":hashlib.sha256(reference.read_bytes()).hexdigest(),
            "E5":hashlib.sha256(e5_path.read_bytes()).hexdigest()},
        "not_a_prospective_test":True}
    (out/"C2_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main():
    p=argparse.ArgumentParser()
    for a in ("reference-oof","c1-oof","e5-2024","output-dir"):
        p.add_argument("--"+a,type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(run(args.reference_oof,args.c1_oof,args.e5_2024,
                         args.output_dir),indent=2))


if __name__=="__main__":
    main()
