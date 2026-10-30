"""Pre-frozen XGB F3 learning curve and inner-only correction scaling.

No 2023 outer label enters 2023 hyperparameter/alpha choices.
No 2024 outer label enters 2024 choices. Development data, final odds only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_information_expansion_v1 import _build_pipeline
from run_input_learning_a_v1 import (
    check_reference,
    feature_registry,
    prepare_dataset,
)
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from keiba_place_lab.model_selection import (
    candidate_score,
    paired_delta,
    select_with_incumbent_one_se,
)

# Registered in docs/INPUT_LEARNING_V1_SPEC.md before any A/B scores.
CHECKPOINTS=(60,120,240,400)
INNER_YEARS=(2021,2022,2023)
ALPHA_PENALTY=0.0001


def _offset_margin(p: np.ndarray) -> np.ndarray:
    p=np.asarray(p,dtype=float)
    if np.any(~np.isfinite(p)) or np.any(p<=0) or np.any(p>=1):
        raise ValueError("Non-finite/degenerate chronological market OOF")
    return logit(p)


def fit_curve(train: pd.DataFrame, target: pd.DataFrame,
              numeric: list[str], categorical: list[str]
              ) -> dict[int,np.ndarray]:
    """Train 400 trees once, predict prefixes without target-year fitting."""
    if train.race_date.max()>=target.race_date.min():
        raise ValueError("Leaked current/future target labels into XGB training")
    a=train.copy()
    b=target.copy()
    for col in numeric:
        a[col]=pd.to_numeric(a[col],errors="coerce").to_numpy(
            dtype=float,na_value=np.nan)
        b[col]=pd.to_numeric(b[col],errors="coerce").to_numpy(
            dtype=float,na_value=np.nan)
    for col in categorical:
        a[col]=a[col].astype("object").where(a[col].notna(),None)
        b[col]=b[col].astype("object").where(b[col].notna(),None)
    fit=_build_pipeline(numeric,categorical)
    fit.named_steps["model"].set_params(n_estimators=max(CHECKPOINTS))
    size=a.groupby("race_id").race_id.transform("size").to_numpy(float)
    w=1/size
    w*=len(w)/w.sum()
    columns=numeric+categorical
    fit.fit(
        a[columns],a.top3_label.astype(int),
        model__sample_weight=w,
        model__base_margin=_offset_margin(a.p_market_oof.to_numpy(float)),
    )
    X=fit.named_steps["preprocess"].transform(b[columns])
    margin=_offset_margin(b.p_market_oof.to_numpy(float))
    results={}
    for rounds in CHECKPOINTS:
        p=fit.named_steps["model"].predict_proba(
            X,base_margin=margin,iteration_range=(0,rounds))[:,1]
        if not np.isfinite(p).all() or np.any((p<=0)|(p>=1)):
            raise ValueError(f"Nonprobability prediction at {rounds} rounds")
        results[rounds]=p
    return results


def chronological_inner_oof(frame: pd.DataFrame
                           ) -> tuple[pd.DataFrame,pd.DataFrame]:
    numeric,categorical=feature_registry()["F3_market_field"]
    predicted=[]
    scores=[]
    for year in INNER_YEARS:
        train=frame.loc[frame.race_date.dt.year.between(2018,year-1)].copy()
        target=frame.loc[frame.race_date.dt.year.eq(year)].copy()
        if train.empty or target.empty:
            raise ValueError(f"Cannot produce inner chronological fold {year}")
        p=fit_curve(train,target,numeric,categorical)
        output=target[["race_date","race_id","horse_id","horse_no",
                       "top3_label","p_market_oof"]].copy()
        for rounds in CHECKPOINTS:
            output["p_rounds_"+str(rounds)]=p[rounds]
            metric=candidate_score(target,p[rounds],name=str(rounds))
            scores.append({"year":year,"rounds":rounds,"races":metric.races,
                           "rows":metric.rows,
                           "race_macro_brier":metric.race_macro_brier,
                           "race_macro_log_loss":metric.race_macro_log_loss})
        predicted.append(output)
    result=pd.concat(predicted,ignore_index=True).sort_values(
        ["race_date","race_id","horse_no"],kind="stable").reset_index(drop=True)
    return result,pd.DataFrame(scores)


def select_rounds(scores: pd.DataFrame, outer_year: int) -> int:
    if outer_year not in (2023,2024):
        raise ValueError("Unsupported preregistered outer year")
    available=(2021,2022) if outer_year==2023 else INNER_YEARS
    older=scores.loc[scores.year.isin(available)].copy()
    if set(older.year)!=set(available):
        raise ValueError("Missing required pre-outer inner year")
    # Equal weight per race across inner years, no weight by horse count.
    older["weighted_brier"]=older.race_macro_brier*older.races
    aggregated=older.groupby("rounds",sort=False).agg(
        total=("weighted_brier","sum"), races=("races","sum"))
    aggregated["average"]=aggregated.total/aggregated.races
    winner=aggregated.reset_index().sort_values(["average","rounds"]).iloc[0]["rounds"]
    return int(winner)


def alpha_fit_from_prior_oof(inner: pd.DataFrame, outer_year: int,
                            rounds: int) -> float:
    usable=inner.loc[inner.race_date.dt.year.lt(outer_year)].copy()
    if usable.empty or usable.race_date.max()>=pd.Timestamp(f"{outer_year}-01-01"):
        raise ValueError("Future outer labels entered alpha")
    m=_offset_margin(usable.p_market_oof.to_numpy(float))
    x=_offset_margin(usable["p_rounds_"+str(rounds)].to_numpy(float))
    target=usable.top3_label.to_numpy(int)
    def objective(alpha):
        pred=expit(m+alpha*(x-m))
        return candidate_score(usable,pred,name="alpha").race_macro_brier + (
            ALPHA_PENALTY*(alpha-1)**2)
    fit=minimize_scalar(objective,bounds=(0,1.5),method="bounded",
                        options={"xatol":1e-7,"maxiter":150})
    if not fit.success:
        raise ValueError("Failed bounded inner-only alpha optimization")
    candidates=[(objective(0.0),0.0),(objective(1.5),1.5),
                (objective(float(fit.x)),float(fit.x))]
    return float(min(candidates,key=lambda z:(z[0],abs(z[1]-1)))[1])


def run(frame: pd.DataFrame,source: Path,e5:Path,out: Path) -> dict:
    check_reference(frame,source)
    inner,scores=chronological_inner_oof(frame)
    out.mkdir(parents=True,exist_ok=True)
    scores.to_csv(out/"B_inner_curves.csv",index=False)
    # Inner predictions are derived from strict OOF, safe to share summaries.
    inner.to_csv(out/"B_inner_oof_predictions.csv",index=False)
    numeric,categorical=feature_registry()["F3_market_field"]
    rows=[]
    choices=[]
    for outer_year in (2023,2024):
        rounds=select_rounds(scores,outer_year)
        alpha=alpha_fit_from_prior_oof(inner,outer_year,rounds)
        train=frame.loc[frame.race_date.dt.year.between(2018,outer_year-1)]
        target=frame.loc[frame.race_date.dt.year.eq(outer_year)]
        curves=fit_curve(train,target,numeric,categorical)
        result=target[["race_date","race_id","horse_id","horse_no",
                       "top3_label","surface","distance_m","race_class"]].copy()
        result["p_M0"]=target.p_market_oof.to_numpy(float)
        result["p_B_raw"]=curves[rounds]
        result["p_B_scaled"]=expit(
            _offset_margin(result.p_M0.to_numpy(float))+
            alpha*(_offset_margin(result.p_B_raw.to_numpy(float))-
                   _offset_margin(result.p_M0.to_numpy(float)))
        )
        rows.append(result)
        choices.append({"outer_year":outer_year,"rounds":rounds,"alpha":alpha,
                        "alpha_trained_through":outer_year-1,
                        "rounds_chosen_from_years":[2021,2022] if outer_year==2023
                               else [2021,2022,2023],
                        "outer_eval_rows":len(result)})
    all_oof=pd.concat(rows,ignore_index=True).sort_values(
        ["race_date","race_id","horse_no"],kind="stable").reset_index(drop=True)
    all_oof.to_csv(out/"B_outer_oof_predictions.csv",index=False)
    test=all_oof.loc[all_oof.race_date.dt.year.eq(2024)].copy().reset_index(drop=True)
    old=pd.read_csv(e5,dtype={"race_id":str,"horse_id":str})
    old.race_date=pd.to_datetime(old.race_date)
    old=old.sort_values(["race_date","race_id","horse_no"],kind="stable").reset_index(drop=True)
    if not test[["race_date","race_id","horse_id","horse_no"]].equals(
        old[["race_date","race_id","horse_id","horse_no"]]
    ):
        raise ValueError("B candidate/E5 evaluation sample mismatch")
    probs={
        "M0":test.p_M0.to_numpy(float),
        "X2":old.p_X2.to_numpy(float),
        "E5":old.p_E5.to_numpy(float),
        "B_raw":test.p_B_raw.to_numpy(float),
        "B_scaled":test.p_B_scaled.to_numpy(float),
    }
    if not np.allclose(probs["M0"],old.p_M0,atol=1e-10,rtol=0):
        raise ValueError("E5/market reference mismatch")
    verdict,table=select_with_incumbent_one_se(
        test,probs,incumbent="E5")
    table.to_csv(out/"B_2024_E5_comparison.csv",index=False)
    paired=[]
    for name,p in probs.items():
        delta=paired_delta(test,probs["E5"],p,
                           candidate_name="E5",reference_name=name)
        paired.append({"candidate":name,"e5_minus_candidate":delta.mean_brier_delta,
                       "date_cluster_se":delta.se_brier_delta,
                       "passes_one_se":bool(delta.mean_brier_delta>delta.se_brier_delta)})
    pd.DataFrame(paired).to_csv(out/"B_2024_paired.csv",index=False)
    manifest={"status":"FINAL_WIN_ODDS_KNOWN_DEVELOPMENT_ONLY",
              "outer_choices":choices,
              "2024_E5_incumbent_retained":bool(verdict.incumbent_retained),
              "2024_best":verdict.point_brier_best,
              "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
              "no_production_promotion":True}
    (out/"B_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--historical-dir",type=Path,required=True)
    p.add_argument("--results-csv",type=Path,required=True)
    p.add_argument("--races-csv",type=Path,required=True)
    p.add_argument("--reference-oof",type=Path,required=True)
    p.add_argument("--e5-2024",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    frame,_=prepare_dataset(args.historical_dir,args.results_csv,args.races_csv)
    verdict=run(frame,args.reference_oof,args.e5_2024,args.output_dir)
    print(json.dumps(verdict,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
