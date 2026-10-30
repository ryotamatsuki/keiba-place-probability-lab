"""Preregistered F0–F3 feature-block ablation on identical outer OOF rows.

No changes to XGBoost parameters, loss, or train years. Fail-closed F0 parity
against PR32's original X2 OOF. True T-10 market not available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import __version__ as xgb_version

from keiba_place_lab.historical_market import reconstruct_historical_market
from keiba_place_lab.input_learning_v1 import (
    MARKET_FIELD, MARKET_SPLITS, add_calibrated_market_splits,
    attach_information,
)
from keiba_place_lab.model_selection import (
    candidate_score, paired_delta, race_loss_table, select_with_incumbent_one_se,
)
from keiba_place_lab.nonmarket import calibration_table
from keiba_place_lab.stage4_successor import calibration_intercept_slope
from keiba_place_lab.stage4_successor_phase3a import RELATIVE_FEATURES

from run_information_expansion_v1 import (
    SOURCE_SHA, KEYS, _features, _fit_xgb, _sort_keyed, build_feature_panel,
    eligibility, fold_calibrated_market, hashfile, load_full_panel,
    merge_complete_market,
)
from keiba_place_lab.information_expansion_v1 import NEW_HISTORY, NEW_STRENGTH

REFERENCE_YEAR_FINGERPRINT = "cb984b2553647d423e05742d0625988da3129dab3183f79b6fb84c74776a582f"
F_NAMES=("F0_X2","F1_market_surface","F2_relative","F3_market_field")


def prepare_dataset(historical_dir: Path, results: Path, races: Path
                    ) -> tuple[pd.DataFrame,dict]:
    full,hashes=load_full_panel(historical_dir)
    rich,raw_audit=build_feature_panel(full,results,races)
    eligible=eligibility(rich)
    paired,market_audit=merge_complete_market(eligible,results)
    wanted=set(paired.race_id.astype(str))
    # Reconstruct full-field market, not only the eligible subset; paired is
    # unchanged from the parent X2 experiment to preserve parity.
    full_market,confirm=reconstruct_historical_market(results,wanted)
    if confirm["incomplete_market_races"]!=0:
        raise ValueError("Market complete-field cross-audit failed")
    enriched,coverage=attach_information(paired,rich,full_market)
    enriched=_sort_keyed(enriched)
    enriched["p_market_oof"]=fold_calibrated_market(enriched)
    enriched=add_calibrated_market_splits(enriched)
    return enriched,{"source_sha256":{**hashes,"keiba_results.csv":hashfile(results),
                                     "keiba_races.csv":hashfile(races)},
                     "raw_summary":{"rows":raw_audit["total_official_rows"],
                                    "matched":raw_audit["matched_raw_runner_rows"]},
                     "market_audit":market_audit,
                     "market_context_audit":confirm,
                     "relative_coverage":coverage.to_dict(orient="records")}


def feature_registry() -> dict:
    numeric,categorical=_features()
    base=list(dict.fromkeys([*numeric,*NEW_HISTORY,*NEW_STRENGTH]))
    return {
        "F0_X2":(base,list(categorical)),
        "F1_market_surface":(
            [*base,*MARKET_SPLITS],[*categorical,"surface"]),
        "F2_relative":(
            [*base,*MARKET_SPLITS,*RELATIVE_FEATURES],[*categorical,"surface"]),
        "F3_market_field":(
            [*base,*MARKET_SPLITS,*RELATIVE_FEATURES,*MARKET_FIELD],
            [*categorical,"surface"]),
    }


def row_fingerprint(frame: pd.DataFrame) -> str:
    rows=frame[KEYS].sort_values(KEYS,kind="stable").copy()
    rows["race_date"]=pd.to_datetime(rows.race_date).dt.strftime("%Y-%m-%d")
    return hashlib.sha256(rows.to_csv(index=False,lineterminator="\n").encode()).hexdigest()


def check_reference(frame: pd.DataFrame, reference: Path) -> pd.DataFrame:
    ref=pd.read_csv(reference,dtype={"race_id":str,"horse_id":str})
    ref.race_date=pd.to_datetime(ref.race_date)
    ref=_sort_keyed(ref)
    if row_fingerprint(ref)!=REFERENCE_YEAR_FINGERPRINT:
        raise ValueError("Original PR32 OOF fingerprint changed")
    check=_sort_keyed(frame.loc[frame.race_date.dt.year.isin([2023,2024])])
    if not check[KEYS].equals(ref[KEYS]):
        raise ValueError("New feature candidate evaluation sample differs from PR32")
    if not check.top3_label.astype(int).equals(ref.top3_label.astype(int)):
        raise ValueError("Outcome mismatch between historical/PR32 rows")
    return ref


def do_outer(frame: pd.DataFrame, reference: pd.DataFrame
             ) -> tuple[pd.DataFrame,dict]:
    registry=feature_registry()
    eval_data=_sort_keyed(
        frame.loc[frame.race_date.dt.year.isin([2023,2024])].copy())
    out=eval_data[KEYS+["top3_label","surface","distance_m","race_class"]].copy()
    out["p_M0"]=eval_data.p_market_oof.to_numpy(float)
    for label in F_NAMES:
        out["p_"+label]=np.nan
    folds=[]
    for year in (2023,2024):
        train=frame.loc[frame.race_date.dt.year.between(2018,year-1)]
        test=_sort_keyed(eval_data.loc[eval_data.race_date.dt.year.eq(year)])
        if train.empty or test.empty or train.p_market_oof.isna().any():
            raise ValueError("Missing prior-only training market OOF")
        if not train.race_date.max()<test.race_date.min():
            raise ValueError("Outer fold training touches evaluation period")
        mask=out.race_date.dt.year.eq(year)
        if not out.loc[mask,KEYS].reset_index(drop=True).equals(
            test[KEYS].reset_index(drop=True)
        ):
            raise ValueError("Prediction row order changed")
        for label,(numeric,categorical) in registry.items():
            p=_fit_xgb(train,test,numeric,categorical)
            out.loc[mask,"p_"+label]=p
        folds.append({"outer_year":year,"training_end_date":str(
            train.race_date.max().date()),"training_rows":len(train),
            "eval_races":int(test.race_id.nunique()),"eval_rows":len(test)})
    if out.isna().any(axis=0).loc[[f"p_{x}" for x in F_NAMES]].any():
        raise ValueError("Some predictions missing")
    if not np.allclose(
        out.p_F0_X2.to_numpy(float),
        reference.p_X2_nonlinear_strength.to_numpy(float),
        atol=1e-8,rtol=0,
    ):
        maxdiff=np.max(np.abs(
            out.p_F0_X2.to_numpy(float)-
            reference.p_X2_nonlinear_strength.to_numpy(float)))
        raise ValueError(f"F0 fails numerical parity against frozen X2: {maxdiff}")
    if not np.allclose(
        out.p_M0.to_numpy(float),
        reference.p_M0_market.to_numpy(float),
        atol=1e-10,rtol=0,
    ):
        raise ValueError("Existing market-only OOF probability changed")
    return out,{"folds":folds,"F0_reproduces_frozen_X2":True,
                 "market_reproduces_frozen_M0":True,
                 "num_features":{k:len(v[0]) for k,v in registry.items()},
                 "categorical_features":{k:v[1] for k,v in registry.items()}}


def evaluate_results(frame: pd.DataFrame, e5_path: Path, out: Path) -> dict:
    if len(frame)!=63276 or frame.race_id.nunique()!=5731:
        raise ValueError("Source common sample size changed")
    preds={"M0":frame.p_M0.to_numpy(float)}
    preds.update({x:frame["p_"+x].to_numpy(float) for x in F_NAMES})
    if row_fingerprint(frame)!=REFERENCE_YEAR_FINGERPRINT:
        raise ValueError("Wrong pooled outer cohort fingerprint")
    selection,scores=select_with_incumbent_one_se(
        frame,preds,incumbent="F0_X2")
    scores.to_csv(out/"A_all_years_scores.csv",index=False)
    rec=[]
    for name,p in preds.items():
        diff=paired_delta(frame,preds["F0_X2"],p,
                          candidate_name="F0_X2",reference_name=name)
        rec.append({"candidate":name,"baseline_x2_minus_candidate":diff.mean_brier_delta,
                    "date_cluster_se":diff.se_brier_delta,
                    "pairwise_one_se_pass":diff.mean_brier_delta>diff.se_brier_delta,
                    **candidate_score(frame,p,name=name).__dict__})
    pd.DataFrame(rec).to_csv(out/"A_all_years_paired.csv",index=False)
    frame.to_csv(out/"A_outer_oof_predictions.csv",index=False)
    test=_sort_keyed(frame.loc[frame.race_date.dt.year.eq(2024)])
    # This is just a reference from PR33, and NEVER enters candidate training
    # or inner-year calibration. E5 was fit on 2023 OOF in PR33.
    e5=pd.read_csv(e5_path,dtype={"race_id":str,"horse_id":str})
    e5.race_date=pd.to_datetime(e5.race_date)
    e5=_sort_keyed(e5)
    if not test[KEYS].equals(e5[KEYS]):
        raise ValueError("Original E5 comparison sample keys differ")
    if not np.allclose(test.p_F0_X2,e5.p_X2,atol=1e-10,rtol=0):
        raise ValueError("E5 comparison refers to different X2 predictions")
    comp={"M0":test.p_M0.to_numpy(float),"E5_frozen":e5.p_E5.to_numpy(float)}
    comp.update({name:test["p_"+name].to_numpy(float) for name in F_NAMES})
    verdict,score=select_with_incumbent_one_se(test,comp,incumbent="E5_frozen")
    score.to_csv(out/"A_2024_E5_comparison.csv",index=False)
    paired=[]
    losses=[]
    bins=[]
    calibration=[]
    subgroups=[]
    for name,p in comp.items():
        diff=paired_delta(test,comp["E5_frozen"],p,
                          candidate_name="E5_frozen",reference_name=name)
        paired.append({"candidate":name,"e5_minus_candidate":diff.mean_brier_delta,
                       "date_cluster_se":diff.se_brier_delta,
                       "exceeds_one_se":bool(diff.mean_brier_delta>diff.se_brier_delta)})
        l=race_loss_table(test,p)
        l["candidate"]=name
        losses.append(l)
        rel=calibration_table(test,p)
        rel["candidate"]=name
        bins.append(rel)
        a,b=calibration_intercept_slope(test.top3_label.to_numpy(int),p)
        calibration.append({"candidate":name,"intercept":a,"slope":b})
        groups={
            "surface":test.surface.astype(str),
            "distance_band":pd.cut(test.distance_m,[999,1400,2000,2600],
               labels=["1000_1400","1401_2000","2001_2600"]).astype(str),
            "race_class":test.race_class.fillna("MISSING").astype(str),
        }
        for group,values in groups.items():
            for value in sorted(values.unique()):
                selector=values.eq(value)
                score=candidate_score(test.loc[selector],p[selector.to_numpy()],
                                      name=name)
                subgroups.append({"candidate":name,"group":group,"value":value,
                                  **score.__dict__})
    pd.DataFrame(paired).to_csv(out/"A_2024_E5_paired.csv",index=False)
    pd.DataFrame(calibration).to_csv(out/"A_2024_calibration.csv",index=False)
    pd.DataFrame(subgroups).to_csv(out/"A_2024_subgroups.csv",index=False)
    pd.concat(bins).to_csv(out/"A_2024_reliability.csv",index=False)
    pd.concat(losses).to_csv(out/"A_2024_race_losses.csv",index=False)
    return {"outer_years":{"winner":selection.winner,"incumbent_retained":
                              bool(selection.incumbent_retained)},
            "2024_vs_E5":{"winner":verdict.winner,"incumbent_retained":
                          bool(verdict.incumbent_retained)},
            "2024_fingerprint":row_fingerprint(test)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--historical-dir",type=Path,required=True)
    p.add_argument("--results-csv",type=Path,required=True)
    p.add_argument("--races-csv",type=Path,required=True)
    p.add_argument("--reference-oof",type=Path,required=True)
    p.add_argument("--e5-2024",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    print("A: audit original source and create prior-only full-field features",flush=True)
    frame,audit=prepare_dataset(
        args.historical_dir,args.results_csv,args.races_csv)
    (out/"A_source_audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False))
    ref=check_reference(frame,args.reference_oof)
    print("A: train frozen XGBoost F0..F3 on 2023/24 outer folds",flush=True)
    pred,train=do_outer(frame,ref)
    print("A: evaluate common Brier and frozen E5 comparator",flush=True)
    verdict=evaluate_results(pred,args.e5_2024,out)
    manifest={"status":"HISTORICAL_FINAL_ODDS_DEVELOPMENT_ONLY",
              "pre_registration_sha":"f4d12b571b27a0309f760ba60317f5eee277cbdb",
              "training":train,"verdict":verdict,
              "source_oof_sha256":hashfile(args.reference_oof),
              "e5_oof_sha256":hashfile(args.e5_2024),
              "source_audit":audit,
              "script_sha256":hashfile(Path(__file__)),
              "python":platform.python_version(),"xgboost":xgb_version,
              "seed":20261031,"no_production_change":True}
    (out/"A_manifest.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":manifest["status"],"verdict":verdict,
                      "training":train},indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
