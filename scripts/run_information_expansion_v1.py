"""Full all-distance/dual-surface final-odds market-offset research V1.

All fitting is strictly walk-forward. Uses frozen Stage3.6 official-day panels.
Output cannot be used to revise canonical live prediction locks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from xgboost import __version__ as xgboost_version

from keiba_place_lab.ensemble import (
    apply_logit_calibrator,
    fit_logit_calibrator,
    probability_logit,
)
from keiba_place_lab.historical_market import reconstruct_historical_market
from keiba_place_lab.information_expansion_v1 import (
    NEW_HISTORY,
    NEW_STRENGTH,
    add_historical_result_features,
    add_opponent_elo,
    raw_observations,
)
from keiba_place_lab.model_selection import (
    paired_delta,
    race_loss_table,
    select_with_incumbent_one_se,
)
from keiba_place_lab.nonmarket import (
    calibration_table,
    engineer_features,
    feature_columns,
)
from keiba_place_lab.probability_improvement_v1 import (
    fit_market_offset,
    predict_market_offset,
)
from keiba_place_lab.scope_expansion import straight_course_mask
from keiba_place_lab.stage4_successor import calibration_intercept_slope

SOURCE_SHA = "fb9345273b21a7c23d41260dc77e45134a1bc2fe756899f3416b24ac0dd51de3"
KEYS = ["race_date","race_id","horse_id","horse_no"]
FIXED_MODEL = {
    "n_estimators":120,"max_depth":2,"learning_rate":0.03,
    "min_child_weight":60,"subsample":0.8,"colsample_bytree":0.8,
    "reg_lambda":20.0,"reg_alpha":2.0,"tree_method":"hist","n_jobs":2,
    "random_state":20261031,"objective":"binary:logistic","eval_metric":"logloss",
}
CANDIDATES = ("M0_market","L0_linear_old","X0_nonlinear_old",
              "X1_nonlinear_history","X2_nonlinear_strength")


def hashfile(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def load_full_panel(folder: Path) -> tuple[pd.DataFrame, dict]:
    files = ("jra_flat_warmup_v1.parquet","jra_flat_train_v1.parquet",
             "jra_flat_validation_v1.parquet")
    parts, hashes = [], {}
    for name in files:
        path = folder/name
        if not path.exists():
            raise FileNotFoundError(f"Missing audited Stage3.6 partition: {path}")
        hashes[name] = hashfile(path)
        parts.append(pd.read_parquet(path))
    full = pd.concat(parts,ignore_index=True,sort=False)
    full["race_date"] = pd.to_datetime(full.race_date)
    for col in ("race_id","horse_id"):
        full[col] = full[col].astype(str)
    if full.race_date.dt.year.min()!=2010 or full.race_date.dt.year.max()!=2024:
        raise ValueError("Historical panel must span 2010..2024 exactly")
    if full.duplicated(["race_id","horse_id"]).any():
        raise ValueError("Duplicate official full starter identities")
    return full, hashes


def eligibility(full: pd.DataFrame) -> pd.DataFrame:
    years = full.race_date.dt.year
    mask = (
        years.between(2016,2024)
        & full.surface.isin(("turf","dirt"))
        & full.distance_m.between(1000,2600)
        & full.field_size.ge(8)
        & full.career_starts.ge(3)
        & ~straight_course_mask(full)
    )
    return full.loc[mask].copy()


def merge_complete_market(eligible: pd.DataFrame, source: Path) -> tuple[pd.DataFrame,dict]:
    if hashfile(source)!=SOURCE_SHA:
        raise ValueError("Final-odds source SHA256 mismatch")
    wanted=set(eligible.race_id)
    market, audit = reconstruct_historical_market(source,wanted)
    joined=eligible.merge(
        market,on=["race_id","horse_no"],how="left",
        validate="one_to_one",indicator=True,
    )
    missing=joined["_merge"].ne("both") | joined.market_p_win.isna() | (
        joined.market_field_size != joined.field_size
    )
    invalid_races=set(joined.loc[missing,"race_id"])
    joined=joined.loc[~joined.race_id.isin(invalid_races)].copy()
    joined.drop(columns=["_merge"],inplace=True)
    if joined.empty or joined.duplicated(KEYS).any():
        raise ValueError("No common eligible final market")
    audit["excluded_races_after_full_field_check"]=len(invalid_races)
    audit["paired_eligible_races"]=int(joined.race_id.nunique())
    audit["paired_eligible_rows"]=len(joined)
    return joined,audit


def build_feature_panel(full: pd.DataFrame, result_csv: Path,
                        race_csv: Path) -> tuple[pd.DataFrame,dict]:
    raw,audit = raw_observations(full,result_csv,race_csv)
    history = add_historical_result_features(raw)
    rich = add_opponent_elo(history)
    # The prior is fixed from *2016 only*; each OOF training year starts in 2018.
    prior = float(rich.loc[rich.race_date.dt.year.eq(2016),"top3_label"].mean())
    if not (0.05<prior<0.5):
        raise ValueError("Invalid strictly pre-2018 prior")
    rich = engineer_features(rich,prior_mean=prior)
    audit["2016_training_prior"]=prior
    audit["jra_2010_2024_races"]=int(rich.race_id.nunique())
    audit["jra_2010_2024_rows"]=len(rich)
    audit["information_coverage_by_year_surface"]=(
        rich.assign(year=rich.race_date.dt.year)
        .groupby(["year","surface"],dropna=False)
        .agg(
            rows=("race_id","size"),
            last3f=("raw_last3f",lambda s:int(s.notna().sum())),
            passing=("raw_passing_gain",lambda s:int(s.notna().sum())),
            mass=("raw_mass",lambda s:int(s.notna().sum())),
            going=("raw_wet",lambda s:int(s.notna().sum())),
            jockey=("jockey_name",lambda s:int(s.notna().sum())),
            trainer=("trainer",lambda s:int(s.notna().sum())),
        ).reset_index().to_dict(orient="records")
    )
    return rich,audit


def _sort_keyed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(KEYS,kind="stable").reset_index(drop=True)


def fold_calibrated_market(frame: pd.DataFrame) -> np.ndarray:
    """Yearly chronological OOF logit calibrators for both meta-train/eval."""
    output=np.full(len(frame),np.nan)
    years=frame.race_date.dt.year.to_numpy()
    for year in range(2018,2025):
        fit=frame.loc[(years>=2016)&(years<year)]
        target=np.where(years==year)[0]
        if not len(target):
            continue
        if fit.empty or fit.race_date.max() >= pd.Timestamp(f"{year}-01-01"):
            raise ValueError("Market calibration crosses year fold")
        cal=fit_logit_calibrator(
            fit.market_p_top3.to_numpy(float),
            fit.top3_label.to_numpy(int),
        )
        output[target]=apply_logit_calibrator(
            frame.iloc[target].market_p_top3.to_numpy(float),cal,
        )
    return output


def _features():
    numeric,categorical=feature_columns()
    numeric=list(dict.fromkeys(numeric))
    categorical=list(dict.fromkeys(categorical))
    return numeric,categorical


def _build_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    # Old/new offset fit uses identical preprocessing and XGB settings.
    preprocess=ColumnTransformer([
        ("num","passthrough",numeric),
        ("cat",Pipeline([
            ("impute",SimpleImputer(strategy="constant",fill_value="MISSING")),
            ("onehot",OneHotEncoder(
                handle_unknown="ignore",min_frequency=5,sparse_output=False,
            )),
        ]),categorical),
    ],remainder="drop",sparse_threshold=0.0)
    model=XGBClassifier(**FIXED_MODEL)
    return Pipeline([("preprocess",preprocess),("model",model)])


def _fit_xgb(train: pd.DataFrame, evaluate: pd.DataFrame,
             predictors: list[str],categorical: list[str]) -> np.ndarray:
    train=train.copy()
    evaluate=evaluate.copy()
    for col in predictors:
        train[col]=pd.to_numeric(train[col],errors="coerce")
        evaluate[col]=pd.to_numeric(evaluate[col],errors="coerce")
    for col in categorical:
        train[col]=train[col].astype("object").where(train[col].notna(),None)
        evaluate[col]=evaluate[col].astype("object").where(evaluate[col].notna(),None)
    matrix=predictors+categorical
    fit=_build_pipeline(predictors,categorical)
    # Every race has unit mass to align training objective with race-macro Brier.
    race_counts=train.groupby("race_id").race_id.transform("size").to_numpy(float)
    w=1.0/race_counts
    w*=len(w)/w.sum()
    fit.fit(
        train[matrix],train.top3_label.astype(int),
        model__sample_weight=w,
        model__base_margin=probability_logit(train.p_market_oof.to_numpy(float)),
    )
    transformed=fit.named_steps["preprocess"].transform(evaluate[matrix])
    result=fit.named_steps["model"].predict_proba(
        transformed,base_margin=probability_logit(
            evaluate.p_market_oof.to_numpy(float)))[:,1]
    if not np.isfinite(result).all() or ((result<0)|(result>1)).any():
        raise ValueError("XGBoost offset predictions not probabilities")
    return np.asarray(result,float)


def _fit_linear(train: pd.DataFrame,evaluate: pd.DataFrame,
                predictors: list[str]) -> np.ndarray:
    a=train[predictors].apply(pd.to_numeric,errors="coerce")
    b=evaluate[predictors].apply(pd.to_numeric,errors="coerce")
    # Imputation fit only on pre-evaluation training rows.
    impute=SimpleImputer(strategy="median",add_indicator=True)
    x=impute.fit_transform(a)
    z=impute.transform(b)
    fit=fit_market_offset(
        train.p_market_oof.to_numpy(float),x,
        train.top3_label.to_numpy(int),train.race_date.to_numpy(),
        max_training_date=str(evaluate.race_date.min().date()),
        penalty=1.0,
    )
    return predict_market_offset(evaluate.p_market_oof.to_numpy(float),z,fit)


def predict_outer(frame: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    numeric,categorical=_features()
    if set(NEW_HISTORY+NEW_STRENGTH).intersection(numeric+categorical):
        raise ValueError("Extra history features already present in baseline")
    needed=numeric+categorical+list(NEW_HISTORY)+list(NEW_STRENGTH)
    absent=set(needed)-set(frame)
    if absent:
        raise ValueError(f"Missing historical/old features: {sorted(absent)}")
    frame=_sort_keyed(frame)
    frame["p_market_oof"]=fold_calibrated_market(frame)
    eval_rows=frame.loc[frame.race_date.dt.year.isin((2023,2024))].copy()
    eval_rows=_sort_keyed(eval_rows)
    result=eval_rows[KEYS+["top3_label","surface","distance_m","race_class"]].copy()
    for label in CANDIDATES:
        result["p_"+label]=np.nan
    manifest={"outer_folds":[],"model_spec":FIXED_MODEL}
    for year in (2023,2024):
        train=frame.loc[
            frame.race_date.dt.year.between(2018,year-1)
        ].copy()
        test=eval_rows.loc[eval_rows.race_date.dt.year.eq(year)].copy()
        if train.empty or test.empty:
            raise ValueError(f"Empty outer fold {year}")
        if train.p_market_oof.isna().any() or test.p_market_oof.isna().any():
            raise ValueError("Missing chronology-safe market OOF input")
        predictors={
            "M0_market":[],
            "L0_linear_old":numeric,
            "X0_nonlinear_old":numeric,
            "X1_nonlinear_history":numeric+list(NEW_HISTORY),
            "X2_nonlinear_strength":numeric+list(NEW_HISTORY)+list(NEW_STRENGTH),
        }
        output={
            "M0_market":test.p_market_oof.to_numpy(float),
            "L0_linear_old":_fit_linear(train,test,predictors["L0_linear_old"]),
        }
        for label in ("X0_nonlinear_old","X1_nonlinear_history",
                      "X2_nonlinear_strength"):
            output[label]=_fit_xgb(train,test,predictors[label],categorical)
        mask=result.race_date.dt.year.eq(year)
        original_keys=result.loc[mask,KEYS].reset_index(drop=True)
        if not original_keys.equals(test[KEYS].reset_index(drop=True)):
            raise ValueError("Outer predictions do not align with exact row keys")
        for name,p in output.items():
            result.loc[mask,"p_"+name]=p
        manifest["outer_folds"].append({
            "year":year,
            "train_2018_through":year-1,"train_rows":len(train),
            "train_max_date":str(train.race_date.max().date()),
            "test_rows":len(test),"test_races":int(test.race_id.nunique()),
        })
    if result[[f"p_{n}" for n in CANDIDATES]].isna().any().any():
        raise ValueError("OOF prediction missing in candidate")
    return result,manifest


def report(pred: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True,exist_ok=True)
    pred=_sort_keyed(pred)
    probs={name:pred[f"p_{name}"].to_numpy(float) for name in CANDIDATES}
    decision,scores=select_with_incumbent_one_se(
        pred,probs,incumbent="M0_market")
    pred.to_csv(out/"outer_oof_predictions.csv",index=False)
    scores.to_csv(out/"joint_scores.csv",index=False)
    paired,losses,calibration,reliability,subgroups=[],[],[],[],[]
    for name,p in probs.items():
        delta=paired_delta(pred,probs["M0_market"],p,
                           candidate_name="M0_market",reference_name=name)
        paired.append({
            "candidate":name,"market_minus_candidate_brier":delta.mean_brier_delta,
            "paired_date_clustered_se":delta.se_brier_delta,
            "pairwise_one_se_pass":delta.mean_brier_delta>delta.se_brier_delta,
            "market_minus_candidate_logloss":delta.mean_log_loss_delta,
        })
        l=race_loss_table(pred,p)
        l["candidate"]=name
        losses.append(l)
        intercept,slope=calibration_intercept_slope(
            pred.top3_label.to_numpy(int),p)
        calibration.append({
            "candidate":name,"intercept":intercept,"slope":slope,
        })
        table=calibration_table(pred,p)
        table["candidate"]=name
        reliability.append(table)
        grouped={
            "year":pred.race_date.dt.year.astype(str),
            "surface":pred.surface.astype(str),
            "distance_band":pd.cut(
                pred.distance_m,[999,1400,2000,2600],
                labels=["1000_1400","1401_2000","2001_2600"],
            ).astype(str),
            "race_class":pred.race_class.fillna("MISSING").astype(str),
        }
        for group,values in grouped.items():
            for value in sorted(values.unique()):
                mask=values.eq(value)
                s=pred.loc[mask].copy()
                if s.empty:
                    continue
                from keiba_place_lab.model_selection import candidate_score
                score=candidate_score(s,p[mask.to_numpy()],name=name)
                subgroups.append({
                    "candidate":name,"group":group,"value":value,
                    "race_macro_brier":score.race_macro_brier,
                    "race_macro_log_loss":score.race_macro_log_loss,
                    "races":score.races,"rows":score.rows,
                })
    pd.DataFrame(paired).to_csv(out/"paired_vs_market.csv",index=False)
    pd.DataFrame(calibration).to_csv(out/"calibration.csv",index=False)
    pd.DataFrame(subgroups).to_csv(out/"subgroups.csv",index=False)
    pd.concat(reliability).to_csv(out/"reliability.csv",index=False)
    loss=pd.concat(losses,ignore_index=True)
    loss.to_csv(out/"race_losses.csv",index=False)
    m=loss.loc[loss.candidate.eq("M0_market"),
                ["race_id","block","brier","log_loss"]].rename(
        columns={"brier":"market_brier","log_loss":"market_logloss"})
    differences=loss.merge(m,on=["race_id","block"],validate="many_to_one")
    differences["market_minus_candidate_brier"]=(
        differences.market_brier-differences.brier)
    differences.to_csv(out/"race_differences.csv",index=False)
    canonical=pred[KEYS].copy()
    canonical.race_date=canonical.race_date.dt.strftime("%Y-%m-%d")
    digest=hashlib.sha256(canonical.to_csv(index=False,lineterminator="\n").encode()).hexdigest()
    return {
        "winner":decision.winner,
        "incumbent_retained":bool(decision.incumbent_retained),
        "point_brier_best":decision.point_brier_best,
        "development_rows":len(pred),"development_races":int(pred.race_id.nunique()),
        "paired_row_fingerprint":digest,
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--historical-dir",type=Path,required=True)
    p.add_argument("--results-csv",type=Path,required=True)
    p.add_argument("--races-csv",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    print("Stage 1: frozen historical full starter panel",flush=True)
    full,source_hashes=load_full_panel(args.historical_dir)
    print("Stage 2: exact raw JRA reconciliation",flush=True)
    enriched,raw_audit=build_feature_panel(
        full,args.results_csv,args.races_csv)
    eligible=eligibility(enriched)
    print("Stage 3: reconstruct FINAL win market and all-runner exclusions",flush=True)
    paired,market_audit=merge_complete_market(eligible,args.results_csv)
    print("Stage 4: strictly chronological outer folds",flush=True)
    predictions,training=predict_outer(paired)
    selection=report(predictions,out)
    manifest={
        "status":"HISTORICAL_FINAL_ODDS_RESEARCH_ONLY",
        "pre_registered_spec":"docs/INFORMATION_EXPANSION_V1_SPEC.md",
        "source_sha256":{
            **source_hashes,"keiba_results.csv":hashfile(args.results_csv),
            "keiba_races.csv":hashfile(args.races_csv),
        },
        "code_sha256":{
            "runner":hashfile(Path(__file__)),
            "features":hashfile(Path(__file__).resolve().parents[1]/
                "src/keiba_place_lab/information_expansion_v1.py"),
        },
        "raw_audit":raw_audit,"market_audit":market_audit,
        "training":training,"selection":selection,
        "dependencies":{
            "python":platform.python_version(),
            "numpy":np.__version__,"pandas":pd.__version__,
            "sklearn":sklearn.__version__,"xgboost":xgboost_version,
            "seed":20261031,
        },
        "claim_limit":"Final win odds are not observed T-10 market snapshots",
        "production_unchanged":True,
    }
    (out/"manifest.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "selection":selection,"market_audit":market_audit
    },indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
