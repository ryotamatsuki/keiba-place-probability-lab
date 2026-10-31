"""Unchanged frozen XGB800+CatBoost400 independent 2025 JRA trio holdout.

Load THE ORIGINAL frozen November F3 binary artifact. No train/fit calls.
2025 horse results and final-win odds are out-of-time; previous-race histories
are lagged, and model calibration is the frozen 2016-2024 object.

Historical win odds here are FINAL, NOT archived T-10 observations.
Only complete, source-matched and fully eligible starter fields are scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool
from scipy.special import expit, logit

from keiba_place_lab.ensemble import apply_logit_calibrator
from keiba_place_lab.historical_market import reconstruct_historical_market
from keiba_place_lab.historical_panel import build_historical_panel
from keiba_place_lab.information_expansion_v1 import (
    add_historical_result_features, add_opponent_elo, raw_observations,
)
from keiba_place_lab.input_learning_v1 import (
    add_calibrated_market_splits, attach_information, full_starter_market_context,
)
from keiba_place_lab.live_history import load_frozen_history
from keiba_place_lab.market import harville_top_k_probabilities, normalized_win_probabilities
from keiba_place_lab.nonmarket import engineer_features
from keiba_place_lab.scope_expansion import straight_course_mask
from keiba_place_lab.tennosho_forward import locked_bundle
from keiba_place_lab.trio_bridge_v1 import (
    invert_subset_marginals, plackett_luce_trio, project_exactly_three,
)
from keiba_place_lab.trio_residual_v2 import (
    CANDIDATE_SETS, _prepare, residual_distribution, score_probabilities,
)

# This cut excludes the 4 independently documented late-2025 cards where
# the original Kaggle source has missing runners and the frozen standardized
# supplement contains 7 verified mismatched horse identities. No corrections.
SOURCE_TEST_LAST_DATE = pd.Timestamp("2025-12-26")
TARGET_RACE = "202605041111"
TOLERANCE_FORWARD_PARITY = 1e-8


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_jra_id(value: object) -> str:
    a = str(value).removeprefix("jra:")
    if re.fullmatch(r"\d{10}", a) is None:
        raise ValueError("Cannot reconcile a frozen historical JRA horse ID")
    return "jra:" + a


def frozen_model_probabilities(model: dict, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Exact inference algorithm from keiba_place_lab.tennosho_forward, no fitting."""
    num, cat = model["numeric"], model["categorical"]
    absent = set(num+cat) - set(frame)
    if absent:
        raise ValueError("Incomplete F3 feature input " + str(sorted(absent)))
    if "market_p_top3" not in frame:
        raise ValueError("Missing full-field market Harville baseline")
    matrix = frame[num+cat].copy()
    for name in num:
        matrix[name] = pd.to_numeric(matrix[name], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
    for name in cat:
        matrix[name] = matrix[name].astype("object").where(
            matrix[name].notna(), None
        )
    pcal = apply_logit_calibrator(
        frame.market_p_top3.to_numpy(float), model["market_calibrator"]
    )
    if "market_calibrated_top3_p" in matrix:
        assert np.max(np.abs(matrix.market_calibrated_top3_p - pcal)) < 1e-10
    offset = logit(pcal)
    pipeline = model["xgb"]
    transformed = pipeline.named_steps["preprocess"].transform(matrix)
    xgb = pipeline.named_steps["model"]
    p400 = xgb.predict_proba(transformed, base_margin=offset,
                             iteration_range=(0,400))[:,1]
    p800 = xgb.predict_proba(transformed, base_margin=offset,
                             iteration_range=(0,800))[:,1]
    native = matrix.copy()
    for name in cat:
        native[name] = native[name].astype("string").fillna("MISSING").astype("object")
    cbo = model["catboost"]
    with_offset = cbo.predict(Pool(native, cat_features=cat, baseline=offset),
                              prediction_type="RawFormulaVal")
    no_offset = cbo.predict(Pool(native, cat_features=cat),
                            prediction_type="RawFormulaVal")
    raw = np.asarray(with_offset, dtype=float).reshape(-1)
    raw_zero = np.asarray(no_offset, dtype=float).reshape(-1)
    if np.allclose(raw-raw_zero,offset,atol=1e-5,rtol=1e-5):
        total_logit = raw
    elif np.allclose(raw,raw_zero,atol=1e-7,rtol=0):
        total_logit = raw_zero+offset
    else:
        raise ValueError("Frozen CatBoost offset is not applied exactly once")
    pcat = expit(total_logit)
    pjoint = expit(offset + .75*(logit(p800)-offset)
                   + .25*(logit(pcat)-offset))
    if not np.all(np.isfinite(pjoint)) or np.any((pjoint<=0)|(pjoint>=1)):
        raise ValueError("Wrong frozen F3 probabilities")
    return {
        "p_XGB400":p400, "p_XGB800":p800,
        "p_CatBoost400":pcat, "p_XC_75_25_logit":pjoint,
        "p_market_frozen_cal":pcal,
    }


def verify_2026_existing_preview_parity(
    model: dict, preview_csv: Path, prior_csv: Path,
) -> dict:
    p = pd.read_csv(preview_csv, dtype={"race_id":str,"horse_id":str})
    prior = pd.read_csv(prior_csv, dtype={"race_id":str,"horse_id":str})
    key = ["race_id","horse_id","horse_no"]
    if (len(p)!=17 or len(prior)!=17 or p.duplicated(key).any()
            or prior.duplicated(key).any() or set(p.race_id)!={TARGET_RACE}):
        raise ValueError("Cannot verify original pre-off 17-runner parity")
    x=prior.merge(p[key+["win_odds"]],on=key,how="inner",validate="one_to_one")
    if len(x)!=17 or x.horse_no.nunique()!=17:
        raise ValueError("17-runner preview and frozen prior mismatch")
    x=x.sort_values("horse_no").reset_index(drop=True)
    win, _=normalized_win_probabilities(x.win_odds.to_numpy(float))
    harville=harville_top_k_probabilities(win,k=3)
    market=x[key].copy()
    market["market_p_win"]=win
    market["market_p_top3"]=harville
    market["market_field_size"]=len(x)
    f=full_starter_market_context(market)
    x=x.merge(f,on=["race_id","horse_no"],how="left",validate="one_to_one")
    x["market_p_top3"]=harville
    calibrated=apply_logit_calibrator(harville,model["market_calibrator"])
    x["market_calibrated_top3_p"]=calibrated
    x["market_calibrated_top3_logit"]=logit(calibrated)
    expected=p.sort_values("horse_no").reset_index(drop=True)
    out=frozen_model_probabilities(model,x)
    deviations={}
    for got, ref in (("p_XGB400","p_B_raw_400"),("p_XGB800","p_XGB800_raw"),
                     ("p_CatBoost400","p_CatBoost400_raw"),
                     ("p_XC_75_25_logit","p_XC_75_25_logit")):
        deviations[ref]=float(np.max(np.abs(out[got]-expected[ref].to_numpy(float))))
    if max(deviations.values())>TOLERANCE_FORWARD_PARITY:
        raise ValueError(f"2025 holdout prediction does not match frozen 2026 F3: {deviations}")
    return {"exact_frozen_model_parity_max_abs":deviations,
            "preview_sha256":sha(preview_csv),"prior_sha256":sha(prior_csv)}


def build_2025_holdout(
    standardized_parquet: Path, results: Path, races_csv: Path, model: dict,
) -> tuple[pd.DataFrame, dict]:
    frozen=load_frozen_history(standardized_parquet)
    frozen["race_date"]=pd.to_datetime(frozen.race_date)
    frozen["race_id"]=frozen.race_id.astype(str)
    # Strictly forbid original-source-missing last four 2025 JRA cards.
    frozen=frozen.loc[frozen.race_date<=SOURCE_TEST_LAST_DATE].copy()
    frozen["horse_id"]=frozen.horse_id.map(as_jra_id)
    if frozen.duplicated(["race_id","horse_id"]).any():
        raise ValueError("Frozen JRA source has repeated horse/race IDs")
    if frozen.race_date.dt.year.min()!=2010 or frozen.race_date.dt.year.max()!=2025:
        raise ValueError("Historical test source missing 2010-2025 history")
    raw=build_historical_panel(frozen)
    observations,audit=raw_observations(raw,results,races_csv,
                                       horse_id_normalizer=as_jra_id)
    if observations.loc[observations.race_date.dt.year.le(2024),
                        "source_horse_no"].isna().any():
        raise ValueError("2010-2024 original JRA raw history no longer matches")
    # Source data must be complete for every 2025 sample participant. Other
    # invalid races are excluded and fully audited before joint probability.
    rich=add_historical_result_features(observations)
    rich=add_opponent_elo(rich)
    past2016=rich.loc[rich.race_date.dt.year.eq(2016)]
    prior=float(past2016.top3_label.mean())
    if not .05<prior<.5:
        raise ValueError("Frozen historical prior cannot be reconstructed")
    rich=engineer_features(rich,prior_mean=prior)
    test=rich.loc[rich.race_date.dt.year.eq(2025)].copy()
    all_groups=test.groupby("race_id",sort=False)
    required=(
        test.surface.isin(["turf","dirt"])
        & test.distance_m.between(1000,2600)
        & test.field_size.ge(8)
        & test.career_starts.ge(3)
        & ~straight_course_mask(test)
    )
    eligible=test.loc[required].copy()
    counts=eligible.groupby("race_id").size()
    actual=all_groups.size()
    full_ids=counts.index[counts.eq(actual.reindex(counts.index))]
    eligible=eligible.loc[eligible.race_id.isin(full_ids)]
    source_verified=eligible.groupby("race_id").apply(
        lambda g: bool(
            g.source_horse_no.notna().all()
            and g.jockey_name.notna().all()
            and g.trainer.notna().all()
            and g.raw_finish_position.notna().sum()>=3
            and g.top3_label.sum()==3
            and set(g.loc[g.top3_label.eq(1),"raw_finish_position"].astype(int))
                =={1,2,3}
            and g.raw_finish_position.between(1,3).eq(
                g.top3_label.astype(bool)).all()
        ),include_groups=False
    )
    eligible=eligible.loc[eligible.race_id.isin(
        source_verified.index[source_verified]
    )].copy()
    if eligible.empty:
        raise ValueError("No 2025 fully matched starter races")
    market, market_audit=reconstruct_historical_market(
        results,set(eligible.race_id)
    )
    mcount=market.groupby("race_id").size()
    fcount=eligible.groupby("race_id").size()
    stable=mcount.index[mcount.eq(fcount.reindex(mcount.index))]
    eligible=eligible.loc[eligible.race_id.isin(stable)].copy()
    market=market.loc[market.race_id.isin(stable)].copy()
    if eligible.empty:
        raise ValueError("No 2025 eligible race with full source win odds")
    joined,coverage=attach_information(eligible,rich,market)
    market_subset=market[["race_id","horse_no","market_p_top3","market_p_win",
                          "historical_win_odds"]]
    joined=joined.merge(
        market_subset,on=["race_id","horse_no"],
        how="left",validate="one_to_one",indicator=True,
    )
    if not joined._merge.eq("both").all():
        raise ValueError("2025 market source and eligible horse not aligned")
    joined.drop(columns="_merge",inplace=True)
    joined["p_market_oof"]=apply_logit_calibrator(
        joined.market_p_top3.to_numpy(float),model["market_calibrator"]
    )
    joined=add_calibrated_market_splits(joined)
    joined=joined.sort_values(["race_date","race_id","horse_no"]).reset_index(drop=True)
    if set(joined.race_date.dt.year)!={2025}:
        raise ValueError("2025 test contains other years")
    cohort={
        "all_2025_source_races":int(test.race_id.nunique()),
        "all_2025_source_rows":len(test),
        "full_2025_cohort_before_rank_source_check":int(len(full_ids)),
        "full_after_strict_original_rank_and_identity_check":int(source_verified.sum()),
        "full_after_market_check":int(joined.race_id.nunique()),
        "2025_scored_rows":len(joined),
        "first_scored_date":str(joined.race_date.min().date()),
        "last_scored_date":str(joined.race_date.max().date()),
        "raw_audit":audit,
        "market_audit":market_audit,
        "relative_coverage":coverage.to_dict(orient="records"),
        "standardized_2010_2025_source_sha256":sha(standardized_parquet),
        "raw_results_sha256":sha(results),
        "raw_races_sha256":sha(races_csv),
        "exclude_late_incomplete_2025_cards":True,
        "historical_win_odds_are_FINAL_not_T10":True,
    }
    return joined,cohort


def race_joint_summary(frame:pd.DataFrame, fitted:dict,
                       model_coefs:dict) -> tuple[pd.DataFrame,dict]:
    historic=frame.copy()
    pred=frozen_model_probabilities(fitted,historic)
    historic["p_f3_frozen"]=pred["p_XC_75_25_logit"]
    historic["p_xgb800_frozen"]=pred["p_XGB800"]
    historic["p_cat400_frozen"]=pred["p_CatBoost400"]
    # The 2023-trained trio residual is a proxy-domain transfer hypothesis,
    # not a refit to 2025 or a post-hoc replacement for the F3 bundle.
    betas={
        "v1_raw": ((),np.array([],dtype=float)),
        "v2_2023_transfer": (
            CANDIDATE_SETS["race_interactions"],
            np.asarray(model_coefs["model_candidates"]["race_interactions"]["beta"],float)
        ),
        "v2_market_hybrid_transfer": (
            CANDIDATE_SETS["market_hybrid"],
            np.asarray(model_coefs["model_candidates"]["market_hybrid"]["beta"],float)
        ),
    }
    records=[]
    preview=[]
    for i,(race_id,g) in enumerate(historic.groupby("race_id",sort=False)):
        g=g.sort_values("horse_no").reset_index(drop=True)
        horses=g.horse_no.to_numpy(int)
        q=project_exactly_three(g.p_f3_frozen.to_numpy(float))
        weights=invert_subset_marginals(q)
        shares=g.market_p_win.to_numpy(float)
        if not np.isclose(shares.sum(),1,atol=1e-10):
            raise ValueError("2025 entire market full race is missing")
        market_pl=plackett_luce_trio(shares,horses)
        item=_prepare(
            horses,weights,q,
            g.recent4_early_pos_pct_mean.to_numpy(float),
            g.draw_pct.to_numpy(float),
            market_pl.probability.to_numpy(float),
            market_pl[["horse_1","horse_2","horse_3"]].to_numpy(int),
        )
        winner=tuple(sorted(horses[g.top3_label.to_numpy(int).astype(bool)].tolist()))
        idx=np.flatnonzero(np.all(item["trios"]==winner,axis=1))
        if len(idx)!=1:
            raise ValueError("Actual trio not included in 2025 race")
        row={
            "race_date":g.race_date.iloc[0].date().isoformat(),
            "race_id":race_id, "field_size":len(g),
            "distance_m":int(g.distance_m.iloc[0]),
            "surface":str(g.surface.iloc[0]),
            "actual_trio":"-".join(map(str,winner)),
            "f3_probability_sum":float(g.p_f3_frozen.sum()),
            "f3_max_marginal_projection_shift":float(np.max(
                np.abs(q-g.p_f3_frozen.to_numpy(float))
            )),
        }
        pp=market_pl.probability.to_numpy(float)
        for name,p in (("market_pl",pp),):
            for key,v in score_probabilities(p,int(idx[0])).items():
                # PL probability row order differs! Align exact ticket keys first.
                pass
        # Align market PL to trio enumeration (market_pl sorted descending).
        market_map={
            tuple(sorted((int(x.horse_1),int(x.horse_2),int(x.horse_3)))):float(x.probability)
            for x in market_pl.itertuples(index=False)
        }
        pp=np.array([market_map[tuple(v)] for v in item["trios"]])
        for key,v in score_probabilities(pp,int(idx[0])).items():
            row["market_pl_"+key]=v
        for name,(features,beta) in betas.items():
            probability=residual_distribution(item,beta,features)
            for key,v in score_probabilities(probability,int(idx[0])).items():
                row[name+"_"+key]=v
        records.append(row)
        if i<3:
            # Safety/inspectability only; NOT selection or model fitting.
            preview.append({"race_id":race_id,"count":len(g),"actual":winner})
    scores=pd.DataFrame(records)
    from research_trio_f3_v1 import paired_day_ci
    methods=["market_pl","v1_raw","v2_2023_transfer","v2_market_hybrid_transfer"]
    summary={name:{
        "joint_logloss":float(scores[name+"_joint_logloss"].mean()),
        "joint_brier":float(scores[name+"_joint_brier"].mean()),
        "top10_coverage":float(scores[name+"_top10_hit"].mean()),
        "top10_mass":float(scores[name+"_top10_mass"].mean()),
    } for name in methods}
    differences={}
    for name in ("v1_raw","v2_2023_transfer","v2_market_hybrid_transfer"):
        differences["market_pl_minus_"+name]=paired_day_ci(
            scores,"market_pl_joint_logloss",name+"_joint_logloss",draws=2500
        )
    for name in ("v2_2023_transfer","v2_market_hybrid_transfer"):
        differences["v1_raw_minus_"+name]=paired_day_ci(
            scores,"v1_raw_joint_logloss",name+"_joint_logloss",draws=2500
        )
        differences["v1_raw_minus_"+name]["joint_brier_improvement"]=float(
            summary["v1_raw"]["joint_brier"]-summary[name]["joint_brier"]
        )
    # Race-level Brier; no per-ticket payout or ROI available.
    return scores,{"methods":summary,"paired_day_ci":differences,
                   "valid_2025_full_races":len(records),"examples":preview}


def main():
    p=argparse.ArgumentParser()
    for name in (
        "standardized","results","races","bundle","frozen-manifest",
        "preview","prior","v2-2023-fit","out"
    ):
        p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError("Must not overwrite existing research outputs")
    frozen,meta=locked_bundle(a.bundle,a.frozen_manifest)
    if meta["bundle_sha256"]!=sha(a.bundle):
        raise ValueError("Frozen ensemble joblib file hash changed")
    if meta.get("training_max_date")!="2024-12-28":
        raise ValueError("Model training includes future time leakage")
    parity=verify_2026_existing_preview_parity(frozen,a.preview,a.prior)
    print("EXACT_FROZEN_2026_PREVIEW_PARITY="+json.dumps(parity),flush=True)
    frame,audit=build_2025_holdout(a.standardized,a.results,a.races,frozen)
    print("2025_INDEPENDENT_SOURCE_COHORT="+json.dumps({
        "eligible_races":audit["full_after_market_check"],
        "runners":audit["2025_scored_rows"],
        "dates":[audit["first_scored_date"],audit["last_scored_date"]],
        "raw_unmatched":audit["raw_audit"]["unmatched_raw_runner_rows"],
    }),flush=True)
    fitted=json.loads(a.v2_2023_fit.read_text())
    if (fitted["fit_only_year"]!=2023 or
            fitted["model"]!="trio_v2_2023_historical_f3_120tree_PROXY"):
        raise ValueError("Invalid historical 2023 fit used for V2 transfer")
    scores,experiment=race_joint_summary(frame,frozen,fitted)
    a.out.mkdir(parents=True,exist_ok=True)
    scores.to_csv(a.out/"2025_frozen_F3_trio_race_scores.csv",index=False)
    scored=frame[["race_id","race_date","horse_id","horse_no",
                   "top3_label","historical_win_odds"]].copy()
    probabilities=frozen_model_probabilities(frozen,frame)
    for name,prob in probabilities.items():
        scored[name]=prob
    scored.to_csv(a.out/"2025_frozen_F3_runner_predictions.csv",index=False)
    audit.update({
        "status":"STRICT_2025_OUT_OF_TIME_SAME_FROZEN_ENSEMBLE_FINAL_ODDS_ONLY",
        "source_model_sha":meta["bundle_sha256"],
        "frozen_at_utc":meta["frozen_at_utc"],
        "frozen_F3_model_versions":meta["model_rounds"],
        "frozen_2026_preview_parity":parity,
        "2025_trio_model":experiment,
        "v2_transfer_trained_only_on_2023_120tree_proxy":True,
        "2025_race_result_used_only_as_ground_truth_and_lagged_past":True,
        "historical_market_final_odds_not_T10":True,
        "2026_tennosho_f3_lock_unmodified":True,
        "no_betting_or_model_promotion":True,
        "model_fit_and_predict_code_unchanged":True,
    })
    (a.out/"validation.json").write_text(
        json.dumps(audit,indent=2,ensure_ascii=False)+"\n"
    )
    print("FROZEN_2025_HOLDOUT_RESULT="+json.dumps({
        "cohort":audit["full_after_market_check"],
        "scores":experiment["methods"],
        "paired":experiment["paired_day_ci"],
    }),ensure_ascii=False,flush=True)


if __name__=="__main__":
    main()
