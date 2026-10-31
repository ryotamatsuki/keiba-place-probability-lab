"""First-priority V4: post-V1 arithmetic mixing on original 2023/2024 matched OOF.

No new learner training; weights selected from 2023 race outcomes alone.
2024 exact PR40 1336 races are descriptive development, never pristine test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research_trio_f3_v1 import paired_day_ci

from keiba_place_lab.trio_pool_v4 import checked_pool, v1_joint, weight_grid_two
from keiba_place_lab.trio_residual_v2 import score_probabilities

EXPECTED_2023 = 1355
EXPECTED_2024 = 1336
SEED = 20261031


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_races(oo: pd.DataFrame, keep: pd.DataFrame, year: int) -> list[dict]:
    keys = set(keep.race_id.astype(str))
    if len(keep) != len(keys):
        raise ValueError("Reference has duplicate race IDs")
    group = oo.loc[oo.race_id.isin(keys)].groupby("race_id", sort=True)
    if set(group.groups) != keys:
        raise ValueError("Missing PR40 exact reference races")
    result=[]
    for race_id, x in group:
        g = x.sort_values("horse_no").reset_index(drop=True)
        if (len(g) != int(g.field_size.iloc[0]) or
                g.horse_no.duplicated().any() or g.top3_label.sum()!=3 or
                pd.to_datetime(g.race_date.iloc[0]).year != year):
            raise ValueError(f"Mismatch complete race: {race_id}")
        indices, p_xgb = v1_joint(
            g.p_matched_xgb800.to_numpy(float),g.horse_no.to_numpy(int)
        )
        cats, p_cat = v1_joint(
            g.p_matched_cat400.to_numpy(float),g.horse_no.to_numpy(int)
        )
        f3idx, p_f3 = v1_joint(
            g.p_matched_f3.to_numpy(float),g.horse_no.to_numpy(int)
        )
        if not np.array_equal(indices,cats) or not np.array_equal(indices,f3idx):
            raise ValueError("Trio coordinate mismatch between models")
        true = tuple(sorted(g.loc[g.top3_label.eq(1),"horse_no"].astype(int)))
        actual = np.flatnonzero(np.all(indices==true, axis=1))
        if len(actual)!=1:
            raise ValueError("Actual 3 entrants are not available")
        result.append({
            "race_id":str(race_id),
            "race_date":str(pd.Timestamp(g.race_date.iloc[0]).date()),
            "horses":g.horse_no.tolist(),
            "trios":indices,"actual_idx":int(actual[0]),
            "xgb":p_xgb,"cat":p_cat,"incumbent":p_f3,
            "xgb_cat_equal":checked_pool([p_xgb,p_cat],[.5,.5]),
            "xgb_cat_75":checked_pool([p_xgb,p_cat],[.75,.25]),
        })
    return result


def main() -> None:
    p=argparse.ArgumentParser()
    for field in ("oof","pr40-2024","pr40-2023","out"):
        p.add_argument("--"+field, type=Path, required=True)
    a=p.parse_args()
    out=a.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Cannot overwrite versioned immutable results")
    started=time.perf_counter()
    oof=pd.read_csv(a.oof,dtype={"race_id":str,"horse_id":str})
    for key in ("race_id","race_date","horse_no","top3_label","field_size",
                "p_matched_xgb800","p_matched_cat400","p_matched_f3"):
        if key not in oof:
            raise ValueError("Missing matched 800/400 original OOF: "+key)
    if oof.duplicated(["race_id","horse_no"]).any():
        raise ValueError("Duplicate OOF starter IDs")
    train_ref=pd.read_csv(a.pr40_2023,dtype={"race_id":str})
    test_ref=pd.read_csv(a.pr40_2024,dtype={"race_id":str})
    if len(train_ref)!=EXPECTED_2023 or len(test_ref)!=EXPECTED_2024:
        raise ValueError("PR40 strict race population changed")
    train=build_races(oof,train_ref,2023)
    test=build_races(oof,test_ref,2024)
    print(f"PREPARED 2023={len(train)}, 2024={len(test)}",flush=True)
    if np.max(np.abs(np.asarray([
        -np.log(x["incumbent"][x["actual_idx"]]) for x in test
    ])-test_ref.v1_joint_logloss.to_numpy(float)))>5e-6:
        # Audit is keyed, not assumed in source file order.
        ref=test_ref.set_index("race_id")
        distance=max(abs(-np.log(x["incumbent"][x["actual_idx"]])-
                         float(ref.loc[x["race_id"],"v1_joint_logloss"])) for x in test)
        if distance>5e-6:
            raise ValueError(f"Existing PR40 V1 probability parity broken {distance}")
    fitted=weight_grid_two(train,"xgb","cat")
    for x in train+test:
        x["xgb_cat_learned"]=checked_pool(
            [x["xgb"],x["cat"]],
            [fitted["weight_left"],fitted["weight_right"]]
        )
    models=("xgb","cat","xgb_cat_equal","xgb_cat_75","xgb_cat_learned")
    train_scores={name:float(np.mean([
        -np.log(x[name][x["actual_idx"]]) for x in train
    ])) for name in ("incumbent",*models)}
    secondary=min(models,key=lambda name:(train_scores[name],name))
    # Different joint structure can complement incumbent despite weaker solo NLL.
    option_weights=(0.0,.25,.5,.75,1.0)
    choices=[]
    for alpha in option_weights:
        mean=float(np.mean([
            -np.log(checked_pool(
                [x["incumbent"],x[secondary]],[1-alpha,alpha]
            )[x["actual_idx"]]) for x in train
        ]))
        choices.append((mean,alpha))
    _,chosen=min(choices,key=lambda x:(x[0],x[1]))
    observations=[]
    ticket_rows=[]
    for x in test:
        variants={"incumbent":x["incumbent"]}
        variants.update({n:x[n] for n in models})
        variants["incumbent_plus_selected"]=checked_pool(
            [x["incumbent"],x[secondary]],[1-chosen,chosen]
        )
        row={"race_id":x["race_id"],"race_date":x["race_date"],
             "field_size":len(x["horses"])}
        for name, probs in variants.items():
            z=score_probabilities(probs,x["actual_idx"])
            row.update({name+"_"+k:v for k,v in z.items()})
        observations.append(row)
        # Complete 2024 unordered joint distribution, no ticket-ID leakage.
        for i,(h1,h2,h3) in enumerate(x["trios"]):
            ticket_rows.append({
                "race_id":x["race_id"],"horse_1":int(h1),
                "horse_2":int(h2),"horse_3":int(h3),
                "actual":int(i==x["actual_idx"]),
                **{name:float(probs[i]) for name,probs in variants.items()},
            })
    scores=pd.DataFrame(observations)
    stats={}
    for name in ("incumbent",*models,"incumbent_plus_selected"):
        stats[name]={metric:float(scores[name+"_"+metric].mean())
                     for metric in ("joint_logloss","joint_brier","top1_hit","top5_hit",
                                    "top10_hit","top1_mass","top10_mass")}
    diffs={}
    for name in (*models,"incumbent_plus_selected"):
        delta=paired_day_ci(scores,"incumbent_joint_logloss",
                             name+"_joint_logloss",draws=2500)
        delta["joint_brier_delta"]=stats["incumbent"]["joint_brier"]-stats[name]["joint_brier"]
        hit=paired_day_ci(scores,"incumbent_top10_hit",
                          name+"_top10_hit",draws=2500)
        diffs[name]={"nll":delta,"top10_hit":hit}
    summary={
        "status":"2024_DEVELOPMENT_NOT_UNTOUCHED",
        "preregistered_model_first":"post-V1 arithmetic joint mixing",
        "OOF_fit_year":2023,"evaluation_year":2024,
        "train_races":len(train),"test_races":len(test),
        "model_budget":"zero extra base learner fits, 21-grid w, 5-grid alpha",
        "trained_xgb_cat_weights":fitted,
        "2023_training_joint_scores":train_scores,
        "best_secondary_chosen_in_2023":secondary,
        "incumbent_mix_alpha_selected_2023":chosen,
        "2023_incumbent_mix_candidate_grid":choices,
        "2024_all_model_scores":stats,
        "2024_paired_day_delta":diffs,
        "source_sha256":{"oof":sha(a.oof),"2024_ref":sha(a.pr40_2024),
                         "2023_ref":sha(a.pr40_2023)},
        "runtime_seconds":time.perf_counter()-started,
        "cpu_device":"github-actions-ubuntu-latest",
        "random_seed":SEED,"no_training_on_2024_2025":True,
        "no_production_change":True,"no_real_trio_market_odds":True,
    }
    out.mkdir(parents=True,exist_ok=True)
    scores.to_csv(out/"2024_post_v1_racewise_scores.csv",index=False)
    pd.DataFrame(ticket_rows).to_csv(
        out/"2024_post_v1_all_trio_predictions.csv.gz",
        index=False,compression="gzip"
    )
    (out/"2024_post_v1_experiment.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n"
    )
    print("TRIO_POST_V1_REAL_DATA="+json.dumps({
        "train":len(train),"test":len(test),"fit":fitted,
        "secondary":secondary,"alpha":chosen,
        "scores":stats,"differences":diffs,
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
