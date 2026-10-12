"""Post-lock interpretation of the frozen Oct 12 2026 Swan Stakes model.
No new fitting, no postrace inputs, and no changes to locked probabilities.
XGBoost pred_contribs are log-odds-scale TreeSHAP contributions.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost
from scipy.special import expit

from keiba_place_lab.live_history import load_frozen_history
from keiba_place_lab.scope_features import scope_features
from keiba_place_lab.stage4_production_v3 import build_live_context, validate_roster

ROOT=Path(".")
OUT=ROOT/"analysis/predictions/2026-10-12_swan/explanation"
OUT.mkdir(parents=True,exist_ok=True)
model_path=ROOT/"models/stage4_scope_v3/bundle.joblib"
manifest=json.loads((model_path.parent/"manifest.json").read_text())
assert hashlib.sha256(model_path.read_bytes()).hexdigest()==manifest["bundle_sha256"]
bundle=joblib.load(model_path)
fit=bundle["models"]["turf_G0"]
assert fit["blocks"]==[] and fit["surface"]=="turf"

roster=validate_roster(pd.read_csv("analysis/predictions/2026-10-12_swan/roster.csv",dtype={"race_id":"string","horse_id":"string"}))
assert len(roster)==18
old=pd.read_csv("analysis/predictions/2026-10-12_swan/shadow_probabilities.csv",dtype={"horse_id":"string"})
pointer=json.loads((ROOT/"data/live_history/2026/current.json").read_text())
assert pointer["complete_through"]=="2026-10-04"
livepath=ROOT/"data/live_history/2026/snapshots"/pointer["snapshot_id"]/"jra_flat_history.parquet"
base=load_frozen_history(Path("/tmp/stage36/data/historical_processed/stage36_freeze_v1/standardized_jra_flat_source_v1.parquet"))
live=pd.read_parquet(livepath)
history=pd.concat([base,live],ignore_index=True,sort=False)
assert pd.to_datetime(history.race_date).max().strftime("%Y-%m-%d")=="2026-10-04"
assert not history.duplicated(["race_id","horse_id"]).any()

context=build_live_context(history,roster,max_history_age_days=8)
assert context.career_starts.ge(3).all()
engineered=scope_features(context,context,prior_mean=fit["prior_mean"],blocks=(),surface="turf")
pipe=fit["model"]
pre=pipe.named_steps["preprocess"]
x=pre.transform(engineered)
feat_names=list(pre.get_feature_names_out())
model=pipe.named_steps["model"]
booster=model.get_booster()
contrib=booster.predict(xgboost.DMatrix(np.asarray(x)),pred_contribs=True)
assert contrib.shape==(len(roster),len(feat_names)+1)
p=expit(contrib.sum(axis=1))
pred=pipe.predict_proba(engineered)[:,1]
assert np.max(np.abs(p-pred))<1e-6,(p,pred)
joined=roster[["horse_id","horse_no","horse_name"]].copy()
joined["p"]=pred
check=joined.merge(old[["horse_id","p_nonmarket"]],on="horse_id",validate="one_to_one")
assert np.max(np.abs(check.p-check.p_nonmarket))<1e-6

shap=pd.DataFrame(contrib[:,:-1],columns=feat_names)
shap.insert(0,"horse_name",roster.horse_name.values)
shap.insert(0,"horse_no",roster.horse_no.values)
shap.to_csv(OUT/"tree_shap_logit_all_horses.csv",index=False)
raw=engineered[["horse_no","horse_name"]+[
  "career_starts","career_top3_shrunk","turf_starts","turf_top3_shrunk",
  "same_distance_starts","same_distance_top3_shrunk","same_course_starts","same_course_top3_shrunk",
  "recent3_finish_pct_mean","recent3_top3_count","recent3_open_plus_count","recent3_graded_count",
  "recent3_relative_time_mean","recent4_early_pos_pct_mean",
  "rel_career_top3_vs_others","rel_same_distance_top3_vs_others",
  "rel_same_course_top3_vs_others","rel_recent3_finish_vs_others","rel_recent3_time_vs_others",
  "age","assigned_weight_kg","days_since_prev","distance_change_from_prev_m","draw_pct"]].copy()
raw["days_since_prev"]=context["days_since_prev"].values
raw.to_csv(OUT/"engineered_features_by_horse.csv",index=False)

top=[]
for i,row in shap.iterrows():
    v=contrib[i,:-1]
    pos=np.argsort(-v)[:10]
    neg=np.argsort(v)[:10]
    for direction,indices in (("positive",pos),("negative",neg)):
        for rank,j in enumerate(indices,1):
            top.append({"horse_no":int(row.horse_no),"horse_name":row.horse_name,
                        "p_top3":float(p[i]),"direction":direction,"rank":rank,
                        "feature":feat_names[j],"tree_shap_logit":float(v[j]),
                        "feature_value_transformed":float(np.asarray(x)[i,j])})
pd.DataFrame(top).to_csv(OUT/"top_contributions_by_horse.csv",index=False)

importance=pd.DataFrame({"feature":feat_names,
                         "mean_abs_tree_shap_logit":np.abs(contrib[:,:-1]).mean(axis=0),
                         "mean_tree_shap_logit":contrib[:,:-1].mean(axis=0)})
importance=importance.sort_values("mean_abs_tree_shap_logit",ascending=False)
importance.to_csv(OUT/"global_feature_importance_for_this_race.csv",index=False)
summary={"model_id":"turf_G0","model_sha256":manifest["bundle_sha256"],
         "prediction_max_abs_diff":float(np.max(np.abs(check.p-check.p_nonmarket))),
         "shap_reconstruction_max_abs_diff":float(np.max(np.abs(p-pred))),
         "contributions_are":"XGBoost TreeSHAP to raw log odds; signed effects are model-internal, not causal",
         "source_history_through":"2026-10-04",
         "target_horses":[1,9,11,13,15,17,18],
         "missing_new_distance_blocks":True}
(OUT/"manifest.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
print("PARITY_OK; selected top 8 absolute effects per horse:")
for number in [15,17,11,9,18,1,13]:
    i=roster.index[roster.horse_no.eq(number)][0]
    vv=contrib[i,:-1]
    idx=np.argsort(-np.abs(vv))[:8]
    print(number,roster.iloc[i].horse_name,f"p={p[i]:.4f}",[(feat_names[j],round(float(vv[j]),3)) for j in idx],flush=True)
print("GLOBAL TOP:",importance.head(15).to_dict("records"),flush=True)
