"""Exact XGBoost contribution audit for Kyoto Daishoten live shadow predictions."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from keiba_place_lab.scope_features import scope_features

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models/stage4_scope_v3/bundle.joblib"
CONTEXT = ROOT / "analysis/live_trial_20261004/morning_feature_context.csv"
OUT = ROOT / "analysis/live_trial_20261004/kyoto_model_attribution.csv"
SUMMARY = ROOT / "analysis/live_trial_20261004/kyoto_model_attribution_summary.json"

RACE_ID = "202608040211"
MODEL_ID = "turf_R3_2001_2600"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    bundle = joblib.load(BUNDLE)
    fit = bundle["models"][MODEL_ID]
    context = pd.read_csv(CONTEXT)
    context["race_id"] = context["race_id"].astype("string")
    context["horse_id"] = context["horse_id"].astype("string")
    context["race_date"] = pd.to_datetime(context["race_date"])
    context = context.loc[context["race_id"].eq(RACE_ID)].reset_index(drop=True)
    if len(context) != 18:
        raise ValueError(f"Expected 18 Kyoto starters, got {len(context)}")

    engineered = scope_features(
        context,
        context,
        prior_mean=float(fit["prior_mean"]),
        blocks=fit["blocks"],
        surface=fit["surface"],
    )

    pipeline = fit["model"]
    pre = pipeline.named_steps["preprocess"]
    clf = pipeline.named_steps["model"]
    X = pre.transform(engineered)
    names = np.asarray(pre.get_feature_names_out(), dtype=object)

    booster = clf.get_booster()
    contrib = booster.predict(xgb.DMatrix(X, feature_names=booster.feature_names), pred_contribs=True)
    if contrib.shape[1] != len(names) + 1:
        raise ValueError((contrib.shape, len(names)))

    margin = contrib.sum(axis=1)
    probability = sigmoid(margin)
    model_probability = clf.predict_proba(X)[:, 1]
    max_error = float(np.max(np.abs(probability - model_probability)))
    if max_error > 1e-8:
        raise ValueError(f"Contribution parity failure: {max_error}")

    rows = []
    for i, horse in context.iterrows():
        for feature, value in zip(names, contrib[i, :-1], strict=True):
            rows.append({
                "horse_no": int(horse["horse_no"]),
                "horse_name": horse["horse_name"],
                "feature": str(feature),
                "contribution_log_odds": float(value),
                "abs_contribution": float(abs(value)),
                "bias_log_odds": float(contrib[i, -1]),
                "prediction_log_odds": float(margin[i]),
                "prediction_probability": float(probability[i]),
            })
    pd.DataFrame(rows).to_csv(OUT, index=False)

    indices = {int(row.horse_no): i for i, row in context.iterrows()}
    i4, i12 = indices[4], indices[12]
    diff = contrib[i4, :-1] - contrib[i12, :-1]
    compare = pd.DataFrame({
        "feature": names,
        "hedentall_contribution": contrib[i4, :-1],
        "ecolo_contribution": contrib[i12, :-1],
        "hedentall_minus_ecolo_log_odds": diff,
    })
    compare = compare.reindex(compare["hedentall_minus_ecolo_log_odds"].abs().sort_values(ascending=False).index)

    def top(i, positive=True, n=12):
        vals = pd.DataFrame({"feature": names, "contribution": contrib[i, :-1]})
        if positive:
            vals = vals.loc[vals.contribution > 0].sort_values("contribution", ascending=False)
        else:
            vals = vals.loc[vals.contribution < 0].sort_values("contribution")
        return vals.head(n).to_dict(orient="records")

    summary = {
        "model_id": MODEL_ID,
        "prior_mean": float(fit["prior_mean"]),
        "blocks": list(fit["blocks"]),
        "parity_max_abs_error": max_error,
        "horse_4": {
            "horse_name": str(context.iloc[i4].horse_name),
            "probability": float(probability[i4]),
            "log_odds": float(margin[i4]),
            "bias_log_odds": float(contrib[i4, -1]),
            "top_positive": top(i4, True),
            "top_negative": top(i4, False),
        },
        "horse_12": {
            "horse_name": str(context.iloc[i12].horse_name),
            "probability": float(probability[i12]),
            "log_odds": float(margin[i12]),
            "bias_log_odds": float(contrib[i12, -1]),
            "top_positive": top(i12, True),
            "top_negative": top(i12, False),
        },
        "largest_pairwise_differences": compare.head(20).to_dict(orient="records"),
        "probability_gap": float(probability[i4] - probability[i12]),
        "log_odds_gap": float(margin[i4] - margin[i12]),
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
