"""Freeze F3 XGB800/400 prefix and CAT400 shadow bundle BEFORE first live T-10.

This is a NEW, forward research bundle trained only on historical FINAL-odds data.
No production promotion; no direct prediction without audited live prior features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.special import expit, logit

from keiba_place_lab.ensemble import fit_logit_calibrator
from run_information_expansion_v1 import _build_pipeline
from run_input_learning_a_v1 import feature_registry, prepare_dataset
from run_model_families_c1 import CATBOOST_PARAMS, _categorical_input, _race_weight

FROZEN_TRAINING_YEARS = (2018, 2024)
TRAIN_CUTOFF = "2024-12-31"
XGB_COUNTS = (400, 800)
CAT_COUNT = 400
SEED = 20261031


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def numeric_categorical(frame: pd.DataFrame, numeric: list[str],
                        categorical: list[str]) -> pd.DataFrame:
    x = frame[numeric + categorical].copy()
    for name in numeric:
        x[name] = pd.to_numeric(x[name], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan,
        )
    for name in categorical:
        x[name] = x[name].astype("object").where(x[name].notna(), None)
    return x


def train(data: pd.DataFrame) -> tuple[dict, dict]:
    """No time-of-day claims: train target stops in 2024, results known."""
    data = data.copy()
    data.race_date = pd.to_datetime(data.race_date)
    if data.race_date.max() > pd.Timestamp(TRAIN_CUTOFF):
        raise ValueError("Training source contains a 2025+ event")
    train_rows = data.loc[data.race_date.dt.year.between(*FROZEN_TRAINING_YEARS)].copy()
    fit_market = data.loc[data.race_date.dt.year.between(2016, 2024)].copy()
    if (train_rows.empty or fit_market.empty
            or train_rows.top3_label.isna().any()
            or fit_market.top3_label.isna().any()):
        raise ValueError("Empty or incomplete historical training cohort")
    if train_rows.p_market_oof.isna().any():
        raise ValueError("Historical training market must be prior-year OOF")
    if train_rows.race_date.max() >= pd.Timestamp("2025-01-01"):
        raise ValueError("Cannot fit on date after frozen cutoff")
    numerical, categorical = feature_registry()["F3_market_field"]
    if len(set(numerical + categorical)) != len(numerical + categorical):
        raise ValueError("Duplicate F3 column names")
    tr = train_rows.sort_values(
        ["race_date", "race_id", "horse_id", "horse_no"], kind="stable",
    ).reset_index(drop=True)
    x = numeric_categorical(tr, numerical, categorical)
    offset = logit(tr.p_market_oof.to_numpy(float))
    weights = _race_weight(tr)
    y = tr.top3_label.to_numpy(int)
    start = time.perf_counter()
    pipe = _build_pipeline(numerical, categorical)
    pipe.named_steps["model"].set_params(n_estimators=800)
    pipe.fit(x, y, model__sample_weight=weights, model__base_margin=offset)
    xgb_seconds = time.perf_counter() - start
    start = time.perf_counter()
    model = CatBoostClassifier(**{**CATBOOST_PARAMS, "iterations": CAT_COUNT})
    c = _categorical_input(tr, numerical, categorical)
    model.fit(Pool(
        c, label=y, cat_features=categorical,
        weight=weights, baseline=offset,
    ))
    cat_seconds = time.perf_counter() - start
    cal = fit_logit_calibrator(
        fit_market.market_p_top3.to_numpy(float),
        fit_market.top3_label.to_numpy(int),
    )
    information = {
        "version": "f3_20261101_research_forward_v1",
        "training_min_date": str(tr.race_date.min().date()),
        "training_max_date": str(tr.race_date.max().date()),
        "training_rows": len(tr),
        "training_races": int(tr.race_id.nunique()),
        "training_cutoff": TRAIN_CUTOFF,
        "train_market": "historical FINAL odds, prior-year OOF market logit",
        "forward_market": "T-10 real win odds; final-odds calibration extrapolation",
        "market_calibrator_fit_latest_date": str(
            fit_market.race_date.max().date()
        ),
        "market_calibrator_intercept": float(cal.intercept),
        "market_calibrator_slope": float(cal.slope),
        "model_rounds": {
            "B_raw_400": 400,
            "XGB800_raw": 800,
            "CatBoost400_raw": CAT_COUNT,
        },
        "candidate_weight": {"XGB800_raw": 0.75, "CatBoost400_raw": 0.25},
        "feature_numeric": numerical,
        "feature_categorical": categorical,
        "seed": SEED,
        "xgb_train_seconds": float(xgb_seconds),
        "cat_train_seconds": float(cat_seconds),
        "no_2025_2026_train_labels": True,
        "no_live_adoption": True,
    }
    bundle = {
        "xgb": pipe,
        "catboost": model,
        "numeric": numerical,
        "categorical": categorical,
        "market_calibrator": cal,
        "training_cutoff": TRAIN_CUTOFF,
        "version": information["version"],
    }
    return bundle, information


def main() -> None:
    import catboost
    import xgboost
    import sklearn

    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--races-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError("Do not overwrite an already frozen forward bundle")
    df, provenance = prepare_dataset(
        a.historical_dir, a.results_csv, a.races_csv,
    )
    bundle, info = train(df)
    a.out.mkdir(parents=True, exist_ok=True)
    artifact = a.out / "f3_bundle.joblib"
    joblib.dump(bundle, artifact, compress=3)
    reloaded = joblib.load(artifact)
    if reloaded["training_cutoff"] != TRAIN_CUTOFF:
        raise ValueError("Frozen bundle deserialization mismatch")
    info.update({
        "source_audit": provenance,
        "bundle_sha256": sha(artifact),
        "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "python": platform.python_version(),
        "xgboost": xgboost.__version__,
        "catboost": catboost.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "source_script_sha256": sha(Path(__file__)),
    })
    (a.out / "manifest.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps({
        "status": "FORWARD_SHADOW_MODEL_FROZEN",
        "bundle_sha256": info["bundle_sha256"],
        "history_max": info["training_max_date"],
        "training_races": info["training_races"],
        "feature_numeric": len(info["feature_numeric"]),
        "feature_categorical": len(info["feature_categorical"]),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
