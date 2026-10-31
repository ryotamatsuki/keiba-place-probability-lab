"""Strict pre-off F3 research candidate forecasts from a frozen model bundle.

Data must be real all-starter pre-off WIN odds, and prior-only F3 feature rows.
Past-final-odds market calibration is an explicit extrapolation at T-minus-10.
No current race results or raw odds source records are serialized in predictions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from keiba_place_lab.ensemble import apply_logit_calibrator
from keiba_place_lab.input_learning_v1 import (
    MARKET_FIELD,
    MARKET_SPLITS,
    full_starter_market_context,
)
from keiba_place_lab.market import (
    harville_top_k_probabilities,
    normalized_win_probabilities,
)

EXPECTED_MODEL_VERSION = "f3_20261101_research_forward_v1"
TARGET_RACE = "202605041111"
TARGET_DATE = pd.Timestamp("2026-11-01")
REQUIRED_CANDIDATES = (
    "p_M0_live_raw", "p_M0_final_odds_calibrated",
    "p_B_raw_400", "p_XGB800_raw", "p_CatBoost400_raw",
    "p_XC_75_25_logit",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def locked_bundle(bundle_file: Path, manifest_file: Path) -> tuple[dict, dict]:
    meta = json.loads(manifest_file.read_text())
    if (meta.get("version") != EXPECTED_MODEL_VERSION
            or meta.get("bundle_sha256") != sha256(bundle_file)
            or meta.get("training_cutoff") != "2024-12-31"
            or meta.get("no_2025_2026_train_labels") is not True
            or meta.get("no_live_adoption") is not True):
        raise ValueError("Missing/changed F3 pre-frozen bundle audit")
    obj = joblib.load(bundle_file)
    if (obj.get("version") != EXPECTED_MODEL_VERSION
            or obj.get("training_cutoff") != "2024-12-31"
            or obj.get("numeric") != meta.get("feature_numeric")
            or obj.get("categorical") != meta.get("feature_categorical")):
        raise ValueError("Unpickled bundle differs from frozen F3 manifest")
    return obj, meta


def market_context(odds: pd.DataFrame, market_calibrator,
                   *, required_horses: set[str]) -> tuple[pd.DataFrame, float]:
    """Compute public Stage5 M0 and F3 offset from EXACT same full win-odds field."""
    required = {"race_id", "horse_id", "horse_no", "horse_name", "win_odds"}
    if not required.issubset(odds):
        raise ValueError("Full-field win odds missing required columns")
    x = odds.copy().reset_index(drop=True)
    x["race_id"] = x.race_id.astype(str)
    x["horse_id"] = x.horse_id.astype(str)
    if (set(x.race_id) != {TARGET_RACE}
            or len(x) != len(required_horses)
            or set(x.horse_id) != required_horses
            or x.horse_id.duplicated().any()
            or x.horse_no.duplicated().any()):
        raise ValueError("Missing/duplicate/mismatched full-starter odds snapshot")
    x["horse_no"] = pd.to_numeric(x.horse_no, errors="raise").astype(int)
    x = x.sort_values("horse_no", kind="stable").reset_index(drop=True)
    win = pd.to_numeric(x.win_odds, errors="raise").to_numpy(float)
    if not np.isfinite(win).all() or np.any(win <= 1):
        raise ValueError("Invalid, missing or non-current win odds")
    shares, overround = normalized_win_probabilities(win)
    harville = harville_top_k_probabilities(shares)
    calibrated = apply_logit_calibrator(harville, market_calibrator)
    if (not np.isfinite(calibrated).all()
            or np.any((calibrated <= 0) | (calibrated >= 1))):
        raise ValueError("Invalid prior-calibrated live market")
    market = x[["race_id", "horse_id", "horse_no", "horse_name"]].copy()
    market["market_p_win"] = shares
    market["market_p_top3"] = harville
    market["market_field_size"] = len(x)
    field = full_starter_market_context(market)
    if len(field) != len(x):
        raise ValueError("Market all-starter field information incomplete")
    market = market.merge(
        field, on=["race_id", "horse_no"],
        validate="one_to_one", how="left", sort=False,
    )
    market["p_M0_live_raw"] = harville
    market["p_M0_final_odds_calibrated"] = calibrated
    market["market_calibrated_top3_p"] = calibrated
    market["market_calibrated_top3_logit"] = logit(calibrated)
    return market, float(overround)


def _prior_features(prior: pd.DataFrame, manifest: dict, *, required_horses: set[str],
                    expected_numeric: list[str],
                    expected_categorical: list[str]) -> pd.DataFrame:
    forbidden = {
        "finish_position", "top3_label", "raw_finish_position",
        "current_result", "future_result", "final_odds", "payout",
        "win_odds", "market_p_top3", "p_market_oof",
    }
    if forbidden.intersection(prior.columns):
        raise ValueError("Postrace/market features appeared in prior-only context")
    features = prior.copy()
    features["race_id"] = features.race_id.astype(str)
    features["horse_id"] = features.horse_id.astype(str)
    if (len(features) != len(required_horses)
            or set(features.race_id) != {TARGET_RACE}
            or set(features.horse_id) != required_horses
            or features.horse_id.duplicated().any()
            or features.horse_no.duplicated().any()):
        raise ValueError("Prior feature file not all active starters")
    if (manifest.get("no_future_rows") is not True
            or manifest.get("history_complete_for_all_entrants") is not True
            or pd.Timestamp(manifest.get("history_cutoff", "9999-01-01")) >= TARGET_DATE
            or manifest.get("race_id") != TARGET_RACE
            or not isinstance(manifest.get("history_sha256"), str)
            or len(manifest["history_sha256"]) != 64):
        raise ValueError("Prior-only historical context unverified for all entrants")
    if not set(MARKET_FIELD).isdisjoint(features.columns):
        raise ValueError("Historical context contains same-race market features")
    if not set(MARKET_SPLITS).isdisjoint(features.columns):
        raise ValueError("Historical context contains same-race market calibration")
    missing = set(expected_numeric + expected_categorical).difference(
        set(features) | set(MARKET_SPLITS) | set(MARKET_FIELD)
    )
    if missing:
        raise ValueError("Cannot predict F3 with incomplete prior features: "
                         + ",".join(sorted(missing)))
    if not features["race_date"].astype(str).eq(str(TARGET_DATE.date())).all():
        raise ValueError("Feature-row race date changed")
    if features.duplicated(["horse_no", "horse_id"]).any():
        raise ValueError("Target prior features duplicate active starter")
    return features


def predict_from_frozen(
    bundle: dict, manifest: dict, prior_features: pd.DataFrame,
    prior_manifest: dict, odds: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Never train or use labels here; return only same-snapshot probabilities."""
    if bundle.get("version") != EXPECTED_MODEL_VERSION:
        raise ValueError("Wrong frozen model version")
    required = set(odds.horse_id.astype(str))
    number, categorical = bundle["numeric"], bundle["categorical"]
    prior = _prior_features(
        prior_features, prior_manifest,
        required_horses=required, expected_numeric=number,
        expected_categorical=categorical,
    )
    market, overround = market_context(
        odds, bundle["market_calibrator"], required_horses=required,
    )
    work = prior.merge(
        market.drop(columns=["market_field_size"]),
        on=["race_id", "horse_id", "horse_no"],
        how="inner", validate="one_to_one", suffixes=("", "_live"),
    )
    work = work.sort_values("horse_no", kind="stable").reset_index(drop=True)
    if len(work) != len(odds):
        raise ValueError("Live market/current prior features not perfectly paired")
    if (
        "horse_name" in work and "horse_name_live" in work
        and not work.horse_name.astype(str).equals(
            work.horse_name_live.astype(str),
        )
    ):
        raise ValueError("Frozen roster name differed from live odds")
    matrix = work[number + categorical].copy()
    for column in number:
        matrix[column] = pd.to_numeric(
            matrix[column], errors="coerce",
        ).to_numpy(dtype=float, na_value=np.nan)
    for column in categorical:
        matrix[column] = matrix[column].astype("object").where(
            matrix[column].notna(), None,
        )
    pbase = work.p_M0_final_odds_calibrated.to_numpy(float)
    margin = logit(pbase)
    pipeline = bundle["xgb"]
    transformed = pipeline.named_steps["preprocess"].transform(matrix)
    model = pipeline.named_steps["model"]
    p400 = model.predict_proba(
        transformed, base_margin=margin, iteration_range=(0, 400),
    )[:, 1]
    p800 = model.predict_proba(
        transformed, base_margin=margin, iteration_range=(0, 800),
    )[:, 1]
    from catboost import Pool
    native = matrix.copy()
    for column in categorical:
        native[column] = native[column].astype("string").fillna(
            "MISSING",
        ).astype("object")
    with_offset = Pool(native, cat_features=categorical, baseline=margin)
    no_offset = Pool(native, cat_features=categorical)
    cat = bundle["catboost"]
    with_raw = np.asarray(
        cat.predict(with_offset, prediction_type="RawFormulaVal"),
        dtype=float,
    ).reshape(-1)
    without_raw = np.asarray(
        cat.predict(no_offset, prediction_type="RawFormulaVal"),
        dtype=float,
    ).reshape(-1)
    if np.allclose(with_raw - without_raw, margin, atol=1e-5, rtol=1e-5):
        raw = with_raw
        cat_offset = "baseline_included"
    elif np.allclose(with_raw, without_raw, atol=1e-7, rtol=0):
        raw = without_raw + margin
        cat_offset = "explicit_offset_added_once"
    else:
        raise ValueError("CatBoost baseline not correct on actual live inputs")
    pcat = expit(raw)
    pblend = expit(
        margin + .75 * (logit(p800) - margin)
        + .25 * (logit(pcat) - margin)
    )
    scored = work[["race_id", "horse_id", "horse_no"]].copy()
    if "horse_name" in work:
        scored["horse_name"] = work.horse_name
    else:
        scored["horse_name"] = work.horse_name_live
    scored["p_M0_live_raw"] = work.p_M0_live_raw.to_numpy(float)
    scored["p_M0_final_odds_calibrated"] = pbase
    scored["p_B_raw_400"] = p400
    scored["p_XGB800_raw"] = p800
    scored["p_CatBoost400_raw"] = pcat
    scored["p_XC_75_25_logit"] = pblend
    p = scored[list(REQUIRED_CANDIDATES)].to_numpy(float)
    if not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError("Non-finite or out-of-range forward probability")
    return scored, {
        "model_version": manifest["version"],
        "model_sha256": manifest["bundle_sha256"],
        "history_sha256": prior_manifest["history_sha256"],
        "history_cutoff": prior_manifest["history_cutoff"],
        "market_source_domain": "same-snapshot T-10 win odds, not final odds",
        "market_calibration_training_domain": "2016-2024 final win odds",
        "market_probability_convention": "all starters normalized win Harville",
        "catboost_offset_contract": cat_offset,
        "overround": overround,
        "all_entrants_prior_history_audited": True,
    }
