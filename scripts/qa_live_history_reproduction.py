"""Reproduce the 2026-10-04 35-horse V3 morning trial from the common live DB."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from keiba_place_lab.stage4_production_v3 import (
    build_live_context,
    score_scope_target,
    validate_roster,
)

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "analysis/live_trial_20261004"
BUNDLE = ROOT / "models/stage4_scope_v3/bundle.joblib"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_live(root: Path):
    pointer = json.loads((root / "current.json").read_text())
    snap = root / "snapshots" / pointer["snapshot_id"]
    manifest = json.loads((snap / "manifest.json").read_text())
    history_path = snap / "jra_flat_history.parquet"
    if manifest["snapshot_id"] != pointer["snapshot_id"]:
        raise ValueError("Snapshot pointer mismatch")
    if sha256(history_path) != manifest["history_sha256"]:
        raise ValueError("Live history checksum mismatch")
    if pd.Timestamp(manifest["complete_through"]) < pd.Timestamp("2026-10-03"):
        raise ValueError("Live history is not complete through 2026-10-03")
    return pd.read_parquet(history_path), manifest


def compare_context(actual: pd.DataFrame, expected: pd.DataFrame) -> float:
    keys = ["race_id", "horse_id"]
    actual = actual.copy()
    expected = expected.copy()
    for frame in (actual, expected):
        frame["race_id"] = frame["race_id"].astype("string")
        frame["horse_id"] = frame["horse_id"].astype("string")
    expected_cols = list(expected.columns)
    missing = set(expected_cols) - set(actual.columns)
    if missing:
        raise ValueError(f"Rebuilt context missing columns: {sorted(missing)}")
    a = actual[expected_cols].sort_values(keys).reset_index(drop=True)
    e = expected[expected_cols].sort_values(keys).reset_index(drop=True)
    if a[keys].astype(str).to_dict("records") != e[keys].astype(str).to_dict("records"):
        raise ValueError("Context identity mismatch")
    max_err = 0.0
    for col in expected_cols:
        if col in keys:
            continue
        an = pd.to_numeric(a[col], errors="coerce")
        en = pd.to_numeric(e[col], errors="coerce")
        numeric_like = an.notna().sum() + en.notna().sum() > 0 and (
            pd.api.types.is_numeric_dtype(e[col]) or pd.api.types.is_numeric_dtype(a[col])
        )
        if numeric_like:
            av = an.to_numpy(float)
            ev = en.to_numpy(float)
            same_nan = np.isnan(av) == np.isnan(ev)
            if not same_nan.all():
                raise ValueError(f"Feature missingness changed: {col}")
            mask = np.isfinite(av) & np.isfinite(ev)
            err = float(np.max(np.abs(av[mask] - ev[mask]))) if mask.any() else 0.0
            max_err = max(max_err, err)
            if err > 1e-10:
                raise ValueError(f"Feature changed: {col}, max_abs_error={err}")
        else:
            av = a[col].fillna("<NA>").astype(str).tolist()
            ev = e[col].fillna("<NA>").astype(str).tolist()
            if av != ev:
                raise ValueError(f"Feature changed: {col}")
    return max_err


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-root", type=Path, default=ROOT / "data/live_history/2026")
    parser.add_argument(
        "--output",
        type=Path,
        default=TRIAL / "common_live_history_reproduction.json",
    )
    args = parser.parse_args()

    old_target_history = pd.read_parquet(TRIAL / "target_history.parquet")
    old_target_history["race_date"] = pd.to_datetime(old_target_history["race_date"])
    historical_base = old_target_history.loc[
        old_target_history["race_date"] < pd.Timestamp("2026-01-01")
    ].copy()

    live, live_manifest = load_live(args.live_root)
    live["race_date"] = pd.to_datetime(live["race_date"])
    history = pd.concat([historical_base, live], ignore_index=True, sort=False)
    history["race_id"] = history["race_id"].astype("string")
    history["horse_id"] = history["horse_id"].astype("string")
    if history.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Historical base/live DB overlap")

    expected_context = pd.read_csv(
        TRIAL / "morning_feature_context.csv",
        dtype={"race_id": "string", "horse_id": "string"},
    )
    expected_predictions = pd.read_csv(
        TRIAL / "morning_nonmarket_predictions.csv",
        dtype={"race_id": "string", "horse_id": "string"},
    )
    bundle = joblib.load(BUNDLE)

    contexts = []
    predictions = []
    for race_id in ("202608040211", "202605040211"):
        roster = validate_roster(
            pd.read_csv(
                TRIAL / f"{race_id}_roster.csv",
                dtype={"race_id": "string", "horse_id": "string"},
            )
        )
        target_date = pd.Timestamp(roster.race_date.iloc[0]).normalize()
        # Explicit caller-side as-of filter, followed by build_live_context's strict barrier.
        asof = history.loc[history["race_date"] < target_date].copy()
        context = build_live_context(asof, roster, max_history_age_days=7)
        score = score_scope_target(bundle, context)
        contexts.append(context)
        predictions.append(score)

    rebuilt_context = pd.concat(contexts, ignore_index=True)
    rebuilt_predictions = pd.concat(predictions, ignore_index=True)
    feature_error = compare_context(rebuilt_context, expected_context)

    got = rebuilt_predictions[["race_id", "horse_id", "p_nonmarket"]].copy()
    want = expected_predictions[["race_id", "horse_id", "p_nonmarket"]].copy()
    got = got.sort_values(["race_id", "horse_id"]).reset_index(drop=True)
    want = want.sort_values(["race_id", "horse_id"]).reset_index(drop=True)
    if got[["race_id", "horse_id"]].astype(str).to_dict("records") != want[
        ["race_id", "horse_id"]
    ].astype(str).to_dict("records"):
        raise ValueError("Prediction identity mismatch")
    prediction_error = float(
        np.max(np.abs(got.p_nonmarket.to_numpy(float) - want.p_nonmarket.to_numpy(float)))
    )
    if prediction_error > 1e-7:
        raise ValueError(f"Prediction reproduction failure: {prediction_error}")

    report = {
        "status": "PASS",
        "snapshot_id": live_manifest["snapshot_id"],
        "schema_version": live_manifest["schema_version"],
        "complete_through": live_manifest["complete_through"],
        "live_history_sha256": live_manifest["history_sha256"],
        "runners": len(got),
        "races": int(got.race_id.nunique()),
        "feature_max_abs_error": feature_error,
        "prediction_max_abs_error": prediction_error,
        "model_bundle_sha256": sha256(BUNDLE),
        "model_changed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
