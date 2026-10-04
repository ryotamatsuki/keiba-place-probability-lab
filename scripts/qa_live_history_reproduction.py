"""Reproduce the 2026-10-04 35-horse V3 morning trial from the common live DB."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from keiba_place_lab.live_history import load_frozen_history
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
    if manifest.get("source_parser_contract") != "time_margin_separated_v1":
        raise ValueError("Live-history snapshot requires corrected time-parser reparse")
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
    parser.add_argument("--historical-base", type=Path,
                        help="Also verify against the exact audited full 2010–2025 source")
    parser.add_argument("--reference-dir", type=Path, default=TRIAL,
                        help="Captured morning reference, or separately audited corrected reference")
    parser.add_argument(
        "--output",
        type=Path,
        default=TRIAL / "common_live_history_reproduction.json",
    )
    args = parser.parse_args()

    old_target_history = pd.read_parquet(TRIAL / "target_history.parquet")
    old_target_history["race_date"] = pd.to_datetime(old_target_history["race_date"])
    if args.historical_base is not None:
        historical_base = load_frozen_history(args.historical_base)
    else:
        historical_base = old_target_history.loc[
            old_target_history["race_date"] < pd.Timestamp("2026-01-01")
        ].copy()

    live, live_manifest = load_live(args.live_root)
    live["race_date"] = pd.to_datetime(live["race_date"])
    reference_history_error = None
    reference_history_rows = None
    reference_history_path = args.reference_dir / "target_history.parquet"
    if args.reference_dir != TRIAL and reference_history_path.exists():
        reference_history = pd.read_parquet(reference_history_path)
        reference_history = reference_history.loc[pd.to_datetime(reference_history.race_date).dt.year.eq(2026)]
        physical = ["race_id", "horse_id", "race_date", "horse_no", "racecourse", "surface", "distance_m",
                    "race_class", "is_open_plus", "is_graded", "sex", "age", "finish_position", "finish_status",
                    "race_time_seconds", "early_position", "field_size", "assigned_weight_kg"]
        subset = live.loc[live.race_id.astype(str).isin(reference_history.race_id.astype(str))]
        reference_history_error = compare_context(subset[physical], reference_history[physical])
        reference_history_rows = len(reference_history)
    history = pd.concat([historical_base, live], ignore_index=True, sort=False)
    history["race_id"] = history["race_id"].astype("string")
    history["horse_id"] = history["horse_id"].astype("string")
    if history.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Historical base/live DB overlap")

    expected_context = pd.read_csv(
        args.reference_dir / "morning_feature_context.csv",
        dtype={"race_id": "string", "horse_id": "string"},
    )
    expected_predictions = pd.read_csv(
        args.reference_dir / "morning_nonmarket_predictions.csv",
        dtype={"race_id": "string", "horse_id": "string"},
    )
    frozen_manifest = json.loads((BUNDLE.parent / "manifest.json").read_text())
    bundle_sha = sha256(BUNDLE)
    if bundle_sha != frozen_manifest["bundle_sha256"]:
        raise ValueError("Frozen V3 model checksum changed")
    correction_path = args.reference_dir / "correction_audit.json"
    correction = json.loads(correction_path.read_text()) if correction_path.exists() else None
    if correction:
        for filename, key in (("target_history.parquet", "corrected_history_sha256"),
                              ("morning_feature_context.csv", "corrected_context_sha256"),
                              ("morning_nonmarket_predictions.csv", "corrected_prediction_sha256")):
            if sha256(args.reference_dir / filename) != correction[key]:
                raise ValueError("Audited corrected reference checksum changed")
        if correction["model_bundle_sha256"] != bundle_sha:
            raise ValueError("Corrected reference used another frozen model")
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
        "model_bundle_sha256": bundle_sha,
        "model_changed": False,
        "historical_base_mode": "full_frozen_base" if args.historical_base else "trial_prior_race_subset",
        "historical_base_sha256": sha256(args.historical_base) if args.historical_base else sha256(TRIAL / "target_history.parquet"),
        "reference_directory": str(args.reference_dir.relative_to(ROOT)) if args.reference_dir.is_relative_to(ROOT) else str(args.reference_dir),
        "reference_context_sha256": sha256(args.reference_dir / "morning_feature_context.csv"),
        "reference_predictions_sha256": sha256(args.reference_dir / "morning_nonmarket_predictions.csv"),
        "independent_reference_history_rows": reference_history_rows,
        "independent_reference_history_max_abs_error": reference_history_error,
        "reference_correction_audit_sha256": sha256(correction_path) if correction else None,
        "original_locked_reference_gate_passed": correction["original_1e_minus7_reproduction_gate_passed"] if correction else True,
        "original_locked_prediction_max_abs_delta": correction["original_prediction_max_abs_delta"] if correction else prediction_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
