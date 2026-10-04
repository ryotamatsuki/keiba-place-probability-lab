"""Audit numeric-margin parsing against the original captured result HTML.

Preserve the locked morning trial. Write a distinct corrected reference for
common-history reproduction; this is an input correction, not a new model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from keiba_place_lab.live_public import parse_results
from keiba_place_lab.stage4_production_v3 import (
    build_live_context,
    score_scope_target,
    validate_roster,
)

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "analysis/live_trial_20261004"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/raw/live_trial_20261004")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/live_history_2026/corrected_trial_reference")
    args = parser.parse_args()
    history = pd.read_parquet(TRIAL / "target_history.parquet")
    original = history.copy()
    current = pd.to_datetime(history.race_date).dt.year.eq(2026)
    evidence = []
    for race_id in sorted(history.loc[current, "race_id"].unique()):
        url = "https://sports.yahoo.co.jp/keiba/race/result/" + str(race_id)[2:]
        key = hashlib.sha256(url.encode()).hexdigest()
        html_path = args.cache / (key + ".html")
        capture = json.loads((args.cache / (key + ".json")).read_text())
        if capture["url"] != url or capture["sha256"] != sha(html_path):
            raise ValueError("Original source capture checksum mismatch")
        parsed = parse_results(html_path.read_text(), str(race_id)[2:])
        selected = history.race_id.eq(race_id)
        old = history.loc[selected].set_index("horse_id")
        new = parsed.set_index("horse_id").reindex(old.index)
        if set(old.index) != set(parsed.horse_id):
            raise ValueError("Original full-field identity changed")
        # Only time interpretation may change. Other result facts remain fixed.
        for col in ("horse_no", "finish_position", "early_position", "field_size", "assigned_weight_kg"):
            if not old[col].fillna(-999).eq(new[col].fillna(-999)).all():
                raise ValueError(f"Unexpected non-time correction: {race_id}/{col}")
        history.loc[selected, "race_time_seconds"] = new.race_time_seconds.to_numpy()
        evidence.append({"race_id": str(race_id), "url": url, "sha256": capture["sha256"],
                         "retrieved_at": capture["retrieved_at"]})
    bundle_path = ROOT / "models/stage4_scope_v3/bundle.joblib"
    manifest = json.loads((bundle_path.parent / "manifest.json").read_text())
    if sha(bundle_path) != manifest["bundle_sha256"]:
        raise ValueError("Frozen model checksum changed")
    bundle = joblib.load(bundle_path)
    contexts, predictions = [], []
    for race_id in ("202608040211", "202605040211"):
        roster = validate_roster(pd.read_csv(TRIAL / f"{race_id}_roster.csv",
                                            dtype={"race_id": "string", "horse_id": "string"}))
        date = pd.Timestamp(roster.race_date.iloc[0]).normalize()
        context = build_live_context(history.loc[pd.to_datetime(history.race_date).lt(date)].copy(), roster,
                                     max_history_age_days=7)
        contexts.append(context)
        predictions.append(score_scope_target(bundle, context))
    context = pd.concat(contexts, ignore_index=True)
    prediction = pd.concat(predictions, ignore_index=True)
    keys = ["race_id", "horse_id"]
    original_prediction = pd.read_csv(TRIAL / "morning_nonmarket_predictions.csv", dtype=dict.fromkeys(keys, "string"))
    original_context = pd.read_csv(TRIAL / "morning_feature_context.csv", dtype=dict.fromkeys(keys, "string"))
    a = context.sort_values(keys).reset_index(drop=True)
    b = original_context.sort_values(keys).reset_index(drop=True)
    changes = {}
    for col in b:
        if pd.api.types.is_numeric_dtype(b[col]):
            av, bv = pd.to_numeric(a[col]).to_numpy(float), pd.to_numeric(b[col]).to_numpy(float)
            if not np.array_equal(np.isnan(av), np.isnan(bv)):
                raise ValueError(f"Unexpected missingness change: {col}")
            mask = np.isfinite(av) & np.isfinite(bv)
            delta = float(np.max(np.abs(av[mask] - bv[mask]))) if mask.any() else 0.0
            if delta > 1e-10:
                changes[col] = delta
        elif a[col].fillna("<NA>").astype(str).tolist() != b[col].fillna("<NA>").astype(str).tolist():
            raise ValueError(f"Unexpected categorical change: {col}")
    p = prediction.sort_values(keys).reset_index(drop=True)
    q = original_prediction.sort_values(keys).reset_index(drop=True)
    error = float(np.max(np.abs(p.p_nonmarket.to_numpy(float) - q.p_nonmarket.to_numpy(float))))
    report = {"status": "CORRECTED_REFERENCE", "reason": "Numeric margin text was concatenated to race time without a separator",
              "original_trial_preserved": True, "original_history_sha256": sha(TRIAL / "target_history.parquet"),
              "original_prediction_sha256": sha(TRIAL / "morning_nonmarket_predictions.csv"),
              "changed_history_times": int(original.race_time_seconds.fillna(-999).ne(history.race_time_seconds.fillna(-999)).sum()),
              "changed_context_columns": changes, "original_prediction_max_abs_delta": error,
              "original_1e_minus7_reproduction_gate_passed": error <= 1e-7,
              "runners": len(prediction), "model_bundle_sha256": sha(bundle_path), "source_captures": evidence}
    args.output.mkdir(parents=True, exist_ok=True)
    context.to_csv(args.output / "morning_feature_context.csv", index=False)
    prediction.to_csv(args.output / "morning_nonmarket_predictions.csv", index=False)
    history.to_parquet(args.output / "target_history.parquet", index=False)
    report["corrected_history_sha256"] = sha(args.output / "target_history.parquet")
    report["corrected_context_sha256"] = sha(args.output / "morning_feature_context.csv")
    report["corrected_prediction_sha256"] = sha(args.output / "morning_nonmarket_predictions.csv")
    (args.output / "correction_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_captures"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
