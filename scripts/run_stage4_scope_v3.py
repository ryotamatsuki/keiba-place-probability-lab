"""Train frozen V3 scope artifacts or score a real as-of roster."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from run_scope_expansion_completion import ROOT, cohort, digest, json_write

from keiba_place_lab.scope_expansion import full_context_for_races
from keiba_place_lab.scope_features import fit_scope_candidate, predict_scope_candidate
from keiba_place_lab.stage4_production_v3 import (
    build_live_context,
    route_model,
    score_scope_target,
    validate_roster,
)


def compile_routing(surfaces):
    routing = {"version": "stage4_scope_v3", "stage5_canonical": "market-only",
               "train_years": [2016, 2025], "surfaces": {}}
    specs = {}
    for surface in surfaces:
        path = ROOT / f"analysis/scope_expansion_{surface}/decisions.json"
        decisions = json.loads(path.read_text())
        policy = {"routes": [], "decision_sha256": digest(path)}
        for band in decisions["routes"]:
            lo, hi = band["min_distance_m"], band["max_distance_m"]
            model_id = f"{surface}_{band['model']}" + (f"_{lo}_{hi}" if band["model"] == "R3" else "")
            train_scope = [lo, hi] if band["model"] == "R3" else [1000, 2600]
            specs[model_id] = {"surface": surface, "scope": train_scope, "blocks": band["blocks"]}
            policy["routes"].append({**band, "model_id": model_id})
        if "1200_override" in decisions:
            override = decisions["1200_override"]
            model_id = "turf_1200_override"
            policy["1200_override"] = {**override, "model_id": model_id}
            specs[model_id] = {"surface": surface, "scope": override["scope"], "blocks": override["blocks"]}
        routing["surfaces"][surface] = policy
    return routing, specs


def train(args):
    panel = pd.read_parquet(args.panel)
    panel["race_date"] = pd.to_datetime(panel.race_date)
    for k in ("race_id", "horse_id"):
        panel[k] = panel[k].astype("string")
    if not all(f"{b}_starts" in panel for b in ("near", "regime", "course_distance")):
        raise ValueError("Use the audited scope_augmented.parquet cache")
    surfaces = ["turf", "dirt"] if args.surface == "all" else [args.surface]
    routing, specs = compile_routing(surfaces)
    source_hash = digest(args.panel)
    bundle = {"routing": routing, "models": {}}
    # No search or selection against 2025: refit only frozen configurations.
    metadata = {"routing": routing, "source_sha256": source_hash,
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "spec_sha256": digest(ROOT / "docs/SCOPE_EXPANSION_COMPLETION_SPEC.md"), "models": {}}
    import sklearn
    import xgboost
    metadata["versions"] = {"numpy": np.__version__, "pandas": pd.__version__,
                            "sklearn": sklearn.__version__, "xgboost": xgboost.__version__}
    for name, spec in specs.items():
        fit_rows = cohort(panel, spec["surface"], *spec["scope"])
        fit_rows = fit_rows.loc[fit_rows.race_date.dt.year.between(2016, 2025)]
        context = full_context_for_races(panel, set(fit_rows.race_id))
        print(f"Production refit {name}: {len(fit_rows)} rows", flush=True)
        model, prior = fit_scope_candidate(fit_rows, context, blocks=spec["blocks"], surface=spec["surface"])
        record = {**spec, "prior_mean": prior, "train_rows": len(fit_rows),
                  "train_races": fit_rows.race_id.nunique(),
                  "train_min_date": str(fit_rows.race_date.min().date()),
                  "train_max_date": str(fit_rows.race_date.max().date())}
        bundle["models"][name] = {**record, "model": model}
        metadata["models"][name] = record
    args.output.mkdir(parents=True, exist_ok=True)
    bundle_path = args.output / "bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    loaded = joblib.load(bundle_path)
    # Verify persistence on held training rows, not an extra validation claim.
    parity = {}
    for name, spec in specs.items():
        sample = cohort(panel, spec["surface"], *spec["scope"]).tail(32)
        context = full_context_for_races(panel, set(sample.race_id))
        fit = bundle["models"][name]
        expected = predict_scope_candidate(fit["model"], sample, context,
                                          prior_mean=fit["prior_mean"], blocks=fit["blocks"], surface=fit["surface"])
        actual = predict_scope_candidate(loaded["models"][name]["model"], sample, context,
                                        prior_mean=fit["prior_mean"], blocks=fit["blocks"], surface=fit["surface"])
        if not np.array_equal(expected, actual):
            raise ValueError("Serialized probability changed")
        parity[name] = {"rows": len(sample), "max_abs_error": float(np.max(np.abs(expected - actual)))}
    metadata["bundle_sha256"] = digest(bundle_path)
    metadata["serialization_qa"] = parity
    json_write(args.output / "manifest.json", metadata)
    json_write(args.output / "routing.json", routing)
    print(f"Saved {len(specs)} fitted configurations, serialization parity PASS", flush=True)


def predict(args):
    manifest = json.loads((args.bundle.parent / "manifest.json").read_text())
    if digest(args.bundle) != manifest["bundle_sha256"]:
        raise ValueError("Bundle checksum mismatch")
    # Load only the trusted, hash-verified repository model artifact.
    bundle = joblib.load(args.bundle)
    roster = validate_roster(pd.read_csv(args.roster, dtype={"horse_id": "string", "race_id": "string"}))
    if route_model(roster, bundle["routing"]) is None:
        scored = score_scope_target(bundle, roster)
    else:
        if args.history is None:
            raise ValueError("Supported scope needs updated standardized history")
        history = pd.read_parquet(args.history) if args.history.suffix == ".parquet" else pd.read_csv(args.history)
        context = build_live_context(history, roster, max_history_age_days=args.max_history_age_days)
        scored = score_scope_target(bundle, context)
    if args.output.exists():
        raise ValueError("Prediction output already exists; use a new version")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)
    json_write(args.output.with_suffix(".manifest.json"), {
        "prediction_sha256": digest(args.output),
        "model_sha256": digest(args.bundle), "roster_sha256": digest(args.roster),
        "history_sha256": digest(args.history) if args.history else None,
        "history_date_max": str(pd.to_datetime(history.race_date).max().date()) if args.history and route_model(roster, bundle["routing"]) else None,
        "stage5_canonical": "market-only", "shadow_only": True,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--panel", type=Path, required=True)
    fit.add_argument("--surface", choices=("turf", "dirt", "all"), default="all")
    fit.add_argument("--output", type=Path, default=ROOT / "models/stage4_scope_v3")
    live = sub.add_parser("predict")
    live.add_argument("--bundle", type=Path, default=ROOT / "models/stage4_scope_v3/bundle.joblib")
    live.add_argument("--roster", type=Path, required=True)
    live.add_argument("--history", type=Path)
    live.add_argument("--max-history-age-days", type=int, default=7)
    live.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(args) if args.command == "train" else predict(args)


if __name__ == "__main__":
    main()
