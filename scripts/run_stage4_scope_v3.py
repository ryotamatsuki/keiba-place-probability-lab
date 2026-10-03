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


def _load_live_snapshot(root: Path):
    pointer_path = root / "current.json"
    if not pointer_path.exists():
        raise ValueError(f"Live-history current pointer missing: {pointer_path}")
    pointer = json.loads(pointer_path.read_text())
    snapshot_dir = root / "snapshots" / pointer["snapshot_id"]
    manifest_path = snapshot_dir / "manifest.json"
    history_path = snapshot_dir / "jra_flat_history.parquet"
    if not manifest_path.exists() or not history_path.exists():
        raise ValueError("Live-history snapshot files are incomplete")
    live_manifest = json.loads(manifest_path.read_text())
    if live_manifest["snapshot_id"] != pointer["snapshot_id"]:
        raise ValueError("Live-history pointer/manifest snapshot mismatch")
    if live_manifest["schema_version"] != pointer["schema_version"]:
        raise ValueError("Live-history schema mismatch")
    if digest(history_path) != live_manifest["history_sha256"]:
        raise ValueError("Live-history snapshot checksum mismatch")
    return pd.read_parquet(history_path), live_manifest, history_path


def _prediction_history(args, roster):
    target_date = pd.Timestamp(roster.race_date.iloc[0]).normalize()
    source_meta = {}
    if args.history is not None:
        if args.historical_base is not None or args.live_history_root is not None:
            raise ValueError("--history cannot be combined with --historical-base/--live-history-root")
        history = pd.read_parquet(args.history) if args.history.suffix == ".parquet" else pd.read_csv(args.history)
        source_meta = {
            "mode": "standalone",
            "history_sha256": digest(args.history),
        }
    else:
        if args.historical_base is None or args.live_history_root is None:
            raise ValueError(
                "Supported scope needs either --history or both --historical-base and --live-history-root"
            )
        base = (
            pd.read_parquet(args.historical_base)
            if args.historical_base.suffix == ".parquet"
            else pd.read_csv(args.historical_base)
        )
        live, live_manifest, live_path = _load_live_snapshot(args.live_history_root)
        complete_through = pd.Timestamp(live_manifest["complete_through"])
        required_through = target_date - pd.Timedelta(days=1)
        if complete_through < required_through:
            raise ValueError(
                f"Live history is not certified complete through the day before target: "
                f"{complete_through.date()} < {required_through.date()}"
            )
        base["race_id"] = base["race_id"].astype("string")
        base["horse_id"] = base["horse_id"].astype("string")
        live["race_id"] = live["race_id"].astype("string")
        live["horse_id"] = live["horse_id"].astype("string")
        overlap = base[["race_id", "horse_id"]].merge(
            live[["race_id", "horse_id"]],
            on=["race_id", "horse_id"],
            how="inner",
        )
        if not overlap.empty:
            raise ValueError("Historical base and live snapshot overlap")
        history = pd.concat([base, live], ignore_index=True, sort=False)
        source_meta = {
            "mode": "historical_base_plus_live_snapshot",
            "base_history_sha256": digest(args.historical_base),
            "live_history_snapshot_id": live_manifest["snapshot_id"],
            "live_history_schema_version": live_manifest["schema_version"],
            "live_history_sha256": live_manifest["history_sha256"],
            "live_history_complete_through": live_manifest["complete_through"],
            "live_history_latest_race_date": live_manifest["latest_race_date"],
            "live_history_path": str(live_path),
        }

    history["race_date"] = pd.to_datetime(history["race_date"], errors="raise")
    history["race_id"] = history["race_id"].astype("string")
    history["horse_id"] = history["horse_id"].astype("string")
    if history.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Combined prediction history contains duplicate race_id x horse_id")

    # Explicit as-of slice at the caller boundary. build_live_context retains its own
    # strict no-on/after-target validation as a second information barrier.
    rows_before = len(history)
    history = history.loc[history["race_date"] < target_date].copy()
    source_meta["rows_before_asof_filter"] = rows_before
    source_meta["rows_after_asof_filter"] = len(history)
    source_meta["asof_exclusive"] = target_date.date().isoformat()
    return history, source_meta


def predict(args):
    manifest = json.loads((args.bundle.parent / "manifest.json").read_text())
    if digest(args.bundle) != manifest["bundle_sha256"]:
        raise ValueError("Bundle checksum mismatch")
    # Load only the trusted, hash-verified repository model artifact.
    bundle = joblib.load(args.bundle)
    roster = validate_roster(pd.read_csv(args.roster, dtype={"horse_id": "string", "race_id": "string"}))
    history = None
    history_meta = None
    if route_model(roster, bundle["routing"]) is None:
        scored = score_scope_target(bundle, roster)
    else:
        history, history_meta = _prediction_history(args, roster)
        context = build_live_context(
            history,
            roster,
            max_history_age_days=args.max_history_age_days,
        )
        scored = score_scope_target(bundle, context)
    if args.output.exists():
        raise ValueError("Prediction output already exists; use a new version")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)
    json_write(args.output.with_suffix(".manifest.json"), {
        "prediction_sha256": digest(args.output),
        "model_sha256": digest(args.bundle),
        "roster_sha256": digest(args.roster),
        "history_sources": history_meta,
        "history_date_max": (
            str(pd.to_datetime(history.race_date).max().date())
            if history is not None and not history.empty
            else None
        ),
        "stage5_canonical": "market-only",
        "shadow_only": True,
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
    live.add_argument("--historical-base", type=Path)
    live.add_argument("--live-history-root", type=Path)
    live.add_argument("--max-history-age-days", type=int, default=7)
    live.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(args) if args.command == "train" else predict(args)


if __name__ == "__main__":
    main()
