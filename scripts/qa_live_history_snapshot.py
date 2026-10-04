"""Independently audit an immutable live snapshot against official race inventory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from keiba_place_lab.live_history import expected_results, sha256_file
from keiba_place_lab.stage4_successor_phase3a import assert_full_field_context

ROOT = Path(__file__).resolve().parents[1]


def audit_snapshot(root: Path, ledger_path: Path, snapshot_id: str | None = None):
    pointer = json.loads((root / "current.json").read_text())
    snapshot_id = snapshot_id or pointer["snapshot_id"]
    snap = root / "snapshots" / snapshot_id
    manifest = json.loads((snap / "manifest.json").read_text())
    if manifest["snapshot_id"] != snapshot_id:
        raise ValueError("Snapshot identity mismatch")
    for filename, key in (("jra_flat_history.parquet", "history_sha256"),
                          ("entry_audit.parquet", "entry_audit_sha256"),
                          ("race_ledger.csv", "race_ledger_sha256")):
        if sha256_file(snap / filename) != manifest[key]:
            raise ValueError(f"Snapshot checksum mismatch: {filename}")
    if snapshot_id == pointer["snapshot_id"] and any(pointer[k] != manifest[k] for k in ("complete_through", "history_sha256")):
        raise ValueError("Current pointer provenance mismatch")
    cutoff = pd.Timestamp(manifest["complete_through"])
    coverage = json.loads(ledger_path.with_suffix(".manifest.json").read_text())
    if sha256_file(ledger_path) != coverage["ledger_sha256"] or pd.Timestamp(coverage["through"]) < cutoff:
        raise ValueError("Independent inventory cannot certify cutoff")
    ledger = pd.read_csv(ledger_path, dtype={"race_id": "string"})
    ledger["race_date"] = pd.to_datetime(ledger.race_date)
    ledger = ledger.loc[ledger.race_date.le(cutoff)].copy()
    expected = expected_results(ledger).set_index("race_id")
    history = pd.read_parquet(snap / "jra_flat_history.parquet")
    entries = pd.read_parquet(snap / "entry_audit.parquet")
    assert_full_field_context(history)
    for frame in (history, entries):
        if frame.duplicated(["race_id", "horse_id"]).any() or frame.duplicated(["race_id", "horse_no"]).any():
            raise ValueError("Duplicate identities in snapshot")
        if not frame.outcome_confirmed.eq(True).all():
            raise ValueError("Unresolved outcomes in snapshot")
        dates = pd.to_datetime(frame.race_date)
        if not dates.dt.year.eq(cutoff.year).all() or dates.gt(cutoff).any():
            raise ValueError("Outside-cutoff rows in snapshot")
        if set(frame.race_id.astype(str)) != set(expected.index.astype(str)):
            raise ValueError("Official inventory/full field race set mismatch")
    if not history.is_starter.eq(True).all() or not history.finish_status.isin(["finished", "dnf", "disqualified"]).all():
        raise ValueError("Nonstarters in starter history")
    if not entries.finish_status.isin(["finished", "dnf", "disqualified", "scratched", "excluded"]).all():
        raise ValueError("Unknown entry outcome status")
    actual_entries = entries.loc[entries.finish_status.isin(["finished", "dnf", "disqualified"])]
    keys = ["race_id", "horse_id", "horse_no"]
    if set(history[keys].itertuples(index=False, name=None)) != set(actual_entries[keys].itertuples(index=False, name=None)):
        raise ValueError("Declared-entry/result starter identities differ")
    qa = {item["race_id"]: item for item in manifest["race_qa"]}
    for race_id, race in history.groupby("race_id", sort=False):
        declared = entries.loc[entries.race_id.eq(race_id)]
        if not race.field_size.eq(len(race)).all() or not race.declared_field_size.eq(len(declared)).all():
            raise ValueError(f"Field size mismatch: {race_id}")
        if len(race) != qa[race_id]["actual_starters"] or len(declared) != qa[race_id]["declared_entries"]:
            raise ValueError(f"Manifest field QA mismatch: {race_id}")
        official = expected.loc[race_id]
        for column, official_column in (("race_date", "race_date"), ("racecourse", "racecourse"),
                                        ("surface", "official_surface"), ("distance_m", "official_distance_m")):
            got = pd.to_datetime(race[column]) if column == "race_date" else race[column]
            if not got.eq(official[official_column]).all():
                raise ValueError(f"Official condition mismatch: {race_id}/{column}")
        finished = race.loc[race.finish_status.eq("finished")]
        if finished.race_time_seconds.isna().any() or not finished.race_time_seconds.gt(0).all():
            raise ValueError(f"Invalid confirmed race time: {race_id}")
        # JRA publishes these result times to one decimal second. More digits
        # are a signal of the numeric-margin concatenation bug, not precision.
        tenths = finished.race_time_seconds * 10
        if (tenths - tenths.round()).abs().gt(1e-8).any():
            raise ValueError(f"Race time includes unexpected sub-tenth digits: {race_id}")
        if not race.finish_position.eq(1).any():
            raise ValueError(f"No winning starter: {race_id}")
    counts = {"expected_races": len(expected), "confirmed_races": history.race_id.nunique(),
              "expected_race_days": expected.race_date.nunique(), "confirmed_race_days": history.race_date.nunique(),
              "rows": len(history), "entry_rows": len(entries)}
    if manifest["missing_race_ids"] or any(int(manifest[k]) != int(v) for k, v in counts.items()):
        raise ValueError("Manifest completeness totals mismatch")
    return {"status": "PASS", "snapshot_id": snapshot_id, "complete_through": str(cutoff.date()),
            **{k: int(v) for k, v in counts.items()}, "missing_race_ids": [],
            "latest_race_date": str(pd.to_datetime(history.race_date).max().date()),
            "starter_status_counts": {str(k): int(v) for k, v in history.finish_status.value_counts().items()},
            "entry_status_counts": {str(k): int(v) for k, v in entries.finish_status.value_counts().items()},
            "source_race_counts": history.groupby("race_id").result_source_url.first().str.extract(r"https://([^/]+)")[0].value_counts().to_dict(),
            "history_sha256": manifest["history_sha256"], "entry_audit_sha256": manifest["entry_audit_sha256"],
            "race_ledger_sha256": manifest["race_ledger_sha256"], "independent_inventory_sha256": coverage["ledger_sha256"],
            "source_parser_contract": manifest.get("source_parser_contract"),
            "parser_source_sha256": manifest.get("parser_source_sha256"),
            "auditor_sha256": sha256_file(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/live_history/2026")
    parser.add_argument("--ledger", type=Path, default=ROOT / "analysis/live_history_2026/official_race_ledger.csv")
    parser.add_argument("--snapshot-id")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/live_history_2026/snapshot_audit.json")
    args = parser.parse_args()
    report = audit_snapshot(args.root, args.ledger, args.snapshot_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
