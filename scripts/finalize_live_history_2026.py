"""One-shot audited finalization helpers for the 2026-10-03 live-history snapshot.

The shard command reparses every race in a disjoint date range from published pages.
It prefers Yahoo, then retries only failed races through Umanity while reusing successful
Yahoo captures. The merge command combines all shard snapshots, verifies that reparsing
did not alter pre-existing non-time facts through September, and publishes one immutable
full-year-to-date snapshot through 2026-10-03.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from keiba_place_lab.live_history import (
    DENMA_URL,
    RESULT_URL,
    FetchRecord,
    expected_results,
    publish_snapshot,
    sha256_bytes,
    sha256_file,
    update_live_history,
)

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "analysis/live_history_2026/official_race_ledger.csv"
LIVE_ROOT = ROOT / "data/live_history/2026"
CHECKPOINTS = ROOT / "analysis/live_history_2026/checkpoints.json"
FINAL_REPORT = ROOT / "analysis/live_history_2026/finalization_report.json"


def _load_ledger(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={"race_id": "string", "provider_id": "string", "meeting_day_id": "string"},
    )
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="raise")
    return frame


def _filtered_ledger(full: pd.DataFrame, start: pd.Timestamp, through: pd.Timestamp, work: Path) -> Path:
    subset = full.loc[full.race_date.between(start, through)].copy()
    if subset.empty:
        raise ValueError(f"No ledger rows in shard {start.date()}..{through.date()}")
    path = work / "official_race_ledger.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(path, index=False)
    manifest = {
        "through": through.date().isoformat(),
        "rows": int(len(subset)),
        "ledger_sha256": sha256_file(path),
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return path


def _drop_failed_yahoo_cache(cache: Path, subset: pd.DataFrame, failed_ids: set[str]) -> None:
    lookup = subset.assign(_rid=subset.race_id.astype(str)).set_index("_rid")
    for rid in sorted(failed_ids):
        if rid not in lookup.index:
            raise ValueError(f"Failed race absent from shard ledger: {rid}")
        provider_id = str(lookup.loc[rid, "provider_id"])
        for url in (RESULT_URL.format(provider_id=provider_id), DENMA_URL.format(provider_id=provider_id)):
            key = sha256_bytes(url.encode("utf-8"))
            for suffix in (".html", ".json"):
                (cache / f"{key}{suffix}").unlink(missing_ok=True)


def shard(args: argparse.Namespace) -> None:
    start = pd.Timestamp(args.start).normalize()
    through = pd.Timestamp(args.through).normalize()
    if start.year != 2026 or through.year != 2026 or start > through:
        raise ValueError("Shard bounds must be ordered 2026 dates")

    out = args.output.resolve()
    if out.exists():
        shutil.rmtree(out)
    work = out.parent / f".{out.name}-work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    full = _load_ledger(args.ledger)
    subset = full.loc[full.race_date.between(start, through)].copy()
    shard_ledger = _filtered_ledger(full, start, through, work)
    root = work / "root"
    cache = work / "cache"
    fallback_ids: list[str] = []

    try:
        manifest = update_live_history(
            year=2026,
            through=through,
            root=root,
            cache_dir=cache,
            recheck_days=0,
            max_workers=args.max_workers,
            source="yahoo",
            ledger_path=shard_ledger,
            reparse_all=True,
        )
    except RuntimeError:
        failure_path = root / "last_failure.json"
        if not failure_path.exists():
            raise
        failure = json.loads(failure_path.read_text())
        fallback_ids = sorted(str(x) for x in failure["failed_races"])
        if not fallback_ids:
            raise
        _drop_failed_yahoo_cache(cache, subset, set(fallback_ids))
        failure_path.unlink(missing_ok=True)
        manifest = update_live_history(
            year=2026,
            through=through,
            root=root,
            cache_dir=cache,
            recheck_days=0,
            max_workers=args.max_workers,
            source="umanity",
            ledger_path=shard_ledger,
            reparse_all=True,
        )

    if manifest.get("source_parser_contract") != "time_margin_separated_v1":
        raise ValueError("Shard did not satisfy corrected time-parser contract")
    pointer = json.loads((root / "current.json").read_text())
    snap = root / "snapshots" / pointer["snapshot_id"]
    shutil.copytree(snap, out)
    summary = {
        "start": start.date().isoformat(),
        "through": through.date().isoformat(),
        "snapshot_id": manifest["snapshot_id"],
        "expected_races": int(manifest["expected_races"]),
        "confirmed_races": int(manifest["confirmed_races"]),
        "rows": int(manifest["rows"]),
        "entry_rows": int(manifest["entry_rows"]),
        "fallback_race_ids": fallback_ids,
        "history_sha256": manifest["history_sha256"],
    }
    (out / "shard_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    shutil.rmtree(work)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _same_values(a: pd.Series, b: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(a) or pd.api.types.is_numeric_dtype(b):
        av = pd.to_numeric(a, errors="coerce").to_numpy(float)
        bv = pd.to_numeric(b, errors="coerce").to_numpy(float)
        if not np.array_equal(np.isnan(av), np.isnan(bv)):
            return False
        mask = np.isfinite(av) & np.isfinite(bv)
        return bool(np.allclose(av[mask], bv[mask], rtol=0.0, atol=1e-10)) if mask.any() else True
    return a.fillna("<NA>").astype(str).tolist() == b.fillna("<NA>").astype(str).tolist()


def _assert_preexisting_facts_stable(old: pd.DataFrame, new: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    keys = ["race_id", "horse_id"]
    old = old.copy()
    new = new.loc[pd.to_datetime(new.race_date).le(cutoff)].copy()
    for frame in (old, new):
        frame["race_id"] = frame.race_id.astype("string")
        frame["horse_id"] = frame.horse_id.astype("string")
    old = old.sort_values(keys).reset_index(drop=True)
    new = new.sort_values(keys).reset_index(drop=True)
    if old[keys].astype(str).to_dict("records") != new[keys].astype(str).to_dict("records"):
        raise ValueError("Reparse changed pre-existing starter identities")

    ignored = {
        "race_time_seconds",
        "result_source_url",
        "result_retrieved_at",
        "entry_source_url",
        "entry_retrieved_at",
    }
    stable = [c for c in old.columns if c in new.columns and c not in ignored]
    changed = [c for c in stable if not _same_values(old[c], new[c])]
    if changed:
        raise ValueError(f"Reparse changed non-time starter facts: {changed}")

    old_t = pd.to_numeric(old.race_time_seconds, errors="coerce").to_numpy(float)
    new_t = pd.to_numeric(new.race_time_seconds, errors="coerce").to_numpy(float)
    missing_changed = ~np.equal(np.isnan(old_t), np.isnan(new_t))
    finite = np.isfinite(old_t) & np.isfinite(new_t)
    delta = np.zeros(len(old_t), dtype=float)
    delta[finite] = np.abs(old_t[finite] - new_t[finite])
    changed_mask = missing_changed | (delta > 1e-10)
    old_bad = np.isfinite(old_t) & (np.abs(old_t * 10 - np.round(old_t * 10)) > 1e-8)
    new_bad = np.isfinite(new_t) & (np.abs(new_t * 10 - np.round(new_t * 10)) > 1e-8)
    if new_bad.any():
        raise ValueError("Corrected reparse still contains sub-tenth race times")
    return {
        "preexisting_rows": int(len(old)),
        "changed_time_rows": int(changed_mask.sum()),
        "changed_time_races": int(old.loc[changed_mask, "race_id"].nunique()),
        "legacy_subtenth_rows": int(old_bad.sum()),
        "corrected_subtenth_rows": int(new_bad.sum()),
        "max_abs_time_delta_seconds": float(delta.max()) if len(delta) else 0.0,
        "non_time_columns_changed": [],
    }


def merge(args: argparse.Namespace) -> None:
    full_ledger = _load_ledger(args.ledger)
    through = pd.Timestamp(args.through).normalize()
    full_ledger = full_ledger.loc[full_ledger.race_date.le(through)].copy()
    expected = expected_results(full_ledger)
    if len(expected) != 2544 or expected.race_date.nunique() != 83:
        raise ValueError(
            f"Official ledger invariant changed: races={len(expected)}, days={expected.race_date.nunique()}"
        )

    shard_dirs = sorted({p.parent for p in args.shards.rglob("jra_flat_history.parquet")})
    if len(shard_dirs) != 10:
        raise ValueError(f"Expected 10 reparse shards, found {len(shard_dirs)}")

    histories: list[pd.DataFrame] = []
    entries: list[pd.DataFrame] = []
    qas: list[dict] = []
    provenance: list[FetchRecord] = []
    shard_summaries: list[dict] = []
    for directory in shard_dirs:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("source_parser_contract") != "time_margin_separated_v1":
            raise ValueError(f"Uncorrected shard: {directory}")
        histories.append(pd.read_parquet(directory / "jra_flat_history.parquet"))
        entries.append(pd.read_parquet(directory / "entry_audit.parquet"))
        qas.extend(manifest["race_qa"])
        provenance.extend(FetchRecord(**item) for item in manifest.get("source_fetches", []))
        summary_path = directory / "shard_summary.json"
        shard_summaries.append(
            json.loads(summary_path.read_text()) if summary_path.exists()
            else {"snapshot_id": manifest["snapshot_id"]}
        )

    history = pd.concat(histories, ignore_index=True, sort=False)
    entry = pd.concat(entries, ignore_index=True, sort=False)
    for frame in (history, entry):
        frame["race_id"] = frame.race_id.astype("string")
        frame["horse_id"] = frame.horse_id.astype("string")
        frame["race_date"] = pd.to_datetime(frame.race_date, errors="raise")
    history = history.sort_values(["race_date", "race_id", "horse_no"]).reset_index(drop=True)
    entry = entry.sort_values(["race_date", "race_id", "horse_no"]).reset_index(drop=True)
    if history.duplicated(["race_id", "horse_id"]).any() or entry.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Duplicate identities across reparse shards")
    if set(history.race_id.astype(str)) != set(expected.race_id.astype(str)):
        missing = sorted(set(expected.race_id.astype(str)) - set(history.race_id.astype(str)))
        extra = sorted(set(history.race_id.astype(str)) - set(expected.race_id.astype(str)))
        raise ValueError(f"Shard race coverage mismatch missing={missing[:10]} extra={extra[:10]}")

    old_pointer = json.loads((args.live_root / "current.json").read_text())
    old_snap = args.live_root / "snapshots" / old_pointer["snapshot_id"]
    old_manifest = json.loads((old_snap / "manifest.json").read_text())
    old_history = pd.read_parquet(old_snap / "jra_flat_history.parquet")
    old_entries = pd.read_parquet(old_snap / "entry_audit.parquet")
    old_cutoff = pd.Timestamp(old_manifest["complete_through"])
    impact = _assert_preexisting_facts_stable(old_history, history, old_cutoff)

    old_entry_keys = old_entries[["race_id", "horse_id", "horse_no", "finish_status"]].copy()
    new_entry_keys = entry.loc[entry.race_date.le(old_cutoff), ["race_id", "horse_id", "horse_no", "finish_status"]].copy()
    for frame in (old_entry_keys, new_entry_keys):
        frame["race_id"] = frame.race_id.astype(str)
        frame["horse_id"] = frame.horse_id.astype(str)
    old_entry_keys = old_entry_keys.sort_values(["race_id", "horse_id"]).reset_index(drop=True)
    new_entry_keys = new_entry_keys.sort_values(["race_id", "horse_id"]).reset_index(drop=True)
    if old_entry_keys.fillna("<NA>").astype(str).to_dict("records") != new_entry_keys.fillna("<NA>").astype(str).to_dict("records"):
        raise ValueError("Reparse changed pre-existing declared-entry identities/statuses")

    manifest = publish_snapshot(
        root=args.live_root,
        through=through,
        expected_ledger=full_ledger,
        starters=history,
        entries=entry,
        race_qas=qas,
        provenance=provenance,
    )
    if manifest.get("source_parser_contract") != "time_margin_separated_v1":
        raise ValueError("Merged snapshot lacks corrected parser contract")
    if int(manifest["rows"]) != 35148:
        raise ValueError(
            f"Final starter-row invariant changed: rows={manifest['rows']}"
        )

    checkpoints = json.loads(args.checkpoints.read_text()) if args.checkpoints.exists() else []
    summary_keys = (
        "complete_through", "latest_race_date", "expected_races", "confirmed_races",
        "missing_race_ids", "expected_race_days", "confirmed_race_days", "rows",
        "entry_rows", "snapshot_id", "history_sha256", "abandoned_race_ids",
    )
    summary = {k: manifest[k] for k in summary_keys}
    checkpoints = [x for x in checkpoints if x.get("complete_through") != summary["complete_through"]]
    checkpoints.append(summary)
    checkpoints.sort(key=lambda x: x["complete_through"])
    args.checkpoints.write_text(json.dumps(checkpoints, ensure_ascii=False, indent=2) + "\n")

    report = {
        "status": "MERGED_PENDING_EXTERNAL_QA",
        "through": through.date().isoformat(),
        "previous_snapshot_id": old_manifest["snapshot_id"],
        "final_snapshot_id": manifest["snapshot_id"],
        "expected_races": int(manifest["expected_races"]),
        "confirmed_races": int(manifest["confirmed_races"]),
        "expected_race_days": int(manifest["expected_race_days"]),
        "confirmed_race_days": int(manifest["confirmed_race_days"]),
        "rows": int(manifest["rows"]),
        "entry_rows": int(manifest["entry_rows"]),
        "missing_race_ids": manifest["missing_race_ids"],
        "history_sha256": manifest["history_sha256"],
        "entry_audit_sha256": manifest["entry_audit_sha256"],
        "race_ledger_sha256": manifest["race_ledger_sha256"],
        "source_parser_contract": manifest["source_parser_contract"],
        "time_correction_impact_through_previous_cutoff": impact,
        "shards": shard_summaries,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("shard")
    p.add_argument("--start", required=True)
    p.add_argument("--through", required=True)
    p.add_argument("--ledger", type=Path, default=LEDGER)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-workers", type=int, default=6)

    p = sub.add_parser("merge")
    p.add_argument("--through", default="2026-10-03")
    p.add_argument("--ledger", type=Path, default=LEDGER)
    p.add_argument("--live-root", type=Path, default=LIVE_ROOT)
    p.add_argument("--shards", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, default=CHECKPOINTS)
    p.add_argument("--report", type=Path, default=FINAL_REPORT)

    args = parser.parse_args()
    {"shard": shard, "merge": merge}[args.command](args)


if __name__ == "__main__":
    main()
