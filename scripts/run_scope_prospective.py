"""Create pre-off locks; verify their GitHub publication; review official outcomes."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from run_scope_expansion_completion import ROOT, diagnostics, digest, json_write

from keiba_place_lab.model_selection import paired_delta, race_loss_table
from keiba_place_lab.nonmarket import calibration_table
from keiba_place_lab.prospective_scope import make_lock, review_predictions, utc_timestamp

REPOSITORY = "ryotamatsuki/keiba-place-probability-lab"
API = f"https://api.github.com/repos/{REPOSITORY}"
SPEC = ROOT / "docs/SCOPE_PROSPECTIVE_V3_SPEC.md"


def get_json(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def verify_publication(directory, run_id, repository_path):
    """Server-created push-run timestamp is stronger than a backdatable commit date."""
    manifest = json.loads((directory / "manifest.json").read_text())
    run = get_json(f"{API}/actions/runs/{int(run_id)}")
    if run["event"] != "push" or run["head_branch"] != "main" or run["repository"]["full_name"] != REPOSITORY:
        raise ValueError("Publication proof must be a push event in the canonical repository")
    published = utc_timestamp(run["created_at"])
    if published > utc_timestamp(manifest["off_at"]) or published < utc_timestamp(manifest["locked_at"]):
        raise ValueError("GitHub publication did not occur between lock and off")
    commit = run["head_sha"]
    for filename in ("manifest.json", "predictions.csv"):
        path = quote(f"{repository_path.rstrip('/')}/{filename}", safe="/")
        remote = get_json(f"{API}/contents/{path}?ref={commit}")
        content = base64.b64decode(remote["content"])
        if content != (directory / filename).read_bytes():
            raise ValueError("Published lock differs from local lock")
    return {"run_id": int(run_id), "head_sha": commit, "published_at": published.isoformat(),
            "repository_path": repository_path, "repository": REPOSITORY}


def lock(args):
    scored = pd.read_csv(args.predictions, dtype={"race_id": "string", "horse_id": "string"})
    market = pd.read_csv(args.market, dtype={"race_id": "string", "horse_id": "string"})
    market_meta = json.loads(args.market_manifest.read_text())
    if market_meta.get("canonical_stage5_version") != "stage5_v2_market_only" or market_meta.get("snapshot_sha256") != digest(args.market):
        raise ValueError("Market input needs a canonical Stage 5 version and matching snapshot hash")
    keys = ["race_id", "horse_id"]
    if market.duplicated(keys).any() or len(market) != len(scored):
        raise ValueError("Market and nonmarket rosters disagree")
    matched = scored.merge(market[keys + ["p_market"]], on=keys, how="outer", validate="one_to_one", indicator=True)
    if not matched._merge.eq("both").all():
        raise ValueError("Market and nonmarket identities disagree")
    matched = matched.drop(columns="_merge")
    # Copy roster metadata from the hashed prediction input rather than outcomes.
    prediction_meta = json.loads(args.prediction_manifest.read_text())
    if prediction_meta.get("prediction_sha256") != digest(args.predictions):
        raise ValueError("Missing/mismatched production prediction provenance")
    if prediction_meta.get("shadow_only") is not True:
        raise ValueError("Prospective predictions must be V3 shadow output")
    payload, manifest = make_lock(
        matched, off_at=args.off_at, market_asof=market_meta["snapshot_at"],
        model_hash=prediction_meta["model_sha256"], protocol_hash=digest(SPEC),
    )
    manifest.update({"prediction_sha256": digest(args.predictions), "market_sha256": digest(args.market),
                     "market_manifest_sha256": digest(args.market_manifest),
                     "prediction_manifest_sha256": digest(args.prediction_manifest)})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "predictions.csv").write_bytes(payload)
    json_write(args.output / "manifest.json", manifest)
    print("Lock created. Publish both files before off, then verify the push-run evidence.")


def verify(args):
    proof = verify_publication(args.lock, args.run_id, args.repository_path)
    path = args.lock / "publication.json"
    if path.exists():
        raise ValueError("Publication evidence exists; immutable lock")
    json_write(path, proof)
    print(json.dumps(proof))


def review(args):
    if args.output.exists():
        raise ValueError("Review output exists; do not overwrite recorded decisions")
    prefix = str(args.locks.resolve().relative_to(ROOT)) + "/"
    tree = get_json(f"{API}/git/trees/main?recursive=1")
    if tree.get("truncated"):
        raise ValueError("Cannot verify complete lock ledger")
    remote_locks = {x["path"] for x in tree["tree"] if x["path"].startswith(prefix) and x["path"].endswith("/manifest.json")}
    local_locks = {str(p.resolve().relative_to(ROOT)) for p in args.locks.rglob("manifest.json")}
    if remote_locks != local_locks:
        raise ValueError("Local lock set differs from entire published main ledger")
    outcomes = pd.read_csv(args.outcomes, dtype={"race_id": "string", "horse_id": "string"})
    if not {"race_id", "horse_id", "top3_label"}.issubset(outcomes) or outcomes.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Official outcomes require unique race/horse keys and labels")
    events = pd.read_csv(args.events, dtype={"race_id": "string"}) if args.events else pd.DataFrame(columns=["race_id", "event_at", "reason"])
    if not {"race_id", "event_at", "reason"}.issubset(events):
        raise ValueError("Invalid cancellation event log")
    parts, exclusions, model_hashes, seen = [], [], set(), set()
    for path in sorted(args.locks.rglob("manifest.json")):
        directory = path.parent
        manifest = json.loads(path.read_text())
        if manifest.get("version") != "stage5_scope_v3_prospective":
            continue
        if manifest["protocol_hash"] != digest(SPEC):
            raise ValueError("Changed prospective protocol")
        proof = json.loads((directory / "publication.json").read_text())
        verify_publication(directory, proof["run_id"], proof["repository_path"])
        payload = directory / "predictions.csv"
        if hashlib.sha256(payload.read_bytes()).hexdigest() != manifest["payload_sha256"]:
            raise ValueError("Immutable lock changed")
        frame = pd.read_csv(payload, dtype={"race_id": "string", "horse_id": "string"})
        race = manifest["race_id"]
        if race in seen:
            raise ValueError("Duplicate lock versions for a race: resolve in a new preregistered period")
        seen.add(race)
        model_hashes.add(manifest["model_hash"])
        event = events.loc[events.race_id.eq(race)]
        if not event.empty:
            if not event.reason.isin(["late_cancellation", "race_cancelled"]).all() or event.reason.isna().any():
                raise ValueError("Unrecognized exclusion event")
            if any(utc_timestamp(t) < utc_timestamp(manifest["locked_at"]) for t in event.event_at):
                raise ValueError("Pre-lock cancellation should regenerate roster before lock")
            exclusions.append({"race_id": race, "reason": ",".join(event.reason)})
            continue
        official = outcomes.loc[outcomes.race_id.eq(race)]
        joined = frame.merge(official[["race_id", "horse_id", "top3_label"]], on=["race_id", "horse_id"], how="outer", validate="one_to_one", indicator=True)
        if not joined._merge.eq("both").all() or joined.top3_label.isna().any():
            exclusions.append({"race_id": race, "reason": "missing_official_outcomes"})
            continue
        if not joined.top3_label.isin([0, 1]).all():
            raise ValueError("Invalid official labels")
        parts.append(joined.drop(columns="_merge"))
    if len(model_hashes) > 1:
        raise ValueError("Changed fitted model during frozen prospective period")
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["race_date", "race_id"])
    report, scores = review_predictions(frame)
    report.update({"locked_races": len(seen), "excluded_races": exclusions, "ledger_tree_sha": tree["sha"],
                   "official_outcomes_sha256": digest(args.outcomes), "model_hashes": sorted(model_hashes)})
    args.output.mkdir(parents=True)
    json_write(args.output / "review.json", report)
    if scores is not None:
        frame = frame.loc[pd.to_datetime(frame.race_date).le(pd.Timestamp(report["window_end_date"]))].copy()
        scores.to_csv(args.output / "scores.csv", index=False)
        paired, losses, calibration, subgroup = [], [], [], []
        for c in [c for c in frame if c.startswith("p_w")]:
            loss = race_loss_table(frame, frame[c].to_numpy())
            loss["candidate"] = c
            losses.append(loss)
            bins = calibration_table(frame, frame[c].to_numpy())
            bins["candidate"] = c
            calibration.append(bins)
            subgroup.append(diagnostics(frame, frame[c].to_numpy(), c))
            for key in ("surface", "distance_m"):
                for value, group in frame.groupby(key):
                    subgroup.append(diagnostics(group, group[c].to_numpy(), c, key, value))
            if c != "p_w000":
                paired.append(paired_delta(frame, frame[c].to_numpy(), frame.p_w000.to_numpy(), candidate_name=c, reference_name="p_w000").__dict__)
        pd.DataFrame(paired).to_csv(args.output / "paired.csv", index=False)
        combined = pd.concat(losses)
        combined.to_csv(args.output / "race_losses.csv", index=False)
        ref = combined.loc[combined.candidate.eq("p_w000"), ["race_id", "brier", "log_loss"]]
        difference = combined.merge(ref, on="race_id", suffixes=("", "_market"), validate="many_to_one")
        difference["brier_delta"] = difference.brier - difference.brier_market
        difference["log_loss_delta"] = difference.log_loss - difference.log_loss_market
        difference.to_csv(args.output / "race_differences.csv", index=False)
        pd.concat(calibration).to_csv(args.output / "calibration.csv", index=False)
        pd.DataFrame(subgroup).to_csv(args.output / "subgroup_scores.csv", index=False)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("lock")
    pre.add_argument("--predictions", type=Path, required=True)
    pre.add_argument("--prediction-manifest", type=Path, required=True)
    pre.add_argument("--market", type=Path, required=True)
    pre.add_argument("--market-manifest", type=Path, required=True)
    pre.add_argument("--off-at", required=True)
    pre.add_argument("--output", type=Path, required=True)
    proof = sub.add_parser("verify")
    proof.add_argument("--lock", type=Path, required=True)
    proof.add_argument("--run-id", type=int, required=True)
    proof.add_argument("--repository-path", required=True)
    post = sub.add_parser("review")
    post.add_argument("--locks", type=Path, default=ROOT / "prospective/locks")
    post.add_argument("--outcomes", type=Path, required=True)
    post.add_argument("--events", type=Path)
    post.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"lock": lock, "verify": verify, "review": review}[args.command](args)


if __name__ == "__main__":
    main()
