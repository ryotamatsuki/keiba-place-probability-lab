"""Materialize a Stage 6 retrospective prediction-lock rehearsal."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from keiba_place_lab.lock import REQUIRED_LOCK_COLUMNS, validate_lock_frame

LOCK_MODE = "RETROSPECTIVE_DRY_RUN"
EXPECTED_AS_OF = "2026-10-03T08:35:00+09:00"
EXPECTED_RUNNERS = 18


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage5", type=Path, required=True)
    parser.add_argument("--nonmarket", type=Path, required=True)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stage5 = pd.read_csv(args.stage5)
    nonmarket = pd.read_csv(args.nonmarket)
    market = pd.read_csv(args.market)

    if len(stage5) != EXPECTED_RUNNERS:
        raise ValueError("Stage 5 input must have 18 runners")
    if stage5["as_of_time"].nunique() != 1 or stage5["as_of_time"].iloc[0] != EXPECTED_AS_OF:
        raise ValueError("Stage 5 target market timestamp changed")

    lock = stage5[
        [
            "horse_no",
            "horse_name",
            "as_of_time",
            "p_market_stage2",
            "p_nonmarket_calibrated_raw",
            "p_ensemble",
            "uncertainty_low",
            "uncertainty_high",
            "rank",
            "model_version",
        ]
    ].merge(
        nonmarket[["horse_no", "horse_name", "raw_probability"]],
        on=["horse_no", "horse_name"],
        how="inner",
        validate="one_to_one",
    ).merge(
        market[["horse_no", "horse_name"]],
        on=["horse_no", "horse_name"],
        how="inner",
        validate="one_to_one",
    )
    if len(lock) != EXPECTED_RUNNERS:
        raise ValueError("Frozen Stage 2/4/5 inputs do not align to 18 runners")

    lock = lock.rename(
        columns={
            "p_market_stage2": "p_market",
            "raw_probability": "p_model_raw",
            "p_nonmarket_calibrated_raw": "p_model_calibrated",
        }
    )
    lock["locked_commit_sha"] = args.source_commit
    lock = lock[REQUIRED_LOCK_COLUMNS].sort_values("rank").reset_index(drop=True)
    validate_lock_frame(lock)

    out_csv = args.output_dir / "probability_estimates.csv"
    lock.to_csv(out_csv, index=False, float_format="%.9f")

    fingerprints = {
        "stage5_ensemble.csv": sha256_file(args.stage5),
        "nonmarket_baseline.csv": sha256_file(args.nonmarket),
        "market_baseline.csv": sha256_file(args.market),
        "probability_estimates.csv": sha256_file(out_csv),
    }
    manifest = {
        "status": "PASS",
        "lock_mode": LOCK_MODE,
        "source_commit_sha": args.source_commit,
        "market_as_of_time": EXPECTED_AS_OF,
        "runner_count": len(lock),
        "sum_p_ensemble": float(lock["p_ensemble"].sum()),
        "model_version": lock["model_version"].iloc[0],
        "target_outcome_loaded": False,
        "final_target_odds_loaded": False,
        "payout_loaded": False,
        "popularity_loaded": False,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "fingerprints": fingerprints,
    }
    (args.output_dir / "stage6_lock_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    top = lock.head(6)
    lines = [
        "# Stage 6 — Retrospective Pre-race Lock Rehearsal",
        "",
        f"Status: **PASS — {LOCK_MODE}**",
        "",
        "This is an operational rehearsal of the pre-race locking step. It was executed",
        "after the scheduled start and must not be represented as a genuine pre-start lock.",
        "",
        "## Locked source",
        "",
        f"- source Stage 5 commit: `{args.source_commit}`",
        f"- frozen market timestamp: `{EXPECTED_AS_OF}`",
        f"- model version: `{lock['model_version'].iloc[0]}`",
        f"- runners: {len(lock)}",
        f"- sum P(top3): `{lock['p_ensemble'].sum():.12f}`",
        "",
        "## Information barriers",
        "",
        "- target outcome loaded: **no**",
        "- later/final target odds loaded: **no**",
        "- payout loaded: **no**",
        "- popularity loaded: **no**",
        "- source inputs are only the already-frozen Stage 2 / Stage 4 / Stage 5 files",
        "",
        "## Locked leading probabilities",
        "",
        "| rank | horse_no | horse_name | p_ensemble | low | high |",
        "|---:|---:|:---|---:|---:|---:|",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"| {row.rank} | {row.horse_no} | {row.horse_name} | "
            f"{row.p_ensemble:.6f} | {row.uncertainty_low:.6f} | "
            f"{row.uncertainty_high:.6f} |"
        )
    lines += [
        "",
        "## File fingerprints",
        "",
        *[f"- `{name}`: `{digest}`" for name, digest in fingerprints.items()],
        "",
        "## Decision",
        "",
        "The values in probability_estimates.csv are treated as immutable rehearsal-lock",
        "values from this point forward. Future result evaluation must compare against these",
        "values and must not rewrite them.",
        "",
        "For the next live race, this exact step must run and commit before scheduled start.",
    ]
    (args.output_dir / "STAGE6_LOCK_DECISION_LOG.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
