"""Post-registered diagnostics for the preregistered Information Expansion V1.

Does not fit, tune, or select any model. All inferences use original frozen
out-of-time 2023/24 runner predictions; no 2025 or target G1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from keiba_place_lab.model_selection import (
    candidate_score,
    paired_delta,
    select_with_incumbent_one_se,
)

KEYS = ["race_date", "race_id", "horse_id", "horse_no"]
CANDIDATES = (
    "M0_market",
    "L0_linear_old",
    "X0_nonlinear_old",
    "X1_nonlinear_history",
    "X2_nonlinear_strength",
)
COMPARISONS = (
    ("M0_market", "L0_linear_old", "linear_vs_market"),
    ("L0_linear_old", "X0_nonlinear_old", "nonlinear_vs_linear_old_info"),
    ("M0_market", "X0_nonlinear_old", "nonlinear_old_vs_market"),
    ("X0_nonlinear_old", "X1_nonlinear_history", "raw_history_increment"),
    ("X1_nonlinear_history", "X2_nonlinear_strength", "opponent_elo_increment"),
    ("M0_market", "X2_nonlinear_strength", "all_improvements_vs_market"),
)


def load_verified(folder: Path) -> tuple[pd.DataFrame, dict, dict]:
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("status") != "HISTORICAL_FINAL_ODDS_RESEARCH_ONLY":
        raise ValueError("Unverified model domain")
    if manifest.get("selection", {}).get("development_races") != 5731:
        raise ValueError("Unexpected preregistered cohort size")
    frame = pd.read_csv(
        folder / "outer_oof_predictions.csv",
        dtype={"race_id": str, "horse_id": str},
    )
    frame["race_date"] = pd.to_datetime(frame.race_date)
    frame["horse_no"] = frame.horse_no.astype(int)
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate evaluation keys")
    if set(frame.race_date.dt.year) != {2023, 2024}:
        raise ValueError("Known 2025 labels or missing evaluation fold")
    if frame.race_id.nunique() != manifest["selection"]["development_races"]:
        raise ValueError("Mismatch with signed run manifest")
    normalized = frame[KEYS].sort_values(KEYS, kind="stable").copy()
    normalized.race_date = normalized.race_date.dt.strftime("%Y-%m-%d")
    check = hashlib.sha256(
        normalized.to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()
    if check != manifest["selection"]["paired_row_fingerprint"]:
        raise ValueError("Evaluation fingerprint drift")
    required = [f"p_{c}" for c in CANDIDATES]
    if set(required).difference(frame):
        raise ValueError("Missing candidate OOF predictions")
    for col in required:
        p = frame[col].to_numpy(float)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError(f"Invalid probabilities: {col}")
    predictions = {c: frame["p_" + c].to_numpy(float) for c in CANDIDATES}
    return frame, predictions, manifest


def paired_comparison(frame: pd.DataFrame, predictions: dict,
                      baseline: str, challenger: str, label: str) -> dict:
    lhs = predictions[baseline]
    rhs = predictions[challenger]
    if len(lhs) != len(frame) or len(rhs) != len(frame):
        raise ValueError("Comparison row mismatch")
    difference = paired_delta(
        frame, lhs, rhs,
        candidate_name=baseline, reference_name=challenger,
    )
    return {
        "factor": label,
        "baseline": baseline,
        "challenger": challenger,
        "races": int(frame.race_id.nunique()),
        "rows": len(frame),
        "baseline_brier": candidate_score(
            frame, lhs, name=baseline
        ).race_macro_brier,
        "challenger_brier": candidate_score(
            frame, rhs, name=challenger
        ).race_macro_brier,
        "baseline_minus_challenger_brier": difference.mean_brier_delta,
        "date_clustered_paired_se": difference.se_brier_delta,
        "passes_pairwise_one_se": bool(
            difference.mean_brier_delta > difference.se_brier_delta
        ),
        "baseline_minus_challenger_log_loss": difference.mean_log_loss_delta,
        "paired_log_loss_se": difference.se_log_loss_delta,
    }


def run(folder: Path) -> dict:
    frame, preds, source = load_verified(folder)
    rows = [
        paired_comparison(frame, preds, first, next_, label)
        for first, next_, label in COMPARISONS
    ]
    pd.DataFrame(rows).to_csv(folder / "factor_attribution.csv", index=False)
    grouped = []
    for title, mask in (
        ("2023", frame.race_date.dt.year.eq(2023)),
        ("2024", frame.race_date.dt.year.eq(2024)),
        ("turf", frame.surface.eq("turf")),
        ("dirt", frame.surface.eq("dirt")),
    ):
        section = frame.loc[mask].copy()
        local = {k: value[mask.to_numpy()] for k, value in preds.items()}
        for first, next_, label in COMPARISONS:
            record = paired_comparison(section, local, first, next_, label)
            record["subgroup"] = title
            grouped.append(record)
    pd.DataFrame(grouped).to_csv(
        folder / "factor_attribution_subgroups.csv", index=False
    )
    verdict, table = select_with_incumbent_one_se(
        frame, preds, incumbent="M0_market"
    )
    if verdict.winner != source["selection"]["winner"]:
        raise ValueError("Selection changed while computing diagnostics")
    if verdict.incumbent_retained != source["selection"]["incumbent_retained"]:
        raise ValueError("Incumbent retention changed")
    table.to_csv(folder / "factor_verified_full_selection.csv", index=False)
    out = {
        "status": "COMPLETED_FINAL_ODDS_HISTORICAL_DEVELOPMENT_ONLY",
        "candidate_selection_reproduced": verdict.winner,
        "market_only_incumbent_retained": bool(verdict.incumbent_retained),
        "rows": len(frame),
        "races": int(frame.race_id.nunique()),
        "source_row_fingerprint": source["selection"]["paired_row_fingerprint"],
        "factors": rows,
        "do_not_confuse": (
            "Incremental information vs incumbent improvement; "
            "no out-of-time T-10 odds or forward validation yet."
        ),
    }
    (folder / "factor_manifest.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.result_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
