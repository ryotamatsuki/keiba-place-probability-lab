"""Preregistered cross-experiment combinations: NO exploratory weight grid.

All candidates compare on the exact same 2024 runner keys against calibrated
historical final win-odds market. Results cannot update canonical live Stage5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from keiba_place_lab.ensemble import convex_blend
from keiba_place_lab.model_selection import (
    paired_delta,
    race_loss_table,
    select_with_incumbent_one_se,
)
from keiba_place_lab.nonmarket import calibration_table
from keiba_place_lab.stage4_successor import calibration_intercept_slope

KEYS = ["race_date", "race_id", "horse_id", "horse_no"]


def compare(a: pd.DataFrame, c: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    for frame in (a, c):
        frame["race_date"] = pd.to_datetime(frame.race_date)
        frame["race_id"] = frame.race_id.astype(str)
        frame["horse_id"] = frame.horse_id.astype(str)
        frame["horse_no"] = frame.horse_no.astype(int)
    a = a.loc[a.race_date.dt.year.eq(2024)].sort_values(KEYS).reset_index(drop=True)
    c = c.loc[c.race_date.dt.year.eq(2024)].sort_values(KEYS).reset_index(drop=True)
    if not a[KEYS].equals(c[KEYS]):
        raise ValueError("A/C outer evaluation row mismatch; no intersection allowed")
    if not a.top3_label.equals(c.top3_label):
        raise ValueError("A/C outcome conflict")
    baseline = a.p_A1_calibrated.to_numpy(float)
    preds = {
        "market_A1": baseline,
        "A1_C0_w005": convex_blend(baseline, c.p_C0.to_numpy(float), 0.05),
        "A1_C1_w005": convex_blend(baseline, c.p_C1.to_numpy(float), 0.05),
        "A1_C2_w005": convex_blend(baseline, c.p_C2.to_numpy(float), 0.05),
        "A2_C0_w005": convex_blend(a.p_A2_corrected.to_numpy(float),
                                    c.p_C0.to_numpy(float), 0.05),
    }
    decision, table = select_with_incumbent_one_se(a, preds, incumbent="market_A1")
    output = a[KEYS + ["top3_label", "surface", "distance_m", "race_class"]].copy()
    losses = []
    paired = []
    calibrations = []
    for name, p in preds.items():
        output[f"p_{name}"] = p
        loss = race_loss_table(a, p)
        loss["candidate"] = name
        losses.append(loss)
        delta = paired_delta(a, baseline, p, candidate_name="market_A1",
                             reference_name=name)
        intercept, slope = calibration_intercept_slope(a.top3_label.to_numpy(int), p)
        paired.append({"candidate": name, "market_minus_candidate": delta.mean_brier_delta,
                       "paired_date_se": delta.se_brier_delta,
                       "calibration_intercept": intercept, "calibration_slope": slope})
        bins = calibration_table(a, p)
        bins["candidate"] = name
        calibrations.append(bins)
    outcome = {
        "status": "HISTORICAL_FINAL_ODDS_RESEARCH_ONLY",
        "winner": decision.winner,
        "incumbent_retained": decision.incumbent_retained,
        "rows": len(a), "races": int(a.race_id.nunique()),
        "comparisons": paired,
    }
    result = pd.concat(losses, ignore_index=True)
    result.attrs["reliability"] = pd.concat(calibrations, ignore_index=True)
    return output, {"selection": outcome, "scores": table}, result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a-oof", type=Path, required=True)
    p.add_argument("--c-oof", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    a = pd.read_csv(args.a_oof)
    c = pd.read_csv(args.c_oof)
    oof, result, losses = compare(a, c)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out / "AC_oof_predictions.csv", index=False)
    result["scores"].to_csv(out / "AC_scores.csv", index=False)
    losses.to_csv(out / "AC_race_losses.csv", index=False)
    losses.attrs["reliability"].to_csv(out / "AC_reliability.csv", index=False)
    manifest = result["selection"]
    manifest["a_oof_sha256"] = hashlib.sha256(args.a_oof.read_bytes()).hexdigest()
    manifest["c_oof_sha256"] = hashlib.sha256(args.c_oof.read_bytes()).hexdigest()
    (out / "AC_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
