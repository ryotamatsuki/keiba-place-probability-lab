"""Strict common-2024 joint selection across ALL preregistered candidates.

A pairwise improvement can pass one SE yet fail the global candidate-selection
gate against the point-Brier-best. Never confuse the two.
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
    race_loss_table,
    select_with_incumbent_one_se,
)
from keiba_place_lab.nonmarket import calibration_table
from keiba_place_lab.stage4_successor import calibration_intercept_slope

KEYS = ["race_date", "race_id", "horse_id", "horse_no"]
BASE = "A1_market_only"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_evaluation_keys(base: pd.DataFrame, extra: pd.DataFrame, name: str) -> None:
    for frame in (base, extra):
        frame["race_date"] = pd.to_datetime(frame.race_date)
        frame["race_id"] = frame.race_id.astype(str)
        frame["horse_id"] = frame.horse_id.astype(str)
        frame["horse_no"] = frame.horse_no.astype(int)
        if frame.duplicated(KEYS).any():
            raise ValueError(f"Duplicate evaluation rows in {name}")
    if not base[KEYS].equals(extra[KEYS]):
        raise ValueError(f"{name} differs from frozen 2024 evaluation keys")
    if not base.top3_label.astype(int).equals(extra.top3_label.astype(int)):
        raise ValueError(f"{name} outcome labels differ")


def ordered_2024(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    x["race_date"] = pd.to_datetime(x.race_date)
    return x.loc[x.race_date.dt.year.eq(2024)].sort_values(KEYS).reset_index(drop=True)


def _probability(frame: pd.DataFrame, col: str) -> np.ndarray:
    p = frame[col].to_numpy(dtype=float)
    if len(p) != len(frame) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError(f"Invalid probability in {col}")
    return p


def joint_predictions(folder: Path, *, with_c: bool) -> tuple[pd.DataFrame, dict]:
    frame = ordered_2024(folder / "A_oof_predictions.csv")
    shadow = ordered_2024(folder / "B_shadow_v2_oof_predictions.csv")
    b2 = ordered_2024(folder / "B2_oof_predictions.csv")
    same_evaluation_keys(frame, shadow, "B1 shadow")
    same_evaluation_keys(frame, b2, "B2 offset")
    p = {
        BASE: _probability(frame, "p_A1_calibrated"),
        "A0_raw": _probability(frame, "p_A0_raw"),
        "A2_corrected": _probability(frame, "p_A2_corrected"),
        "B0_blend005": _probability(shadow, "p_B0_blend005"),
        "B0_blend010": _probability(shadow, "p_B0_blend010"),
        "B1_logit_stack_V2": _probability(shadow, "p_B1_logit_stack"),
        "B2_offset": _probability(b2, "p_B2_offset"),
    }
    if not np.allclose(p[BASE], _probability(shadow, "p_B0_market"),
                       rtol=0, atol=1e-12):
        raise ValueError("A1/B market-only probability mismatch")
    if not np.allclose(p[BASE], _probability(b2, "p_B0_market"),
                       rtol=0, atol=1e-12):
        raise ValueError("A1/B2 market-only probability mismatch")
    if with_c:
        c = ordered_2024(folder / "C_oof_predictions.csv")
        combos = ordered_2024(folder / "AC_oof_predictions.csv")
        same_evaluation_keys(frame, c, "C windows")
        same_evaluation_keys(frame, combos, "A/C combinations")
        for name in ("C0","C1","C2"):
            p[name + "_nonmarket"] = _probability(c, "p_" + name)
        for name in ("A1_C0_w005","A1_C1_w005","A1_C2_w005","A2_C0_w005"):
            p[name] = _probability(combos, "p_" + name)
        if not np.allclose(p[BASE], _probability(combos, "p_market_A1"),
                           rtol=0, atol=1e-12):
            raise ValueError("Market-only probability mismatch in A/C combinations")
    return frame, p


def run(folder: Path, out: Path, *, with_c: bool) -> dict:
    frame, preds = joint_predictions(folder, with_c=with_c)
    out.mkdir(parents=True, exist_ok=True)
    decision, scores = select_with_incumbent_one_se(frame, preds, incumbent=BASE)
    scores.to_csv(out / "JOINT_selection.csv", index=False)
    prediction = frame[KEYS + ["top3_label", "surface", "distance_m", "race_class"]].copy()
    scores_extra = []
    losses = []
    reliability = []
    subgroup_results = []
    band = pd.cut(
        frame.distance_m.astype(float),
        bins=[0.0, 1400.0, 2000.0, 2600.0],
        labels=["up_to_1400", "1401_2000", "2001_2600"],
        include_lowest=True,
    )
    for name, p in preds.items():
        prediction["p_" + name] = p
        paired = paired_delta(frame, preds[BASE], p, candidate_name=BASE,
                              reference_name=name)
        ci, cs = calibration_intercept_slope(frame.top3_label.to_numpy(int), p)
        scores_extra.append({"candidate": name,
                             "market_minus_candidate_brier": paired.mean_brier_delta,
                             "paired_date_clustered_se": paired.se_brier_delta,
                             "pairwise_one_se_pass": (
                                 paired.mean_brier_delta > paired.se_brier_delta),
                             "calibration_intercept": ci,
                             "calibration_slope": cs})
        race = race_loss_table(frame, p)
        race["candidate"] = name
        losses.append(race)
        rel = calibration_table(frame, p)
        rel["candidate"] = name
        reliability.append(rel)
        grouping = {
            "surface": frame.surface.astype(str),
            "distance_band": band.astype(str),
            "race_class": frame.race_class.fillna("MISSING").astype(str),
        }
        for group_name, group_values in grouping.items():
            for value in sorted(group_values.unique()):
                mask = group_values.eq(value).to_numpy()
                score = candidate_score(frame.loc[mask], p[mask], name=name)
                subgroup_results.append({
                    "candidate": name, "group": group_name, "value": value,
                    "rows": int(mask.sum()), "races": int(frame.loc[mask].race_id.nunique()),
                    "race_macro_brier": score.race_macro_brier,
                    "race_macro_log_loss": score.race_macro_log_loss,
                })
    detail = pd.DataFrame(scores_extra)
    detail.to_csv(out / "JOINT_paired_and_calibration.csv", index=False)
    prediction.to_csv(out / "JOINT_2024_oof_predictions.csv", index=False)
    losses = pd.concat(losses, ignore_index=True)
    losses.to_csv(out / "JOINT_race_losses.csv", index=False)
    pd.concat(reliability).to_csv(out / "JOINT_reliability.csv", index=False)
    pd.DataFrame(subgroup_results).to_csv(out / "JOINT_subgroups.csv", index=False)
    first = losses.loc[losses.candidate.eq(BASE),
                        ["race_id","block","brier","log_loss"]].rename(columns={
                            "brier":"baseline_brier","log_loss":"baseline_log_loss"})
    all_differences = losses.merge(first,on=["race_id","block"],validate="many_to_one")
    all_differences["market_minus_candidate_brier"] = (
        all_differences.baseline_brier - all_differences.brier)
    all_differences.to_csv(out / "JOINT_race_differences.csv", index=False)
    row_keys = frame[KEYS].copy()
    row_keys["race_date"] = row_keys.race_date.dt.strftime("%Y-%m-%d")
    digest = hashlib.sha256(row_keys.to_csv(index=False,lineterminator="\n").encode()).hexdigest()
    manifest = {
        "status": "RESEARCH_FINAL_ODDS_ONLY",
        "cohort": "turf1200 Phase-A 2024 complete odds",
        "candidate_selection": "one-SE vs point-Brier-best over registered candidates",
        "winner": decision.winner,
        "incumbent_retained": decision.incumbent_retained,
        "point_brier_best": decision.point_brier_best,
        "market_baseline": BASE,
        "rows": len(frame),
        "races": int(frame.race_id.nunique()),
        "row_fingerprint": digest,
        "with_scope_v3_c": with_c,
        "candidate_names": list(preds),
        "input_sha256": {
            path.name: sha(path) for path in folder.glob("*_oof_predictions.csv")
            if path.name in (
                "A_oof_predictions.csv",
                "B_shadow_v2_oof_predictions.csv",
                "B2_oof_predictions.csv",
                "C_oof_predictions.csv",
                "AC_oof_predictions.csv",
            )
        },
        "no_live_adoption": True,
        "forward_protocol": ">=1000 races; >=60 dates; >=90 elapsed days",
    }
    (out / "JOINT_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--with-c", action="store_true")
    args = p.parse_args()
    print(json.dumps(run(args.input_dir,args.output_dir,with_c=args.with_c), indent=2))


if __name__ == "__main__":
    main()
