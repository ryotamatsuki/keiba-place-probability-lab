"""Independent Nov-1 F3 trio preview: immutable 17-horse joint probabilities.

No race labels, no newly fitted parameters, no calibration from future.
Uncalibrated constrained maximum-entropy joint distribution only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from keiba_place_lab.market import normalized_win_probabilities
from keiba_place_lab.three_subset_v1 import top3_subset_marginals
from keiba_place_lab.trio_bridge_v1 import (
    invert_subset_marginals,
    plackett_luce_trio,
    project_exactly_three,
    trio_distribution,
)

RACE_ID = "202605041111"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", required=True, type=Path)
    parser.add_argument("--bundle-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Immutable trio preview output already exists")
    d = pd.read_csv(args.preview, dtype={"race_id": str, "horse_id": str})
    d = d.sort_values("horse_no").reset_index(drop=True)
    if (len(d) != 17 or d.horse_id.nunique() != 17
            or set(d.race_id) != {RACE_ID}
            or set(d.horse_no.to_numpy(int)) != set(range(1, 18))):
        raise ValueError("Missing 17-entry frozen F3 forward prediction")
    required = {"p_XC_75_25_logit", "win_odds", "horse_name"}
    if not required.issubset(d):
        raise ValueError("Missing F3 and source win-odds fields")
    p = d.p_XC_75_25_logit.to_numpy(float)
    q = project_exactly_three(p)
    w = invert_subset_marginals(q)
    error = np.max(np.abs(top3_subset_marginals(w) - q))
    if error > 3e-5:
        raise ValueError("F3 marginals not reproduced by trio joint model")
    horses = d.horse_no.to_numpy(int)
    joint = trio_distribution(w, horses)
    market, overround = normalized_win_probabilities(d.win_odds.to_numpy(float))
    pl = plackett_luce_trio(market, horses)
    keys = ["horse_1", "horse_2", "horse_3"]
    joint = joint.rename(columns={"probability": "p_f3_trio_raw"})
    joint = joint.merge(pl.rename(columns={"probability": "p_market_pl_trio"}),
                        on=keys, validate="one_to_one")
    joint["break_even_odds_f3_raw"] = 1 / joint.p_f3_trio_raw
    joint = joint.sort_values("p_f3_trio_raw", ascending=False).reset_index(drop=True)
    if len(joint) != 680 or not np.isclose(joint.p_f3_trio_raw.sum(), 1.0):
        raise ValueError("Wrong total trio probability or number of tickets")
    args.output.mkdir(parents=True, exist_ok=True)
    csv = args.output / "f3_trio_680.csv"
    joint.to_csv(csv, index=False, float_format="%.12g")
    frozen = json.loads(args.bundle_manifest.read_text())
    meta = {
        "status": "F3_TRIO_PROVISIONAL_20261031_NOT_NOV1_T10",
        "race_id": RACE_ID,
        "full_starter_count": len(d),
        "combination_count": len(joint),
        "preview_input_file": args.preview.name,
        "preview_input_sha256": sha(args.preview),
        "frozen_f3_model_sha256": frozen["bundle_sha256"],
        "f3_original_sum": float(p.sum()),
        "f3_projected_sum": float(q.sum()),
        "maximum_f3_marginal_projection_shift": float(np.max(np.abs(q-p))),
        "max_marginal_inverse_error": float(error),
        "f3_joint_sum": float(joint.p_f3_trio_raw.sum()),
        "market_pl_joint_sum": float(joint.p_market_pl_trio.sum()),
        "win_odds_overround": float(overround),
        "historical_trio_calibration_proven": False,
        "frozen_f3_not_modified": True,
        "actual_trio_odds_not_supplied": True,
        "top_10": joint.head(10).to_dict(orient="records"),
        "trio_csv_sha256": sha(csv),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n"
    )
    print("TRIO_PREVIEW_SUMMARY=" + json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
