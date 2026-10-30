"""Frozen final-odds-domain research A, and strictly OOF 2024 B diagnostics.

Never mutates any Stage 4/5 freeze, existing lock, or model artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from xgboost import __version__ as xgb_version

from keiba_place_lab.ensemble import apply_logit_calibrator, convex_blend, fit_logit_calibrator
from keiba_place_lab.historical_market import reconstruct_historical_market
from keiba_place_lab.model_selection import (
    candidate_score, paired_delta, race_loss_table, select_with_incumbent_one_se,
)
from keiba_place_lab.nonmarket import calibration_table
from keiba_place_lab.probability_improvement_v1 import (
    corrected_harville_top3, euclidean_top3_projection, fit_logit_stack,
    predict_logit_stack,
)
from keiba_place_lab.stage4_successor import calibration_intercept_slope

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs/PROBABILITY_IMPROVEMENT_V1_SPEC.md"
EXPECTED_KAGGLE_SHA = "fb9345273b21a7c23d41260dc77e45134a1bc2fe756899f3416b24ac0dd51de3"
GRID = (0.75, 1.0, 1.25)
KEYS = ["race_date", "race_id", "horse_id", "horse_no"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def input_frames(base: Path, oof_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = ("phase_a_turf_1200_train_v1.parquet",
             "phase_a_turf_1200_validation_v1.parquet")
    for name in names:
        if not (base / name).exists():
            raise FileNotFoundError(f"Audited Stage3.6 input unavailable: {base/name}")
    train = pd.read_parquet(base / names[0])
    validation = pd.read_parquet(base / names[1])
    oof = pd.read_csv(oof_path)
    for d in (train, validation, oof):
        d["race_date"] = pd.to_datetime(d["race_date"])
        d["race_id"] = d["race_id"].astype(str)
        d["horse_id"] = d["horse_id"].astype(str)
        d["horse_no"] = d["horse_no"].astype(int)
        if d.duplicated(KEYS).any():
            raise ValueError("Duplicate historical/evaluation keys")
    if train.race_date.dt.year.max() != 2022:
        raise ValueError("Train history extends beyond 2022")
    if set(oof.race_date.dt.year.unique()) != {2023, 2024}:
        raise ValueError("Frozen outer OOF years missing")
    if "p_challenger" not in oof:
        raise ValueError("Audited prior Stage4 OOF missing")
    if not set(oof["top3_label"].unique()).issubset({0, 1}):
        raise ValueError("Invalid OOF labels")
    validation = validation.loc[validation.race_date.dt.year.isin((2023, 2024))]
    # This is an explicit row-contract check, not a selective inner join.
    x = validation[KEYS].sort_values(KEYS).reset_index(drop=True)
    y = oof[KEYS].sort_values(KEYS).reset_index(drop=True)
    if not x.equals(y):
        raise ValueError("Frozen OOF and validation cohorts disagree; do not intersect silently")
    return pd.concat([train, validation], ignore_index=True), oof.sort_values(KEYS).reset_index(drop=True)


def market_with_full_fields(results: Path, wanted_races: set[str]
                            ) -> tuple[pd.DataFrame, dict]:
    if sha(results) != EXPECTED_KAGGLE_SHA:
        raise ValueError("Kaggle final-odds source SHA differs from original Stage5 freeze")
    market, audit = reconstruct_historical_market(results, wanted_races)
    return market, audit


def aligned_panel(panel: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    joined = panel.merge(market, on=["race_id", "horse_no"], how="left",
                         validate="one_to_one", indicator=True)
    incomplete = joined.groupby("race_id")["_merge"].transform(
        lambda x: not x.eq("both").all()
    )
    if incomplete.any():
        joined = joined.loc[~incomplete].copy()
    if joined.empty:
        raise ValueError("No complete-market eligible rows")
    joined.drop(columns=["_merge"], inplace=True)
    if joined["market_p_win"].isna().any():
        raise ValueError("Missing market after filtering complete races")
    return joined


def all_field_corrected(market: pd.DataFrame, gamma: float, delta: float
                        ) -> pd.DataFrame:
    records = []
    for race_id, block in market.groupby("race_id", sort=False):
        block = block.sort_values("horse_no")
        p = corrected_harville_top3(block["market_p_win"].to_numpy(float), gamma, delta)
        records.extend(zip([race_id] * len(block), block.horse_no.astype(int), p))
    return pd.DataFrame(records, columns=["race_id", "horse_no", "probability"])


def matched_predictions(frame: pd.DataFrame, raw: pd.DataFrame) -> np.ndarray:
    work = frame[["race_id", "horse_no"]].copy()
    work["_row_order"] = np.arange(len(work))
    merged = work.merge(raw, on=["race_id", "horse_no"], validate="one_to_one",
                        how="left").sort_values("_row_order")
    if len(merged) != len(frame) or merged.probability.isna().any():
        raise ValueError("Candidate predictions do not cover exact common rows")
    return merged.probability.to_numpy(float)


def fingerprint(frame: pd.DataFrame) -> str:
    x = frame[KEYS].sort_values(KEYS).copy()
    x["race_date"] = x.race_date.dt.strftime("%Y-%m-%d")
    return hashlib.sha256(x.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def a2_grid_choice(panel: pd.DataFrame, cached: dict, outer_year: int
                   ) -> tuple[float, float, pd.DataFrame]:
    """Selection uses only earlier 2020-2022 walk-forward inner folds."""
    selection = []
    for gamma in GRID:
        for delta in GRID:
            p_all = cached[(gamma, delta)]
            frame = panel.copy()
            frame["p"] = p_all
            loss = []
            for year in (2020, 2021, 2022):
                fit = frame.loc[frame.race_date.dt.year.between(2016, year - 1)]
                test = frame.loc[frame.race_date.dt.year.eq(year)]
                if fit.empty or test.empty:
                    raise ValueError("Missing inner chronological fold")
                cal = fit_logit_calibrator(fit.p.to_numpy(float),
                                           fit.top3_label.to_numpy(int))
                pred = apply_logit_calibrator(test.p.to_numpy(float), cal)
                score = candidate_score(test, pred, name=f"{gamma}/{delta}")
                loss.append((score.race_macro_brier, score.races))
            # Equal race weighting across inner folds, not equal year weighting.
            metric = sum(v * n for v, n in loss) / sum(n for _, n in loss)
            selection.append({"gamma": gamma, "delta": delta, "inner_brier": metric,
                              "distance_from_one": abs(gamma-1)+abs(delta-1)})
    ranking = pd.DataFrame(selection).sort_values(
        ["inner_brier", "distance_from_one", "gamma", "delta"])
    win = ranking.iloc[0]
    return float(win.gamma), float(win.delta), ranking


def report_scores(frame: pd.DataFrame, probs: dict[str, np.ndarray],
                  incumbent: str, out: Path, label: str) -> dict:
    decision, table = select_with_incumbent_one_se(frame, probs, incumbent=incumbent)
    table.to_csv(out / f"{label}_scores.csv", index=False)
    reliability = []
    diagnostics = []
    race_losses = []
    for name, p in probs.items():
        intercept, slope = calibration_intercept_slope(
            frame.top3_label.to_numpy(int), p)
        bins = calibration_table(frame, p)
        bins["candidate"] = name
        reliability.append(bins)
        loss = race_loss_table(frame, p)
        loss["candidate"] = name
        race_losses.append(loss)
        for subgroup in ("all", "year", "surface", "distance_m", "race_class"):
            values = (pd.Series(["all"] * len(frame), index=frame.index)
                      if subgroup == "all" else
                      frame.race_date.dt.year if subgroup == "year" else frame[subgroup])
            for value in values.dropna().unique():
                mask = values.eq(value).to_numpy()
                sub = frame.loc[mask]
                score = candidate_score(sub, p[mask], name=name)
                diagnostics.append({"candidate": name, "subgroup": subgroup,
                                    "value": str(value), **score.__dict__,
                                    "calibration_intercept": intercept if subgroup=="all" else np.nan,
                                    "calibration_slope": slope if subgroup=="all" else np.nan})
    pd.concat(reliability).to_csv(out / f"{label}_reliability.csv", index=False)
    pd.concat(race_losses).to_csv(out / f"{label}_race_losses.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out / f"{label}_diagnostics.csv", index=False)
    for name, p in probs.items():
        frame[f"p_{name}"] = p
    frame.to_csv(out / f"{label}_oof_predictions.csv", index=False)
    deltas = []
    for name, p in probs.items():
        difference = paired_delta(frame, probs[incumbent], p, candidate_name=incumbent,
                                  reference_name=name)
        deltas.append({"candidate": name,
                       "incumbent_minus_candidate_brier": difference.mean_brier_delta,
                       "paired_date_se": difference.se_brier_delta})
    pd.DataFrame(deltas).to_csv(out / f"{label}_paired.csv", index=False)
    return {"winner": decision.winner, "incumbent_retained": decision.incumbent_retained,
            "point_brier_best": decision.point_brier_best,
            "rows": len(frame), "races": int(frame.race_id.nunique()),
            "row_fingerprint": fingerprint(frame)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--phase3a-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    panel, frozen = input_frames(args.historical_dir, args.phase3a_oof)
    market, market_audit = market_with_full_fields(args.results_csv,
                                                   set(panel.race_id.astype(str)))
    panel = aligned_panel(panel, market)
    eval_frame = aligned_panel(frozen, market).sort_values(KEYS).reset_index(drop=True)
    if not set(eval_frame.race_id).issubset(set(panel.race_id)):
        raise ValueError("No matching training/validation race panel")
    panel = panel.sort_values(KEYS).reset_index(drop=True)
    # Purely market-only experiment A, full-field odds from original source.
    allp = {(g, d): all_field_corrected(market, g, d)
            for g in GRID for d in GRID}
    selected_grids = []
    outer_predictions = {name: np.full(len(eval_frame), np.nan)
                         for name in ("A0_raw", "A1_calibrated", "A2_corrected")}
    raw = eval_frame.market_p_top3.to_numpy(float)
    outer_predictions["A0_raw"][:] = raw
    for year in (2023, 2024):
        train = panel.loc[panel.race_date.dt.year < year].copy()
        mask = eval_frame.race_date.dt.year.eq(year).to_numpy()
        test = eval_frame.loc[mask]
        if test.empty:
            raise ValueError("Missing outer evaluation fold")
        cal1 = fit_logit_calibrator(train.market_p_top3.to_numpy(float),
                                    train.top3_label.to_numpy(int))
        outer_predictions["A1_calibrated"][mask] = apply_logit_calibrator(
            test.market_p_top3.to_numpy(float), cal1)
        cache_train = {}
        for key, field in allp.items():
            cache_train[key] = matched_predictions(train, field)
        gamma, delta, ranking = a2_grid_choice(train, cache_train, year)
        raw_a2_train = cache_train[(gamma, delta)]
        cal2 = fit_logit_calibrator(raw_a2_train, train.top3_label.to_numpy(int))
        raw_a2_test = matched_predictions(test, allp[(gamma, delta)])
        outer_predictions["A2_corrected"][mask] = apply_logit_calibrator(raw_a2_test, cal2)
        selected_grids.append({"outer_year": year, "gamma": gamma, "delta": delta,
                               "train_rows": len(train), "test_rows": len(test),
                               "a1_intercept": cal1.intercept, "a1_slope": cal1.slope,
                               "a2_intercept": cal2.intercept, "a2_slope": cal2.slope})
        ranking.to_csv(out / f"A2_inner_grid_{year}.csv", index=False)
    if any(not np.isfinite(p).all() for p in outer_predictions.values()):
        raise ValueError("Some outer OOF probabilities missing")
    if not np.allclose(matched_predictions(eval_frame, allp[(1.0,1.0)]),
                       eval_frame.market_p_top3.to_numpy(float), atol=1e-9):
        raise ValueError("A2 gamma=delta=1 does not reproduce frozen A0")
    aresult = report_scores(eval_frame.copy(), outer_predictions, "A1_calibrated", out, "A")
    # 2023 OOF is legitimate meta training; 2024 only is legitimate B evaluation.
    fit_mask = eval_frame.race_date.dt.year.eq(2023).to_numpy()
    test_mask = eval_frame.race_date.dt.year.eq(2024).to_numpy()
    market_2023 = outer_predictions["A1_calibrated"][fit_mask]
    market_2024 = outer_predictions["A1_calibrated"][test_mask]
    pnon_2023 = eval_frame.loc[fit_mask, "p_challenger"].to_numpy(float)
    pnon_2024 = eval_frame.loc[test_mask, "p_challenger"].to_numpy(float)
    stack = fit_logit_stack(market_2023, pnon_2023,
                            eval_frame.loc[fit_mask, "top3_label"].to_numpy(int),
                            eval_frame.loc[fit_mask, "race_date"].to_numpy(),
                            max_training_date="2024-01-01")
    bp = {"B0_market": market_2024,
          "B0_blend005": convex_blend(market_2024, pnon_2024, 0.05),
          "B0_blend010": convex_blend(market_2024, pnon_2024, 0.10),
          "B1_logit_stack": predict_logit_stack(stack, market_2024, pnon_2024)}
    bresult = report_scores(eval_frame.loc[test_mask].copy().reset_index(drop=True),
                            bp, "B0_market", out, "B_shadow_v2")
    # S does not change the current subset eligibility; only full-field A0 diagnostic.
    projection = []
    for rid, block in market.groupby("race_id"):
        if len(block) < 4 or not np.isclose(block.market_p_top3.sum(), 3, atol=1e-8):
            continue
        p = block.market_p_top3.to_numpy(float)
        proj = euclidean_top3_projection(p)
        projection.append({"race_id": rid, "field_size": len(p),
                           "max_abs_market_projection": float(np.max(np.abs(p-proj)))})
    pd.DataFrame(projection).to_csv(out / "S_market_projection.csv", index=False)
    manifest = {"status": "PARTIAL_RESULTS_NOT_LIVE_ADOPTION",
                "domain": "historical_FINAL_odds_only",
                "base_main_sha": "a82cb4788190d9d82f9684cedc228a60d1077311",
                "spec_sha256": sha(SPEC), "kaggle_sha256": sha(args.results_csv),
                "phase3a_sha256": sha(args.phase3a_oof), "market_audit": market_audit,
                "A": aresult, "B_shadow_stage4_v2": bresult,
                "A2_grid": selected_grids,
                "B2": "BLOCKED: awaiting audited prior-feature OOF and same ScopeV3 model",
                "C": "BLOCKED: training-window refits must be run from complete as-of historical panel",
                "S": "market projection only; labelled full-field outcomes not joined",
                "xgboost": xgb_version, "sklearn": sklearn.__version__,
                "numpy": np.__version__, "pandas": pd.__version__,
                "python": platform.python_version(), "seed": 20261031}
    save_json(out / "manifest.json", manifest)
    (out / "RESULT.md").write_text(
        "# Probability Improvement V1 — historical final-odds research\n\n"
        "No 10-minute-before-off market-performance claim; no production promotion.\n\n"
        + "## A — common 2023/2024 development rows\n\n"
        + pd.read_csv(out/"A_scores.csv").to_markdown(index=False)
        + "\n\n## B — 2024 diagnostic only, frozen earlier Stage4 V2 OOF\n\n"
        + pd.read_csv(out/"B_shadow_v2_scores.csv").to_markdown(index=False)
        + "\n\n**B2 and C unavailable** until the complete leakage-safe feature/context "
        "and prior OOF are built. Stage4 V2 B result is NOT evidence for Scope V3. "
        "2025 has not been reselected. Market-only is retained live.\n",
        encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
