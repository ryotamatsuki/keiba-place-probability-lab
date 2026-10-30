"""Pre-registered B_raw-anchored model-diversity ensembles on prior OOF.

Historical FINAL win odds only; 2023 labels determine two ridge mixtures,
2024 outcomes are never used for fitting weights. Both years are previously
seen development periods, NOT a new untouched test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import minimize
from scipy.special import expit, logit

from keiba_place_lab.model_selection import (
    candidate_score,
    paired_delta,
    race_loss_table,
    select_with_incumbent_one_se,
)
from keiba_place_lab.nonmarket import calibration_table
from keiba_place_lab.stage4_successor import calibration_intercept_slope

KEYS = ["race_date", "race_id", "horse_id", "horse_no"]
EXPECTED_FINGERPRINT = (
    "cb984b2553647d423e05742d0625988da3129dab3183f79b6fb84c74776a582f"
)
EXPECTED_2024_FINGERPRINT = (
    "2291807521499019cb2a0c62f416c1e9e4f43f0b421711371d5f41aa97c8941a"
)
SEED = 20261031
PENALTY = 0.0005
TOLERANCE = 1e-10
REFERENCE_NAMES = (
    "M0", "X2", "E5", "B_raw", "LGB_raw", "CAT_raw",
    "LGB_scaled", "CAT_scaled", "D3_old",
)
NEW_NAMES = (
    "BL_prob_75_25", "BL_logit_75_25", "BL_logit_50_50",
    "BC_logit_75_25", "BC_logit_50_50", "BLC_logit_50_25_25",
    "BLC_logit_equal", "BL_scalL_logit_75_25",
    "BLC_ridge_2023", "BL_ridge_2023",
)
ALL_NAMES = (*REFERENCE_NAMES, *NEW_NAMES)
FIXED_WEIGHTS = {
    "BL_logit_75_25": (0.75, 0.25, 0.0),
    "BL_logit_50_50": (0.50, 0.50, 0.0),
    "BC_logit_75_25": (0.75, 0.0, 0.25),
    "BC_logit_50_50": (0.50, 0.0, 0.50),
    "BLC_logit_50_25_25": (0.50, 0.25, 0.25),
    "BLC_logit_equal": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
}


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def row_fingerprint(frame: pd.DataFrame) -> str:
    rows = frame[KEYS].sort_values(KEYS, kind="stable").copy()
    rows.race_date = pd.to_datetime(rows.race_date).dt.strftime("%Y-%m-%d")
    return hashlib.sha256(rows.to_csv(
        index=False, lineterminator="\n",
    ).encode()).hexdigest()


def read_oof(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path, dtype={"race_id": str, "horse_id": str},
    )
    frame.race_date = pd.to_datetime(frame.race_date)
    frame.horse_no = frame.horse_no.astype(int)
    if frame.duplicated(KEYS).any():
        raise ValueError(f"Duplicate horse/race OOF rows in {path}")
    return frame.sort_values(KEYS, kind="stable").reset_index(drop=True)


def check_alignment(source: pd.DataFrame, other: pd.DataFrame,
                    label: str) -> None:
    if len(source) != len(other) or not source[KEYS].equals(other[KEYS]):
        raise ValueError(f"Cross-source runner keys differ: {label}")
    if not np.array_equal(
        source.top3_label.to_numpy(int), other.top3_label.to_numpy(int),
    ):
        raise ValueError(f"Cross-source target labels differ: {label}")


def check_probability(name: str, p: np.ndarray) -> np.ndarray:
    a = np.asarray(p, dtype=float)
    if a.ndim != 1 or not np.isfinite(a).all() or np.any(
        (a <= 0) | (a >= 1)
    ):
        raise ValueError(f"Invalid P(top3) for {name}")
    return a


def check_same_prob(name: str, p: np.ndarray, other: np.ndarray) -> None:
    if len(p) != len(other) or not np.allclose(
        p, other, atol=TOLERANCE, rtol=0,
    ):
        raise ValueError(f"Immutable prediction mismatch: {name}")


def source_panel(files: dict[str, Path]) -> tuple[pd.DataFrame, dict]:
    base = read_oof(files["x2"])
    if (row_fingerprint(base) != EXPECTED_FINGERPRINT
            or len(base) != 63276 or base.race_id.nunique() != 5731
            or set(base.race_date.dt.year) != {2023, 2024}):
        raise ValueError("Original PR32 development cohort was modified")

    b = read_oof(files["b"])
    c = read_oof(files["c"])
    check_alignment(base, b, "B_raw full 2023/24 OOF")
    check_alignment(base, c, "C1 full 2023/24 OOF")
    market = check_probability("PR32 M0", base.p_M0_market)
    check_same_prob("B vs M0", market, b.p_M0)
    check_same_prob("C vs M0", market, c.p_M0)

    panel = base[KEYS + [
        "top3_label", "surface", "distance_m", "race_class",
    ]].copy()
    panel["p_M0"] = market
    panel["p_X2"] = check_probability(
        "X2", base.p_X2_nonlinear_strength,
    )
    panel["p_B_raw"] = check_probability("B_raw", b.p_B_raw)
    panel["p_LGB_raw"] = check_probability(
        "LightGBM", c.p_LightGBM_C1,
    )
    panel["p_CAT_raw"] = check_probability(
        "CatBoost", c.p_CatBoost_C1,
    )

    subset = panel.loc[panel.race_date.dt.year.eq(2024)].copy()
    if (row_fingerprint(subset) != EXPECTED_2024_FINGERPRINT
            or len(subset) != 31225 or subset.race_id.nunique() != 2844):
        raise ValueError("2024 historical cohort changed")

    e5 = read_oof(files["e5"])
    c2 = read_oof(files["c2"])
    d3 = read_oof(files["d3"])
    for name, source in (("E5", e5), ("C2", c2), ("D3", d3)):
        check_alignment(subset.reset_index(drop=True), source, name)
        check_same_prob(
            name + " market", subset.p_M0.to_numpy(float), source.p_M0,
        )
    check_same_prob("E5 X2", subset.p_X2.to_numpy(float), e5.p_X2)
    check_same_prob("D3 old X2", subset.p_X2.to_numpy(float), d3.p_X2)
    check_same_prob("C2 raw LightGBM",
                    subset.p_LGB_raw.to_numpy(float), c2.p_LightGBM)
    check_same_prob("C2 raw CatBoost",
                    subset.p_CAT_raw.to_numpy(float), c2.p_CatBoost)
    check_same_prob("D3 old LightGBM",
                    subset.p_LGB_raw.to_numpy(float), d3.p_LightGBM)
    check_same_prob("D3 old CatBoost",
                    subset.p_CAT_raw.to_numpy(float), d3.p_CatBoost)

    extra = subset[KEYS].copy()
    extra["p_E5"] = check_probability("E5", e5.p_E5)
    extra["p_LGB_scaled"] = check_probability(
        "LGB scaled", c2.p_LightGBM_scaled,
    )
    extra["p_CAT_scaled"] = check_probability(
        "CAT scaled", c2.p_CatBoost_scaled,
    )
    extra["p_D3_old"] = check_probability("old D3", d3.p_D3)
    panel = panel.merge(
        extra, on=KEYS, how="left", validate="one_to_one", sort=False,
    )
    if panel.loc[panel.race_date.dt.year.eq(2024), [
        "p_E5", "p_LGB_scaled", "p_CAT_scaled", "p_D3_old",
    ]].isna().any().any():
        raise ValueError("2024 fixed comparator predictions incomplete")

    audit = {
        "input_sha256": {name: file_sha(path) for name, path in files.items()},
        "full_row_fingerprint": row_fingerprint(base),
        "2024_row_fingerprint": row_fingerprint(subset),
        "2023_train_rows": int(panel.race_date.dt.year.eq(2023).sum()),
        "2024_eval_rows": len(subset),
        "2023_train_max_date": str(
            panel.loc[panel.race_date.dt.year.eq(2023), "race_date"].max().date()
        ),
        "2024_eval_min_date": str(subset.race_date.min().date()),
        "all_sources_identical_runner_rows": True,
        "all_sources_identical_market_probability": True,
    }
    return panel, audit


def correction_array(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    market = check_probability("M0", frame.p_M0.to_numpy(float))
    m = logit(market)
    columns = ["p_B_raw", "p_LGB_raw", "p_CAT_raw"]
    residual = np.column_stack([
        logit(check_probability(col, frame[col].to_numpy(float))) - m
        for col in columns
    ])
    return m, residual


def residual_pool(frame: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.shape != (3,) or not np.isfinite(w).all():
        raise ValueError("Need three fixed finite residual weights")
    market, residual = correction_array(frame)
    return check_probability("residual pool", expit(market + residual @ w))


def fit_train_only_ridge(train: pd.DataFrame, *, use_cat: bool) -> dict:
    if set(train.race_date.dt.year) != {2023}:
        raise ValueError("2023 OOF only can fit ensemble weights")
    if train.race_date.max() >= pd.Timestamp("2024-01-01"):
        raise ValueError("Future outer labels enter stacking")
    logits, residual = correction_array(train)
    y = train.top3_label.to_numpy(float)
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("Invalid meta target label")

    # Exactly the repository's race-macro loss, without per-iteration groupby.
    count = train.groupby("race_id").race_id.transform("size").to_numpy(float)
    n_races = int(train.race_id.nunique())
    row_mass = 1.0 / (count * n_races)
    init = np.array([1.0, 0.0, 0.0], dtype=float)
    free = np.array([0, 1, 2] if use_cat else [0, 1], dtype=int)
    bounds = [(0.0, 1.5)] * len(free)

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        w = init.copy()
        w[free] = params
        if not use_cat:
            w[2] = 0.0
        pred = expit(logits + residual @ w)
        error = pred - y
        value = float(np.sum(row_mass * error * error) +
                      PENALTY * np.sum((w - init) ** 2))
        jacfull = (
            2.0 * (row_mass * error * pred * (1.0 - pred)) @ residual
            + 2.0 * PENALTY * (w - init)
        )
        return value, jacfull[free]

    initial = init[free].copy()
    result = minimize(
        objective, x0=initial, method="SLSQP", jac=True,
        bounds=bounds,
        constraints=[{
            "type": "ineq",
            "fun": lambda u: 2.0 - np.sum(u),
            "jac": lambda u: -np.ones_like(u),
        }],
        options={"maxiter": 300, "ftol": 1e-12},
    )
    if not result.success:
        raise ValueError(f"2023-only ridge failed: {result.message}")
    w = init.copy()
    w[free] = result.x
    if not use_cat:
        w[2] = 0.0
    if (np.any(w < -1e-8) or np.any(w > 1.50000001)
            or w.sum() > 2.00000001):
        raise ValueError("Invalid final 2023 ridge constraints")
    measured = candidate_score(
        train, residual_pool(train, w), name="ridge_train",
    ).race_macro_brier
    value = objective(result.x)[0]
    if not np.isclose(value, measured + PENALTY * np.square(w - init).sum(),
                      atol=1e-10, rtol=0):
        raise ValueError("Race-macro objective misaligned with baseline")
    return {
        "weights_B_LGB_CAT": w.tolist(),
        "training_races": n_races,
        "training_rows": len(train),
        "training_max_date": str(train.race_date.max().date()),
        "regularized_train_objective": float(value),
        "unregularized_train_brier": float(measured),
        "optimizer_success": bool(result.success),
    }


def candidates_2024(test: pd.DataFrame, fitted: dict) -> dict[str, np.ndarray]:
    pred: dict[str, np.ndarray] = {
        name: check_probability(name, test["p_" + name].to_numpy(float))
        for name in REFERENCE_NAMES
    }
    pred["BL_prob_75_25"] = check_probability(
        "BL_prob_75_25", 0.75 * pred["B_raw"] + 0.25 * pred["LGB_raw"],
    )
    for name, weights in FIXED_WEIGHTS.items():
        pred[name] = residual_pool(test, np.asarray(weights, dtype=float))
    offset = logit(pred["M0"])
    pred["BL_scalL_logit_75_25"] = expit(
        offset + 0.75 * (logit(pred["B_raw"]) - offset)
        + 0.25 * (logit(pred["LGB_scaled"]) - offset)
    )
    pred["BLC_ridge_2023"] = residual_pool(
        test, np.asarray(fitted["BLC_ridge_2023"]["weights_B_LGB_CAT"]),
    )
    pred["BL_ridge_2023"] = residual_pool(
        test, np.asarray(fitted["BL_ridge_2023"]["weights_B_LGB_CAT"]),
    )
    if set(pred) != set(ALL_NAMES):
        raise ValueError(f"Candidate registry drift: {set(pred) ^ set(ALL_NAMES)}")
    if any(len(p) != len(test) for p in pred.values()):
        raise ValueError("Runner probabilities not aligned")
    return pred


def write_result(panel: pd.DataFrame, out: Path, audit: dict) -> dict:
    train = panel.loc[panel.race_date.dt.year.eq(2023)].copy()
    test = panel.loc[panel.race_date.dt.year.eq(2024)].copy().reset_index(drop=True)
    if not train.race_date.max() < test.race_date.min():
        raise ValueError("Ensemble training crosses outer 2024 boundary")

    fitted = {
        "BLC_ridge_2023": fit_train_only_ridge(train, use_cat=True),
        "BL_ridge_2023": fit_train_only_ridge(train, use_cat=False),
    }
    predictions = candidates_2024(test, fitted)
    incumbent = predictions["B_raw"]
    decision, scores = select_with_incumbent_one_se(
        test, predictions, incumbent="B_raw",
    )

    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "2024_all_candidate_scores.csv", index=False)
    forecast = test[KEYS + [
        "top3_label", "surface", "distance_m", "race_class",
    ]].copy()
    paired = []
    losses = []
    calibration = []
    reliability = []
    subgroups = []
    for name in ALL_NAMES:
        p = predictions[name]
        forecast["p_" + name] = p
        delta = paired_delta(
            test, incumbent, p,
            candidate_name="B_raw", reference_name=name,
        )
        e5_delta = paired_delta(
            test, predictions["E5"], p,
            candidate_name="E5", reference_name=name,
        )
        paired.append({
            "candidate": name,
            "B_raw_minus_candidate_brier": delta.mean_brier_delta,
            "B_raw_paired_date_SE": delta.se_brier_delta,
            "B_raw_minus_candidate_logloss": delta.mean_log_loss_delta,
            "B_raw_paired_logloss_SE": delta.se_log_loss_delta,
            "passes_pairwise_B_raw_one_se":
                bool(delta.mean_brier_delta > delta.se_brier_delta),
            "E5_minus_candidate_brier": e5_delta.mean_brier_delta,
            "E5_paired_date_SE": e5_delta.se_brier_delta,
        })
        l = race_loss_table(test, p)
        l["candidate"] = name
        losses.append(l)
        intercept, slope = calibration_intercept_slope(
            test.top3_label.to_numpy(int), p,
        )
        calibration.append({
            "candidate": name, "intercept": intercept, "slope": slope,
        })
        rel = calibration_table(test, p)
        rel["candidate"] = name
        reliability.append(rel)
        groups = {
            "surface": test.surface.astype(str),
            "distance_band": pd.cut(
                test.distance_m, [999, 1400, 2000, 2600],
                labels=["1000_1400", "1401_2000", "2001_2600"],
            ).astype(str),
            "race_class": test.race_class.fillna("MISSING").astype(str),
        }
        for group, values in groups.items():
            for value in sorted(values.unique()):
                selected = values.eq(value).to_numpy()
                metric = candidate_score(
                    test.loc[selected], p[selected], name=name,
                )
                subgroups.append({
                    "candidate": name,
                    "group": group,
                    "value": value,
                    **metric.__dict__,
                })
    forecast.to_csv(out / "2024_paired_predictions.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "2024_paired_vs_B_raw.csv", index=False)
    pd.DataFrame(calibration).to_csv(out / "2024_calibration.csv", index=False)
    pd.DataFrame(subgroups).to_csv(out / "2024_subgroups.csv", index=False)
    pd.concat(reliability).to_csv(out / "2024_reliability.csv", index=False)
    race = pd.concat(losses, ignore_index=True)
    race.to_csv(out / "2024_race_losses.csv", index=False)
    baseline = race.loc[race.candidate.eq("B_raw"), [
        "race_id", "block", "brier", "log_loss",
    ]].rename(columns={
        "brier": "B_raw_brier", "log_loss": "B_raw_log_loss",
    })
    diffs = race.merge(
        baseline, on=["race_id", "block"], validate="many_to_one",
    )
    diffs["B_raw_minus_candidate_brier"] = (
        diffs.B_raw_brier - diffs.brier
    )
    diffs.to_csv(out / "2024_race_differences.csv", index=False)

    _, f = correction_array(test)
    corr = pd.DataFrame(
        f, columns=["B_raw", "LightGBM", "CatBoost"],
    ).corr()
    corr.to_csv(out / "2024_market_residual_correlations.csv")
    loss_wide = race.pivot(index="race_id", columns="candidate", values="brier")
    loss_wide.corr().to_csv(out / "2024_race_loss_correlations.csv")

    manifest = {
        "status": "HISTORICAL_FINAL_ODDS_KNOWN_2024_DEVELOPMENT_ONLY",
        "pre_registration": "docs/BRAW_ENSEMBLE_V1_SPEC.md",
        "pre_registration_commit": "52d0ae7306b864a6cff3ad770bd8dedb4abdd494",
        "source_audit": audit,
        "evaluation_races": int(test.race_id.nunique()),
        "evaluation_rows": len(test),
        "same_race_same_horse_all_candidates": True,
        "candidate_registry": list(ALL_NAMES),
        "candidate_registry_sha256": hashlib.sha256(
            "\n".join(ALL_NAMES).encode(),
        ).hexdigest(),
        "fixed_correction_weights": FIXED_WEIGHTS,
        "ridge": fitted,
        "incumbent": "B_raw",
        "selection": {
            "winner": decision.winner,
            "point_brier_best": decision.point_brier_best,
            "B_raw_retained": bool(decision.incumbent_retained),
            "reason": decision.reason,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "seed": SEED,
        },
        "runner_file_sha256": file_sha(Path(__file__)),
        "source_is_final_odds_not_T_minus_10": True,
        "development_period_not_unseen": True,
        "live_production_unchanged": True,
        "forward_gate": "1000 races / 60 race dates / 90 days, frozen model",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b-oof", type=Path, required=True)
    parser.add_argument("--c-oof", type=Path, required=True)
    parser.add_argument("--x2-oof", type=Path, required=True)
    parser.add_argument("--e5-2024", type=Path, required=True)
    parser.add_argument("--c2-2024", type=Path, required=True)
    parser.add_argument("--d3-2024", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    files = {
        "b": args.b_oof,
        "c": args.c_oof,
        "x2": args.x2_oof,
        "e5": args.e5_2024,
        "c2": args.c2_2024,
        "d3": args.d3_2024,
    }
    panel, audit = source_panel(files)
    result = write_result(panel, args.output_dir, audit)
    print(json.dumps({
        "selection": result["selection"],
        "sample": [result["evaluation_races"], result["evaluation_rows"]],
        "ridge": result["ridge"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
