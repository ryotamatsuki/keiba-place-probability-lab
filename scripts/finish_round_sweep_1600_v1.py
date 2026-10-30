"""Finalize frozen 1600-round family sweep with same-runner B_raw one-SE gate.

Can produce transparent category D if some per-family checkpoint is missing.
All numeric scoring uses Python. This is FINAL win-odds historical development.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
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

from run_round_sweep_1600_v1 import (
    CHECKPOINTS,
    FAMILIES,
    KEYS,
    choose_rounds,
    file_sha,
    fit_alpha,
    row_fingerprint,
)

EXPECTED_YEAR_2024 = (
    "2291807521499019cb2a0c62f416c1e9e4f43f0b421711371d5f41aa97c8941a"
)
PENALTY = 0.0005
REFERENCE_NAMES = ("M0", "X2", "E5", "B_raw", "LightGBM120", "CatBoost120")


def read_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"race_id": str, "horse_id": str})
    df.race_date = pd.to_datetime(df.race_date)
    df.horse_no = df.horse_no.astype(int)
    if df.duplicated(KEYS).any():
        raise ValueError("Duplicate original/experimental horse keys: " + str(path))
    return df.sort_values(KEYS, kind="stable").reset_index(drop=True)


def aligned(left: pd.DataFrame, right: pd.DataFrame, name: str) -> None:
    if len(left) != len(right) or not left[KEYS].equals(right[KEYS]):
        raise ValueError(name + ": horse/date/race rows do not match")
    if not np.array_equal(
        left.top3_label.to_numpy(int), right.top3_label.to_numpy(int),
    ):
        raise ValueError(name + ": actual top3 labels do not match")


def market_match(a: np.ndarray, b: np.ndarray, name: str) -> None:
    if len(a) != len(b) or not np.allclose(a, b, atol=1e-10, rtol=0):
        raise ValueError(name + ": market benchmark changes between candidates")


def confirmed_family(root: Path, family: str) -> tuple[
    dict[int, pd.DataFrame], dict[int, dict], pd.DataFrame
]:
    location = root / family
    if not (location / "source_manifest.json").exists():
        raise FileNotFoundError(family + ": source data was not prepared")
    source = json.loads((location / "source_manifest.json").read_text())
    outer: dict[int, pd.DataFrame] = {}
    manifests: dict[int, dict] = {}
    for year in (2023, 2024):
        path = location / f"outer_{year}.csv"
        report = location / f"outer_{year}_manifest.json"
        if not path.exists() or not report.exists():
            raise FileNotFoundError(family + f": outer_{year} checkpoint missing")
        m = json.loads(report.read_text())
        if (m["family"] != family or m["outer_year"] != year
                or m["source_signature"] != source["source_signature"]
                or m["output_sha256"] != file_sha(path)
                or m["original_fixed_parity_max_abs_error"] > 1e-8):
            raise ValueError(family + f": failed original prediction parity {year}")
        outer[year] = read_frame(path)
        manifests[year] = m
    full_metrics = []
    for year in (2021, 2022, 2023):
        path = location / f"inner_{year}.csv"
        metric_path = location / f"inner_{year}_metrics.csv"
        record_path = location / f"inner_{year}_manifest.json"
        if not path.exists() or not metric_path.exists() or not record_path.exists():
            raise FileNotFoundError(family + f": inner_{year} checkpoint missing")
        record = json.loads(record_path.read_text())
        if (record["prediction_sha256"] != file_sha(path)
                or record["source_signature"] != source["source_signature"]
                or record["family"] != family
                or record["max_fitted_rounds"] != 1600):
            raise ValueError(family + ": source/inner-data checkpoint hash mismatch")
        metrics = pd.read_csv(metric_path)
        if set(metrics.rounds) != set(CHECKPOINTS) or set(metrics.year) != {year}:
            raise ValueError(family + ": wrong registered checkpoints/year")
        full_metrics.append(metrics)
    metrics_all = pd.concat(full_metrics, ignore_index=True)
    for year, record in manifests.items():
        selected, _ = choose_rounds(metrics_all, year)
        if selected != record["selected_rounds"]:
            raise ValueError(family + ": outer round count not inner-preselected")
        preyears = (2021, 2022) if year == 2023 else (2021, 2022, 2023)
        before = []
        for inner_year in preyears:
            d = read_frame(location / f"inner_{inner_year}.csv")
            before.append(d)
        calibration = pd.concat(before, ignore_index=True)
        calibration = calibration.sort_values(KEYS, kind="stable").reset_index(drop=True)
        rebuilt = fit_alpha(calibration, selected, year)
        if not np.isclose(
            rebuilt["alpha"], record["alpha"]["alpha"], atol=1e-7, rtol=0,
        ):
            raise ValueError(family + ": stored alpha is not reproducible")
        if not set(outer[year].race_date.dt.year) == {year}:
            raise ValueError(family + ": outer labels have incorrect year")
        if outer[year].race_date.min() <= pd.Timestamp(record["training"]["train_max_date"]):
            raise ValueError(family + ": training must end before evaluation")
        if not np.isfinite(outer[year][["p_raw","p_scaled"]].to_numpy(float)).all():
            raise ValueError(family + ": non-finite predictions")
    return outer, manifests, metrics_all


def correction(frame: pd.DataFrame, col: str) -> np.ndarray:
    m = frame.p_M0.to_numpy(float)
    q = frame[col].to_numpy(float)
    if np.any((m <= 0) | (m >= 1)) or np.any((q <= 0) | (q >= 1)):
        raise ValueError("Non-interior market correction probabilities")
    return logit(q) - logit(m)


def optional_ensemble_gate(train: pd.DataFrame) -> dict:
    if set(train.race_date.dt.year) != {2023}:
        raise ValueError("Only 2023 strict OOF may trigger optional mixtures")
    base = candidate_score(train, train.p_xgb_raw, name="xgb").race_macro_brier
    losses = {
        k: candidate_score(train, train[f"p_{k}_raw"], name=k).race_macro_brier
        for k in ("lgb", "cat")
    }
    c_x = correction(train, "p_xgb_raw")
    corrs = {
        k: float(np.corrcoef(c_x, correction(train, f"p_{k}_raw"))[0, 1])
        for k in ("lgb", "cat")
    }
    triggers = {
        k: bool(losses[k] < base or (
            corrs[k] < 0.90 and losses[k] <= base + 0.00010
        ))
        for k in ("lgb", "cat")
    }
    return {
        "2023_xgb_brier": base,
        "2023_candidate_brier": losses,
        "2023_market_residual_correlations": corrs,
        "individual_trigger": triggers,
        "run_three_preregistered_ensembles": bool(any(triggers.values())),
    }


def learned_weights(train: pd.DataFrame) -> dict:
    if set(train.race_date.dt.year) != {2023}:
        raise ValueError("2023 OOF is the ONLY valid stack weight period")
    market = logit(train.p_M0.to_numpy(float))
    f = np.column_stack([
        correction(train, name)
        for name in ("p_xgb_raw", "p_lgb_raw", "p_cat_raw")
    ])
    y = train.top3_label.to_numpy(int)
    counts = train.groupby("race_id").race_id.transform("size").to_numpy(float)
    per_row = 1 / (counts * train.race_id.nunique())
    initial = np.array([1.0, 0.0, 0.0])

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        p = expit(market + f @ w)
        residual = p - y
        loss = np.sum(per_row * residual**2)
        value = loss + PENALTY * np.square(w - initial).sum()
        grad = (
            2.0 * (per_row * residual * p * (1 - p)) @ f
            + 2.0 * PENALTY * (w - initial)
        )
        return float(value), grad

    result = minimize(
        objective, initial, jac=True, method="SLSQP",
        bounds=[(0.0, 1.5)]*3,
        constraints=[{
            "type":"ineq",
            "fun":lambda w: 2.0 - np.sum(w),
            "jac":lambda w: -np.ones(3),
        }],
        options={"ftol":1e-12, "maxiter":300},
    )
    if not result.success:
        raise ValueError("2023 pre-frozen SLSQP stack optimizer failed")
    if np.any(result.x < -1e-8) or result.x.sum() > 2.00000001:
        raise ValueError("2023 pre-frozen stack constraints violated")
    return {
        "weights_xgb_lgb_cat": result.x.tolist(),
        "regularized_train_objective": float(result.fun),
        "fit_years": [2023],
        "fit_rows": len(train),
        "fit_races": int(train.race_id.nunique()),
        "fit_max_date": str(train.race_date.max().date()),
    }


def mix_logits(test: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.shape != (3,) or not np.isfinite(w).all():
        raise ValueError("Only preregistered three-way logit residual weights")
    z = logit(test.p_M0.to_numpy(float)) + np.column_stack([
        correction(test, k)
        for k in ("p_xgb_raw","p_lgb_raw","p_cat_raw")
    ]) @ w
    p = expit(z)
    if not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError("Invalid ensemble probability")
    return p


def evaluated(test: pd.DataFrame, predictions: dict[str, np.ndarray],
              output: Path) -> dict:
    baseline = predictions["B_raw"]
    decision, scores = select_with_incumbent_one_se(
        test, predictions, incumbent="B_raw",
    )
    scores.to_csv(output / "2024_candidate_scores.csv", index=False)
    paired = []
    all_losses = []
    calibration = []
    reliability = []
    subgroup = []
    predicted = test[KEYS + [
        "top3_label", "surface", "distance_m", "race_class",
    ]].copy()
    for name, p in predictions.items():
        predicted[f"p_{name}"] = p
        delta = paired_delta(
            test, baseline, p, candidate_name="B_raw", reference_name=name,
        )
        paired.append({
            "candidate": name,
            "B_raw_minus_candidate_brier": delta.mean_brier_delta,
            "date_cluster_SE_brier": delta.se_brier_delta,
            "B_raw_minus_candidate_log_loss": delta.mean_log_loss_delta,
            "date_cluster_SE_log_loss": delta.se_log_loss_delta,
            "beats_B_raw_pairwise_one_se": bool(
                delta.mean_brier_delta > delta.se_brier_delta
            ),
        })
        lo = race_loss_table(test, p)
        lo["candidate"] = name
        all_losses.append(lo)
        ci, cs = calibration_intercept_slope(test.top3_label.to_numpy(int), p)
        calibration.append({
            "candidate": name, "intercept": ci, "slope": cs,
        })
        rel = calibration_table(test, p)
        rel["candidate"] = name
        reliability.append(rel)
        by = {
            "surface": test.surface.astype(str),
            "distance_band": pd.cut(
                test.distance_m, [999,1400,2000,2600],
                labels=["1000_1400", "1401_2000", "2001_2600"],
            ).astype(str),
            "race_class": test.race_class.fillna("MISSING").astype(str),
        }
        for group, values in by.items():
            for v in sorted(values.unique()):
                mask = values.eq(v).to_numpy()
                s = candidate_score(test.loc[mask], p[mask], name=name)
                subgroup.append({
                    "candidate": name, "group": group, "value": v,
                    **s.__dict__,
                })
    predicted.to_csv(output / "2024_oof_predictions.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "2024_paired_vs_B_raw.csv", index=False)
    pd.DataFrame(calibration).to_csv(output / "2024_calibration.csv", index=False)
    pd.DataFrame(subgroup).to_csv(output / "2024_subgroups.csv", index=False)
    pd.concat(reliability).to_csv(output / "2024_reliability.csv", index=False)
    losses = pd.concat(all_losses, ignore_index=True)
    losses.to_csv(output / "2024_race_losses.csv", index=False)
    base_loss = losses.loc[losses.candidate.eq("B_raw"), [
        "race_id", "block", "brier", "log_loss",
    ]].rename(columns={
        "brier": "B_raw_brier", "log_loss": "B_raw_log_loss",
    })
    loss_diff = losses.merge(
        base_loss, on=["race_id","block"], validate="many_to_one",
    )
    loss_diff["B_raw_minus_candidate_brier"] = (
        loss_diff.B_raw_brier - loss_diff.brier
    )
    loss_diff.to_csv(output / "2024_race_loss_differences.csv", index=False)
    return {
        "winner": decision.winner,
        "point_brier_best": decision.point_brier_best,
        "B_raw_retained": bool(decision.incumbent_retained),
        "selection_reason": decision.reason,
    }


def summarize(
    folder: Path, references: dict[str, Path], output: Path,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    original = read_frame(references["b"])
    source = read_frame(references["x2"])
    if not original[KEYS].equals(source[KEYS]) or not np.array_equal(
        original.top3_label, source.top3_label,
    ):
        raise ValueError("Frozen B_raw and X2 source mismatch")
    base = original.loc[original.race_date.dt.year.eq(2024)].copy()
    base = base.reset_index(drop=True)
    if row_fingerprint(base) != EXPECTED_YEAR_2024:
        raise ValueError("Frozen B_raw reference 2024 runner fingerprint changed")

    status = {}
    available: dict[str, dict[int, pd.DataFrame]] = {}
    meta = {}
    curves = []
    for name in FAMILIES:
        try:
            outer, manifests, metrics = confirmed_family(folder, name)
            for year in (2023, 2024):
                check = outer[year]
                ref = original.loc[original.race_date.dt.year.eq(year)].copy()
                ref = ref.reset_index(drop=True)
                aligned(ref, check, f"{name} outer{year}")
                market_match(ref.p_M0.to_numpy(float), check.p_M0.to_numpy(float),
                             f"{name} outer{year}")
            available[name] = outer
            meta[name] = manifests
            curves.append(metrics)
            status[name] = "COMPLETE"
        except (FileNotFoundError, ValueError, KeyError) as exc:
            status[name] = "INCOMPLETE: " + str(exc)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(
            output / "inner_all_family_curves.csv", index=False,
        )
    if not available:
        report = {"category": "D", "status": status,
                  "reason": "No complete learner has produced both outer years"}
        (output / "manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)+"\n",
        )
        return report

    e5 = read_frame(references["e5"])
    c1 = read_frame(references["c1"])
    e5_rows = e5.copy()
    c1_rows = c1.loc[c1.race_date.dt.year.eq(2024)].copy().reset_index(drop=True)
    aligned(base, e5_rows, "E5 2024")
    aligned(base, c1_rows, "C1 2024")
    market_match(base.p_M0.to_numpy(float), e5_rows.p_M0.to_numpy(float), "E5")
    market_match(base.p_M0.to_numpy(float), c1_rows.p_M0.to_numpy(float), "C1")
    market_match(base.p_M0.to_numpy(float), source.loc[
        source.race_date.dt.year.eq(2024), "p_M0_market"
    ].to_numpy(float), "X2")
    pred = {
        "M0": base.p_M0.to_numpy(float),
        "X2": e5.p_X2.to_numpy(float),
        "E5": e5.p_E5.to_numpy(float),
        "B_raw": base.p_B_raw.to_numpy(float),
        "LightGBM120": c1_rows.p_LightGBM_C1.to_numpy(float),
        "CatBoost120": c1_rows.p_CatBoost_C1.to_numpy(float),
    }
    for name, outer in available.items():
        pred[name + "_selected_raw"] = outer[2024].p_raw.to_numpy(float)
        pred[name + "_selected_scaled"] = outer[2024].p_scaled.to_numpy(float)

    gate = {"status": "SKIPPED: not all three families completed"}
    fitted = None
    if set(available) == set(FAMILIES):
        past = available["xgb"][2023][KEYS+["top3_label","p_M0"]].copy()
        for name in FAMILIES:
            other = available[name][2023]
            aligned(past, other, "2023 optional ensemble data")
            market_match(past.p_M0, other.p_M0, "2023 optional market")
            past[f"p_{name}_raw"] = other.p_raw.to_numpy(float)
        gate = optional_ensemble_gate(past)
        if gate["run_three_preregistered_ensembles"]:
            test = base[KEYS+["top3_label"]].copy()
            test["p_M0"] = base.p_M0.to_numpy(float)
            for name in FAMILIES:
                test[f"p_{name}_raw"] = pred[name + "_selected_raw"]
            pred["XL_75_25_logit"] = mix_logits(test, [.75,.25,0])
            pred["XC_75_25_logit"] = mix_logits(test, [.75,0,.25])
            fitted = learned_weights(past)
            pred["XLC_ridge_2023"] = mix_logits(
                test, fitted["weights_xgb_lgb_cat"],
            )
    decision = evaluated(base, pred, output)
    selected_rows = []
    for name, reports in meta.items():
        for year, detail in reports.items():
            selected_rows.append({
                "family": name,
                "outer_year": year,
                "selected_rounds": detail["selected_rounds"],
                "alpha": detail["alpha"]["alpha"],
                "alpha_fit_years": ",".join(
                    str(v) for v in detail["alpha"]["fit_years"]
                ),
                "train_max_date": detail["training"]["train_max_date"],
                "eval_min_date": detail["training"]["target_min_date"],
                "selected_model_fit_seconds": detail["training"]["elapsed_seconds"],
                "baseline_fit_seconds": detail["baseline_retrain_seconds"],
                "original_fixed_parity_max_error":
                    detail["original_fixed_parity_max_abs_error"],
                "upper_bound_selected": detail["selected_is_upper_bound"],
            })
    pd.DataFrame(selected_rows).to_csv(
        output / "outer_selected_rounds_and_alpha.csv", index=False,
    )
    data_complete = set(available) == set(FAMILIES)
    if not data_complete:
        category = "D"
    elif (
        decision["winner"] != "B_raw"
        and decision["winner"] in [
            x + suffix for x in FAMILIES
            for suffix in ("_selected_raw","_selected_scaled")
        ]
    ):
        category = "A"
    elif decision["winner"] != "B_raw":
        category = "A" if any(
            candidate_score(base, pred[x], name=x).race_macro_brier
            < candidate_score(base, pred["B_raw"], name="B_raw").race_macro_brier
            for x in pred if x.endswith("_selected_raw")
        ) else "B"
    elif decision["point_brier_best"] != "B_raw":
        category = "B"
    else:
        category = "C"
    report = {
        "category": category,
        "status": status,
        "status_explanation": {
            "A": "Selected boosting count produces a candidate passing B_raw one-SE",
            "B": "A candidate improves point Brier but B_raw one-SE incumbent retained",
            "C": "No incremental point gain within frozen learner/round grid",
            "D": "Incomplete data or resources; no comprehensive three-family verdict",
        }[category],
        "selection": decision,
        "optional_ensemble_2023_gate": gate,
        "optional_ensemble_2023_weights": fitted,
        "selected_rounds_and_alpha": selected_rows,
        "input_sha256": {
            k: file_sha(v) for k, v in references.items()
        },
        "2024_runner_fingerprint": row_fingerprint(base),
        "evaluation_rows": len(base),
        "evaluation_races": int(base.race_id.nunique()),
        "candidate_names": list(pred),
        "not_a_prospective_test": True,
        "market_is_final_odds_not_T_minus_10": True,
        "production_model_unchanged": True,
        "forward_gate": "1000 complete races, >=60 distinct dates, >=90 days",
    }
    (output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--b-oof", type=Path, required=True)
    parser.add_argument("--c1-oof", type=Path, required=True)
    parser.add_argument("--x2-oof", type=Path, required=True)
    parser.add_argument("--e5-2024", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = summarize(
        args.folder,
        {
            "b": args.b_oof, "c1": args.c1_oof,
            "x2": args.x2_oof, "e5": args.e5_2024,
        }, args.out,
    )
    print(json.dumps({
        "category": results["category"],
        "selection": results.get("selection"),
        "status": results["status"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
