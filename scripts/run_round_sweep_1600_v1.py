"""Frozen 60..1600 boosting-round experiment for identical F3 data.

Run one stage at a time; each stage saves an independently resumable OOF CSV.
Python 3.11; historical FINAL odds only. No production forecasting changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from run_information_expansion_v1 import _build_pipeline, _sort_keyed
from run_input_learning_a_v1 import (
    check_reference,
    feature_registry,
    prepare_dataset,
    row_fingerprint,
)
from run_model_families_c1 import (
    CATBOOST_PARAMS,
    LIGHTGBM_PARAMS,
    _categorical_input,
    _market_margin,
    _race_weight,
)
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from keiba_place_lab.model_selection import candidate_score

KEYS = ["race_date", "race_id", "horse_id", "horse_no"]
CHECKPOINTS = (60, 120, 240, 400, 800, 1600)
INNER_YEARS = (2021, 2022, 2023)
OUTER_YEARS = (2023, 2024)
SOURCE_FULL_SHA = (
    "cb984b2553647d423e05742d0625988da3129dab3183f79b6fb84c74776a582f"
)
ALPHA_PENALTY = 0.0001
FAMILIES = ("xgb", "lgb", "cat")
SEED = 20261031


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_predict_prefixes(
    train: pd.DataFrame, evaluate: pd.DataFrame, family: str,
    counts: tuple[int, ...],
) -> tuple[dict[int, np.ndarray], dict]:
    """One fit through largest checkpoint, original F3, equal-race weights."""
    if family not in FAMILIES:
        raise ValueError("Unregistered learner family")
    if (not counts or any(c <= 0 for c in counts)
            or len(set(counts)) != len(counts)):
        raise ValueError("Invalid boosting checkpoints")
    if pd.to_datetime(train.race_date).max() >= pd.to_datetime(evaluate.race_date).min():
        raise ValueError("Train-to-evaluation chronology violated")
    numbers, categorical = feature_registry()["F3_market_field"]
    largest = max(counts)
    tr = _sort_keyed(train)
    ev = _sort_keyed(evaluate)
    if not ev[KEYS].reset_index(drop=True).equals(
        evaluate[KEYS].reset_index(drop=True)
    ):
        raise ValueError("Prediction target order does not match frozen canonical rows")
    begin = time.perf_counter()

    if family == "xgb":
        a = tr[numbers + categorical].copy()
        b = ev[numbers + categorical].copy()
        for col in numbers:
            a[col] = pd.to_numeric(a[col], errors="coerce").to_numpy(
                dtype=float, na_value=np.nan,
            )
            b[col] = pd.to_numeric(b[col], errors="coerce").to_numpy(
                dtype=float, na_value=np.nan,
            )
        for col in categorical:
            a[col] = a[col].astype("object").where(a[col].notna(), None)
            b[col] = b[col].astype("object").where(b[col].notna(), None)
        pipeline = _build_pipeline(numbers, categorical)
        pipeline.named_steps["model"].set_params(n_estimators=largest)
        size = tr.groupby("race_id").race_id.transform("size").to_numpy(float)
        weight = 1.0 / size
        weight *= len(weight) / weight.sum()
        pipeline.fit(
            a, tr.top3_label.to_numpy(int),
            model__sample_weight=weight,
            model__base_margin=_market_margin(tr),
        )
        transformed = pipeline.named_steps["preprocess"].transform(b)
        model = pipeline.named_steps["model"]
        margin = _market_margin(ev)
        result = {
            n: model.predict_proba(
                transformed, base_margin=margin, iteration_range=(0, n),
            )[:, 1]
            for n in counts
        }
    elif family == "lgb":
        import lightgbm as lgb

        preprocess = _build_pipeline(
            numbers, categorical,
        ).named_steps["preprocess"]
        a = _categorical_input(tr, numbers, categorical)
        b = _categorical_input(ev, numbers, categorical)
        X = preprocess.fit_transform(a)
        Z = preprocess.transform(b)
        d = lgb.Dataset(
            X, label=tr.top3_label.to_numpy(int),
            init_score=_market_margin(tr),
            weight=_race_weight(tr), free_raw_data=False,
        )
        model = lgb.train(
            dict(LIGHTGBM_PARAMS), train_set=d, num_boost_round=largest,
        )
        margin = _market_margin(ev)
        result = {
            n: expit(
                np.asarray(
                    model.predict(
                        Z, raw_score=True, num_iteration=n, num_threads=2,
                    ), dtype=float,
                ) + margin
            )
            for n in counts
        }
    else:
        from catboost import CatBoostClassifier, Pool

        a = _categorical_input(tr, numbers, categorical)
        b = _categorical_input(ev, numbers, categorical)
        m_train, m_test = _market_margin(tr), _market_margin(ev)
        fitpool = Pool(
            a, label=tr.top3_label.to_numpy(int),
            cat_features=categorical, baseline=m_train,
            weight=_race_weight(tr),
        )
        model = CatBoostClassifier(**{
            **CATBOOST_PARAMS, "iterations": largest,
        })
        model.fit(fitpool)
        test_with = Pool(b, cat_features=categorical, baseline=m_test)
        test_without = Pool(b, cat_features=categorical)
        result = {}
        baseline_semantics = None
        for n in counts:
            raw_with = np.asarray(
                model.predict(
                    test_with, prediction_type="RawFormulaVal",
                    ntree_start=0, ntree_end=n,
                ), float,
            ).reshape(-1)
            raw_without = np.asarray(
                model.predict(
                    test_without, prediction_type="RawFormulaVal",
                    ntree_start=0, ntree_end=n,
                ), float,
            ).reshape(-1)
            if np.allclose(raw_with - raw_without, m_test, rtol=1e-5, atol=1e-5):
                raw = raw_with
                semantics = "baseline_included_in_raw_with"
            elif np.allclose(raw_with, raw_without, rtol=0, atol=1e-7):
                raw = raw_without + m_test
                semantics = "explicit_baseline_added_once"
            else:
                raise ValueError("CatBoost prediction market baseline failed exact audit")
            if baseline_semantics is None:
                baseline_semantics = semantics
            if baseline_semantics != semantics:
                raise ValueError("CatBoost baseline semantics differ across checkpoints")
            result[n] = expit(raw)

    for n, p in result.items():
        if (len(p) != len(ev) or not np.isfinite(p).all()
                or np.any((p <= 0) | (p >= 1))):
            raise ValueError(f"Invalid {family} probabilities for {n} rounds")
    info = {
        "family": family,
        "max_fitted_rounds": largest,
        "predicted_prefixes": list(counts),
        "train_rows": len(tr),
        "train_races": int(tr.race_id.nunique()),
        "target_rows": len(ev),
        "target_races": int(ev.race_id.nunique()),
        "train_max_date": str(tr.race_date.max().date()),
        "target_min_date": str(ev.race_date.min().date()),
        "elapsed_seconds": float(time.perf_counter() - begin),
        "seed": SEED,
    }
    return result, info


def choose_rounds(metrics: pd.DataFrame, outer_year: int) -> tuple[int, pd.DataFrame]:
    if outer_year not in OUTER_YEARS:
        raise ValueError("Invalid outer year")
    years = (2021, 2022) if outer_year == 2023 else INNER_YEARS
    eligible = metrics.loc[metrics.year.isin(years)].copy()
    if set(eligible.year) != set(years):
        raise ValueError("Missing eligible pre-outer inner years")
    if eligible.duplicated(["year", "rounds"]).any():
        raise ValueError("Duplicate inner checkpoints")
    if set(eligible.rounds) != set(CHECKPOINTS) or len(eligible) != (
        len(years) * len(CHECKPOINTS)
    ):
        raise ValueError("Incomplete registered checkpoint grid")
    eligible["loss_times_races"] = (
        eligible.race_macro_brier * eligible.races
    )
    table = eligible.groupby("rounds", as_index=False).agg(
        sum_loss=("loss_times_races", "sum"),
        races=("races", "sum"),
    )
    table["mean_brier"] = table.sum_loss / table.races
    table = table.sort_values(["mean_brier", "rounds"]).reset_index(drop=True)
    return int(table.iloc[0].rounds), table


def fit_alpha(inner: pd.DataFrame, selected: int, outer_year: int) -> dict:
    if selected not in CHECKPOINTS:
        raise ValueError("Alpha checkpoint not preregistered")
    valid_years = {2021, 2022} if outer_year == 2023 else {2021, 2022, 2023}
    if set(inner.race_date.dt.year) != valid_years:
        raise ValueError("Alpha-training OOF years violate chronological boundary")
    if inner.race_date.max() >= pd.Timestamp(f"{outer_year}-01-01"):
        raise ValueError("Outer label leakage into scaling")
    market = logit(inner.p_M0.to_numpy(float))
    raw = logit(inner[f"p_{selected}"].to_numpy(float))
    correction = raw - market

    def objective(alpha: float) -> float:
        p = expit(market + alpha * correction)
        return candidate_score(inner, p, name="alpha").race_macro_brier + (
            ALPHA_PENALTY * (alpha - 1.0)**2
        )

    sol = minimize_scalar(
        objective, bounds=(0.0, 1.5), method="bounded",
        options={"xatol": 1e-7, "maxiter": 200},
    )
    if not sol.success:
        raise ValueError("Inner-year alpha fitting failed")
    score, alpha = min(
        [(objective(0.0), 0.0), (objective(1.5), 1.5),
         (objective(float(sol.x)), float(sol.x))],
        key=lambda row: (row[0], abs(row[1] - 1.0)),
    )
    return {
        "alpha": alpha, "regularized_inner_loss": float(score),
        "fit_years": sorted(valid_years),
        "fit_rows": len(inner),
        "fit_races": int(inner.race_id.nunique()),
        "fit_max_date": str(inner.race_date.max().date()),
    }


def _load_prepared(panel_path: Path) -> pd.DataFrame:
    if not panel_path.exists():
        raise FileNotFoundError(f"Prepare stage required: {panel_path}")
    df = pd.read_parquet(panel_path)
    df.race_date = pd.to_datetime(df.race_date)
    return df


def stage_prepare(args: argparse.Namespace) -> None:
    if args.historical_dir is None or args.results_csv is None or args.races_csv is None:
        raise ValueError("Prepare requires actual historical input files")
    start = time.perf_counter()
    frame, audit = prepare_dataset(
        args.historical_dir, args.results_csv, args.races_csv,
    )
    original = check_reference(frame, args.reference_oof)
    if row_fingerprint(original) != SOURCE_FULL_SHA:
        raise ValueError("Original 2023–24 reference hash mismatch")
    if len(original) != 63276:
        raise ValueError("Original target rows mismatched")
    Path(args.panel).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.panel, index=False)
    args.out.mkdir(parents=True, exist_ok=True)
    record = {
        "reference_sha256": file_sha(args.reference_oof),
        "source": audit,
        "panel_sha256": file_sha(Path(args.panel)),
        "registry": "F3_market_field",
        "feature_columns": feature_registry()["F3_market_field"],
        "panel_rows": len(frame),
        "prepare_seconds": float(time.perf_counter() - start),
        "status": "READ_ONLY_HISTORICAL_PANEL_CREATED_LOCAL_NO_SOURCE_COMMIT",
    }
    (args.out / "source_manifest.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps({
        "stage": "prepare", "family": args.family,
        "rows": len(frame), "time_seconds": record["prepare_seconds"],
    }), flush=True)


def stage_inner(args: argparse.Namespace) -> None:
    year = args.year
    if year not in INNER_YEARS:
        raise ValueError("Invalid inner year")
    frame = _load_prepared(args.panel)
    train = frame.loc[frame.race_date.dt.year.between(2018, year-1)].copy()
    test = _sort_keyed(frame.loc[frame.race_date.dt.year.eq(year)].copy())
    preds, info = fit_predict_prefixes(
        train, test, args.family, CHECKPOINTS,
    )
    result = test[KEYS + ["top3_label"]].copy()
    result["p_M0"] = test.p_market_oof.to_numpy(float)
    metrics = []
    for rounds in CHECKPOINTS:
        result[f"p_{rounds}"] = preds[rounds]
        m = candidate_score(test, preds[rounds], name=str(rounds))
        metrics.append({
            "family": args.family, "year": year, "rounds": rounds,
            **m.__dict__,
        })
    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / f"inner_{year}.csv", index=False)
    pd.DataFrame(metrics).to_csv(
        args.out / f"inner_{year}_metrics.csv", index=False,
    )
    info.update({
        "prediction_sha256": file_sha(args.out / f"inner_{year}.csv"),
        "source_manifest_sha256": file_sha(args.out / "source_manifest.json"),
        "result_target_year": year,
        "kind": "STRICT_PAST_INNER_OOF",
    })
    (args.out / f"inner_{year}_manifest.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps({
        "stage": "inner", "family": args.family,
        "year": year, "elapsed_seconds": info["elapsed_seconds"],
        "best_checkpoint_on_this_year": min(
            metrics, key=lambda x: x["race_macro_brier"],
        )["rounds"],
    }), flush=True)


def _inner_bundle(args: argparse.Namespace, year: int
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = (2021, 2022) if year == 2023 else INNER_YEARS
    pred, metrics = [], []
    for iy in years:
        source = args.out / f"inner_{iy}.csv"
        meta = args.out / f"inner_{iy}_manifest.json"
        if not source.is_file() or not meta.is_file():
            raise FileNotFoundError(f"Required inner chronology checkpoint missing: {iy}")
        record = json.loads(meta.read_text())
        if (record["family"] != args.family
                or record["prediction_sha256"] != file_sha(source)
                or record["source_manifest_sha256"] != file_sha(
                    args.out / "source_manifest.json"
                )
                or record["max_fitted_rounds"] != 1600):
            raise ValueError(f"Inner-{iy} provenance failed")
        df = pd.read_csv(source, dtype={"race_id": str, "horse_id": str})
        df.race_date = pd.to_datetime(df.race_date)
        if set(df.race_date.dt.year) != {iy}:
            raise ValueError("Inner OOF labels from wrong year")
        pred.append(df)
        metrics.append(pd.read_csv(
            args.out / f"inner_{iy}_metrics.csv",
        ))
    full = pd.concat(pred, ignore_index=True)
    full = _sort_keyed(full)
    return full, pd.concat(metrics, ignore_index=True)


def stage_outer(args: argparse.Namespace) -> None:
    year = args.year
    if year not in OUTER_YEARS:
        raise ValueError("Invalid outer year")
    frame = _load_prepared(args.panel)
    inner, metrics = _inner_bundle(args, year)
    selected, table = choose_rounds(metrics, year)
    alpha = fit_alpha(inner, selected, year)
    train = frame.loc[frame.race_date.dt.year.between(2018, year-1)].copy()
    test = _sort_keyed(frame.loc[frame.race_date.dt.year.eq(year)].copy())
    original = check_reference(frame, args.reference_oof)
    year_ref = original.loc[original.race_date.dt.year.eq(year)].copy()
    if not test[KEYS].equals(year_ref[KEYS]):
        raise ValueError("Strict outer original runner identities changed")
    if not np.array_equal(test.top3_label, year_ref.top3_label):
        raise ValueError("Outer truth mismatched frozen labels")
    if not np.allclose(
        test.p_market_oof, year_ref.p_M0_market,
        rtol=0, atol=1e-10,
    ):
        raise ValueError("Outer calibrated market drift")

    predicted, duration = fit_predict_prefixes(
        train, test, args.family, (selected,),
    )
    raw = predicted[selected]
    existing = pd.read_csv(
        args.braw_oof if args.family == "xgb" else args.c1_oof,
        dtype={"race_id": str, "horse_id": str},
    )
    existing.race_date = pd.to_datetime(existing.race_date)
    existing = _sort_keyed(existing)
    reference = existing.loc[existing.race_date.dt.year.eq(year)].copy()
    if not test[KEYS].equals(reference[KEYS]):
        raise ValueError("Old family baseline evaluation identities changed")
    fixed = 400 if args.family == "xgb" else 120
    reference_col = {
        "xgb": "p_B_raw", "lgb": "p_LightGBM_C1",
        "cat": "p_CatBoost_C1",
    }[args.family]
    original_raw = reference[reference_col].to_numpy(float)
    if selected == fixed:
        reproduced = raw
        reference_secs = 0.0
    else:
        standalone, reference_info = fit_predict_prefixes(
            train, test, args.family, (fixed,),
        )
        reproduced = standalone[fixed]
        reference_secs = reference_info["elapsed_seconds"]
    max_delta = float(np.max(np.abs(reproduced - original_raw)))
    if not np.allclose(reproduced, original_raw, rtol=0, atol=1e-8):
        raise ValueError(
            f"{args.family} {fixed}-round original frozen OOF PARITY FAILED "
            f"({max_delta}); cannot score new candidate"
        )
    if selected == fixed and not np.allclose(
        raw, original_raw, rtol=0, atol=1e-8,
    ):
        raise ValueError("Selected-round model baseline differs from frozen reference")
    market_logit = _market_margin(test)
    scaled = expit(
        market_logit + alpha["alpha"] * (logit(raw) - market_logit),
    )
    if not np.isfinite(scaled).all() or np.any((scaled <= 0) | (scaled >= 1)):
        raise ValueError("Invalid outer calibrated model probabilities")
    output = test[KEYS + [
        "top3_label", "surface", "distance_m", "race_class",
    ]].copy()
    output["p_M0"] = test.p_market_oof.to_numpy(float)
    output["p_raw"] = raw
    output["p_scaled"] = scaled
    output["p_fixed_old_reproduced"] = reproduced
    args.out.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out / f"outer_{year}.csv", index=False)
    table.to_csv(args.out / f"outer_{year}_selection_curve.csv", index=False)
    info = {
        "family": args.family,
        "outer_year": year,
        "inner_years": [2021, 2022] if year == 2023 else [2021, 2022, 2023],
        "selected_rounds": selected,
        "original_fixed_rounds": fixed,
        "original_fixed_parity_max_abs_error": max_delta,
        "original_frozen_prediction_hash": file_sha(
            args.braw_oof if args.family == "xgb" else args.c1_oof,
        ),
        "training": duration,
        "baseline_retrain_seconds": float(reference_secs),
        "alpha": alpha,
        "selected_is_upper_bound": selected == 1600,
        "raw_score": candidate_score(test, raw, name="raw").__dict__,
        "scaled_score": candidate_score(test, scaled, name="scaled").__dict__,
        "output_sha256": file_sha(args.out / f"outer_{year}.csv"),
        "source_manifest_sha256": file_sha(args.out / "source_manifest.json"),
    }
    (args.out / f"outer_{year}_manifest.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps({
        "stage": "outer", "family": args.family, "year": year,
        "selected_rounds": selected, "alpha": alpha["alpha"],
        "baseline_max_diff": max_delta,
        "elapsed_seconds": duration["elapsed_seconds"] + reference_secs,
    }), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--stage", required=True, choices=("prepare", "inner", "outer"))
    parser.add_argument("--year", type=int)
    parser.add_argument("--historical-dir", type=Path)
    parser.add_argument("--results-csv", type=Path)
    parser.add_argument("--races-csv", type=Path)
    parser.add_argument("--reference-oof", type=Path, required=True)
    parser.add_argument("--braw-oof", type=Path, required=True)
    parser.add_argument("--c1-oof", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        stage_prepare(args)
    elif args.stage == "inner":
        stage_inner(args)
    else:
        stage_outer(args)


if __name__ == "__main__":
    main()
