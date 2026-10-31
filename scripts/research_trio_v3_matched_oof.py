"""Strict historical 2023->2024 matched-architecture XGB800+CatBoost400 trio OOF.

Purpose: eliminate transferred 120-tree F3 marginal domain when learning V2/V3.
Fits model EACH YEAR using only <= preceding year. 2025 is NOT used to fit.
No market 3-renpuku historical prices are available in this experiment.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from freeze_tennosho_f3_20261101 import numeric_categorical
from research_trio_f3_v1 import paired_day_ci
from run_information_expansion_v1 import _build_pipeline
from run_input_learning_a_v1 import feature_registry, prepare_dataset
from run_model_families_c1 import CATBOOST_PARAMS, _categorical_input, _race_weight
from scipy.special import logit
from validate_frozen_f3_2025 import frozen_model_probabilities

from keiba_place_lab.ensemble import apply_logit_calibrator, fit_logit_calibrator
from keiba_place_lab.trio_bridge_v1 import (
    invert_subset_marginals,
    plackett_luce_trio,
    project_exactly_three,
)
from keiba_place_lab.trio_context_v3 import (
    CANDIDATES,
    contextual_features,
    fit_contextual,
    joint_probability,
)
from keiba_place_lab.trio_residual_v2 import score_probabilities

YEARS = (2023, 2024)
FROZEN_STOP = pd.Timestamp("2024-12-31")
REQUIRE_MIN_FULL_RACES = 1000
SEED = 20261031
FEATURE_VARIANTS = ("v1", "market_maxent", "market_pl", "v2_matched", "v3_context")


def historical_800_400_fold(df: pd.DataFrame, year: int, expected: dict) -> tuple[pd.DataFrame, dict]:
    """Exact frozen 800/400 architecture, with year-specific training cutoff."""
    train = df.loc[df.race_date.dt.year.between(2018, year-1)].copy()
    test = df.loc[df.race_date.dt.year.eq(year)].copy()
    if not (len(train) > 20000 and len(test) > 15000 and
            train.race_date.max() < test.race_date.min() and
            year <= 2024):
        raise ValueError("Invalid chronology or model source for OOF fold")
    numerics, cats = feature_registry()["F3_market_field"]
    if numerics != expected["feature_numeric"] or cats != expected["feature_categorical"]:
        raise ValueError("Matched model has different F3 features than frozen artifact")
    if not (train.p_market_oof.notna().all() and test.p_market_oof.notna().all()):
        raise ValueError("Training / test market OOF probabilities absent")
    market_train = df.loc[df.race_date.dt.year.between(2016, year-1)]
    market_cal = fit_logit_calibrator(
        market_train.market_p_top3.to_numpy(float),
        market_train.top3_label.to_numpy(int),
    )
    # The fold market is itself a chronological fit to pre-year race results.
    target_cal = apply_logit_calibrator(test.market_p_top3.to_numpy(float), market_cal)
    if np.max(np.abs(target_cal - test.p_market_oof.to_numpy(float))) > 2e-8:
        raise ValueError("OOF market calibration differs from F3 training inputs")
    test["market_calibrated_top3_p"] = target_cal
    test["market_calibrated_top3_logit"] = logit(target_cal)
    tr = train.sort_values(["race_date", "race_id", "horse_id", "horse_no"],
                           kind="stable").reset_index(drop=True)
    x = numeric_categorical(tr, numerics, cats)
    y = tr.top3_label.to_numpy(int)
    offset = logit(tr.p_market_oof.to_numpy(float))
    weights = _race_weight(tr)
    xgb = _build_pipeline(numerics, cats)
    xgb.named_steps["model"].set_params(n_estimators=800)
    xgb.fit(x, y, model__sample_weight=weights, model__base_margin=offset)
    cat = CatBoostClassifier(**{**CATBOOST_PARAMS, "iterations": 400})
    cat.fit(Pool(
        _categorical_input(tr, numerics, cats), label=y,
        cat_features=cats, weight=weights, baseline=offset,
    ))
    model = {
        "xgb": xgb, "catboost": cat, "numeric": numerics,
        "categorical": cats, "market_calibrator": market_cal,
    }
    predictions = frozen_model_probabilities(model, test)
    test["p_matched_f3"] = predictions["p_XC_75_25_logit"]
    test["p_matched_xgb800"] = predictions["p_XGB800"]
    test["p_matched_cat400"] = predictions["p_CatBoost400"]
    return test, {
        "eval_year": year,
        "training_max_date": str(tr.race_date.max().date()),
        "training_races": int(tr.race_id.nunique()),
        "training_rows": len(tr), "evaluation_rows": len(test),
        "xgb_estimators": xgb.named_steps["model"].n_estimators,
        "catboost_iterations": cat.tree_count_,
        "market_calibration_latest": str(market_train.race_date.max().date()),
        "market_calibrator": {
            "intercept": market_cal.intercept,
            "slope": market_cal.slope,
        },
        "2025_outcomes_not_used": True,
    }


def compile_complete_races(oo: pd.DataFrame, historical: pd.DataFrame,
                           ) -> tuple[dict[int, list[dict]], dict]:
    # Source already audited for all 2010–2024 JRA ranks by prepare_dataset.
    all_size = oo.groupby("race_id").size()
    expected = oo.groupby("race_id").field_size.first()
    count_three = oo.groupby("race_id").top3_label.sum()
    complete = set(all_size.index[(all_size.eq(expected)) &
                                  (count_three.eq(3)) &
                                  (all_size.ge(8))])
    source_keys = ["race_id", "horse_id", "horse_no"]
    if historical.duplicated(source_keys).any():
        raise ValueError("Training feature cohort has duplicate source horses")
    source = historical.set_index(source_keys)
    matched = oo.loc[oo.race_id.isin(complete)].copy()
    if not set(matched.set_index(source_keys).index).issubset(set(source.index)):
        raise ValueError("Exact OOF runner source identity mismatch")
    # In addition to exactly three top3 labels, check original raw finish ranks.
    if "raw_finish_position" in matched:
        good = matched.groupby("race_id").apply(
            lambda g: set(g.loc[g.top3_label.eq(1),
                                "raw_finish_position"].dropna().astype(int)) == {1, 2, 3},
            include_groups=False,
        )
        matched = matched.loc[matched.race_id.isin(set(good.index[good]))]
    out = {2023: [], 2024: []}
    for rid, g in matched.groupby("race_id", sort=False):
        g = g.sort_values("horse_no").reset_index(drop=True)
        year = g.race_date.iloc[0].year
        if year not in out:
            raise ValueError("Only 2023 and 2024 can train a chronological trio model")
        horses = g.horse_no.to_numpy(int)
        p = project_exactly_three(g.p_matched_f3.to_numpy(float))
        weights = invert_subset_marginals(p)
        market = project_exactly_three(g.p_market_oof.to_numpy(float))
        market_weights = invert_subset_marginals(market)
        win = g.market_p_win.to_numpy(float)
        if not np.isclose(win.sum(), 1, atol=1e-9):
            raise ValueError("Full-field win probabilities are not normalized")
        pl = plackett_luce_trio(win, horses)
        prepared = contextual_features(
            horse_numbers=horses, logits=weights, projected_top3=p,
            recent_early_pct=g.recent4_early_pos_pct_mean.to_numpy(float),
            draw_pct=g.draw_pct.to_numpy(float),
            market_joint=pl.probability.to_numpy(float),
            market_trios=pl[["horse_1", "horse_2", "horse_3"]].to_numpy(int),
            surface=str(g.surface.iloc[0]),
            distance_m=int(g.distance_m.iloc[0]),
        )
        winner = tuple(sorted(g.loc[g.top3_label.eq(1), "horse_no"].tolist()))
        indices = np.flatnonzero(np.all(prepared["trios"] == winner, axis=1))
        if len(indices) != 1:
            raise ValueError("Winning 3-horse combination missing")
        prepared["actual_trio_idx"] = int(indices[0])
        prepared["race_id"] = rid
        prepared["race_date"] = g.race_date.iloc[0].date().isoformat()
        prepared["surface"] = str(g.surface.iloc[0])
        prepared["distance_m"] = int(g.distance_m.iloc[0])
        prepared["racecourse"] = str(g.racecourse.iloc[0])
        prepared["field_size"] = len(g)
        prepared["projected_market_weights"] = market_weights
        market_pl_map = {
            tuple(sorted((int(row.horse_1), int(row.horse_2), int(row.horse_3)))):
            float(row.probability)
            for row in pl.itertuples(index=False)
        }
        prepared["market_pl_aligned"] = np.asarray(
            [market_pl_map[tuple(v)] for v in prepared["trios"]], dtype=float
        )
        out[year].append(prepared)
    summary = {
        "input_full_field": len(complete),
        "after_raw_rank_audit": int(matched.race_id.nunique()),
        "2023_races": len(out[2023]), "2024_races": len(out[2024]),
        "cohort_is_full_field": True,
        "2025_not_used_for_any_oof_model_fit": True,
    }
    if min(len(out[2023]), len(out[2024])) < REQUIRE_MIN_FULL_RACES:
        raise ValueError("Not enough fully matched 2023/2024 JRA trio labels")
    return out, summary


def evaluate(races: list[dict], trained: dict[str, dict] | None,
             *, predict_2023: bool = False) -> tuple[pd.DataFrame, dict]:
    rows = []
    for race in races:
        lineup = race["trios"]
        truth = race["actual_trio_idx"]
        # Market maximum entropy has identical exactly-three distribution family
        # and comparable historical market recalibration as F3 V1.
        from scipy.special import logsumexp
        w = race["projected_market_weights"]
        horse_numbers = np.sort(np.unique(lineup))
        weights_by_no = dict(zip(horse_numbers.tolist(), w.tolist(), strict=True))
        mlogp = np.asarray(
            [sum(weights_by_no[int(h)] for h in triple) for triple in lineup], dtype=float
        )
        market_subset = np.exp(mlogp - logsumexp(mlogp))
        variants = {
            "market_pl": race["market_pl_aligned"],
            "market_maxent": market_subset,
            "v1": joint_probability(race, np.array([]), ()),
        }
        if trained:
            for name in ("v2_matched", "v3_context"):
                variants[name] = joint_probability(
                    race,
                    np.asarray(trained[name]["coefficients"], dtype=float),
                    CANDIDATES[name],
                )
        result = {
            "race_date": race["race_date"], "race_id": race["race_id"],
            "distance_m": race["distance_m"], "surface": race["surface"],
            "racecourse": race["racecourse"], "field_size": race["field_size"],
            "front_count_prior": race["front_count_prior"],
            "pace_missing_count": race["pace_missing_count"],
        }
        for name, p in variants.items():
            for metric, val in score_probabilities(p, truth).items():
                result[name + "_" + metric] = val
        rows.append(result)
    scores = pd.DataFrame(rows)
    names = ["market_pl", "market_maxent", "v1"]
    if trained:
        names += ["v2_matched", "v3_context"]
    summary = {
        name: {
            "joint_logloss": float(scores[name + "_joint_logloss"].mean()),
            "joint_brier": float(scores[name + "_joint_brier"].mean()),
            "top10_hit": float(scores[name + "_top10_hit"].mean()),
            "top10_probability": float(scores[name + "_top10_mass"].mean()),
        } for name in names
    }
    if predict_2023:
        return scores, summary
    return scores, summary


def main() -> None:
    pa = argparse.ArgumentParser()
    for key in ("historical-dir", "results", "races", "frozen-manifest", "out"):
        pa.add_argument("--" + key, required=True, type=Path)
    a = pa.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError("Research evidence must not overwrite an existing output")
    frozen = json.loads(a.frozen_manifest.read_text())
    frame, audit = prepare_dataset(a.historical_dir, a.results, a.races)
    if not set(frame.race_date.dt.year.unique()).issuperset({2016, 2018, 2023, 2024}):
        raise ValueError("Historical 2016–2024 market and racing history not available")
    folds, manifest = [], []
    for year in YEARS:
        test, record = historical_800_400_fold(frame, year, frozen)
        folds.append(test)
        manifest.append(record)
        print("MATCHED_F3_800_400_YEAR=" + json.dumps(record), flush=True)
    predictions = pd.concat(folds, ignore_index=True)
    all_races, cohort = compile_complete_races(predictions, frame)
    fitted = {}
    for name in ("v2_matched", "v3_context"):
        fitted[name] = fit_contextual(all_races[2023], CANDIDATES[name])
    tr, train_scores = evaluate(all_races[2023], None, predict_2023=True)
    test, summary = evaluate(all_races[2024], fitted)
    # Fair comparator and candidate gates on the EXACT same full 2024 races.
    comparisons = {}
    for reference, candidate in (
        ("market_pl", "market_maxent"),
        ("market_maxent", "v1"),
        ("v1", "v2_matched"),
        ("v1", "v3_context"),
        ("v2_matched", "v3_context"),
    ):
        diff = paired_day_ci(test, reference + "_joint_logloss",
                             candidate + "_joint_logloss", draws=2500)
        diff["joint_brier_baseline_minus_candidate"] = (
            summary[reference]["joint_brier"] -
            summary[candidate]["joint_brier"]
        )
        comparisons[reference + "_minus_" + candidate] = diff
    a.out.mkdir(parents=True, exist_ok=True)
    test.to_csv(a.out / "2024_comparable_complete_trio_scores.csv", index=False)
    tr.to_csv(a.out / "2023_training_year_diagnostics.csv", index=False)
    predictions[["race_date", "race_id", "horse_id", "horse_no", "top3_label",
                 "field_size", "p_market_oof", "p_matched_xgb800",
                 "p_matched_cat400", "p_matched_f3"]].to_csv(
        a.out / "2023_2024_matched_800_400_oof.csv", index=False,
    )
    (a.out / "2023_trio_fitted_coefficients.json").write_text(
        json.dumps({
            "training_year": 2023, "f3_oof": "historical matched 800+400",
            "v2_matched": fitted["v2_matched"],
            "v3_context": fitted["v3_context"],
            "fitted_using_2024_or_2025_labels": False,
            "reference_bundle_sha256": frozen["bundle_sha256"],
        }, ensure_ascii=False, indent=2) + "\n",
    )
    meta = {
        "status": "HISTORICAL_MATCHED_800_400_2023_FIT_2024_OOT_DEVELOPMENT",
        "frozen_F3_bundle_sha256": frozen["bundle_sha256"],
        "folds": manifest,
        "cohort": cohort,
        "2023_base_models": train_scores,
        "2024_scores": summary,
        "2024_paired_day_ci": comparisons,
        "source_sha": audit["source_sha256"],
        "no_frozen_model_change": True,
        "2025_training_labels_used": False,
        "2024_had_prior_model_development_and_is_not_new_untouched_confirmation": True,
        "historical_final_win_odds_not_archived_T10": True,
        "actual_trio_tote_odds_not_in_dataset": True,
        "release_decision": "RESEARCH_SHADOW_ONLY",
        "python": platform.python_version(),
    }
    (a.out / "experiment_manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
    )
    print("TRIO_V3_2024_MATCHED_F3_RESULT=" + json.dumps({
        "folds": manifest, "cohort": cohort,
        "scores": summary, "paired": comparisons,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
