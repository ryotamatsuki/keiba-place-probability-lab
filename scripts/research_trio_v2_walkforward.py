"""Train trio v2 on 2023 previous-year historical F3 OOF, test 2024 ONLY.

All predictions are PRE-race; results are used only as labels in 2023 fit
and 2024 evaluation. Historical F3 120 trees does not match frozen forward F3.
No model promotion or alteration of November pre-off T-10 lock is permitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from research_trio_f3_v1 import (
    paired_day_ci,
    sorted_full_races,
)
from research_trio_f3_v1 import (
    sha256 as original_sha256,
)
from run_information_expansion_v1 import load_full_panel

from keiba_place_lab.historical_market import reconstruct_historical_market
from keiba_place_lab.information_expansion_v1 import raw_observations
from keiba_place_lab.market import normalized_win_probabilities
from keiba_place_lab.trio_bridge_v1 import (
    invert_subset_marginals,
    plackett_luce_trio,
    project_exactly_three,
)
from keiba_place_lab.trio_residual_v2 import (
    CANDIDATE_SETS,
    RIDGE,
    _prepare,
    fit_residual,
    residual_distribution,
    score_probabilities,
)

F3_PROXY = "p_F3_market_field"
MARKET_PROXY = "p_M0"
MODEL_2024_LOGLOSS_EXPECTED = 4.4964765097602095
MAIN_TEST_NAME = "race_interactions"
SOURCE_TARGET_RACE = "202605041111"
LATEST_PREVIEW = "f3_preview_20261031T173406.csv"
FEATURE_FORWARD = "p_XC_75_25_logit"


def _build_history(oof: Path, historic: Path, results: Path, races: Path):
    panel, hashes = load_full_panel(historic)
    panel, audit = raw_observations(panel, results, races)
    if audit["matched_raw_runner_rows"] != audit["total_official_rows"]:
        raise ValueError("Original JRA historical result linkage is incomplete")
    f3 = pd.read_csv(oof, dtype={"race_id":str, "horse_id":str})
    f3["race_date"] = pd.to_datetime(f3.race_date)
    if set(f3.race_date.dt.year.unique()) != {2023, 2024}:
        raise ValueError("Historical F3 OOF years are not exactly 2023 and 2024")
    odds, audit_market = reconstruct_historical_market(results, set(f3.race_id))
    if audit_market["incomplete_market_races"] != 0:
        raise ValueError("Missing historical final-market odds")
    linked, audit_eligibility = sorted_full_races(f3, panel, odds)
    # The following two features are strictly prior horse form and assigned draw.
    # Never use actual race-run pace, finish or post-race passage.
    context = panel[["race_id", "horse_id", "horse_no",
                     "recent4_early_pos_pct_mean", "draw_pct", "racecourse"]]
    linked = linked.merge(context, on=["race_id", "horse_id", "horse_no"],
                          how="left", validate="one_to_one", indicator=True)
    if not linked._merge.eq("both").all():
        raise ValueError("Race past information is incomplete")
    linked.drop(columns="_merge", inplace=True)
    if linked.draw_pct.isna().any():
        raise ValueError("No assigned pre-race draw found")
    return linked, {
        "source_sha256": {**hashes, "jra_results": original_sha256(results),
                          "jra_races": original_sha256(races),
                          "f3_oof": original_sha256(oof)},
        "history": audit,
        "market": audit_market,
        "eligibility": audit_eligibility,
    }


def _build_races(frame: pd.DataFrame) -> dict[int, list[dict]]:
    result = {2023: [], 2024: []}
    for rid, g in frame.groupby("race_id", sort=False):
        g = g.sort_values("horse_no", kind="stable").reset_index(drop=True)
        year = pd.Timestamp(g.race_date.iloc[0]).year
        if year not in result:
            raise ValueError("Unrecognized race date")
        horses = g.horse_no.to_numpy(int)
        baseline = project_exactly_three(g[F3_PROXY].to_numpy(float))
        model_weights = invert_subset_marginals(baseline)
        win = g.market_p_win.to_numpy(float)
        if not np.isclose(win.sum(), 1, atol=1e-10):
            raise ValueError("Actual full field market does not sum to one")
        market = plackett_luce_trio(win, horses)
        prepared = _prepare(
            horses, model_weights, baseline,
            g.recent4_early_pos_pct_mean.to_numpy(float),
            g.draw_pct.to_numpy(float),
            market.probability.to_numpy(float),
            market[["horse_1", "horse_2", "horse_3"]].to_numpy(int),
        )
        outcome = tuple(sorted(horses[g.top3_label.to_numpy(int).astype(bool)].tolist()))
        if len(outcome) != 3:
            raise ValueError("Cannot evaluate tied or nonstandard trio results")
        selected = np.flatnonzero(np.all(prepared["trios"] == outcome, axis=1))
        if len(selected) != 1:
            raise ValueError("Winning trio not present in model")
        prepared.update({
            "actual_trio_idx": int(selected[0]),
            "race_date": pd.Timestamp(g.race_date.iloc[0]).strftime("%Y-%m-%d"),
            "race_id": rid,
            "field_size": len(g),
            "distance_m": int(g.distance_m.iloc[0]),
            "surface": g.surface.iloc[0],
            "racecourse": g.racecourse.iloc[0],
        })
        result[year].append(prepared)
    return result


def _benchmark(races: list[dict], models: dict[str, np.ndarray],
               full_model_2024: bool) -> tuple[pd.DataFrame, dict]:
    records = []
    for race in races:
        row = {
            "race_date": race["race_date"], "race_id": race["race_id"],
            "field_size": race["field_size"],
            "distance_m": race["distance_m"],
            "surface": race["surface"], "racecourse": race["racecourse"],
            "pace_missing_count": race["pace_missing_count"],
        }
        for name, beta in models.items():
            p = residual_distribution(race, beta, CANDIDATE_SETS[name])
            for key, val in score_probabilities(p, race["actual_trio_idx"]).items():
                row[f"{name}_{key}"] = val
        records.append(row)
    scored = pd.DataFrame(records)
    summaries = {}
    for name in models:
        summaries[name] = {
            "joint_logloss": float(scored[f"{name}_joint_logloss"].mean()),
            "joint_brier": float(scored[f"{name}_joint_brier"].mean()),
            "top1_accuracy": float(scored[f"{name}_top1_hit"].mean()),
            "top5_coverage": float(scored[f"{name}_top5_hit"].mean()),
            "top10_observed_coverage": float(scored[f"{name}_top10_hit"].mean()),
            "top10_predicted_mass": float(scored[f"{name}_top10_mass"].mean()),
            "top10_coverage_gap": float(
                scored[f"{name}_top10_hit"].mean() -
                scored[f"{name}_top10_mass"].mean()
            ),
        }
    if full_model_2024:
        observed = summaries["f3_raw"]["joint_logloss"]
        if abs(observed - MODEL_2024_LOGLOSS_EXPECTED) > 2e-8:
            raise ValueError(
                f"Historical baseline data/code drift: expected "
                f"{MODEL_2024_LOGLOSS_EXPECTED}, found {observed}"
            )
        if len(races) != 1336:
            raise ValueError("Historical 2024 sample size changed")
    return scored, summaries


def _independent_evaluation(
    trained: list[dict], test: list[dict], out: Path, source_audit: dict,
) -> dict:
    models = {"f3_raw": np.array([], float)}
    fits = {}
    for name in ("pair_strength", "race_interactions", "market_hybrid"):
        fit = fit_residual(trained, CANDIDATE_SETS[name], ridge=RIDGE)
        models[name] = np.asarray(fit["beta"], float)
        fits[name] = fit
    scored, summaries = _benchmark(test, models, full_model_2024=True)
    deltas = {}
    for name in fits:
        key = "f3_raw_minus_" + name
        deltas[key] = paired_day_ci(
            scored, "f3_raw_joint_logloss",
            name + "_joint_logloss", draws=2500,
        )
        brier_delta = float(
            scored["f3_raw_joint_brier"].mean() -
            scored[name + "_joint_brier"].mean()
        )
        deltas[key]["joint_brier_baseline_minus_candidate"] = brier_delta
    primary = deltas["f3_raw_minus_" + MAIN_TEST_NAME]
    statistically_better = (
        primary["ci95"][0] > 0 and
        primary["joint_brier_baseline_minus_candidate"] >= 0
    )
    subgroup = []
    for surface in ("turf", "dirt"):
        for distance_name, lower, upper in (
            ("short_le1600", 0, 1600),
            ("middle_1601_2399", 1601, 2399),
            ("long_ge2400", 2400, 10000),
        ):
            g = scored.loc[
                scored.surface.eq(surface) &
                scored.distance_m.between(lower, upper)
            ]
            if g.empty:
                continue
            subgroup.append({
                "surface": surface, "band": distance_name,
                "races": len(g),
                "f3_raw_joint_logloss": float(g.f3_raw_joint_logloss.mean()),
                "main_candidate_joint_logloss":
                    float(g[f"{MAIN_TEST_NAME}_joint_logloss"].mean()),
                "improvement": float(
                    g.f3_raw_joint_logloss.mean() -
                    g[f"{MAIN_TEST_NAME}_joint_logloss"].mean()
                ),
            })
    out.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out / "2024_racewise_scores.csv", index=False)
    pd.DataFrame(subgroup).to_csv(out / "2024_subgroup_scores.csv", index=False)
    fitted = {
        "model": "trio_v2_2023_historical_f3_120tree_PROXY",
        "fit_only_year": 2023,
        "train_races": len(trained),
        "model_candidates": fits,
        "main_predeclared_candidate": MAIN_TEST_NAME,
        "forward_model_compatibility": False,
        "no_promote": True,
    }
    (out / "2023_fitted_candidate_coefficients.json").write_text(
        json.dumps(fitted, indent=2, ensure_ascii=False) + "\n")
    report = {
        "status": "RESEARCH_2023_TRAIN_2024_TEST_NO_LIVE_ADOPTION",
        "2023_training_races": len(trained),
        "2024_untouched_test_races": len(test),
        "f3_120tree_historical_not_2026_frozen_xgb800_cat400": True,
        "full_field_verified": True,
        "identical_baseline_2024_logloss": MODEL_2024_LOGLOSS_EXPECTED,
        "candidate_summaries_2024": summaries,
        "comparison_2024": deltas,
        "primary_gate_pass_statistically": statistically_better,
        "promotion_allowed": False,
        "reason_no_promotion": (
            "2024 validation is on historical 120-tree F3 proxy; "
            "Nov-1 frozen 800/400 ensemble and T-minus-10 odds are different domain"
        ),
        "racing_day_cluster_bootstrap": 2500,
        "original_source_integrity": source_audit,
        "limits": [
            "As-of race start snapshots historical FINAL win odds, not actual archived T-10 odds.",
            "No observed live race 3-renpuku pools or historic joint tote prices; ROI cannot be inferred.",
            "Race selection restricted to complete starter OOF and unique top three only.",
            "No 2025 independent locked-ensemble evaluation was performed.",
            "Parameters selected only using the 2023 training outcomes; no 2024 retraining.",
            "Multiple comparison exploratory: only race_interactions designated primary gate.",
        ],
    }
    (out / "model_evaluation.json").write_text(json.dumps(
        report, indent=2, ensure_ascii=False) + "\n")
    print("TRIO_V2_VALIDATION=" + json.dumps({
        "2023_train": len(trained), "2024_test": len(test),
        "main_gate": statistically_better,
        "scores": summaries, "paired_primary": primary, "fits": fits,
    }, ensure_ascii=False), flush=True)
    return {"report": report, "fits": fits}


def _forward_sensitivity(
    preview: Path, frozen_features: Path, frozen_manifest: Path,
    validation: dict, out: Path,
) -> None:
    # The OOF 120-tree training domain is incompatible with the frozen 2026
    # 800/400 ensemble; output only SHADOW sensitivity, NOT trading decision.
    d = pd.read_csv(preview, dtype={"race_id":str, "horse_id":str})
    prior = pd.read_csv(frozen_features, dtype={"race_id":str, "horse_id":str})
    keys = ["race_id", "horse_id", "horse_no"]
    d = d.merge(
        prior[keys + ["recent4_early_pos_pct_mean", "draw_pct"]],
        on=keys, how="inner", validate="one_to_one"
    ).sort_values("horse_no")
    if (len(d) != 17 or d.horse_no.nunique() != 17
            or set(d.race_id) != {SOURCE_TARGET_RACE}
            or set(d.horse_no) != set(range(1, 18))):
        raise ValueError("2026 forward projection has missing or substituted entrant")
    probabilities = project_exactly_three(d[FEATURE_FORWARD].to_numpy(float))
    theta = invert_subset_marginals(probabilities)
    market, _ = normalized_win_probabilities(d.win_odds.to_numpy(float))
    pl = plackett_luce_trio(market, d.horse_no.to_numpy(int))
    item = _prepare(
        d.horse_no.to_numpy(int), theta, probabilities,
        d.recent4_early_pos_pct_mean.to_numpy(float), d.draw_pct.to_numpy(float),
        pl.probability.to_numpy(float),
        pl[["horse_1", "horse_2", "horse_3"]].to_numpy(int),
    )
    selected = CANDIDATE_SETS[MAIN_TEST_NAME]
    beta = np.array(validation["fits"][MAIN_TEST_NAME]["beta"], float)
    v1 = residual_distribution(item, np.array([]), CANDIDATE_SETS["f3_raw"])
    shadow = residual_distribution(item, beta, selected)
    prediction = pd.DataFrame(item["trios"], columns=["horse_1","horse_2","horse_3"])
    prediction["p_f3_raw"] = v1
    prediction["p_trio_v2_proxy_fit_shadow"] = shadow
    prediction["absolute_sensitivity"] = np.abs(v1 - shadow)
    prediction = prediction.sort_values("p_f3_raw", ascending=False)
    prediction.to_csv(out / "2026_17horse_shadow_sensitivity.csv", index=False)
    manifest = json.loads(frozen_manifest.read_text())
    provenance = {
        "status": "SHADOW_ONLY_NEVER_BETTING_PRODUCTION",
        "model_fit_year": 2023,
        "historical_proxy": "120tree",
        "forward_f3": "XGB800+CAT400",
        "incompatible_validation_domain": True,
        "frozen_bundle_hash": manifest["bundle_sha256"],
        "input_f3_preview_hash": original_sha256(preview),
        "input_prior_features_hash": original_sha256(frozen_features),
        "f3_model_and_official_preoff_lock_unchanged": True,
        "number_trios": len(prediction),
        "raw_sum": float(prediction.p_f3_raw.sum()),
        "shadow_sum": float(prediction.p_trio_v2_proxy_fit_shadow.sum()),
        "max_probability_change": float(prediction.absolute_sensitivity.max()),
        "not_a_recommended_bet": True,
    }
    (out / "2026_shadow_manifest.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--historical", required=True, type=Path)
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--races", required=True, type=Path)
    p.add_argument("--f3-oof", required=True, type=Path)
    p.add_argument("--preview", required=True, type=Path)
    p.add_argument("--prior", required=True, type=Path)
    p.add_argument("--frozen-manifest", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("New research output cannot overwrite prior evidence")
    data, audit = _build_history(args.f3_oof, args.historical, args.results, args.races)
    split = _build_races(data)
    if len(split[2023]) != 1355 or len(split[2024]) != 1336:
        raise ValueError("Historical original F3 full field eligibility drift")
    results = _independent_evaluation(split[2023], split[2024], args.out, audit)
    _forward_sensitivity(args.preview, args.prior, args.frozen_manifest,
                         results, args.out)


if __name__ == "__main__":
    main()
