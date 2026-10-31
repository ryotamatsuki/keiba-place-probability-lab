"""Frozen-F3-safe retrospective trio validation and Nov 1 preview.

Historical 2023/24 F3 120-tree OOF is a PROXY, not the November frozen
XGB800+CatBoost400. Historical trio temperature must never be presented
as proven calibration for the latter or auto-promoted to production.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from keiba_place_lab.historical_market import reconstruct_historical_market
from keiba_place_lab.market import normalized_win_probabilities
from keiba_place_lab.information_expansion_v1 import raw_observations
from keiba_place_lab.trio_bridge_v1 import (
    historical_trio_temperature,
    invert_subset_marginals,
    plackett_luce_trio,
    project_exactly_three,
    trio_distribution,
)
from run_information_expansion_v1 import load_full_panel

F3_PROXY = "p_F3_market_field"
FROZEN_P = "p_XC_75_25_logit"
RACE = "202605041111"
SEED = 20261031


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sorted_full_races(oof: pd.DataFrame, panel: pd.DataFrame,
                      market: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    keys = ["race_id", "horse_id", "horse_no"]
    for frame in (oof, panel, market):
        frame["race_id"] = frame["race_id"].astype(str)
        if "horse_id" in frame:
            frame["horse_id"] = frame["horse_id"].astype(str)
        frame["horse_no"] = pd.to_numeric(frame["horse_no"], errors="raise").astype(int)
    if oof.duplicated(keys).any() or panel.duplicated(keys).any():
        raise ValueError("Duplicate historical race starters")
    if not {F3_PROXY, "p_M0", "top3_label", "race_date"}.issubset(oof):
        raise ValueError("Wrong historical OOF format: F3 and M0 required")
    cols = keys + ["field_size", "raw_finish_position"]
    linked = oof.merge(panel[cols], on=keys, how="left",
                       validate="one_to_one", indicator=True)
    if not linked._merge.eq("both").all():
        raise ValueError("Historical F3 OOF and raw results starter identities conflict")
    linked.drop(columns="_merge", inplace=True)
    grouped = linked.groupby("race_id", sort=False)
    n = grouped.size()
    expected = grouped.field_size.first()
    # Restrict to EXACT full starter races; no predictions fabricated for excluded
    # horses. Dead heats and nonstandard top-3 outcomes are out of scope.
    complete = n.index[(n.eq(expected)) & (n.ge(4))]
    linked = linked.loc[linked.race_id.isin(complete)].copy()
    valid = []
    for rid, g in linked.groupby("race_id", sort=False):
        finish = g.loc[g.top3_label.eq(1), "raw_finish_position"]
        if len(finish) == 3 and set(finish.astype(int).tolist()) == {1, 2, 3}:
            valid.append(rid)
    linked = linked.loc[linked.race_id.isin(valid)]
    linked = linked.merge(
        market[["race_id", "horse_no", "market_p_win", "market_field_size"]],
        on=["race_id", "horse_no"], how="left", validate="one_to_one",
    )
    sizes = linked.groupby("race_id").size()
    msize = linked.groupby("race_id").market_field_size.first()
    stable = sizes.index[(sizes.eq(msize)) & linked.groupby("race_id").market_p_win.apply(
        lambda s: bool(s.notna().all())
    )]
    linked = linked.loc[linked.race_id.isin(stable)].copy()
    linked = linked.sort_values(["race_date", "race_id", "horse_no"],
                                kind="stable").reset_index(drop=True)
    count_all = oof.race_id.nunique()
    return linked, {
        "oof_races": int(count_all),
        "full_starter_races_with_exactly_three_ranks": len(valid),
        "complete_market_final_eligible_races": int(linked.race_id.nunique()),
        "excluded_races_due_to_missing_starters_ties_or_market": int(
            count_all - linked.race_id.nunique()
        ),
    }


def compile_races(df: pd.DataFrame) -> dict[int, list[dict]]:
    years: dict[int, list[dict]] = {2023: [], 2024: []}
    for rid, g in df.groupby("race_id", sort=False):
        g = g.sort_values("horse_no")
        year = pd.Timestamp(g.race_date.iloc[0]).year
        if year not in years:
            continue
        f3 = project_exactly_three(g[F3_PROXY].to_numpy(float))
        m0 = project_exactly_three(g.p_M0.to_numpy(float))
        f3w = invert_subset_marginals(f3)
        m0w = invert_subset_marginals(m0)
        shares = g.market_p_win.to_numpy(float)
        if not np.isclose(shares.sum(), 1.0, atol=1e-10):
            raise ValueError("Wrong historical full-starter win market")
        actual = tuple(np.flatnonzero(g.top3_label.to_numpy(int)).tolist())
        if len(actual) != 3:
            raise ValueError("Incorrect historical trio outcome")
        years[year].append({
            "date": pd.Timestamp(g.race_date.iloc[0]),
            "race": rid,
            "horses": g.horse_no.to_numpy(int),
            "actual_idx": actual,
            "f3_weights": f3w, "m0_weights": m0w,
            "market_win": shares,
            "max_f3_marginal_shift": float(np.max(np.abs(
                f3 - g[F3_PROXY].to_numpy(float)
            ))),
        })
    return years


def outcome_probability(dist: pd.DataFrame, picks: tuple[int, int, int]) -> float:
    three = set(picks)
    rows = dist.loc[dist[["horse_1", "horse_2", "horse_3"]].apply(
        lambda r: set(r.tolist()) == three, axis=1
    ), "probability"]
    if len(rows) != 1:
        raise ValueError("Outcome trio absent from distribution")
    return float(rows.iloc[0])


def evaluate_race(item: dict, temperature: float) -> dict:
    horses = item["horses"]
    true = tuple(horses[list(item["actual_idx"])].tolist())
    selections = {
        "f3_unadjusted": trio_distribution(item["f3_weights"], horses),
        "f3_2023_temp": trio_distribution(item["f3_weights"], horses,
                                         temperature=temperature),
        "market_subset": trio_distribution(item["m0_weights"], horses),
        "market_pl": plackett_luce_trio(item["market_win"], horses),
    }
    out = {"race_date": item["date"].date().isoformat(),
           "race_id": item["race"],
           "field_size": len(horses),
           "actual_trio": "-".join(map(str, sorted(true)))}
    for name, d in selections.items():
        p = outcome_probability(d, true)
        out[name + "_logloss"] = float(-np.log(max(p, 1e-300)))
        out[name + "_brier"] = float((d.probability**2).sum() + 1 - 2 * p)
    return out


def paired_day_ci(df: pd.DataFrame, earlier: str, later: str,
                  *, draws: int = 2000) -> dict:
    x = df[["race_date", earlier, later]].copy()
    x["gain"] = x[earlier] - x[later]
    day = x.groupby("race_date", sort=True).gain.agg(["sum", "size"])
    n = len(day)
    rng = np.random.default_rng(SEED + len(earlier) * 17 + len(later))
    sample = rng.integers(0, n, size=(draws, n))
    sums = day["sum"].to_numpy()[sample].sum(axis=1)
    sizes = day["size"].to_numpy()[sample].sum(axis=1)
    estimates = sums / sizes
    return {
        "difference_baseline_minus_candidate": float(x.gain.mean()),
        "cluster_dates": n, "bootstrap_resamples": draws,
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "positive_improvement_bootstrap_share": float(np.mean(estimates > 0)),
    }


def retrospective(oof_csv: Path, historical_dir: Path, results_csv: Path,
                  races_csv: Path, out: Path) -> tuple[dict, dict]:
    full, base_sources = load_full_panel(historical_dir)
    full, raw_audit = raw_observations(full, results_csv, races_csv)
    if raw_audit["unmatched_raw_runner_rows"] != 0:
        raise ValueError("Historical race finish positions not complete")
    oof = pd.read_csv(oof_csv, dtype={"race_id": str, "horse_id": str})
    oof["race_date"] = pd.to_datetime(oof["race_date"])
    if not set(oof.race_date.dt.year.unique()).issubset({2023, 2024}):
        raise ValueError("Historical F3 OOF evaluation contains forbidden years")
    markets, market_report = reconstruct_historical_market(
        results_csv, set(oof.race_id.astype(str))
    )
    linked, subset_report = sorted_full_races(oof, full, markets)
    races = compile_races(linked)
    if len(races[2023]) < 40 or len(races[2024]) < 40:
        raise ValueError(
            f"Not enough complete historical F3 race predictions: "
            f"2023={len(races[2023])}, 2024={len(races[2024])}"
        )
    calibration = historical_trio_temperature(
        [q["f3_weights"] for q in races[2023]],
        [q["actual_idx"] for q in races[2023]],
    )
    predictions = [evaluate_race(z, calibration["temperature"]) for z in races[2024]]
    scored = pd.DataFrame(predictions)
    out.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out / "2024_complete_race_trio_scores.csv", index=False)
    methods = ["f3_unadjusted", "f3_2023_temp", "market_subset", "market_pl"]
    results = {name: {
        "mean_joint_logloss": float(scored[name + "_logloss"].mean()),
        "mean_joint_brier": float(scored[name + "_brier"].mean()),
    } for name in methods}
    paired = {}
    for baseline, candidate in (
        ("market_pl", "f3_unadjusted"),
        ("market_pl", "f3_2023_temp"),
        ("f3_unadjusted", "f3_2023_temp"),
        ("market_subset", "f3_2023_temp"),
    ):
        key = baseline + "_minus_" + candidate
        paired[key] = paired_day_ci(scored, baseline + "_logloss",
                                     candidate + "_logloss")
    meta = {
        "status": "HISTORICAL_F3_120_TREE_PROXY_VALIDATION_NOT_FROZEN_FORWARD",
        "historical_2023_2024_proxy": F3_PROXY,
        "frozen_forward_model_source": "XGB800 + CatBoost400; different from 120-tree OOF",
        "calibration_only_2023": calibration,
        "validation_only_2024": {"races": len(races[2024]),
                                 "all_horses": int(scored.field_size.sum())},
        "complete_race_filter": subset_report, "market_coverage": market_report,
        "scores": results, "paired_day_bootstraps": paired,
        "historical_sources": {**base_sources,
                               "oof_sha256": sha256(oof_csv),
                               "kaggle_results_sha256": sha256(results_csv)},
        "no_promotion": True,
        "limitations": [
            "The historical OOF F3 120-tree proxy is not frozen Nov-1 XGB800+CatBoost400.",
            "Only historical races with every starter's historical F3 OOF and exactly ranks 1,2,3 are scored.",
            "Historical market odds are final odds; forward predictions use provisional pre-off odds.",
            "Market final odds and historical model predictions may have different calibration domains.",
            "Selection of complete races changes the population vs the full 2024 JRA card.",
            "Trio payouts and JRA actual trio odds were not observed: no expected betting returns claimed.",
        ],
    }
    (out / "retrospective_audit.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n"
    )
    print("RETROSPECTIVE_RESULT=" + json.dumps({
        "2023_races": len(races[2023]), "2024_races": len(races[2024]),
        "temperature": calibration["temperature"], "scores": results,
        "paired_bootstrap": paired,
    }, ensure_ascii=False), flush=True)
    return calibration, meta


def forecast(preview_csv: Path, calibration: dict, out: Path,
             frozen_bundle_manifest: Path) -> dict:
    p = pd.read_csv(preview_csv, dtype={"race_id": str, "horse_id": str})
    p = p.sort_values("horse_no", kind="stable")
    numbers = p.horse_no.to_numpy(int)
    if (len(p) != 17 or set(numbers) != set(range(1, 18))
            or set(p.race_id) != {RACE} or p.horse_id.nunique() != 17):
        raise ValueError("Trio forward race does not contain every confirmed entrant")
    if not {FROZEN_P, "p_M0_live_raw", "win_odds"}.issubset(p):
        raise ValueError("F3 forward trio preview needs actual F3 probability and win odds")
    marginal = p[FROZEN_P].to_numpy(float)
    projected = project_exactly_three(marginal)
    weights = invert_subset_marginals(projected)
    reconstructed = __import__(
        "keiba_place_lab.three_subset_v1", fromlist=["top3_subset_marginals"]
    ).top3_subset_marginals(weights)
    residual = float(np.max(abs(projected - reconstructed)))
    if residual > 3e-5:
        raise ValueError("Joint probabilities do not reproduce projected F3 marginals")
    win, overround = normalized_win_probabilities(p.win_odds.to_numpy(float))
    raw = trio_distribution(weights, numbers)
    proxy_temp = trio_distribution(weights, numbers,
                                   temperature=calibration["temperature"])
    market_pl = plackett_luce_trio(win, numbers)
    key = ["horse_1", "horse_2", "horse_3"]
    combined = raw.rename(columns={"probability": "p_f3_joint_raw"})
    combined = combined.merge(
        proxy_temp.rename(columns={"probability": "p_f3_proxy_temperature"}),
        on=key, validate="one_to_one",
    ).merge(
        market_pl.rename(columns={"probability": "p_market_pl"}),
        on=key, validate="one_to_one",
    )
    combined["break_even_odds_raw"] = 1 / combined.p_f3_joint_raw
    combined = combined.sort_values("p_f3_joint_raw", ascending=False,
                                    kind="stable").reset_index(drop=True)
    combined.to_csv(out / "tennosho_20261101_trio_680_preview.csv",
                    index=False, float_format="%.12g")
    if not (len(combined) == 680 and np.isclose(combined.p_f3_joint_raw.sum(), 1)
            and np.isclose(combined.p_f3_proxy_temperature.sum(), 1)
            and np.isclose(combined.p_market_pl.sum(), 1)):
        raise ValueError("Trio distribution is incomplete or not normalized")
    bundle_meta = json.loads(frozen_bundle_manifest.read_text())
    manifest = {
        "status": "EXPLORATORY_20261031_F3_TRIO_PREVIEW_NO_T10_LOCK",
        "race_id": RACE,
        "input_snapshot_preview": preview_csv.name,
        "input_sha256": sha256(preview_csv),
        "frozen_F3_bundle_sha256": bundle_meta.get("bundle_sha256"),
        "frozen_F3_training_cutoff": bundle_meta.get("training_cutoff"),
        "original_F3_sum_top3": float(marginal.sum()),
        "projected_F3_sum_top3": float(projected.sum()),
        "max_projection_shift": float(max(abs(projected - marginal))),
        "max_inverse_marginal_error": residual,
        "trio_combinations": 680,
        "raw_joint_sum": float(combined.p_f3_joint_raw.sum()),
        "market_pl_joint_sum": float(combined.p_market_pl.sum()),
        "temperature_source": "2023 120-tree F3 proxy OOF; INCOMPATIBLE frozen F3",
        "temperature_proxy_2023": float(calibration["temperature"]),
        "calibrated_candidate_not_validated_for_frozen_2026": True,
        "actual_trio_odds_unavailable": True,
        "no_training_or_promotion": True,
        "top10_raw": combined.head(10)[
            key + ["p_f3_joint_raw", "p_f3_proxy_temperature", "p_market_pl"]
        ].to_dict(orient="records"),
        "output_sha256": sha256(out / "tennosho_20261101_trio_680_preview.csv"),
    }
    (out / "tennosho_trio_preview_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print("TENN0SHO_TRIO_PREVIEW=" + json.dumps(manifest, ensure_ascii=False), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-csv", required=True, type=Path)
    parser.add_argument("--historical-dir", required=True, type=Path)
    parser.add_argument("--results-csv", required=True, type=Path)
    parser.add_argument("--races-csv", required=True, type=Path)
    parser.add_argument("--preview-csv", required=True, type=Path)
    parser.add_argument("--bundle-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir
    # Research-only outputs cannot overwrite existing immutable F3 snapshots.
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Trio research output directory must be new")
    calibration, report = retrospective(
        args.oof_csv, args.historical_dir, args.results_csv, args.races_csv, out
    )
    forecast(args.preview_csv, calibration, out, args.bundle_manifest)


if __name__ == "__main__":
    main()
