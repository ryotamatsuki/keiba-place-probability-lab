"""Run Stage 5 v2 with market-only incumbent and out-of-time Stage-4 successor forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from keiba_place_lab.ensemble import (
    apply_logit_calibrator,
    convex_blend,
    fit_logit_calibrator,
)
from keiba_place_lab.ensemble_v2 import blend_registry, select_blend_one_se
from keiba_place_lab.historical_market import reconstruct_historical_market
from keiba_place_lab.model_selection import (
    candidate_score,
    select_with_incumbent_one_se,
)
from keiba_place_lab.nonmarket import (
    calibration_table,
    expected_calibration_error,
    validate_market_free,
)
from keiba_place_lab.stage4_successor import calibration_intercept_slope
from keiba_place_lab.stage4_successor_phase3a import (
    fit_relative_candidate,
    predict_relative_candidate,
)

EXPECTED_RESULTS_SHA256 = "fb9345273b21a7c23d41260dc77e45134a1bc2fe756899f3416b24ac0dd51de3"
PANEL_SHA256 = "cee9ae9a099f521f12b1bcdd371c25a3d7b9ba3555fbcf5a46f2c9098f59a33d"
TARGET_MARKET_AS_OF = "2026-10-03T08:35:00+09:00"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_panel(base: Path, name: str) -> pd.DataFrame:
    path = base / name
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    validate_market_free(frame.columns)
    frame = frame.copy()
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="raise")
    frame["race_id"] = frame["race_id"].astype("string")
    if "horse_id" in frame.columns:
        frame["horse_id"] = frame["horse_id"].astype("string")
    return frame


def load_phase3a_oof(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "race_date",
        "race_id",
        "horse_id",
        "horse_no",
        "top3_label",
        "outer_year",
        "p_challenger",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Phase-3A OOF missing columns: {sorted(missing)}")
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="raise")
    frame["race_id"] = frame["race_id"].astype("string")
    frame["horse_id"] = frame["horse_id"].astype("string")
    if set(frame["outer_year"].astype(int).unique()) != {2023, 2024}:
        raise ValueError("Phase-3A OOF must contain exactly 2023 and 2024")
    frame = frame.rename(columns={"p_challenger": "p_nonmarket_raw"})
    return frame


def make_2025_oof(historical_dir: Path) -> pd.DataFrame:
    train = load_panel(historical_dir, "phase_a_turf_1200_train_v1.parquet")
    validation = load_panel(
        historical_dir,
        "phase_a_turf_1200_validation_v1.parquet",
    )
    test = load_panel(historical_dir, "phase_a_turf_1200_test_v1.parquet")

    full_train = load_panel(historical_dir, "jra_flat_train_v1.parquet")
    full_validation = load_panel(historical_dir, "jra_flat_validation_v1.parquet")
    full_test = load_panel(historical_dir, "jra_flat_test_v1.parquet")

    fit_rows = pd.concat([train, validation], ignore_index=True, sort=False)
    if fit_rows["race_date"].dt.year.max() != 2024:
        raise ValueError("2025 OOF fit must stop at 2024")
    if not test["race_date"].dt.year.eq(2025).all():
        raise ValueError("2025 OOF evaluation must contain only 2025")

    fit_races = set(fit_rows["race_id"])
    fit_context = pd.concat(
        [full_train, full_validation],
        ignore_index=True,
        sort=False,
    )
    fit_context = fit_context.loc[
        fit_context["race_id"].isin(fit_races)
    ].copy()
    eval_races = set(test["race_id"])
    eval_context = full_test.loc[
        full_test["race_id"].isin(eval_races)
    ].copy()

    model, prior_mean, _ = fit_relative_candidate(
        fit_rows,
        fit_context,
    )
    probability, _ = predict_relative_candidate(
        model,
        test,
        eval_context,
        prior_mean=prior_mean,
    )
    out = test[
        [
            "race_date",
            "race_id",
            "horse_id",
            "horse_no",
            "horse_name",
            "top3_label",
        ]
    ].copy()
    out["outer_year"] = 2025
    out["p_nonmarket_raw"] = probability
    return out


def pair_market(
    oof: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    paired = oof.merge(
        market,
        on=["race_id", "horse_no"],
        how="inner",
        validate="many_to_one",
    )
    if paired.empty:
        raise ValueError("No paired market rows")
    if paired.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Paired historical rows are not unique")
    return paired


def row_fingerprint(frame: pd.DataFrame) -> str:
    columns = ["race_date", "race_id", "horse_id", "horse_no"]
    keys = frame[columns].copy()
    keys["race_date"] = pd.to_datetime(keys["race_date"]).dt.strftime("%Y-%m-%d")
    for col in ["race_id", "horse_id"]:
        keys[col] = keys[col].astype("string")
    keys = keys.sort_values(columns, kind="stable").reset_index(drop=True)
    return hashlib.sha256(
        keys.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def diagnostic_row(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    name: str,
) -> dict[str, object]:
    score = candidate_score(frame, probability, name=name)
    intercept, slope = calibration_intercept_slope(
        frame["top3_label"].astype(int).to_numpy(),
        probability,
    )
    rel = calibration_table(frame, probability, bins=10)
    return {
        **score.__dict__,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece_10bin": expected_calibration_error(rel),
    }


def reliability_rows(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    name: str,
    period: str,
) -> pd.DataFrame:
    rel = calibration_table(frame, probability, bins=10)
    rel.insert(0, "model", name)
    rel.insert(0, "period", period)
    return rel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--phase3a-oof", type=Path, required=True)
    parser.add_argument("--target-market", type=Path, required=True)
    parser.add_argument("--target-nonmarket", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_sha = sha256_file(args.results_csv)
    if results_sha != EXPECTED_RESULTS_SHA256:
        raise ValueError(
            f"Unexpected historical market SHA256: {results_sha}"
        )

    oof_2023_2024 = load_phase3a_oof(args.phase3a_oof)
    oof_2025 = make_2025_oof(args.historical_dir)
    all_oof = pd.concat(
        [oof_2023_2024, oof_2025],
        ignore_index=True,
        sort=False,
    )
    if set(all_oof["outer_year"].astype(int).unique()) != {2023, 2024, 2025}:
        raise ValueError("Historical OOF years are incomplete")

    market, market_diagnostics = reconstruct_historical_market(
        args.results_csv,
        set(all_oof["race_id"]),
    )
    paired = pair_market(all_oof, market)
    paired_2023 = paired.loc[paired["outer_year"].eq(2023)].copy()
    paired_2024 = paired.loc[paired["outer_year"].eq(2024)].copy()
    paired_2025 = paired.loc[paired["outer_year"].eq(2025)].copy()
    if paired_2023.empty or paired_2024.empty or paired_2025.empty:
        raise ValueError("Need paired 2023, 2024, and 2025 rows")

    market_cal_2023 = fit_logit_calibrator(
        paired_2023["market_p_top3"].to_numpy(dtype=float),
        paired_2023["top3_label"].to_numpy(dtype=int),
    )
    nonmarket_cal_2023 = fit_logit_calibrator(
        paired_2023["p_nonmarket_raw"].to_numpy(dtype=float),
        paired_2023["top3_label"].to_numpy(dtype=int),
    )

    market_2024_raw = paired_2024["market_p_top3"].to_numpy(dtype=float)
    market_2024_cal = apply_logit_calibrator(
        market_2024_raw,
        market_cal_2023,
    )
    market_gate_decision, market_gate_table = select_with_incumbent_one_se(
        paired_2024,
        {
            "market_raw": market_2024_raw,
            "market_calibrated": market_2024_cal,
        },
        incumbent="market_raw",
    )
    market_base_name = market_gate_decision.winner
    market_2024_base = (
        market_2024_cal
        if market_base_name == "market_calibrated"
        else market_2024_raw
    )

    market_gate_diag = []
    reliability_parts = []
    for name, probability in {
        "market_raw": market_2024_raw,
        "market_calibrated": market_2024_cal,
    }.items():
        market_gate_diag.append(
            {
                "stage": "market_gate_2024",
                **diagnostic_row(paired_2024, probability, name=name),
            }
        )
        reliability_parts.append(
            reliability_rows(
                paired_2024,
                probability,
                name=name,
                period="2024_market_gate",
            )
        )
    market_gate_table = market_gate_table.merge(
        pd.DataFrame(market_gate_diag).drop(columns=["stage"]),
        on="name",
        how="left",
        suffixes=("", "_diag"),
        validate="one_to_one",
    )

    nonmarket_2024_raw = paired_2024["p_nonmarket_raw"].to_numpy(dtype=float)
    nonmarket_2024_cal = apply_logit_calibrator(
        nonmarket_2024_raw,
        nonmarket_cal_2023,
    )
    blend_decision, blend_table, blend_predictions = select_blend_one_se(
        paired_2024,
        market_2024_base,
        nonmarket_2024_cal,
    )

    blend_diagnostics = []
    for name, probability in blend_predictions.items():
        blend_diagnostics.append(
            diagnostic_row(paired_2024, probability, name=name)
        )
        reliability_parts.append(
            reliability_rows(
                paired_2024,
                probability,
                name=name,
                period="2024_blend_selection",
            )
        )
    blend_table = blend_table.merge(
        pd.DataFrame(blend_diagnostics),
        left_on="candidate",
        right_on="name",
        how="left",
        validate="one_to_one",
    )

    selected_weight = blend_decision.selected_nonmarket_weight
    selected_2024_probability = blend_predictions[blend_decision.winner]

    paired_2023_2024 = pd.concat(
        [paired_2023, paired_2024],
        ignore_index=True,
        sort=False,
    )
    market_cal_final = fit_logit_calibrator(
        paired_2023_2024["market_p_top3"].to_numpy(dtype=float),
        paired_2023_2024["top3_label"].to_numpy(dtype=int),
    )
    nonmarket_cal_final = fit_logit_calibrator(
        paired_2023_2024["p_nonmarket_raw"].to_numpy(dtype=float),
        paired_2023_2024["top3_label"].to_numpy(dtype=int),
    )

    market_2025_raw = paired_2025["market_p_top3"].to_numpy(dtype=float)
    if market_base_name == "market_calibrated":
        market_2025_base = apply_logit_calibrator(
            market_2025_raw,
            market_cal_final,
        )
    else:
        market_2025_base = market_2025_raw
    nonmarket_2025_raw = paired_2025["p_nonmarket_raw"].to_numpy(dtype=float)
    nonmarket_2025_cal = apply_logit_calibrator(
        nonmarket_2025_raw,
        nonmarket_cal_final,
    )
    selected_2025 = convex_blend(
        market_2025_base,
        nonmarket_2025_cal,
        selected_weight,
    )

    audit_probabilities = {
        "market_raw": market_2025_raw,
        "market_base_frozen_policy": market_2025_base,
        "nonmarket_raw_stage4_v2": nonmarket_2025_raw,
        "nonmarket_calibrated": nonmarket_2025_cal,
        "selected_historical_domain_blend": selected_2025,
    }
    audit_rows = []
    for name, probability in audit_probabilities.items():
        audit_rows.append(
            diagnostic_row(paired_2025, probability, name=name)
        )
        reliability_parts.append(
            reliability_rows(
                paired_2025,
                probability,
                name=name,
                period="2025_known_outcome_audit",
            )
        )
    audit = pd.DataFrame(audit_rows).sort_values(
        ["race_macro_brier", "race_macro_log_loss", "name"],
        kind="stable",
    )

    target_market = pd.read_csv(args.target_market)
    target_nonmarket = pd.read_csv(args.target_nonmarket)
    target = target_market[
        ["horse_no", "horse_name", "p_top3_winmarket_harville"]
    ].merge(
        target_nonmarket[
            ["horse_no", "horse_name", "p_top3_stage4_v2"]
        ],
        on=["horse_no", "horse_name"],
        how="inner",
        validate="one_to_one",
    )
    if len(target) != len(target_market):
        raise ValueError("Target Stage-4 v2 coverage must match target market for rehearsal")

    target["p_market_live_canonical"] = target["p_top3_winmarket_harville"]
    target["p_nonmarket_stage4_v2"] = target["p_top3_stage4_v2"]
    if market_base_name == "market_calibrated":
        target_market_shadow = apply_logit_calibrator(
            target["p_market_live_canonical"].to_numpy(dtype=float),
            market_cal_final,
        )
    else:
        target_market_shadow = target["p_market_live_canonical"].to_numpy(dtype=float)
    target_nonmarket_shadow = apply_logit_calibrator(
        target["p_nonmarket_stage4_v2"].to_numpy(dtype=float),
        nonmarket_cal_final,
    )
    target["p_historical_domain_market_base_shadow"] = target_market_shadow
    target["p_historical_domain_nonmarket_calibrated_shadow"] = target_nonmarket_shadow
    target["p_historical_domain_shadow_blend"] = convex_blend(
        target_market_shadow,
        target_nonmarket_shadow,
        selected_weight,
    )
    target["historical_domain_nonmarket_weight"] = selected_weight
    target["canonical_live_policy"] = "raw_market_only"
    target["market_as_of"] = TARGET_MARKET_AS_OF
    target["canonical_rank"] = (
        target["p_market_live_canonical"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    target = target.sort_values("canonical_rank", kind="stable").reset_index(drop=True)

    registry = blend_registry()
    registry["market_base_2024"] = market_base_name
    registry["nonmarket_component"] = "2023-fitted logit-calibrated Stage4-v2 OOF"
    registry["historical_market_domain"] = "final win odds"
    registry.to_csv(
        args.output_dir / "stage5_v2_candidate_registry.csv",
        index=False,
    )

    calibration_payload = {
        "fit_period": "2023",
        "market": market_cal_2023.__dict__,
        "nonmarket": nonmarket_cal_2023.__dict__,
        "rows": len(paired_2023),
        "races": int(paired_2023["race_id"].nunique()),
        "paired_row_fingerprint": row_fingerprint(paired_2023),
    }
    (args.output_dir / "stage5_v2_calibration_2023.json").write_text(
        json.dumps(calibration_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    market_gate_table.to_csv(
        args.output_dir / "stage5_v2_market_gate_2024.csv",
        index=False,
    )
    blend_table.to_csv(
        args.output_dir / "stage5_v2_blend_grid_2024.csv",
        index=False,
    )
    blend_table.to_csv(
        args.output_dir / "stage5_v2_selection_2024.csv",
        index=False,
    )
    audit.to_csv(
        args.output_dir / "stage5_v2_2025_audit.csv",
        index=False,
    )
    pd.concat(reliability_parts, ignore_index=True).to_csv(
        args.output_dir / "stage5_v2_reliability.csv",
        index=False,
    )
    target.to_csv(
        args.output_dir / "stage5_v2_target_shadow.csv",
        index=False,
        float_format="%.9f",
    )
    oof_2025.to_csv(
        args.output_dir / "stage5_v2_stage4_oof_2025.csv",
        index=False,
        float_format="%.9f",
    )

    selected_2024_score = candidate_score(
        paired_2024,
        selected_2024_probability,
        name=blend_decision.winner,
    )
    market_2024_score = candidate_score(
        paired_2024,
        market_2024_base,
        name="market_only",
    )
    selected_2025_score = candidate_score(
        paired_2025,
        selected_2025,
        name="selected_historical_domain_blend",
    )
    market_2025_score = candidate_score(
        paired_2025,
        market_2025_base,
        name="market_base_frozen_policy",
    )

    manifest = {
        "status": "PASS",
        "stage": "stage5_v2",
        "historical_panel_sha256": PANEL_SHA256,
        "historical_market_sha256": results_sha,
        "historical_market_semantics": "final win odds",
        "stage4_successor": "XGB01 + full-field relative ability",
        "oof_contract": {
            "2023_train_through": 2022,
            "2024_train_through": 2023,
            "2025_train_through": 2024,
            "production_2016_2025_model_used_for_backprediction": False,
        },
        "calibration_fit_2023": calibration_payload,
        "market_gate_2024": {
            "winner": market_gate_decision.winner,
            "point_brier_best": market_gate_decision.point_brier_best,
            "incumbent_retained": market_gate_decision.incumbent_retained,
            "reason": market_gate_decision.reason,
        },
        "blend_selection_2024": {
            "winner": blend_decision.winner,
            "point_brier_best": blend_decision.point_brier_best,
            "market_only_retained": blend_decision.incumbent_retained,
            "selected_nonmarket_weight": selected_weight,
            "reason": blend_decision.reason,
            "market_only_race_macro_brier": market_2024_score.race_macro_brier,
            "selected_race_macro_brier": selected_2024_score.race_macro_brier,
            "paired_rows": len(paired_2024),
            "paired_races": int(paired_2024["race_id"].nunique()),
            "paired_row_fingerprint": row_fingerprint(paired_2024),
        },
        "known_outcome_audit_2025": {
            "selection_use": False,
            "label": "KNOWN_OUTCOME_AUDIT_NOT_TEST",
            "market_race_macro_brier": market_2025_score.race_macro_brier,
            "selected_blend_race_macro_brier": selected_2025_score.race_macro_brier,
            "paired_rows": len(paired_2025),
            "paired_races": int(paired_2025["race_id"].nunique()),
            "paired_row_fingerprint": row_fingerprint(paired_2025),
        },
        "target_policy": {
            "canonical_live_output": "raw market-only",
            "historical_domain_shadow_blend_reported": True,
            "historical_domain_nonmarket_weight": selected_weight,
            "market_snapshot_as_of": TARGET_MARKET_AS_OF,
            "canonical_market_sum": float(target["p_market_live_canonical"].sum()),
            "stage4_v2_raw_sum": float(target["p_nonmarket_stage4_v2"].sum()),
            "shadow_blend_sum": float(target["p_historical_domain_shadow_blend"].sum()),
            "sum_to_three_adjustment_applied_to_shadow": False,
            "target_outcome_loaded": False,
        },
        "market_diagnostics": market_diagnostics,
        "transport_limit": (
            "Historical market uses final odds while live target market is an 08:35 snapshot. "
            "Historical-domain blend selection cannot authorize live-morning adoption."
        ),
    }
    (args.output_dir / "stage5_v2_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    top_selection = blend_table[
        [
            "candidate",
            "nonmarket_weight",
            "race_macro_brier",
            "race_macro_log_loss",
            "mean_brier_delta_vs_best",
            "se_brier_delta_vs_best",
            "within_one_se_of_best",
        ]
    ].head(8)
    report = [
        "# Stage 5 v2 — Market / Non-market Ensemble",
        "",
        "Status: **PASS — v2 historical-domain selection and known-outcome audit completed**",
        "",
        "## Information boundary",
        "",
        "- 2023 Stage-4 prediction: trained through 2022 only",
        "- 2024 Stage-4 prediction: trained through 2023 only",
        "- 2025 audit prediction: trained through 2024 only",
        "- production model fitted through 2025 used to back-predict history: **no**",
        "- 2025 is a known-outcome audit, not an untouched test",
        "",
        "## 2024 market-only gate",
        "",
        "- mandatory incumbent: market_raw",
        f"- gate winner: **{market_gate_decision.winner}**",
        f"- point-Brier-best: **{market_gate_decision.point_brier_best}**",
        f"- incumbent retained: **{market_gate_decision.incumbent_retained}**",
        f"- reason: {market_gate_decision.reason}",
        "",
        market_gate_table.to_markdown(index=False),
        "",
        "## 2024 blend selection",
        "",
        f"- market base: **{market_base_name}**",
        f"- winner under v2 rule: **{blend_decision.winner}**",
        f"- point-Brier-best: **{blend_decision.point_brier_best}**",
        f"- market-only retained: **{blend_decision.incumbent_retained}**",
        f"- selected historical-domain non-market weight: **{selected_weight:.2f}**",
        f"- market-only race-macro Brier: {market_2024_score.race_macro_brier:.6f}",
        f"- selected race-macro Brier: {selected_2024_score.race_macro_brier:.6f}",
        "",
        top_selection.to_markdown(index=False),
        "",
        "## 2025 known-outcome audit",
        "",
        "This section is descriptive only and cannot change the selected recipe.",
        "",
        audit.to_markdown(index=False),
        "",
        "## Frozen 2026 target rehearsal",
        "",
        "Canonical live Stage-5 v2 remains **raw market-only** because the historical market data are",
        "final odds while the target market is the 08:35 snapshot. The historical-domain blend is",
        "reported only as a shadow forecast.",
        "",
        target[
            [
                "canonical_rank",
                "horse_no",
                "horse_name",
                "p_market_live_canonical",
                "p_nonmarket_stage4_v2",
                "p_historical_domain_shadow_blend",
            ]
        ].to_markdown(index=False),
        "",
        f"- canonical raw-market sum: {target['p_market_live_canonical'].sum():.9f}",
        f"- raw Stage-4 v2 sum: {target['p_nonmarket_stage4_v2'].sum():.9f}",
        f"- historical-domain shadow-blend sum: {target['p_historical_domain_shadow_blend'].sum():.9f}",
        "- shadow sum-to-three adjustment: **none**",
        "",
        "## Operational conclusion",
        "",
        "The historical-final-odds result does not by itself authorize a live 08:35 blend. Live",
        "adoption requires repeated pre-race locks of market-only and challenger forecasts under",
        "time-matched snapshots, followed by the preregistered review rule.",
    ]
    (args.output_dir / "STAGE5_V2_REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
