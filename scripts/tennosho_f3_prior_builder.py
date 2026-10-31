"""Recompute 17 complete prior-only F3 rows using the ORIGINAL F3 source rules.

Never replace missing raw 2026 data with synthetic values. The frozen model
requires all JRA flat result runners/conditions to rebuild opponent Elo,
last-observed jockey/trainer rates and raw historical 3F/wet/mass statistics.
The model's SAME-RACE market fields are NOT part of this output: they are
calculated only from truly observed T-10 odds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_information_expansion_v1 import SOURCE_SHA
from run_input_learning_a_v1 import feature_registry
from tennosho_f3_raw_audit import raw_result

from keiba_place_lab.historical_panel import REQUIRED_COLUMNS, build_historical_panel
from keiba_place_lab.information_expansion_v1 import (
    _parse_3f,
    _parse_mass,
    _parse_passing,
    add_historical_result_features,
    add_opponent_elo,
    raw_observations,
)
from keiba_place_lab.input_learning_v1 import (
    MARKET_FIELD,
    MARKET_SPLITS,
    full_starter_relative,
)
from keiba_place_lab.live_history import (
    FROZEN_HISTORY_SHA256,
    RESULT_URL,
    CachedFetcher,
    _read_current_snapshot,
    load_frozen_history,
)
from keiba_place_lab.live_public import parse_roster
from keiba_place_lab.nonmarket import engineer_features

TARGET = "202605041111"
TARGET_DATE = pd.Timestamp("2026-11-01")
ALLOWED_LATEST_SOURCE = pd.Timestamp("2026-10-30")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _standard_columns(source: pd.DataFrame) -> pd.DataFrame:
    """Keep ONLY the fields understood by the pre-existing historical formula."""
    out = source.copy()
    for col in REQUIRED_COLUMNS:
        if col not in out:
            raise ValueError(f"Official standardized historical source absent: {col}")
    return out[sorted(REQUIRED_COLUMNS)].copy()


def _extra_raw_observations(
    full: pd.DataFrame, extra: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Fill source fields for post-Kaggle prior years, preserving original rules."""
    if extra.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Raw results contain duplicate horse/race key")
    original = full.copy()
    extra = extra.copy()
    for frame in (original, extra):
        for name in ("race_id", "horse_id"):
            frame[name] = frame[name].astype(str)
    for name in ("raw_last3f", "raw_corner_first", "raw_corner_last", "raw_early_pct",
                 "raw_passing_gain", "raw_mass", "raw_mass_change", "raw_wet"):
        if name not in original:
            raise ValueError(f"Original raw postrace source feature missing: {name}")
    if set(extra.race_id).difference(set(original.race_id)):
        raise ValueError("External raw results contain unknown historical race ID")
    keys = ["race_id", "horse_id"]
    original = original.merge(
        extra[keys + ["horse_no", "last_3f", "passing", "horse_weight",
                      "track_condition", "jockey_name", "trainer", "rank"]].rename(
                          columns={k: "extra_" + k for k in (
                              "horse_no", "last_3f", "passing", "horse_weight",
                              "track_condition", "jockey_name", "trainer", "rank"
                          )}
                      ),
        on=keys, how="left", validate="one_to_one", indicator=True,
    )
    supplement = original._merge.eq("both")
    if not supplement.any():
        raise ValueError("No audited 2025/26 results were joined to panel")
    mismatch = supplement & pd.to_numeric(
        original.extra_horse_no, errors="coerce"
    ).ne(pd.to_numeric(original.horse_no))
    if mismatch.any():
        raise ValueError("Independent historical raw horse-number discrepancy")
    # Postrace ranks from the audited 2025/26 source are necessary to update
    # opponent Elo. Missing ranks would silently freeze all 2026 Elo histories.
    supplement_rank = pd.to_numeric(
        original.loc[supplement, "extra_rank"].astype("string").str.extract(
            r"^(\d{1,2})(?:\((?:降|再)\))?$", expand=False
        ),
        errors="coerce",
    )
    confirmed_rank = pd.to_numeric(
        original.loc[supplement, "finish_position"], errors="coerce",
    )
    if not supplement_rank.notna().equals(confirmed_rank.notna()) or not (
        supplement_rank.loc[confirmed_rank.notna()].to_numpy(float)
        == confirmed_rank.loc[confirmed_rank.notna()].to_numpy(float)
    ).all():
        raise ValueError("Audited supplemental finishing ranks disagree with verified results")
    original.loc[supplement, "raw_finish_position"] = supplement_rank.to_numpy(float)

    original.loc[supplement, "jockey_name"] = original.loc[supplement, "extra_jockey_name"]
    original.loc[supplement, "trainer"] = original.loc[supplement, "extra_trainer"]
    original.loc[supplement, "raw_last3f"] = _parse_3f(
        original.loc[supplement, "extra_last_3f"]
    )
    parsed = original.loc[supplement, "extra_passing"].map(_parse_passing)
    first = parsed.map(lambda z: z[0])
    last = parsed.map(lambda z: z[1])
    original.loc[supplement, "raw_corner_first"] = first.to_numpy(float)
    original.loc[supplement, "raw_corner_last"] = last.to_numpy(float)
    valid = first.between(1, original.loc[supplement, "field_size"]) & last.between(
        1, original.loc[supplement, "field_size"]
    )
    denom = original.loc[supplement, "field_size"] - 1
    original.loc[supplement, "raw_early_pct"] = ((first - 1)/denom).where(valid)
    original.loc[supplement, "raw_passing_gain"] = ((first - last)/denom).where(valid)
    mass = original.loc[supplement, "extra_horse_weight"].map(_parse_mass)
    original.loc[supplement, "raw_mass"] = mass.map(lambda z: z[0]).to_numpy(float)
    original.loc[supplement, "raw_mass_change"] = mass.map(
        lambda z: z[1]
    ).to_numpy(float)
    condition = original.loc[supplement, "extra_track_condition"]
    original.loc[supplement, "raw_wet"] = condition.isin(
        ("稍重", "重", "不良")
    ).astype(float).to_numpy()
    if not condition.isin(("良", "稍重", "重", "不良")).all():
        raise ValueError("External result race going unavailable")
    # The original 2018-24 source rule ranks last3f among ALL actual
    # starters in that race (excluding unknown finishers automatically).
    counts = original.groupby("race_id", sort=False).raw_last3f.transform("count")
    ranks = original.groupby("race_id", sort=False).raw_last3f.rank(
        ascending=True, method="average"
    )
    original["raw_last3f_rel"] = ((counts-ranks)/(counts-1)).where(counts.ge(2))
    original = original.drop(columns=[
        x for x in original if x.startswith("extra_")
    ] + ["_merge"])
    return original, {"extra_source_rows": int(supplement.sum()),
                      "extra_source_races": int(extra.race_id.nunique())}


def _supplement_2025(raw: pd.DataFrame, original: pd.DataFrame,
                     cache: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Fetch exactly verified original 2025 races missing from Kaggle v1."""
    missing = raw.loc[
        raw.race_date.dt.year.eq(2025) & raw.source_horse_no.isna(),
        "race_id",
    ].astype(str).unique().tolist()
    if not missing:
        return pd.DataFrame(), []
    if len(missing) > 70:
        raise ValueError("Unexpected 2025 Kaggle history missing >70 races")
    fetcher = CachedFetcher(cache, pause_seconds=.5)
    rows, sources = [], []
    for rid in sorted(missing):
        html, record = fetcher.fetch(RESULT_URL.format(provider_id=rid[2:]))
        part = raw_result(html, rid[2:])
        observed = original.loc[
            original.race_id.astype(str).eq(rid), ["horse_id", "horse_no"]
        ].copy()
        observed.horse_id = observed.horse_id.astype(str)
        if len(part) != len(observed) or set(zip(part.horse_id, part.horse_no)) != set(
            zip(observed.horse_id, observed.horse_no)
        ):
            raise ValueError(f"Unmatched 2025 supplemental JRA race {rid}")
        rows.append(part)
        sources.append({"race_id": rid, "url": record.url, "sha256": record.sha256})
    return pd.concat(rows, ignore_index=True), sources


def rebuild(a):
    if sha(a.results_csv) != SOURCE_SHA:
        raise ValueError("Kaggle version differs from original frozen F3 source")
    if sha(a.standardized_base) != FROZEN_HISTORY_SHA256:
        raise ValueError("2010-2025 base does not match original standardized SHA")
    base = load_frozen_history(a.standardized_base)
    current, live, _, _ = _read_current_snapshot(a.live_root)
    if current is None or pd.Timestamp(current["complete_through"]) < pd.Timestamp("2026-10-25"):
        raise ValueError("2026 all-JRA flat source too stale for 17-horse forecast")
    if pd.Timestamp(current["complete_through"]) > ALLOWED_LATEST_SOURCE:
        raise ValueError("Source beyond preregistered 2026-10-30 cutoff")
    if live.empty or not pd.to_datetime(live.race_date).dt.year.eq(2026).all():
        raise ValueError("2026 complete-flat source missing")
    roster_html = a.roster_html.read_text()
    config = json.loads(a.config.read_text())
    roster, _ = parse_roster(roster_html, config)
    if (
        len(roster) != 17 or roster.horse_id.duplicated().any()
        or {str(z.horse_no): str(z.horse_name) for z in roster.itertuples(index=False)}
        != config["horse_no_to_name"]
    ):
        raise ValueError("As-of-date 17-horse roster mismatch")
    roster["is_open_plus"] = 1
    roster["is_graded"] = 1
    for col in ("finish_position", "early_position", "race_time_seconds"):
        roster[col] = np.nan
    older = pd.concat([_standard_columns(base), _standard_columns(live)],
                      ignore_index=True, sort=False)
    if pd.to_datetime(older.race_date).max() >= TARGET_DATE:
        raise ValueError("Target/future result present before feature generation")
    standard = pd.concat([older, _standard_columns(roster)], ignore_index=True)
    standard["race_date"] = pd.to_datetime(standard.race_date)
    if standard.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Duplicate historical result identity")
    panel = build_historical_panel(standard)
    raw, raw_audit = raw_observations(panel, a.results_csv, a.races_csv)
    raw.race_date = pd.to_datetime(raw.race_date)
    historical_2025, supplement_sources = _supplement_2025(
        raw, older, a.cache
    )
    aux = pd.read_csv(
        a.extra_csv, dtype={"race_id": str, "horse_id": str}
    )
    extra_manifest = json.loads(a.extra_manifest.read_text())
    if (
        extra_manifest.get("status") != "COMPLETE_RAW_PRIOR_RESULT_SOURCE"
        or extra_manifest.get("source_csv_sha256") != sha(a.extra_csv)
        or extra_manifest.get("history_input_sha256") != sha(
            a.live_root / "snapshots" / current["snapshot_id"] / "jra_flat_history.parquet"
        )
    ):
        raise ValueError("2026 source evidence not the exact audited live-history snapshot")
    all_2026_keys = set(zip(
        live.race_id.astype(str), live.horse_id.astype(str)
    ))
    if (
        set(zip(aux.race_id, aux.horse_id)) != all_2026_keys
        or aux.duplicated(["race_id", "horse_id"]).any()
    ):
        raise ValueError("2026 raw results do not cover EVERY JRA flat starter")
    if not historical_2025.empty:
        aux = pd.concat([historical_2025, aux], ignore_index=True)
    raw, injection_audit = _extra_raw_observations(raw, aux)
    enriched = add_opponent_elo(add_historical_result_features(raw))
    prior_mean = float(enriched.loc[
        enriched.race_date.dt.year.eq(2016), "top3_label"
    ].mean())
    engineered = engineer_features(enriched, prior_mean=prior_mean)
    focus = engineered.loc[engineered.race_id.astype(str).eq(TARGET)].copy()
    if len(focus) != 17:
        raise ValueError("Target F3 context lacks 17 real entries")
    context, relative_audit = full_starter_relative(focus, focus)
    numeric, categorical = feature_registry()["F3_market_field"]
    exclude = set(MARKET_SPLITS + MARKET_FIELD)
    wanted = [k for k in numeric if k not in exclude] + categorical
    absent = set(wanted) - set(context.columns)
    if absent:
        raise ValueError(f"Target is missing mandatory source F3 columns: {sorted(absent)}")
    inputs = context[
        ["race_id", "race_date", "horse_id", "horse_name", "horse_no"] + wanted
    ].copy()
    inputs["race_date"] = inputs.race_date.dt.strftime("%Y-%m-%d")
    inputs["horse_no"] = inputs.horse_no.astype(int)
    inputs = inputs.sort_values("horse_no").reset_index(drop=True)
    if inputs.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("F3 target horses duplicated")
    inventory = json.loads(a.inventory.read_text())
    if inventory.get("horse_count") != 17 or inventory.get("race_id") != TARGET:
        raise ValueError("Independent 17-horse history inventory missing")
    coverage = []
    for horse in inventory["horses"]:
        hid = str(horse["horse_id"])
        expected = {str(x["race_id"]) for x in horse["historical_races"]}
        actual = set(older.loc[older.horse_id.astype(str).eq(hid), "race_id"].astype(str))
        if expected != actual or len(expected) < 3:
            raise ValueError(f"Incomplete JRA past starts for {hid}: missing={expected-actual}, extra={actual-expected}")
        coverage.append({"horse_id":hid,"expected":len(expected),"actual":len(actual),
                         "latest_jra_flat_date":horse["latest_jra_flat_date"]})
    if set(inputs.horse_id.astype(str)) != {x["horse_id"] for x in coverage}:
        raise ValueError("Target roster / independent history reconciliation failed")
    # Validate meaningfully observed raw signals, never fill defaults or label
    # synthetic missingness 'complete'. These quantities may be NaN for
    # legitimately unsupported historical races; the presence itself is audited.
    raw_target_coverage = {}
    for name in ("history_last3f_rel_last", "history_mass_last",
                 "history_early_pct3", "strength_elo_prior"):
        raw_target_coverage[name] = int(pd.to_numeric(
            inputs[name], errors="coerce"
        ).notna().sum())
    if raw_target_coverage["strength_elo_prior"] != 17:
        raise ValueError("Target Elo missing for a scheduled horse")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists() or a.manifest.exists():
        raise FileExistsError("Immutable original F3 target artifact already exists")
    inputs.to_csv(a.output, index=False, float_format="%.14g", lineterminator="\n")
    original_sources = {
        "standardized_2010_2025_sha256": sha(a.standardized_base),
        "live_2026_snapshot_sha256": current["history_sha256"],
        "live_2026_snapshot_id": current["snapshot_id"],
        "raw_2026_source_manifest_sha256": sha(a.extra_manifest),
        "kaggle_results_v1_sha256": sha(a.results_csv),
        "kaggle_races_v1_sha256": sha(a.races_csv),
        "inventory_source_sha256": sha(a.inventory),
        "roster_source_sha256": sha(a.roster_html),
    }
    combined_history_sha = hashlib.sha256(
        json.dumps(original_sources, sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        "status": "F3_PRIOR_SOURCE_RECONSTRUCTED_20261030",
        "race_id": TARGET,
        "history_cutoff": str(ALLOWED_LATEST_SOURCE.date()),
        "actual_2026_feed_complete_through": current["complete_through"],
        "history_complete_for_all_entrants": True,
        "no_future_rows": True,
        "no_current_race_results_or_odds": True,
        "source_rebuild_contract": "2010-2025 standardized + all 2026 flat starter raw + old original F3 formulas",
        "feature_csv_sha256": sha(a.output),
        "history_sha256": combined_history_sha,
        "inventory_source_sha256": sha(a.inventory),
        "horse_coverage": coverage,
        "feature_numeric_prior_only": [k for k in numeric if k not in exclude],
        "feature_categorical": categorical,
        "last3f_mass_early_elo_coverage": raw_target_coverage,
        "raw_original_audit": raw_audit,
        "source_injection_audit": injection_audit,
        "2025_supplement_races": supplement_sources,
        "relative_feature_coverage": relative_audit.to_dict(orient="records"),
        "input_source_sha256": original_sources,
        "forwards_2026_11_01_t10_not_yet_locked": True,
    }
    a.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": manifest["status"], "horses":len(inputs),
        "feature_csv_sha256":manifest["feature_csv_sha256"],
        "target_raw_signal":raw_target_coverage,
    }, ensure_ascii=False, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--standardized-base", type=Path, required=True)
    p.add_argument("--live-root", type=Path, required=True)
    p.add_argument("--results-csv", type=Path, required=True)
    p.add_argument("--races-csv", type=Path, required=True)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--roster-html", type=Path, required=True)
    p.add_argument("--extra-csv", type=Path, required=True)
    p.add_argument("--extra-manifest", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path("/tmp/tennosho-f3-raw-cache"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    rebuild(p.parse_args())


if __name__ == "__main__":
    main()
