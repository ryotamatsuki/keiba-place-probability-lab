"""Execute pre-registered feature, turf transport and dirt experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from keiba_place_lab.historical_panel import select_phase_a_cohort
from keiba_place_lab.model_selection import (
    candidate_score,
    paired_delta,
    race_loss_table,
    select_with_incumbent_one_se,
)
from keiba_place_lab.nonmarket import calibration_table
from keiba_place_lab.scope_expansion import full_context_for_races, straight_course_mask
from keiba_place_lab.scope_features import (
    BANDS,
    BLOCKS,
    augment_distance_history,
    fit_scope_candidate,
    predict_scope_candidate,
)
from keiba_place_lab.stage4_successor import calibration_intercept_slope

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs/SCOPE_EXPANSION_COMPLETION_SPEC.md"
KEYS = ["race_id", "horse_id"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def cohort(panel, surface, lo=1000, hi=2600):
    out = select_phase_a_cohort(
        panel, surface=surface, min_field_size=8, min_prior_starts=3,
        min_distance_m=lo, max_distance_m=hi,
    )
    return out.loc[~straight_course_mask(out)].sort_values(
        ["race_date", "race_id", "horse_no"], kind="stable",
    ).reset_index(drop=True)


def oof(panel, train_cohort, evaluation, blocks, surface, name, sizes):
    p = np.full(len(evaluation), np.nan)
    for year in (2023, 2024):
        train = train_cohort.loc[train_cohort.race_date.dt.year.between(2016, year - 1)]
        mask = evaluation.race_date.dt.year.eq(year).to_numpy()
        test = evaluation.loc[mask]
        if train.empty or test.empty:
            raise ValueError("Empty training/evaluation fold")
        context = full_context_for_races(panel, set(train.race_id))
        print(f"Fitting {name} {year}: {len(train)} rows, blocks={blocks}", flush=True)
        model, prior = fit_scope_candidate(train, context, blocks=blocks, surface=surface)
        p[mask] = predict_scope_candidate(
            model, test, full_context_for_races(panel, set(test.race_id)),
            prior_mean=prior, blocks=blocks, surface=surface,
        )
        sizes.append({"candidate": name, "year": year, "train_rows": len(train),
                      "train_races": train.race_id.nunique(), "train_dates": train.race_date.nunique(),
                      "context_rows": len(context), "prior_mean": prior,
                      "blocks": ",".join(blocks)})
    if not np.isfinite(p).all():
        raise ValueError("Incomplete OOF")
    return p


def diagnostics(frame, p, name, subgroup="all", value="all"):
    row = asdict(candidate_score(frame, p, name=name))
    intercept, slope = calibration_intercept_slope(frame.top3_label.to_numpy(int), p)
    return {**row, "subgroup": subgroup, "value": str(value),
            "dates": frame.race_date.nunique(), "intercept": intercept, "slope": slope}


def save_report(out, frame, predictions, reference, decisions, sizes, manifest):
    out.mkdir(parents=True, exist_ok=True)
    cols = ["race_date", "race_id", "horse_id", "horse_no", "top3_label",
            "distance_m", "racecourse", "race_class", "age", "field_size", "surface"]
    saved = frame[cols].copy()
    scores, bins, losses, paired = [], [], [], []
    for name, p in predictions.items():
        saved[f"p_{name}"] = p
        scores.append(diagnostics(frame, p, name))
        for key in ("year", "distance_m", "racecourse", "race_class", "age", "field_size"):
            groups = frame.race_date.dt.year if key == "year" else frame[key]
            for value in sorted(groups.dropna().unique()):
                mask = groups.eq(value).to_numpy()
                scores.append(diagnostics(frame.loc[mask], p[mask], name, key, value))
        b = calibration_table(frame, p)
        b["candidate"] = name
        bins.append(b)
        loss = race_loss_table(frame, p)
        loss["candidate"] = name
        losses.append(loss)
        if name != reference:
            paired.append(asdict(paired_delta(
                frame, p, predictions[reference], candidate_name=name, reference_name=reference,
            )))
    saved.to_csv(out / "predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(scores).to_csv(out / "scores.csv", index=False)
    pd.concat(bins).to_csv(out / "calibration.csv", index=False)
    all_losses = pd.concat(losses)
    all_losses.to_csv(out / "race_losses.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    ref = all_losses.loc[all_losses.candidate.eq(reference), ["race_id", "brier", "log_loss"]]
    differences = all_losses.merge(ref, on="race_id", suffixes=("", "_reference"), validate="many_to_one")
    differences["brier_delta"] = differences.brier - differences.brier_reference
    differences["log_loss_delta"] = differences.log_loss - differences.log_loss_reference
    differences.to_csv(out / "race_differences.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)
    pd.DataFrame(sizes).to_csv(out / "training_folds.csv", index=False)
    json_write(out / "decisions.json", decisions)
    manifest.update({"rows": len(frame), "races": frame.race_id.nunique(),
                     "dates": frame.race_date.nunique(), "evaluation_years": [2023, 2024],
                     "prediction_sha256": digest(out / "predictions.csv.gz")})
    json_write(out / "manifest.json", manifest)
    primary = pd.DataFrame(scores).query("subgroup == 'all'")
    (out / "REPORT.md").write_text(
        "# Scope development results\n\nKnown 2023–2024 development results; prospective confirmation pending.\n\n"
        + primary[["name", "race_macro_brier", "race_macro_log_loss", "rows", "races"]].to_markdown(index=False, floatfmt=".9f")
        + "\n\n```json\n" + json.dumps(decisions, indent=2) + "\n```\n",
    )
    print(primary[["name", "race_macro_brier"]].to_string(index=False), flush=True)
    print(json.dumps(decisions), flush=True)


def run_features(panel, out, manifest):
    evaluation = cohort(panel, "turf", 1200, 1200)
    evaluation = evaluation.loc[evaluation.race_date.dt.year.isin([2023, 2024])].reset_index(drop=True)
    assert len(evaluation) == 6313 and evaluation.race_id.nunique() == 495
    sizes, predictions, retained, chosen, gates = [], {}, {}, {}, []
    scopes = {"A": (1200, 1200), "B": (1000, 1400), "C": (1000, 2600)}
    frozen = pd.read_csv(ROOT / "analysis/scope_expansion_stage1/scope_stage1_outer_predictions.csv", dtype={k: "string" for k in KEYS})
    for family, (lo, hi) in scopes.items():
        train = cohort(panel, "turf", lo, hi)
        name = f"{family}_F0"
        predictions[name] = oof(panel, train, evaluation, (), "turf", name, sizes)
        ref_column = next(c for c in frozen if c.startswith(f"p_{family}_"))
        reference = evaluation[KEYS].merge(frozen[KEYS + [ref_column]], on=KEYS, validate="one_to_one")
        if np.max(np.abs(predictions[name] - reference[ref_column].to_numpy())) > 1e-7:
            raise ValueError(f"F0 {family} does not reproduce Stage 1")
        retained[family], chosen[family] = [], name
        for block in BLOCKS:
            blocks = [*retained[family], block]
            proposed = f"{family}_{'_'.join(blocks)}"
            predictions[proposed] = oof(panel, train, evaluation, blocks, "turf", proposed, sizes)
            decision, _ = select_with_incumbent_one_se(
                evaluation, {chosen[family]: predictions[chosen[family]], proposed: predictions[proposed]},
                incumbent=chosen[family],
            )
            delta = asdict(paired_delta(evaluation, predictions[proposed], predictions[chosen[family]],
                                       candidate_name=proposed, reference_name=chosen[family]))
            gates.append({**delta, "accepted": decision.winner == proposed})
            if decision.winner == proposed:
                retained[family], chosen[family] = blocks, proposed
    finalists = {name: predictions[name] for name in {"A_F0", *chosen.values()}}
    decision, _ = select_with_incumbent_one_se(evaluation, finalists, incumbent="A_F0")
    winner = decision.winner
    family = winner[0]
    decisions = {"family_blocks": retained, "family_winners": chosen,
                 "winner": winner, "winner_scope": scopes[family],
                 "winner_blocks": [] if winner == "A_F0" else retained[family], "block_gates": gates,
                 "selection": asdict(decision)}
    save_report(out, evaluation, predictions, "A_F0", decisions, sizes, manifest)


def run_surface(panel, out, manifest, surface):
    train = cohort(panel, surface)
    evaluation = train.loc[train.race_date.dt.year.isin([2023, 2024])].reset_index(drop=True)
    audit = train.assign(year=train.race_date.dt.year).groupby(["year", "distance_m", "racecourse"]).agg(
        rows=("horse_id", "size"), races=("race_id", "nunique"), dates=("race_date", "nunique"),
    ).reset_index()
    out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out / "audit.csv", index=False)
    train.isna().groupby([train.race_date.dt.year, train.distance_m]).mean().to_csv(out / "missingness.csv")
    sizes = []
    predictions = {"G0": oof(panel, train, evaluation, (), surface, "G0", sizes),
                   "G3": oof(panel, train, evaluation, BLOCKS, surface, "G3", sizes)}
    regime = np.full(len(evaluation), np.nan)
    for lo, hi in BANDS:
        mask = evaluation.distance_m.between(lo, hi).to_numpy()
        regime[mask] = oof(panel, cohort(panel, surface, lo, hi), evaluation.loc[mask],
                           BLOCKS, surface, f"R3_{lo}_{hi}", sizes)
    predictions["R3"] = regime
    predictions["baseline"] = 3 / evaluation.field_size.to_numpy(float)
    global_decision, _ = select_with_incumbent_one_se(
        evaluation, {k: predictions[k] for k in ("G0", "G3")}, incumbent="G0",
    )
    whole_decision, _ = select_with_incumbent_one_se(
        evaluation, {k: predictions[k] for k in ("G0", "G3", "R3")}, incumbent="G0",
    )
    selected_global = global_decision.winner
    routed = predictions[selected_global].copy()
    routes, band_metrics = [], []
    for lo, hi in BANDS:
        mask = evaluation.distance_m.between(lo, hi).to_numpy()
        decision, _ = select_with_incumbent_one_se(
            evaluation.loc[mask], {k: predictions[k][mask] for k in (selected_global, "R3")},
            incumbent=selected_global,
        )
        routes.append({"min_distance_m": lo, "max_distance_m": hi,
                       "model": decision.winner,
                       "blocks": list(BLOCKS) if decision.winner != "G0" else []})
        if decision.winner == "R3":
            routed[mask] = regime[mask]
        delta = asdict(paired_delta(evaluation.loc[mask], regime[mask], predictions[selected_global][mask],
                                   candidate_name="R3", reference_name=selected_global))
        for k in predictions:
            band_metrics.append({**diagnostics(evaluation.loc[mask], predictions[k][mask], k),
                                 "band": f"{lo}-{hi}", **delta})
    decisions = {"surface": surface, "global_selection": asdict(global_decision),
                 "whole_selection": asdict(whole_decision), "routes": routes}
    if surface == "turf":
        features = json.loads((ROOT / "analysis/scope_expansion_features/decisions.json").read_text())
        mask = evaluation.distance_m.eq(1200).to_numpy()
        previous = pd.read_csv(ROOT / "analysis/scope_expansion_features/predictions.csv.gz", dtype={k: "string" for k in KEYS})
        column = f"p_{features['winner']}"
        incumbent = evaluation.loc[mask, KEYS].merge(previous[KEYS + [column]], on=KEYS, validate="one_to_one")[column].to_numpy()
        decision, _ = select_with_incumbent_one_se(
            evaluation.loc[mask], {"scope_route": routed[mask], "stage2": incumbent}, incumbent="stage2",
        )
        decisions["1200_selection"] = asdict(decision)
        if decision.winner == "stage2":
            decisions["1200_override"] = {"scope": features["winner_scope"], "blocks": features["winner_blocks"]}
            routed[mask] = incumbent
    predictions["routed"] = routed
    pd.DataFrame(band_metrics).to_csv(out / "bands.csv", index=False)
    save_report(out, evaluation, predictions, "G0", decisions, sizes, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--stage", choices=("features", "turf", "dirt"), required=True)
    args = parser.parse_args()
    panel = pd.read_parquet(args.panel)
    panel["race_date"] = pd.to_datetime(panel.race_date)
    for k in KEYS:
        panel[k] = panel[k].astype("string")
    cache = ROOT / "data/historical_processed/scope_augmented.parquet"
    source_hash = digest(args.panel)
    meta = cache.with_suffix(".json")
    if cache.exists() and meta.exists() and json.loads(meta.read_text()) == {"source": source_hash, "code": digest(ROOT / "src/keiba_place_lab/scope_features.py")}:
        panel = pd.read_parquet(cache)
    else:
        print("Computing strictly prior-date histories from all 753k starters", flush=True)
        panel = augment_distance_history(panel)
        cache.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(cache, index=False)
        json_write(meta, {"source": source_hash, "code": digest(ROOT / "src/keiba_place_lab/scope_features.py")})
    import sklearn
    import xgboost
    manifest = {"source_sha256": source_hash, "spec_sha256": digest(SPEC),
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (
                    Path(__file__), ROOT / "src/keiba_place_lab/scope_features.py")},
                "versions": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                             "sklearn": sklearn.__version__, "xgboost": xgboost.__version__}}
    out = ROOT / f"analysis/scope_expansion_{args.stage}"
    if args.stage == "features":
        run_features(panel, out, manifest)
    else:
        run_surface(panel, out, manifest, args.stage)


if __name__ == "__main__":
    main()
