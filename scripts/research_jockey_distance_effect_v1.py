"""Exploratory, strictly past-date JRA rider × distance out-of-time ablation.

NOT a causal jockey effect estimate. Horse selection and past market prices
remain confounders. Frozen Nov 1 F3 model and live prediction are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from run_information_expansion_v1 import load_full_panel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from keiba_place_lab.information_expansion_v1 import raw_observations
from keiba_place_lab.nonmarket import engineer_features, feature_columns

SEED = 20261031
SOURCE_RESULTS_SHA = "fb9345273b21a7c23d41260dc77e45134a1bc2fe756899f3416b24ac0dd51de3"
SOURCE_RACES_SHA = "b9ba4423efbaec641f9285ff01d79895c7b3821c2146522a397127245c81fadb"
BANDS = ("short_le1600", "middle_1601_2399", "long_ge2400")
COMMON_RIDER = (
    "rider_prior_top3_rate", "rider_log_prior_starts",
    "rider_surface_top3_rate", "rider_surface_log_starts",
    "rider_pair_log_starts", "rider_same_as_previous",
)
DISTANCE_RIDER = (
    "rider_band_top3_rate", "rider_band_log_starts",
    "rider_band_lift", "rider_band_surface_top3_rate",
    "rider_band_surface_log_starts",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def distance_band(meters: pd.Series) -> pd.Series:
    values = pd.to_numeric(meters, errors="coerce")
    return pd.cut(values, [0, 1600, 2399, 10000], labels=BANDS).astype("string")


def asof_rates(data: pd.DataFrame, keys: list[str], prefix: str,
               prior_mean: float) -> pd.DataFrame:
    """As-of START OF DAY, for every rider/group; current day's results excluded."""
    required = set(keys + ["race_date", "top3_label"])
    if not required.issubset(data):
        raise ValueError("Missing required rider observation fields")
    indexed = data.reset_index(drop=True)
    out = pd.DataFrame(index=indexed.index)
    present = indexed[keys].notna().all(axis=1)
    safe = indexed.loc[present, keys + ["race_date", "top3_label"]].copy()
    days = safe.groupby(keys + ["race_date"], sort=False, observed=True).agg(
        today_starts=("top3_label", "size"),
        today_hits=("top3_label", "sum"),
    ).reset_index()
    days = days.sort_values(keys + ["race_date"], kind="stable")
    per = days.groupby(keys, sort=False, observed=True)
    days["prior_starts"] = per.today_starts.cumsum() - days.today_starts
    days["prior_hits"] = per.today_hits.cumsum() - days.today_hits
    dates = indexed.loc[present, keys + ["race_date"]].copy()
    dates["_source_position"] = np.flatnonzero(present.to_numpy())
    matched = dates.merge(
        days[keys + ["race_date", "prior_starts", "prior_hits"]],
        on=keys + ["race_date"], how="left", validate="many_to_one", sort=False,
    )
    if matched[["prior_starts", "prior_hits"]].isna().any().any():
        raise ValueError("Missing day-strict prior rider evidence")
    counts = np.zeros(len(indexed), dtype=float)
    hits = np.zeros(len(indexed), dtype=float)
    p = matched["_source_position"].to_numpy(int)
    counts[p] = matched.prior_starts.to_numpy(float)
    hits[p] = matched.prior_hits.to_numpy(float)
    if (hits > counts).any() or (counts < 0).any():
        raise ValueError("Corrupt past-day rider rollup")
    out[prefix + "_starts"] = counts
    out[prefix + "_rate"] = (hits + 15.0 * prior_mean) / (counts + 15.0)
    return out


def add_rider_features(raw: pd.DataFrame, prior_mean: float) -> pd.DataFrame:
    x = raw.copy().reset_index(drop=True)
    x["race_date"] = pd.to_datetime(x.race_date)
    x = x.sort_values(["race_date", "race_id", "horse_no"], kind="stable").reset_index(drop=True)
    x["band"] = distance_band(x.distance_m)
    if x["band"].isna().any():
        raise ValueError("Distance outside supported band, invalid source")
    x["jockey_name"] = x.jockey_name.astype("string").str.strip()
    x.loc[x.jockey_name.eq("").fillna(False), "jockey_name"] = pd.NA
    if x.jockey_name.isna().any():
        raise ValueError("Cannot attribute all JRA historical observations to a rider")
    prev = x.groupby("horse_id", sort=False).jockey_name.shift(1)
    x["rider_same_as_previous"] = x.jockey_name.eq(prev).astype(float)
    x.loc[prev.isna(), "rider_same_as_previous"] = np.nan

    configurations = (
        (["jockey_name"], "overall"),
        (["jockey_name", "surface"], "surface"),
        (["jockey_name", "band"], "band"),
        (["jockey_name", "band", "surface"], "band_surface"),
        (["horse_id", "jockey_name"], "pair"),
    )
    for keys, prefix in configurations:
        features = asof_rates(x, keys, prefix, prior_mean)
        x[prefix + "_prior_starts"] = features[prefix + "_starts"].to_numpy()
        x[prefix + "_prior_rate"] = features[prefix + "_rate"].to_numpy()
    x["rider_prior_top3_rate"] = x.overall_prior_rate
    x["rider_log_prior_starts"] = np.log1p(x.overall_prior_starts)
    x["rider_surface_top3_rate"] = x.surface_prior_rate
    x["rider_surface_log_starts"] = np.log1p(x.surface_prior_starts)
    x["rider_pair_log_starts"] = np.log1p(x.pair_prior_starts)
    x["rider_band_top3_rate"] = x.band_prior_rate
    x["rider_band_log_starts"] = np.log1p(x.band_prior_starts)
    x["rider_band_lift"] = x.band_prior_rate - x.overall_prior_rate
    x["rider_band_surface_top3_rate"] = x.band_surface_prior_rate
    x["rider_band_surface_log_starts"] = np.log1p(x.band_surface_prior_starts)
    return x


def fitted_predictions(train: pd.DataFrame, test: pd.DataFrame,
                       numeric: list[str], categorical: list[str]) -> np.ndarray:
    for col in numeric + categorical:
        if col not in train or col not in test:
            raise ValueError("Missing feature " + col)
    a = train[numeric + categorical].copy()
    b = test[numeric + categorical].copy()
    for col in numeric:
        a[col] = pd.to_numeric(a[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        b[col] = pd.to_numeric(b[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    for col in categorical:
        a[col] = a[col].astype("string").fillna("MISSING").astype(str)
        b[col] = b[col].astype("string").fillna("MISSING").astype(str)
    transformer = ColumnTransformer([
        ("numbers", "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=5,
                                      sparse_output=False), categorical),
    ], sparse_threshold=0)
    x = transformer.fit_transform(a)
    z = transformer.transform(b)
    model = XGBClassifier(
        n_estimators=180, max_depth=3, learning_rate=0.04,
        min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=2, reg_lambda=20,
        tree_method="hist", n_jobs=3, objective="binary:logistic",
        eval_metric="logloss", random_state=SEED,
    )
    count = train.groupby("race_id").race_id.transform("size").to_numpy(float)
    weight = 1 / count
    weight *= len(weight) / weight.sum()
    model.fit(x, train.top3_label.to_numpy(int), sample_weight=weight, verbose=False)
    predicted = model.predict_proba(z)[:, 1]
    if not np.isfinite(predicted).all():
        raise ValueError("Invalid rider ablation probability")
    return np.clip(predicted, 1e-7, 1 - 1e-7)


def date_bootstrap(paired: pd.DataFrame, comparison: str,
                   *, draws: int = 2000) -> dict:
    """Resample racing DAYS, shared across distance groups, compare race-macro Brier."""
    if paired.empty:
        return {}
    grouped = paired.groupby(["race_date", "band"], observed=True).agg(
        count=("improvement", "size"),
        total=("improvement", "sum"),
    ).reset_index()
    dates = np.sort(paired.race_date.unique())
    n = len(dates)
    b = {x: i for i, x in enumerate(BANDS)}
    c = np.zeros((n, 3), dtype=float)
    t = np.zeros((n, 3), dtype=float)
    date_map = {x: i for i, x in enumerate(dates)}
    for x in grouped.itertuples(index=False):
        i, j = date_map[x.race_date], b[str(x.band)]
        c[i, j] = x.count
        t[i, j] = x.total
    if np.any(c.sum(axis=0) < 20):
        return {"status": "INSUFFICIENT_RACES_FOR_DISTANCE_CONTRAST",
                "counts": c.sum(axis=0).tolist()}
    rng = np.random.default_rng(SEED + sum(map(ord, comparison)))
    samples = rng.integers(0, n, size=(draws, n))
    # Resampling days preserves shared racing-day shocks and band correlation.
    count_samples = c[samples].sum(axis=1)
    total_samples = t[samples].sum(axis=1)
    rates = np.divide(total_samples, count_samples,
                      out=np.full_like(total_samples, np.nan),
                      where=count_samples > 0)
    observed = t.sum(axis=0) / c.sum(axis=0)
    contrast_samples = rates[:, 2] - rates[:, 0]
    observed_contrast = observed[2] - observed[0]
    valid = contrast_samples[np.isfinite(contrast_samples)]
    if len(valid) != draws:
        raise ValueError("Date bootstrap lost distance-band coverage")
    return {
        "status": "OK", "bootstrap_draws": draws, "cluster": "race_date",
        "n_cluster_dates": n, "comparison": comparison,
        "improvement_by_band": dict(zip(BANDS, observed.tolist())),
        "long_minus_short": float(observed_contrast),
        "contrast_95pct_percentile_CI": np.quantile(valid, [.025, .975]).tolist(),
        "contrast_two_sided_bootstrap_sign_p": float(
            min(1.0, 2 * min(np.mean(valid <= 0), np.mean(valid >= 0)))
        ),
    }


def summarize(pred: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    parts = []
    intervals: dict = {}
    for year in (2023, 2024):
        for surface in ("all", "turf", "dirt"):
            sub = pred.loc[pred.race_date.dt.year.eq(year)]
            if surface != "all":
                sub = sub.loc[sub.surface.eq(surface)]
            for comparison, before, after in (
                ("rider_common_vs_base", "p_base", "p_rider_common"),
                ("rider_distance_vs_base", "p_base", "p_rider_distance"),
                ("distance_specific_vs_common", "p_rider_common", "p_rider_distance"),
            ):
                q = sub[["race_date", "race_id", "band", "top3_label", before, after]].copy()
                if q.empty:
                    continue
                y = q.top3_label.to_numpy(int)
                q["improvement"] = (q[before].to_numpy() - y) ** 2 - (
                    q[after].to_numpy() - y) ** 2
                races = q.groupby(["race_date", "race_id", "band"], observed=True).agg(
                    improvement=("improvement", "mean"),
                    before=("top3_label", "size"),
                ).reset_index()
                for band in ("all", *BANDS):
                    t = races if band == "all" else races.loc[races.band.eq(band)]
                    if t.empty:
                        continue
                    rows = q if band == "all" else q.loc[q.band.eq(band)]
                    parts.append({
                        "year": year, "surface": surface, "band": band,
                        "comparison": comparison,
                        "rows": len(rows), "races": len(t),
                        "dates": int(t.race_date.nunique()),
                        "delta_brier_race_macro": float(t.improvement.mean()),
                        "delta_brier_se_cluster_date": float(
                            (t.groupby("race_date").improvement.mean().std(ddof=1))
                            / np.sqrt(t.race_date.nunique())
                        ),
                    })
                if year == 2024:
                    intervals[surface + ":" + comparison] = date_bootstrap(
                        races[["race_date", "band", "improvement"]],
                        surface + ":" + comparison,
                    )
    return pd.DataFrame(parts), intervals


def evaluate(historical_dir: Path, results_csv: Path, races_csv: Path,
             output: Path) -> dict:
    if sha256(results_csv) != SOURCE_RESULTS_SHA or sha256(races_csv) != SOURCE_RACES_SHA:
        raise ValueError("Original JRA source sha256 changed")
    full, hashes = load_full_panel(historical_dir)
    raw, raw_audit = raw_observations(full, results_csv, races_csv)
    prior = float(raw.loc[raw.race_date.dt.year.eq(2016)].top3_label.mean())
    data = add_rider_features(raw, prior)
    data = engineer_features(data, prior_mean=prior)
    data = data.loc[
        data.race_date.dt.year.between(2018, 2024)
        & data.surface.isin(("turf", "dirt"))
        & data.field_size.ge(8)
        & data.career_starts.ge(3)
        & data.distance_m.between(1000, 3600)
    ].copy()
    if data.empty or data.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("No unique JRA horses for distance experiment")
    cats = ["sex", "racecourse", "race_class", "surface"]
    numerical, _ = feature_columns()
    feature_sets = {
        "base": numerical,
        "rider_common": numerical + list(COMMON_RIDER),
        "rider_distance": numerical + list(COMMON_RIDER) + list(DISTANCE_RIDER),
    }
    out = []
    audits = []
    for year in (2023, 2024):
        training = data.loc[data.race_date.dt.year.between(2018, year-1)].copy()
        test = data.loc[data.race_date.dt.year.eq(year)].copy()
        if (training.empty or test.empty
                or training.race_date.max() >= test.race_date.min()):
            raise ValueError("Future data leaked into training")
        scored = test[["race_date", "race_id", "horse_no", "horse_id", "surface",
                       "distance_m", "band", "top3_label"]].copy()
        for model, columns in feature_sets.items():
            print(f"YEAR {year} FIT {model} train={len(training)} test={len(test)}", flush=True)
            scored["p_" + model] = fitted_predictions(training, test, columns, cats)
        out.append(scored)
        audits.append({
            "outer_year": year,
            "training_latest_date": str(training.race_date.max().date()),
            "evaluation_earliest_date": str(test.race_date.min().date()),
            "train_rows": len(training), "eval_rows": len(test),
            "eval_races": test.race_id.nunique(),
            "rider_column_valid": int(test.jockey_name.notna().sum()),
        })
    preds = pd.concat(out, ignore_index=True)
    report, bootstrap = summarize(preds)
    output.mkdir(parents=True, exist_ok=True)
    report.to_csv(output / "distance_effect_metrics.csv", index=False, float_format="%.12g")
    # Individual runner predictions withheld from Git: save reproducible aggregates only.
    m = {
        "status": "EXPLORATORY_OBSERVATIONAL_OUT_OF_TIME_NOT_CAUSAL",
        "target": "top3_label", "horse_controls": numerical,
        "jockey_common_features": COMMON_RIDER,
        "jockey_distance_specific_features": DISTANCE_RIDER,
        "distance_bands_m": {"short": "<=1600", "middle": "1601..2399", "long": ">=2400"},
        "rider_history": "Jockey performance strictly before the TARGET race date; same-day results omitted",
        "horse_history": "Pre-existing leakage-safe frozen horse prior features",
        "source_sha256": {**hashes, "keiba_results.csv": sha256(results_csv),
                          "keiba_races.csv": sha256(races_csv)},
        "raw_source": raw_audit,
        "folds": audits,
        "bootstrap_2024": bootstrap,
        "model_params": {
            "n_estimators":180, "max_depth":3, "learning_rate":0.04,
            "min_child_weight":50, "subsample":0.8, "colsample_bytree":0.8,
            "reg_alpha":2, "reg_lambda":20, "seed":SEED,
        },
        "limitations": [
            "Predictive ablation DOES NOT identify causal jockey ability: assignment to better horses is confounded.",
            "No live pre-race market odds are controlled, so this cannot establish extra edge beyond betting market.",
            "Outer 2023 is secondary and 2024 is primary. All model settings are fixed, no result-based selection.",
            "Distance bands have different race conditions, rival quality, and sample sizes.",
            "Rider probability estimates use previous outcomes of other horses, only preceding calendar days.",
            "F3 production and Nov 1 frozen experimental model are unchanged.",
        ],
        "code_sha256": sha256(Path(__file__)),
        "python": platform.python_version(),
        "xgboost": xgboost.__version__,
    }
    (output / "audit.json").write_text(json.dumps(m, ensure_ascii=False, indent=2,
                                                 default=lambda x: int(x) if isinstance(x,np.integer)
                                                 else float(x)) + "\n")
    highlighted = report.loc[
        (report.year == 2024) & report.surface.eq("turf")
        & report.band.isin(["all", *BANDS])
    ].copy()
    print("2024_TURF_RESULTS\\n" + highlighted.to_string(index=False), flush=True)
    print("2024_TURF_BOOTSTRAP\\n"
          + json.dumps({k:v for k,v in bootstrap.items() if k.startswith("turf:")},
                       indent=2, ensure_ascii=False), flush=True)
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--historical-dir", required=True, type=Path)
    p.add_argument("--results-csv", required=True, type=Path)
    p.add_argument("--races-csv", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    evaluate(a.historical_dir, a.results_csv, a.races_csv, a.out)


if __name__ == "__main__":
    main()
