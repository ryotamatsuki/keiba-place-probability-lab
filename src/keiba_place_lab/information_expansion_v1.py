"""Strictly prior-date raw JRA history and opponent-adjusted Elo research.

Important: raw result values at a target race are NEVER direct target features.
All rider/trainer identities are last-observed connections, not target lineup.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RAW_RESULT_COLS = (
    "race_id", "horse_id", "rank", "number", "jockey_name", "trainer",
    "last_3f", "passing", "horse_weight",
)
RAW_RACE_COLS = ("race_id", "track_condition")
NEW_HISTORY = (
    "history_last3f_rel3", "history_last3f_rel_last",
    "history_passing_gain3", "history_early_pct3",
    "history_early_stability3", "history_mass_last",
    "history_mass_change_last", "history_wet_starts",
    "history_wet_top3_rate", "history_last_jockey_prior_rate",
    "history_last_trainer_prior_rate", "history_jockey_stability",
)
NEW_STRENGTH = (
    "strength_elo_prior", "strength_elo_vs_field",
    "strength_opp_mean_prior", "strength_previous_opponent",
    "strength_previous_placing", "strength_prior_races",
)


def _read_csv_select(path: Path, fields: tuple[str, ...], requested: set[str],
                     *, chunksize: int = 200_000) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(
        path, encoding="utf-8-sig", dtype=str, usecols=list(fields),
        chunksize=chunksize, on_bad_lines="skip", low_memory=False,
    ):
        part = chunk.loc[chunk["race_id"].isin(requested)]
        if not part.empty:
            parts.append(part)
    if not parts:
        raise ValueError(f"No matching JRA rows in {path.name}")
    return pd.concat(parts, ignore_index=True)


def _parse_3f(series: pd.Series) -> pd.Series:
    numbers = pd.to_numeric(series, errors="coerce")
    return numbers.where(numbers.between(20, 65))


def _parse_passing(value: object) -> tuple[float, float]:
    if not isinstance(value, str) or not re.fullmatch(r"[1-9]\d*(?:-[1-9]\d*){0,3}", value):
        return np.nan, np.nan
    blocks = [float(z) for z in value.split("-")]
    return blocks[0], blocks[-1]


def _parse_mass(value: object) -> tuple[float, float]:
    if not isinstance(value, str):
        return np.nan, np.nan
    match = re.fullmatch(r"(\d{3})\(([+-]?\d+)\)", value.strip())
    if match is None:
        return np.nan, np.nan
    mass, change = float(match.group(1)), float(match.group(2))
    if 300 <= mass <= 700 and -80 <= change <= 80:
        return mass, change
    return np.nan, np.nan


def raw_observations(full: pd.DataFrame, results_path: Path,
                     races_path: Path, *, horse_id_normalizer=None
                     ) -> tuple[pd.DataFrame, dict]:
    """Match immutable official-date starter keys using optional audited ID namespace.

    The default path preserves the frozen historical experiment. The
    preoff-only runner may supply a strict JRA-ID canonicalizer so original
    Kaggle IDs and normalized 2010-2026 history agree on the same horse.
    """
    keys = ["race_id", "horse_id"]
    x = full.copy()
    for key in keys:
        x[key] = x[key].astype(str)
    if x.duplicated(keys).any():
        raise ValueError("Duplicate official JRA runner keys")
    ids = set(x.race_id)
    results = _read_csv_select(results_path, RAW_RESULT_COLS, ids)
    results = results.rename(columns={"number": "source_horse_no"})
    results["horse_id"] = results.horse_id.astype(str)
    if horse_id_normalizer is not None:
        results["horse_id"] = results.horse_id.map(horse_id_normalizer)
    results = results.loc[results.horse_id.isin(set(x.horse_id))]
    if results.duplicated(keys).any():
        raise ValueError("Duplicate raw JRA horse/race keys")
    races = _read_csv_select(races_path, RAW_RACE_COLS, ids)
    if races.race_id.duplicated().any():
        raise ValueError("Duplicate raw JRA race conditions")
    x = x.merge(
        results[keys + ["rank", "source_horse_no", "jockey_name", "trainer",
                        "last_3f", "passing", "horse_weight"]],
        on=keys, how="left", validate="one_to_one", indicator=True,
    )
    matched = int(x["_merge"].eq("both").sum())
    x.drop(columns="_merge", inplace=True)
    x = x.merge(races, on="race_id", how="left", validate="many_to_one")
    source_no = pd.to_numeric(x.source_horse_no, errors="coerce")
    inconsistent = source_no.notna() & source_no.ne(pd.to_numeric(x.horse_no))
    if inconsistent.any():
        raise ValueError("Official runner/source horse number mismatch")
    # Historical rank is absent from the model-safe frozen panel on purpose.
    # Restore it only as a POSTRACE update signal for Elo, never as a feature
    # of the target horse. Reject any conflict with the frozen binary label.
    rank_text=x["rank"].astype("string").str.extract(
        r"^(\d{1,2})(?:\((?:降|再)\))?$", expand=False
    )
    x["raw_finish_position"]=pd.to_numeric(rank_text,errors="coerce")
    outcome_conflict=(
        x.raw_finish_position.notna()
        & x.raw_finish_position.between(1,3).ne(x.top3_label.astype(bool))
    )
    conflicts=int(outcome_conflict.sum())
    x.loc[outcome_conflict,"raw_finish_position"]=np.nan
    x["raw_last3f"] = _parse_3f(x.last_3f)
    pos = x.passing.map(_parse_passing)
    x["raw_corner_first"] = pos.map(lambda z: z[0])
    x["raw_corner_last"] = pos.map(lambda z: z[1])
    valid_positions = x.raw_corner_first.between(1, x.field_size) & (
        x.raw_corner_last.between(1, x.field_size))
    x["raw_early_pct"] = ((x.raw_corner_first - 1) / (x.field_size-1)).where(
        valid_positions)
    x["raw_passing_gain"] = ((x.raw_corner_first-x.raw_corner_last) /
                              (x.field_size-1)).where(valid_positions)
    masses = x.horse_weight.map(_parse_mass)
    x["raw_mass"] = masses.map(lambda z: z[0])
    x["raw_mass_change"] = masses.map(lambda z: z[1])
    x["raw_wet"] = x.track_condition.isin(("稍重", "重", "不良")).astype(float)
    x.loc[x.track_condition.isna() | ~x.track_condition.isin(
        ("良", "稍重", "重", "不良")), "raw_wet"] = np.nan
    # Current-race relative closing time is a POSTRACE observation; shifts
    # below ensure it contributes only to strictly later dates.
    group = x.groupby("race_id", sort=False)["raw_last3f"]
    count = group.transform("count")
    rank = group.rank(ascending=True, method="average")
    x["raw_last3f_rel"] = ((count-rank)/(count-1)).where(count.ge(2))
    audit = {
        "total_official_rows": len(x), "matched_raw_runner_rows": matched,
        "unmatched_raw_runner_rows": int(len(x)-matched),
        "raw_rank_label_conflicts_excluded": conflicts,
        "raw_result_duplicate_keys": 0,
        "raw_race_duplicate_ids": 0,
        "usable": {column: int(x[column].notna().sum()) for column in (
            "raw_last3f", "raw_corner_first", "raw_passing_gain",
            "raw_mass", "raw_mass_change", "raw_wet", "jockey_name", "trainer")},
    }
    return x, audit


def _lagged_feature(x: pd.DataFrame, source: str, *, periods: int) -> pd.Series:
    return x.groupby("horse_id", sort=False)[source].shift(periods)


def _lag_mean(x: pd.DataFrame, source: str, n: int = 3) -> pd.Series:
    parts = [_lagged_feature(x, source, periods=i) for i in range(1, n+1)]
    return pd.concat(parts, axis=1).mean(axis=1)


def _prior_agent_rate(x: pd.DataFrame, name: str) -> pd.Series:
    """Agent rate as of strictly prior date (not this day's later results)."""
    safe = x.loc[x[name].notna(), [name, "race_date", "top3_label"]].copy()
    days = safe.groupby([name, "race_date"], sort=False).agg(
        starts=("top3_label","size"), wins=("top3_label","sum")).reset_index()
    days.sort_values([name,"race_date"], kind="stable", inplace=True)
    per = days.groupby(name, sort=False)
    days["prior_count"] = per.starts.cumsum() - days.starts
    days["prior_wins"] = per.wins.cumsum() - days.wins
    days["prior_agent_rate"] = (days.prior_wins + 3.0)/(days.prior_count + 15.0)
    tagged = x[[name, "race_date"]].merge(
        days[[name,"race_date","prior_agent_rate"]],
        on=[name,"race_date"], how="left", sort=False, validate="many_to_one",
    )
    return pd.Series(tagged.prior_agent_rate.to_numpy(), index=x.index)


def add_historical_result_features(data: pd.DataFrame) -> pd.DataFrame:
    """Chronological historical day-aggregation with strictly previous races."""
    x = data.copy()
    x["race_date"] = pd.to_datetime(x.race_date)
    x.sort_values(["race_date","race_id","horse_no"], kind="stable", inplace=True)
    x.reset_index(drop=True, inplace=True)
    if x.duplicated(["horse_id","race_date"]).any():
        raise ValueError("Horse may not have multiple JRA starts on one date")
    # These source columns are POSTRACE only; all transformed columns have lag>=1.
    x["history_last3f_rel3"] = _lag_mean(x,"raw_last3f_rel")
    x["history_last3f_rel_last"] = _lagged_feature(x,"raw_last3f_rel",periods=1)
    x["history_passing_gain3"] = _lag_mean(x,"raw_passing_gain")
    x["history_early_pct3"] = _lag_mean(x,"raw_early_pct")
    position_lags = pd.concat(
        [_lagged_feature(x,"raw_early_pct",periods=i) for i in (1,2,3)],axis=1)
    x["history_early_stability3"] = position_lags.std(axis=1)
    x["history_mass_last"] = _lagged_feature(x,"raw_mass",periods=1)
    x["history_mass_change_last"] = _lagged_feature(x,"raw_mass_change",periods=1)
    start_wet = x.raw_wet.eq(1.0).astype(int)
    success_wet = (start_wet * x.top3_label).astype(int)
    x["_wet_starts"] = start_wet.where(x.raw_wet.notna(), 0)
    x["_wet_wins"] = success_wet
    grouped = x.groupby("horse_id", sort=False)
    x["history_wet_starts"] = grouped["_wet_starts"].cumsum() - x["_wet_starts"]
    previous_wet_wins = grouped["_wet_wins"].cumsum()-x["_wet_wins"]
    x["history_wet_top3_rate"] = (
        (previous_wet_wins+1.0)/(x.history_wet_starts+5.0)
    )
    for name in ("jockey_name","trainer"):
        x["_agent_rate_"+name] = _prior_agent_rate(x, name)
        x["history_last_"+("jockey" if name=="jockey_name" else "trainer")+
          "_prior_rate"] = _lagged_feature(x,"_agent_rate_"+name,periods=1)
    prev_jockey = _lagged_feature(x,"jockey_name",periods=1)
    second_jockey = _lagged_feature(x,"jockey_name",periods=2)
    x["history_jockey_stability"] = prev_jockey.eq(second_jockey).astype(float)
    x.loc[prev_jockey.isna()|second_jockey.isna(),"history_jockey_stability"] = np.nan
    return x


def add_opponent_elo(data: pd.DataFrame, *, rating_scale: float = 400,
                     k_factor: float = 12.0) -> pd.DataFrame:
    """All horse ratings for a date are read BEFORE any race that date updates.

    A race's actual finishing times/ranks can only update following dates.
    """
    x = data.copy()
    x["race_date"] = pd.to_datetime(x.race_date)
    x = x.sort_values(["race_date","race_id","horse_no"], kind="stable").reset_index(drop=True)
    rating = defaultdict(lambda: 1500.0)
    count = defaultdict(int)
    outputs = {
        key: np.full(len(x),np.nan) for key in (
            "strength_elo_prior","strength_elo_vs_field",
            "strength_opp_mean_prior","strength_prior_races",
        )
    }
    ids = x.horse_id.astype(str).to_numpy()
    for _, day in x.groupby("race_date",sort=False):
        # Snapshot ratings at start of date; do not update until all races read.
        updates = defaultdict(float)
        increments = defaultdict(int)
        for race_id, race in day.groupby("race_id",sort=False):
            idx = race.index.to_numpy()
            horses = ids[idx]
            pre = np.array([rating[h] for h in horses],float)
            n = len(idx)
            if n<2:
                continue
            outputs["strength_elo_prior"][idx] = pre
            outputs["strength_elo_vs_field"][idx] = (
                pre - (pre.sum()-pre)/(n-1))
            outputs["strength_opp_mean_prior"][idx] = (pre.sum()-pre)/(n-1)
            outputs["strength_prior_races"][idx] = [count[h] for h in horses]
            rank = pd.to_numeric(race.raw_finish_position,errors="coerce").to_numpy(float)
            valid = np.isfinite(rank)
            if valid.sum()<2:
                continue
            prob = 1.0/(1.0+10.0**((pre[None,:]-pre[:,None])/rating_scale))
            order = (rank[:,None] < rank[None,:]).astype(float)
            ties = rank[:,None] == rank[None,:]
            outcome = order + 0.5*ties.astype(float)
            pairs = valid[:,None] & valid[None,:]
            np.fill_diagonal(pairs,False)
            scales = pairs.sum(axis=1)
            drift = np.divide(
                ((outcome-prob)*pairs).sum(axis=1), scales,
                out=np.zeros(n),where=scales>0)
            for h,delta,ok in zip(horses,drift,valid):
                if ok:
                    updates[h] += k_factor*delta
                    increments[h]+=1
        for h,delta in updates.items():
            rating[h] += delta
            count[h] += increments[h]
    for key, vals in outputs.items():
        x[key] = vals
    x["strength_previous_opponent"] = _lagged_feature(
        x,"strength_opp_mean_prior",periods=1)
    previous_pct = (pd.to_numeric(x.raw_finish_position,errors="coerce")-1) / (
        pd.to_numeric(x.field_size,errors="coerce")-1)
    x["_previous_placing"] = previous_pct
    x["strength_previous_placing"] = _lagged_feature(
        x,"_previous_placing",periods=1)
    return x
