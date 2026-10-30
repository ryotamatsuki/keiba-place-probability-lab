"""All-starter market structure and prior-only relative-ability features.

Market snapshots are from the same historical FINAL win-odds domain as PR32.
Never imply those final odds were observed at the T−10 prediction lock.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from keiba_place_lab.stage4_successor_phase3a import (
    RELATIVE_FEATURES,
    SOURCE_RULES,
    _leave_one_out_mean,
    assert_full_field_context,
)

MARKET_SPLITS = ("market_calibrated_top3_p", "market_calibrated_top3_logit")
MARKET_FIELD = (
    "market_win_probability",
    "market_win_rank_pct",
    "market_win_hhi",
    "market_top2_win_gap",
    "market_favorite_win_share",
)
FEATURE_NAMES = (*MARKET_SPLITS, *RELATIVE_FEATURES, *MARKET_FIELD)


def full_starter_relative(
    eligible: pd.DataFrame, full_context: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use exact Stage4 source rules, on all actual starters, no fitted data."""
    keys = ["race_id", "horse_id"]
    eligible = eligible.copy()
    full_context = full_context.copy()
    for d in (eligible, full_context):
        for key in keys:
            d[key] = d[key].astype(str)
        if d.duplicated(keys).any():
            raise ValueError("Duplicate race-horse keys")
    wanted = set(eligible.race_id)
    context = full_context.loc[full_context.race_id.isin(wanted)].copy()
    if set(context.race_id) != wanted:
        raise ValueError("Missing full-field race context")
    assert_full_field_context(context)
    if not set(eligible.set_index(keys).index).issubset(
        set(context.set_index(keys).index)
    ):
        raise ValueError("Eligible horse absent from full starter field")
    relative = context[keys].copy()
    coverage = []
    for source, out, direction in SOURCE_RULES:
        if source not in context:
            raise ValueError(f"Missing prior-only source {source}")
        own = pd.to_numeric(context[source], errors="coerce")
        peer = _leave_one_out_mean(context, source)
        relative[out] = own - peer if direction > 0 else peer - own
        coverage.append({
            "feature":out,"full_field_rows":len(context),
            "full_field_nonmissing":int(relative[out].notna().sum()),
        })
    enriched = eligible.merge(relative, on=keys, how="left", validate="one_to_one",
                              sort=False, indicator=True)
    if not enriched["_merge"].eq("both").all() or len(enriched) != len(eligible):
        raise ValueError("Relative feature join fails exact eligible rows")
    return enriched.drop(columns="_merge"), pd.DataFrame(coverage)


def full_starter_market_context(market: pd.DataFrame) -> pd.DataFrame:
    """Every context statistic uses *all* available starter win-odds shares."""
    needed = {"race_id","horse_no","market_p_win","market_p_top3","market_field_size"}
    if needed.difference(market.columns):
        raise ValueError(f"Market incomplete columns: {sorted(needed-set(market))}")
    x = market[sorted(needed)].copy()
    x["race_id"] = x.race_id.astype(str)
    if x.duplicated(["race_id","horse_no"]).any():
        raise ValueError("Duplicate market starters")
    group = x.groupby("race_id", sort=False)
    n = group.horse_no.transform("size").to_numpy(float)
    expected = x.market_field_size.to_numpy(float)
    if not np.array_equal(n, expected):
        raise ValueError("Full-field odds missing at least one starter")
    p=x.market_p_win.to_numpy(float)
    if not np.isfinite(p).all() or ((p <= 0) | (p >= 1)).any():
        raise ValueError("Invalid all-starter market share")
    race_totals=group.market_p_win.transform("sum").to_numpy(float)
    if not np.allclose(race_totals, 1, atol=1e-8):
        raise ValueError("Market shares do not sum to one")
    if not np.allclose(group.market_p_top3.transform("sum"), 3, atol=1e-8):
        raise ValueError("Full-field Harville probabilities must sum to three")
    x["market_win_probability"] = p
    rank = group.market_p_win.rank(method="average",ascending=False).to_numpy(float)
    x["market_win_rank_pct"] = (rank - 1) / (n - 1)
    x["_p_squared"] = p*p
    x["market_win_hhi"] = x.groupby("race_id")["_p_squared"].transform("sum")
    # Two largest win shares from a complete race. Never use results/rank.
    def favorites(s: pd.Series) -> tuple[float,float]:
        a = np.sort(s.to_numpy(float))
        return float(a[-1]), float(a[-1]-a[-2])
    top = group.market_p_win.agg(favorites)
    top = pd.DataFrame(top.tolist(), index=top.index,
                       columns=["market_favorite_win_share","market_top2_win_gap"])
    x = x.join(top, on="race_id", validate="many_to_one")
    if x[list(MARKET_FIELD)].isna().any().any():
        raise ValueError("Market context created non-finite features")
    return x[["race_id","horse_no",*MARKET_FIELD]].copy()


def attach_information(
    eligible: pd.DataFrame,
    full_field: pd.DataFrame,
    market_full: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x, coverage = full_starter_relative(eligible, full_field)
    odds = full_starter_market_context(market_full)
    x["race_id"] = x.race_id.astype(str)
    final = x.merge(
        odds,on=["race_id","horse_no"],how="left",validate="one_to_one",
        sort=False,indicator=True,
    )
    if not final["_merge"].eq("both").all() or len(final)!=len(eligible):
        raise ValueError("Some eligible runners lack all-starter odds context")
    return final.drop(columns="_merge"),coverage


def add_calibrated_market_splits(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply OOF calibrated market only after chronological calibration."""
    if "p_market_oof" not in frame:
        raise ValueError("Chronological OOF market probability not fitted yet")
    x=frame.copy()
    p=pd.to_numeric(x.p_market_oof,errors="coerce").to_numpy(
        dtype=float,na_value=np.nan,
    )
    good=np.isfinite(p) & (p>0) & (p<1)
    x["market_calibrated_top3_p"]=p
    x["market_calibrated_top3_logit"]=np.where(
        good, np.log(np.clip(p,1e-12,1)) -
        np.log1p(-np.clip(p,0,1-1e-12)), np.nan
    )
    return x
