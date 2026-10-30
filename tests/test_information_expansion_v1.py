"""Causal, same-day, raw-source and Elo adversarial checks for information V1."""

import numpy as np
import pandas as pd
import pytest

from keiba_place_lab.information_expansion_v1 import (
    NEW_HISTORY,
    NEW_STRENGTH,
    _parse_3f,
    _parse_mass,
    _parse_passing,
    add_historical_result_features,
    add_opponent_elo,
    raw_observations,
)


def sample_events():
    # Two independent two-runner races on the SAME DATE, then a later date.
    day = ["2023-01-01"] * 4 + ["2023-01-08"] * 2
    return pd.DataFrame({
        "race_date": pd.to_datetime(day),
        "race_id": ["r1","r1","r2","r2","r3","r3"],
        "horse_id": ["a","b","c","d","a","c"],
        "horse_no": [1,2,1,2,1,2],
        "field_size": [2]*6,
        "finish_position": [1,2,1,2,2,1],
        "top3_label": [1,1,1,1,1,1],
        "raw_last3f": [33,35,32,36,34,35],
        "raw_last3f_rel": [1,0,1,0,1,0],
        "raw_passing_gain": [1,-1,1,-1,0,0],
        "raw_early_pct": [0,1,0,1,0,1],
        "raw_mass": [480,490,450,455,477,452],
        "raw_mass_change": [0,0,0,0,-3,2],
        "raw_wet": [0,0,1,1,1,1],
        "jockey_name": ["J1","J2","J3","J4","J9","J3"],
        "trainer": ["T1","T2","T3","T4","T1","T3"],
    })


def test_parse_strictly_no_fabricated_postrace_values():
    assert _parse_passing("9-7-5-2") == (9,2)
    assert all(np.isnan(x) for x in _parse_passing("3-ア-1"))
    assert _parse_mass("456(-9)") == (456,-9)
    assert all(np.isnan(x) for x in _parse_mass("計不"))
    assert pd.isna(_parse_3f(pd.Series(["bad","70.4"]))).all()


def test_historical_features_always_lag_target_observations():
    frame = sample_events()
    baseline = add_historical_result_features(frame)
    later = baseline.loc[baseline.race_id.eq("r3")].set_index("horse_id")
    assert later.loc["a","history_last3f_rel_last"] == 1
    assert later.loc["a","history_mass_last"] == 480
    assert later.loc["c","history_last3f_rel_last"] == 1
    first = baseline.loc[baseline.race_date.eq("2023-01-01")]
    assert first.history_mass_last.isna().all()
    assert first.history_last3f_rel_last.isna().all()
    modified = frame.copy()
    modified.loc[modified.race_id.eq("r3"), "raw_last3f_rel"] = -1000
    modified.loc[modified.race_id.eq("r3"), "raw_mass"] = 99999
    changed = add_historical_result_features(modified)
    pd.testing.assert_frame_equal(
        baseline.loc[:,list(NEW_HISTORY)],
        changed.loc[:,list(NEW_HISTORY)],
    )


def test_all_today_elo_read_before_any_today_result_or_label_change():
    baseline = add_opponent_elo(sample_events())
    assert (baseline.loc[baseline.race_date.eq("2023-01-01"),
                         "strength_elo_prior"] == 1500).all()
    assert (baseline.loc[baseline.race_date.eq("2023-01-01"),
                         "strength_elo_vs_field"] == 0).all()
    future = sample_events()
    future.loc[future.race_date.eq("2023-01-08"),"finish_position"] = [1,2]
    changed = add_opponent_elo(future)
    for col in NEW_STRENGTH:
        np.testing.assert_allclose(
            baseline.loc[baseline.race_date.eq("2023-01-08"),col],
            changed.loc[changed.race_date.eq("2023-01-08"),col],
            equal_nan=True,
        )


def test_elo_stronger_opponents_affect_forward_ability():
    a = sample_events()
    b = a.copy()
    b.loc[b.race_id.eq("r1"),"finish_position"] = [2,1]
    first = add_opponent_elo(a)
    flipped = add_opponent_elo(b)
    v1 = first.loc[first.race_id.eq("r3"),"strength_elo_prior"].to_numpy()
    v2 = flipped.loc[flipped.race_id.eq("r3"),"strength_elo_prior"].to_numpy()
    assert not np.array_equal(v1,v2)


def test_horse_can_not_use_other_race_on_same_calendar_day():
    a=sample_events()
    duplicate=a.iloc[0].copy()
    duplicate["race_id"] = "r2"
    duplicate["horse_no"] = 2
    duplicate["race_date"] = pd.Timestamp("2023-01-01")
    with pytest.raises(ValueError,match="multiple JRA starts"):
        add_historical_result_features(pd.concat([
            a,pd.DataFrame([duplicate])
        ],ignore_index=True))


def test_exact_raw_join_on_official_jra_keys_and_never_source_date(tmp_path):
    full = pd.DataFrame({
        "race_date":pd.to_datetime(["2023-01-01"]*2),
        "race_id":["r1","r1"],"horse_id":["a","b"],
        "horse_no":[1,2],"field_size":[2,2],
    })
    result = pd.DataFrame({
        "race_id":["r1","r1"],"horse_id":["a","b"],
        "rank":["1","2"],"number":["1","2"],
        "jockey_name":["J1","J2"],"trainer":["T1","T2"],
        "last_3f":["33.3","34.9"],"passing":["1-1","2-2"],
        "horse_weight":["450(+0)","計不"],
    })
    conditions = pd.DataFrame({"race_id":["r1"],"track_condition":["稍重"]})
    rp = tmp_path/"result.csv"
    cp = tmp_path/"race.csv"
    result.to_csv(rp,index=False)
    conditions.to_csv(cp,index=False)
    data,audit=raw_observations(full,rp,cp)
    assert audit["matched_raw_runner_rows"]==2
    assert data.race_date.eq(pd.Timestamp("2023-01-01")).all()
    assert data.raw_wet.eq(1).all()
    assert data.loc[data.horse_id.eq("b"),"raw_mass"].isna().all()
    result.loc[0,"number"] = "9"
    result.to_csv(rp,index=False)
    with pytest.raises(ValueError,match="horse number mismatch"):
        raw_observations(full,rp,cp)
