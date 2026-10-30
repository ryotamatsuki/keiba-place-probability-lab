"""Frozen grids and no-future-label invariants for 60..1600-round study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

MODULE = run_path(str(
    Path(__file__).resolve().parents[1] / "scripts/run_round_sweep_1600_v1.py"
))
checkpoints = MODULE["CHECKPOINTS"]
choose_rounds = MODULE["choose_rounds"]
fit_alpha = MODULE["fit_alpha"]
fit_predict_prefixes = MODULE["fit_predict_prefixes"]


def fake_metrics() -> pd.DataFrame:
    rows = []
    for year in (2021, 2022, 2023):
        for n in checkpoints:
            rows.append({
                "year": year, "rounds": n, "races": 100,
                "race_macro_brier": 0.13 + abs(n - 400) / 100000 + (
                    -0.12 if n == 1600 and year == 2023 else 0
                ),
            })
    return pd.DataFrame(rows)


def fake_oof() -> pd.DataFrame:
    rng = np.random.default_rng(2040)
    n = 100
    rows = []
    for year in (2021, 2022, 2023):
        for i in range(n):
            m = rng.uniform(0.1, 0.7)
            row = {
                "race_date": pd.Timestamp(f"{year}-01-15"),
                "race_id": f"r{year}_{i // 5}",
                "horse_id": f"h{year}_{i}", "horse_no": i % 5 + 1,
                "top3_label": int(i % 5 < 3),
                "p_M0": m,
            }
            for rounds in checkpoints:
                row[f"p_{rounds}"] = float(np.clip(
                    m + rng.normal(0, 0.04), 0.01, 0.99,
                ))
            rows.append(row)
    return pd.DataFrame(rows)


def test_registered_grid_and_chronological_round_selection():
    assert checkpoints == (60, 120, 240, 400, 800, 1600)
    metrics = fake_metrics()
    p23, table23 = choose_rounds(metrics, 2023)
    p24, table24 = choose_rounds(metrics, 2024)
    assert p23 == 400
    assert p24 == 1600
    assert len(table23) == len(table24) == 6
    changed = metrics.copy()
    changed.loc[changed.year.eq(2023), "race_macro_brier"] = 99.0
    assert choose_rounds(changed, 2023)[0] == p23
    with pytest.raises(ValueError, match="Incomplete"):
        choose_rounds(metrics.loc[metrics.rounds.ne(1600)], 2023)
    with pytest.raises(ValueError, match="outer"):
        choose_rounds(metrics, 2025)


def test_no_future_alpha_data_and_explicit_identity():
    df = fake_oof()
    early = df.loc[df.race_date.dt.year.lt(2023)].copy()
    a23 = fit_alpha(early, 400, 2023)["alpha"]
    late = df.copy()
    late.loc[late.race_date.dt.year.eq(2023), "top3_label"] = 0
    assert fit_alpha(
        late.loc[late.race_date.dt.year.lt(2023)], 400, 2023,
    )["alpha"] == a23
    with pytest.raises(ValueError, match="years"):
        fit_alpha(late, 400, 2023)
    with pytest.raises(ValueError):
        fit_alpha(early, 123, 2023)
    m = early.p_M0.to_numpy(float)
    z = early.p_400.to_numpy(float)
    np.testing.assert_allclose(
        expit(logit(m) + 0 * (logit(z) - logit(m))), m, atol=1e-12,
    )
    np.testing.assert_allclose(
        expit(logit(m) + 1 * (logit(z) - logit(m))), z, atol=1e-12,
    )


def test_source_sha_identity_is_deterministic(tmp_path):
    payload = {"features": ["f"], "dataset_sha": "abc", "year": 2021}
    a = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    b = hashlib.sha256(json.dumps(
        dict(reversed(list(payload.items()))), sort_keys=True,
    ).encode()).hexdigest()
    assert a == b
    frame = fake_oof()
    path = tmp_path / "oof.csv"
    frame.to_csv(path, index=False)
    reread = pd.read_csv(path, dtype={"race_id": str, "horse_id": str})
    assert len(reread) == len(frame)
    assert np.allclose(reread.p_400.to_numpy(float), frame.p_400)


def test_no_future_model_fitting():
    tr = fake_oof().loc[lambda df: df.race_date.dt.year.eq(2022)].copy()
    with pytest.raises(ValueError, match="chronology"):
        fit_predict_prefixes(tr, tr, "xgb", (60,))
    target = fake_oof().loc[lambda df: df.race_date.dt.year.eq(2023)].copy()
    with pytest.raises(ValueError, match="Unregistered"):
        fit_predict_prefixes(tr, target, "nope", (60,))


def test_real_xgb_prefix_matches_fresh_60_rounds_on_synthetic_f3():
    from run_input_learning_a_v1 import feature_registry

    numeric, categorical = feature_registry()["F3_market_field"]
    rng = np.random.default_rng(401)
    # Keep realistic variance and >8 starters; race-wise weights still equal.
    n = 480
    features = {
        "race_date": pd.Timestamp("2021-06-01"),
        "race_id": [f"r{i // 12}" for i in range(n)],
        "horse_id": [f"h{i}" for i in range(n)],
        "horse_no": np.tile(np.arange(1, 13), 40),
        "top3_label": np.tile([1, 1, 1] + [0]*9, 40),
        "p_market_oof": np.tile(np.linspace(0.08, 0.6, 12), 40),
    }
    for col in numeric:
        features[col] = rng.normal(size=n)
    for col in categorical:
        features[col] = rng.choice(["A", "B"], size=n)
    train = pd.DataFrame(features).sort_values(
        ["race_date", "race_id", "horse_id", "horse_no"],
    ).reset_index(drop=True)
    target = train.iloc[:24].copy()
    target["race_date"] = pd.Timestamp("2022-06-01")
    p, _ = fit_predict_prefixes(train, target, "xgb", (60, 120))
    q, _ = fit_predict_prefixes(train, target, "xgb", (60,))
    np.testing.assert_allclose(p[60], q[60], rtol=0, atol=1e-9)


@pytest.mark.parametrize("family", ["lgb", "cat"])
def test_optional_family_offset_prefix_matches_short_fresh(family):
    pytest.importorskip("lightgbm" if family == "lgb" else "catboost")
    from run_input_learning_a_v1 import feature_registry

    numeric, categorical = feature_registry()["F3_market_field"]
    rng = np.random.default_rng(99)
    n = 160
    data = {
        "race_date": pd.Timestamp("2021-06-01"),
        "race_id": [f"r{i // 8}" for i in range(n)],
        "horse_id": [f"h{i}" for i in range(n)],
        "horse_no": np.tile(np.arange(1, 9), 20),
        "top3_label": np.tile([1, 1, 1, 0, 0, 0, 0, 0], 20),
        "p_market_oof": rng.uniform(0.15, 0.65, size=n),
    }
    for col in numeric:
        data[col] = rng.normal(size=n)
    for col in categorical:
        data[col] = rng.choice(["A", "B"], size=n)
    train = pd.DataFrame(data).sort_values(
        ["race_date", "race_id", "horse_id", "horse_no"],
    ).reset_index(drop=True)
    target = train.iloc[:16].copy()
    target.race_date = pd.Timestamp("2022-06-01")
    long, _ = fit_predict_prefixes(train, target, family, (60, 120))
    short, _ = fit_predict_prefixes(train, target, family, (60,))
    np.testing.assert_allclose(long[60], short[60], atol=1e-7, rtol=0)
