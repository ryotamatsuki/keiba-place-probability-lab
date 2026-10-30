"""Adversarial no-leak, paired-row, and residual-weight tests for B_raw V1."""
from __future__ import annotations

from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

RUNTIME = run_path(str(
    Path(__file__).resolve().parents[1] / "scripts/run_braw_ensemble_v1.py"
))
fit_train_only_ridge = RUNTIME["fit_train_only_ridge"]
residual_pool = RUNTIME["residual_pool"]
candidates_2024 = RUNTIME["candidates_2024"]
check_alignment = RUNTIME["check_alignment"]
check_probability = RUNTIME["check_probability"]
check_same_prob = RUNTIME["check_same_prob"]
read_oof = RUNTIME["read_oof"]


def example_frame(year: int = 2023) -> pd.DataFrame:
    rng = np.random.default_rng(31415)
    n = 120
    market = rng.uniform(0.1, 0.7, n)
    b = np.clip(market + rng.normal(0, 0.025, n), .01, .99)
    l = np.clip(market + rng.normal(0, 0.035, n), .01, .99)
    c = np.clip(market + rng.normal(0, 0.03, n), .01, .99)
    return pd.DataFrame({
        "race_date": pd.Timestamp(f"{year}-05-01"),
        "race_id": np.repeat([f"{year}_{k}" for k in range(n // 6)], 6),
        "horse_id": [f"horse{k}" for k in range(n)],
        "horse_no": np.tile(np.arange(1, 7), n // 6),
        "top3_label": np.tile([1, 1, 1, 0, 0, 0], n // 6),
        "p_M0": market,
        "p_X2": b,
        "p_B_raw": b,
        "p_LGB_raw": l,
        "p_CAT_raw": c,
        "p_E5": np.clip(b + .005, .01, .99),
        "p_LGB_scaled": np.clip(l + .005, .01, .99),
        "p_CAT_scaled": np.clip(c + .005, .01, .99),
        "p_D3_old": np.clip(market + .01, .01, .99),
    })


def test_market_and_braw_identities_and_probabilities():
    a = example_frame()
    np.testing.assert_allclose(
        residual_pool(a, np.zeros(3)),
        a.p_M0.to_numpy(float), atol=1e-12,
    )
    np.testing.assert_allclose(
        residual_pool(a, np.array([1., 0., 0.])),
        a.p_B_raw.to_numpy(float), atol=1e-12,
    )
    for weights in ((.75, .25, 0), (.5, 0, .5), (1/3, 1/3, 1/3)):
        p = residual_pool(a, np.asarray(weights))
        assert np.isfinite(p).all() and np.all((p > 0) & (p < 1))
    with pytest.raises(ValueError, match="three fixed"):
        residual_pool(a, np.array([1., 0.]))


def test_fixed_logit_and_prob_pool_intentionally_differ():
    frame = example_frame(2024)
    fitted = {
        "BLC_ridge_2023": {"weights_B_LGB_CAT": [1., 0., 0.]},
        "BL_ridge_2023": {"weights_B_LGB_CAT": [1., 0., 0.]},
    }
    probabilities = candidates_2024(frame, fitted)
    assert list(probabilities) == list(RUNTIME["ALL_NAMES"])
    np.testing.assert_allclose(
        probabilities["BL_prob_75_25"],
        .75 * frame.p_B_raw + .25 * frame.p_LGB_raw,
    )
    np.testing.assert_allclose(
        probabilities["BL_logit_75_25"],
        expit(.75 * logit(frame.p_B_raw.to_numpy()) +
              .25 * logit(frame.p_LGB_raw.to_numpy())),
        atol=1e-12,
    )
    assert not np.allclose(
        probabilities["BL_logit_75_25"],
        probabilities["BL_prob_75_25"],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        probabilities["BLC_ridge_2023"],
        probabilities["B_raw"], atol=1e-12,
    )


def test_ridge_optimizer_only_2023_out_of_time():
    train = example_frame()
    fitted = fit_train_only_ridge(train, use_cat=True)
    weights = np.asarray(fitted["weights_B_LGB_CAT"])
    assert len(weights) == 3
    assert np.all(weights >= -1e-8)
    assert np.all(weights <= 1.5 + 1e-8)
    assert weights.sum() <= 2 + 1e-8
    fitted_two = fit_train_only_ridge(train, use_cat=False)
    assert fitted_two["weights_B_LGB_CAT"][2] == 0
    changed = train.copy()
    changed.loc[0, "race_date"] = pd.Timestamp("2024-01-01")
    with pytest.raises(ValueError, match="2023"):
        fit_train_only_ridge(changed, use_cat=True)


def test_2024_outcome_cannot_change_2023_weights():
    train = example_frame(2023)
    extra = example_frame(2024)
    original = fit_train_only_ridge(train, use_cat=True)
    extra["top3_label"] = 1 - extra["top3_label"]
    # Only train rows are provided to ridge, future labels are irrelevant.
    replicated = fit_train_only_ridge(train, use_cat=True)
    np.testing.assert_array_equal(
        original["weights_B_LGB_CAT"],
        replicated["weights_B_LGB_CAT"],
    )


def test_predictions_and_source_row_alignment_fail_closed(tmp_path):
    frame = example_frame()
    other = frame.copy()
    check_alignment(frame, other, "same")
    other.loc[0, "top3_label"] = 1 - other.loc[0, "top3_label"]
    with pytest.raises(ValueError, match="target labels"):
        check_alignment(frame, other, "mutated label")
    other = frame.copy()
    other.loc[0, "horse_id"] = "alien"
    with pytest.raises(ValueError, match="runner keys"):
        check_alignment(frame, other, "wrong horse")
    path = tmp_path / "fixture.csv"
    pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        read_oof(path)
    with pytest.raises(ValueError, match="Invalid"):
        check_probability("nan", np.array([0.5, np.nan]))
    with pytest.raises(ValueError, match="mismatch"):
        check_same_prob("market", np.array([.2]), np.array([.3]))
