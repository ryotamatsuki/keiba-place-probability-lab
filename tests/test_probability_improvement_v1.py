"""Fail-closed tests for preregistered probability-improvement primitives."""

import numpy as np
import pandas as pd
import pytest

from keiba_place_lab.market import harville_top_k_probabilities
from keiba_place_lab.probability_improvement_v1 import (
    SNAPSHOT_COLUMNS,
    corrected_harville_top3,
    euclidean_top3_projection,
    fit_logit_stack,
    fit_market_offset,
    predict_market_offset,
    select_snapshot_at,
)


@pytest.mark.parametrize("n", [4, 8, 18])
def test_a2_matches_harville_at_unit_exponents(n):
    q = np.arange(1, n + 1, dtype=float)
    q /= q.sum()
    expected = harville_top_k_probabilities(q, k=3)
    actual = corrected_harville_top3(q, gamma=1, delta=1)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    assert np.isclose(actual.sum(), 3, atol=1e-10)


@pytest.mark.parametrize("gamma,delta", [(0.75, 1.25), (1.25, 0.75), (1, 1)])
def test_a2_full_field_probabilities(gamma, delta):
    q = np.array([0.6, 0.19, 0.08, 0.07, 0.04, 0.02])
    p = corrected_harville_top3(q, gamma, delta)
    assert np.isfinite(p).all()
    assert np.all((p >= 0) & (p <= 1))
    assert np.isclose(p.sum(), 3, atol=1e-10)


@pytest.mark.parametrize("invalid", [[0.4, 0.6], [0.4, 0.4, 0.1, 0.1],
                                           [1.0, 0, 0, 0]])
def test_a2_rejects_bad_q(invalid):
    with pytest.raises(ValueError):
        corrected_harville_top3(np.asarray(invalid))


def test_projection_is_not_proportional_and_nonexpansive_for_full_field():
    raw = np.array([0.95, 0.60, 0.43, 0.25, 0.08])
    proj = euclidean_top3_projection(raw)
    assert np.isclose(proj.sum(), 3, atol=1e-10)
    assert np.all((proj >= 0) & (proj <= 1))
    assert not np.allclose(proj, raw * (3/raw.sum()))
    y = np.array([1, 1, 1, 0, 0], dtype=float)
    assert np.sum((proj-y)**2) <= np.sum((raw-y)**2) + 1e-12
    np.testing.assert_allclose(euclidean_top3_projection(proj), proj, atol=1e-10)


def test_projection_subset_not_guaranteed():
    raw = np.array([0.7, 0.7, 0.7, 0.7, 0.1])
    y = np.array([1, 1, 1, 0, 0], dtype=float)
    proj = euclidean_top3_projection(raw)
    assert np.square(proj-y).sum() <= np.square(raw-y).sum() + 1e-12
    # A restricted subset of rows has no universal non-increase theorem.
    assert np.square(proj[3]-y[3]) > np.square(raw[3]-y[3])


def test_offset_zero_parameters_equal_market_and_fit_is_finite():
    rng = np.random.default_rng(43)
    p = rng.uniform(0.2, 0.75, size=100)
    X = rng.normal(size=(100, 5))
    y = rng.integers(0, 2, size=100)
    dates = np.array(["2023-01-01"] * 100)
    fit = fit_market_offset(p, X, y, dates, max_training_date="2024-01-01")
    assert np.isfinite(fit.coefficients).all()
    from dataclasses import replace
    zero = replace(fit, coefficients=np.zeros_like(fit.coefficients))
    np.testing.assert_allclose(predict_market_offset(p, X, zero), p, atol=1e-12)
    pred = predict_market_offset(p, X, fit)
    assert np.all((pred >= 0) & (pred <= 1))


def test_meta_training_and_offset_reject_outer_labels():
    p = np.tile([0.2, 0.5, 0.7, 0.35], 4)
    X = np.arange(len(p)*5, dtype=float).reshape(len(p), 5)
    y = np.tile([0, 1, 0, 1], 4)
    dates = np.array(["2024-01-01"] * len(p))
    with pytest.raises(ValueError, match="evaluation date"):
        fit_market_offset(p, X, y, dates, max_training_date="2024-01-01")
    with pytest.raises(ValueError, match="outer fold"):
        fit_logit_stack(p, p, y, dates, max_training_date="2024-01-01")


def snapshot(captured, provider, *, odds=2.1, horse="a", snap="s1", bet="win"):
    return {"race_id": "r1", "horse_id": horse, "bet_type": bet, "odds": odds,
            "place_odds_low": 1.2 if bet == "place" else float("nan"),
            "place_odds_high": 1.6 if bet == "place" else float("nan"),
            "provider_updated_at": provider,
            "captured_at": captured, "scheduled_off_at": "2026-11-01T15:40:00+09:00",
            "source": "licensed_export", "snapshot_id": snap}


def test_checkpoint_uses_only_prior_fresh_complete_snapshot():
    d = pd.DataFrame([
        snapshot("2026-11-01T15:29:00+09:00",
                 "2026-11-01T15:28:00+09:00", horse="a"),
        snapshot("2026-11-01T15:29:00+09:00",
                 "2026-11-01T15:28:00+09:00", horse="b"),
        snapshot("2026-11-01T15:31:00+09:00",
                 "2026-11-01T15:31:00+09:00", horse="a", snap="after"),
        snapshot("2026-11-01T15:31:00+09:00",
                 "2026-11-01T15:31:00+09:00", horse="b", snap="after"),
    ])
    assert set(SNAPSHOT_COLUMNS).issubset(d.columns)
    result = select_snapshot_at(d, minutes_before=10, expected_horse_ids={"a", "b"})
    assert set(result.snapshot_id) == {"s1"}
    with pytest.raises(ValueError, match="No complete"):
        select_snapshot_at(d, minutes_before=30, expected_horse_ids={"a", "b"})


def test_timestamp_forgery_and_naive_timestamps_fail():
    d = pd.DataFrame([snapshot("2026-11-01T15:29:00+09:00",
                                "2026-11-01T15:31:00+09:00")])
    with pytest.raises(ValueError, match="later than capture"):
        select_snapshot_at(d, minutes_before=10, expected_horse_ids={"a"})
    d.loc[0, "provider_updated_at"] = "2026-11-01T15:28:00"
    with pytest.raises(ValueError, match="UTC offset"):
        select_snapshot_at(d, minutes_before=10, expected_horse_ids={"a"})


def test_missing_or_duplicate_runner_invalidates_snapshot():
    d = pd.DataFrame([snapshot("2026-11-01T15:29:00+09:00",
                                "2026-11-01T15:28:00+09:00")])
    with pytest.raises(ValueError, match="No complete"):
        select_snapshot_at(d, minutes_before=10, expected_horse_ids={"a", "b"})
    with pytest.raises(ValueError, match="Duplicate"):
        select_snapshot_at(pd.concat([d, d]), minutes_before=10, expected_horse_ids={"a"})


def test_original_historical_pipeline_unchanged():
    from keiba_place_lab.ensemble import convex_blend
    market = np.array([0.1, 0.5, 0.8])
    nonmarket = np.array([0.2, 0.4, 0.7])
    np.testing.assert_array_equal(convex_blend(market, nonmarket, 0.0), market)
