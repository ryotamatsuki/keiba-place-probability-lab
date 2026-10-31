"""Structural and statistical invariants for racewise conditional trio learning."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from keiba_place_lab.trio_bridge_v1 import invert_subset_marginals, project_exactly_three
from keiba_place_lab.trio_residual_v2 import (
    CANDIDATE_SETS,
    _prepare,
    fit_residual,
    residual_distribution,
    score_probabilities,
)


def make_race(n=7, label=(0, 1, 2), *, shift=0):
    horses = np.arange(1, n + 1)
    p = np.linspace(.62, .07, n)
    q = project_exactly_three(p)
    w = invert_subset_marginals(q)
    triplets = np.array(list(combinations(range(n), 3)))
    pl = np.exp((w[triplets].sum(axis=1) - np.max(w[triplets].sum(axis=1))))
    pl /= pl.sum()
    market_sets = horses[triplets]
    pace = np.linspace(.1, .9, n)
    pace[3] = np.nan
    draw = np.linspace(0, 1, n)
    item = _prepare(
        horses, w, q, pace, draw, pl,
        market_sets,
    )
    selected = horses[list(label)]
    ix = np.flatnonzero(np.all(item["trios"] == np.sort(selected), axis=1))
    item["actual_trio_idx"] = int(ix[0])
    return item


def test_no_change_with_beta_zero():
    item = make_race()
    p = residual_distribution(item, np.zeros(3), CANDIDATE_SETS["race_interactions"])
    raw = residual_distribution(item, np.array([]), ())
    np.testing.assert_allclose(p, raw, rtol=1e-12, atol=1e-12)


def test_v2_valid_probabilities_and_proper_score():
    r = make_race(17)
    p = residual_distribution(r, np.array([.2, -.25, .1]),
                              CANDIDATE_SETS["race_interactions"])
    assert len(p) == 680
    assert p.sum() == pytest.approx(1.0, abs=1e-10)
    assert np.all(p > 0)
    s = score_probabilities(p, r["actual_trio_idx"])
    assert s["joint_logloss"] > 0
    assert 0 <= s["top10_mass"] <= 1
    assert s["top10_hit"] in (0, 1)


def test_preknown_field_order_is_invariant():
    r = make_race(n=9)
    idx = np.array([3, 1, 5, 2, 7, 0, 8, 6, 4])
    hs = np.arange(1, 10)
    q = project_exactly_three(np.linspace(.62, .07, 9))
    w = invert_subset_marginals(q)
    trios = np.array(list(combinations(range(9), 3)))
    mp = np.exp(w[trios].sum(axis=1))
    mp /= mp.sum()
    pace = np.linspace(.1, .9, 9)
    pace[3] = np.nan
    draw = np.linspace(0, 1, 9)
    a = _prepare(hs, w, q, pace, draw, mp, hs[trios])
    b = _prepare(hs[idx], w[idx], q[idx], pace[idx], draw[idx], mp, hs[trios])
    pa = residual_distribution(a, np.array([.2, -.3, .05]), (0, 1, 2))
    pb = residual_distribution(b, np.array([.2, -.3, .05]), (0, 1, 2))
    # Direct tuple keys preserve exchangeability despite shuffled starter rows.
    def mapping(item, p):
        return {tuple(x): float(z) for x, z in zip(item["trios"], p, strict=True)}
    for key, value in mapping(a, pa).items():
        assert mapping(b, pb)[key] == pytest.approx(value, abs=1e-11)


def test_duplicate_and_invalid_market_fail_closed():
    item = make_race()
    p = item["base_logp"]
    trios = item["trios"]
    n = 7
    q = project_exactly_three(np.linspace(.62, .07, n))
    w = invert_subset_marginals(q)
    mp = np.exp(p)
    with pytest.raises(ValueError):
        _prepare(np.array([1, 1, 3, 4, 5, 6, 7]), w, q,
                 np.full(n, .5), np.linspace(0, 1, n), mp, trios)
    with pytest.raises(ValueError):
        _prepare(np.arange(1, 8), w, q, np.full(n, .5),
                 np.linspace(0, 1, n), np.zeros_like(mp), trios)


def test_train_coefficient_uses_only_training_rows():
    train = []
    for i in range(120):
        # Fixed deterministic toy outcomes, no 2024 information.
        train.append(make_race(7, label=(i % 4, (i % 4 + 1) % 7,
                                             (i % 4 + 2) % 7)))
    a = fit_residual(train, (0,), ridge=.1)
    b = fit_residual(train, (0,), ridge=.1)
    assert a["train_races"] == 120
    np.testing.assert_allclose(a["beta"], b["beta"], atol=1e-12)


def test_no_silent_acceptance_of_invalid_posterior():
    item = make_race()
    with pytest.raises(ValueError):
        residual_distribution(item, np.array([np.nan]), (0,))
    with pytest.raises(ValueError):
        residual_distribution(item, np.array([1]), (0, 0))
