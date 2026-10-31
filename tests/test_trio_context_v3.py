"""Conditional-trio V3 invariants; uses synthetic races, not leaked outcomes."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from keiba_place_lab.trio_bridge_v1 import invert_subset_marginals, project_exactly_three
from keiba_place_lab.trio_context_v3 import (
    CANDIDATES,
    contextual_features,
    fit_contextual,
    joint_probability,
)


def race(n=9, front_count=3, actual=(1, 2, 3), *, surface="turf", distance=2000):
    horse_numbers = np.arange(1, n+1)
    marginal = project_exactly_three(np.linspace(.59,.08,n))
    strength = invert_subset_marginals(marginal)
    triplets = np.array(list(combinations(horse_numbers, 3)))
    raw = strength[triplets-1].sum(axis=1)
    market = np.exp(raw - np.max(raw))
    market /= market.sum()
    pace = np.linspace(.1,.9,n)
    pace[3:] = np.linspace(.32,.95,n-3)
    pace[3:3+max(0,front_count-3)] = .19
    item = contextual_features(
        horse_numbers=horse_numbers, logits=strength, projected_top3=marginal,
        recent_early_pct=pace, draw_pct=np.linspace(0,1,n),
        market_joint=market, market_trios=triplets,
        surface=surface, distance_m=distance,
    )
    item["actual_trio_idx"] = int(np.flatnonzero(
        np.all(item["trios"]==np.asarray(actual),axis=1)
    )[0])
    return item


def test_all_combinations_and_probability_conservation():
    e=race(17)
    assert e["context_matrix"].shape==(680,9)
    for included in CANDIDATES.values():
        probability=joint_probability(e,np.zeros(len(included)),included)
        assert len(probability)==680
        assert probability.sum()==pytest.approx(1)
        assert np.all(probability>0)


def test_adding_race_context_must_change_trio_distribution():
    a=race(9,surface="turf",distance=1200)
    b=race(9,surface="dirt",distance=2400)
    beta=np.array([.05,.1,-.15,.23,.2,.35,.3,.25,.18])
    p=joint_probability(a,beta,CANDIDATES["v3_context"])
    q=joint_probability(b,beta,CANDIDATES["v3_context"])
    assert not np.allclose(p,q)
    assert np.isclose(p.sum(),1) and np.isclose(q.sum(),1)


def test_fit_is_reproducible_and_nonzero_coefficients():
    training=[race(actual=(1+(i%3),3+(i%3),6)) for i in range(105)]
    fit_a=fit_contextual(training,CANDIDATES["v2_matched"])
    fit_b=fit_contextual(training,CANDIDATES["v2_matched"])
    assert fit_a["training_races"]==105
    np.testing.assert_allclose(fit_a["coefficients"],fit_b["coefficients"],atol=1e-10)
    assert len(fit_a["coefficients"])==3


def test_invalid_context_or_coefficients_fail_closed():
    with pytest.raises(ValueError):
        race(surface="not-a-surface")
    with pytest.raises(ValueError):
        joint_probability(race(),np.array([np.nan]),(0,))


def test_all_context_features_centered_per_race():
    f=race(13)
    np.testing.assert_allclose(f["context_matrix"].mean(axis=0),0,atol=1e-12)
