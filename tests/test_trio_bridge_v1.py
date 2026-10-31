"""F3-derived exact unordered trio distribution: invariants and leakage gates."""
from __future__ import annotations

import math

import numpy as np
import pytest

from keiba_place_lab.three_subset_v1 import top3_subset_marginals
from keiba_place_lab.trio_bridge_v1 import (
    historical_trio_temperature,
    invert_subset_marginals,
    plackett_luce_trio,
    project_exactly_three,
    trio_distribution,
)


@pytest.mark.parametrize("n", [4, 5, 10, 17])
def test_distribution_valid_and_inverts_provided_marginals(n):
    source = np.linspace(-2, 2, n) + .3 * np.sin(np.arange(n))
    p = top3_subset_marginals(source)
    recovered = invert_subset_marginals(p)
    assert np.max(np.abs(top3_subset_marginals(recovered) - p)) < 3e-5
    d = trio_distribution(recovered, np.arange(1, n + 1))
    assert len(d) == math.comb(n, 3)
    assert d.probability.sum() == pytest.approx(1.0, abs=1e-11)
    assert d.probability.ge(0).all()
    # Every joint marginal must agree with the requested input probabilities.
    for i in range(1, n + 1):
        selected = d[["horse_1", "horse_2", "horse_3"]].eq(i).any(axis=1)
        assert d.loc[selected, "probability"].sum() == pytest.approx(p[i-1], abs=3e-5)


def test_logit_projection_enforces_three_and_does_not_mutate_input():
    a = np.array([.67, .53, .3, .22, .12, .07, .01])
    copy = a.copy()
    q = project_exactly_three(a)
    assert a.tolist() == copy.tolist()
    assert q.sum() == pytest.approx(3.0, abs=1e-10)
    assert np.all((q > 0) & (q < 1))
    assert np.array_equal(project_exactly_three(q), q)


def test_permutation_invariance_of_trio_assignments():
    p = top3_subset_marginals(np.linspace(-1, 1, 8))
    horses = np.arange(1, 9)
    w = invert_subset_marginals(p)
    d = trio_distribution(w, horses)
    permutation = [3, 4, 1, 0, 2, 7, 6, 5]
    perm = trio_distribution(w[permutation], horses[permutation])
    def ordered(t):
        return {
            frozenset((int(row.horse_1), int(row.horse_2), int(row.horse_3))):
            float(row.probability)
            for row in t.itertuples()
        }
    a, b = ordered(d), ordered(perm)
    assert a.keys() == b.keys()
    assert max(abs(a[k] - b[k]) for k in a) < 1e-10


def test_uniform_joint_and_pl_baselines():
    h = np.arange(1, 18)
    q = np.full(17, 3 / 17)
    d = trio_distribution(invert_subset_marginals(q), h)
    pl = plackett_luce_trio(np.full(17, 1 / 17), h)
    assert len(d) == len(pl) == 680
    assert d.probability.std() < 1e-9
    assert pl.probability.std() < 1e-9
    assert np.allclose(d.probability, pl.probability)


def test_temperature_does_not_claim_marginal_preservation():
    horses = np.arange(1, 8)
    w = np.arange(7, dtype=float) / 3
    d = trio_distribution(w, horses, temperature=1.6)
    assert d.probability.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("data", [
    [0.0, .2, .3, .4], [1.0, .2, .3, .4],
    [.1, float("nan"), .3, .4], [.5, .5, .5],
])
def test_bad_marginals_fail_closed(data):
    with pytest.raises(ValueError):
        project_exactly_three(np.asarray(data))


def test_joint_histogram_calibration_needs_prior_races():
    with pytest.raises(ValueError, match="Too few"):
        historical_trio_temperature([np.ones(4)] * 3,
                                    [(0, 1, 2)] * 3)


def test_2026_preview_must_contain_exactly_17_horses():
    from keiba_place_lab.trio_bridge_v1 import validate_probability_field
    with pytest.raises(ValueError):
        validate_probability_field(np.ones(2))
