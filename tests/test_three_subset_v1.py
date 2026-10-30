"""Mathematically exact top3 fixed-cardinality probability checks."""
from itertools import combinations

import numpy as np
import pytest
from scipy.special import logsumexp

from keiba_place_lab.three_subset_v1 import (
    subset_log_probability,
    top3_subset_marginals,
)


@pytest.mark.parametrize("n", [3,4,7,18])
def test_all_starter_fixed_three_marginal_sum(n):
    rng=np.random.default_rng(n)
    lw=rng.normal(size=n)
    p=top3_subset_marginals(lw)
    assert np.all((p>=0)&(p<=1))
    assert np.isclose(p.sum(),3,atol=1e-10)
    if n==3:
        np.testing.assert_allclose(p,1,atol=1e-12)
    if n==4 and np.allclose(lw,0):
        np.testing.assert_allclose(p,3/4)


def test_uniform_symmetric_and_extreme_logits():
    np.testing.assert_allclose(
        top3_subset_marginals(np.zeros(8)),np.full(8,3/8),atol=1e-12)
    logits=np.array([1000,500,100,0,-100,-900])
    p=top3_subset_marginals(logits)
    assert np.isclose(p.sum(),3,atol=1e-10)
    assert p[0]>p[-1]


def test_fixed_cardinality_distribution_normalizes_and_matches_marginals():
    w=np.array([-2,0,1,2,3],dtype=float)
    sets=list(combinations(range(len(w)),3))
    logps=np.array([subset_log_probability(w,s) for s in sets])
    assert np.isclose(logsumexp(logps),0,atol=1e-12)
    marginal=top3_subset_marginals(w)
    brute=np.array([sum(np.exp(v) for s,v in zip(sets,logps) if i in s)
                    for i in range(len(w))])
    np.testing.assert_allclose(marginal,brute,atol=1e-12)


def test_weak_inputs_rejected():
    with pytest.raises(ValueError):
        top3_subset_marginals(np.array([0,0]))
    with pytest.raises(ValueError):
        top3_subset_marginals(np.array([0,0,0,np.nan]))
    with pytest.raises(ValueError):
        subset_log_probability(np.zeros(5),[0,0,1])
    with pytest.raises(ValueError):
        subset_log_probability(np.zeros(5),[0,1,7])
