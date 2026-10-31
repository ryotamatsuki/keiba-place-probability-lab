"""Covariate rank model invariants for every unordered top3 ticket."""
from __future__ import annotations

import numpy as np
import pytest

from keiba_place_lab.trio_covariate_rank_v4 import (
    fit_rank_model, ordered_top3_probability, preknown_covariates,
    unordered_top3_distribution,
)


def sample(n=9):
    p=np.linspace(.13,.7,n)
    x=preknown_covariates(p,p[::-1],np.linspace(.11,.8,n))
    return {"x":x,"order":(0,1,2)}


@pytest.mark.parametrize("family,theta",[
    ("pl",[.7,.2,.1]),
    ("fixed_gpl",[.7,.2,.1]),
    ("learned_gpl",[.7,.2,.1,np.log(.8),np.log(.6)]),
])
def test_unordered_sum_of_six_correct(family,theta):
    race=sample(17)
    index,p=unordered_top3_distribution(race["x"],np.asarray(theta),family)
    assert index.shape==(680,3)
    assert p.shape==(680,) and np.all(p>0) and p.sum()==pytest.approx(1)
    original=[tuple(z) for z in index].index((0,1,2))
    from itertools import permutations
    expected=sum(ordered_top3_probability(race["x"],v,np.asarray(theta),family)
                 for v in permutations((0,1,2)))
    assert p[original]==pytest.approx(expected,rel=1e-10)


def test_rank_fit_finite_and_future_labels_not_implicit():
    rows=[sample() for _ in range(102)]
    fitted=fit_rank_model(rows,"pl",.1,maxiter=15)
    assert len(fitted["theta"])==3
    assert fitted["train_races"]==102


def test_rejects_bad_race_feature_and_unexpected_family():
    with pytest.raises(ValueError):
        preknown_covariates(np.array([.2,np.nan,.4,.5]),
                            np.array([.2,.3,.4,.5]),np.array([.2,.3,.4,.5]))
    with pytest.raises(ValueError):
        unordered_top3_distribution(np.ones((4,3)),np.ones(3),"invalid")
