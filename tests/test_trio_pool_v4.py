"""Post-transform joint distributions use true 3-subsets and fixed 2023 weights."""
from __future__ import annotations

import numpy as np
import pytest

from keiba_place_lab.trio_pool_v4 import checked_pool, v1_joint, weight_grid_two


def test_all_combinations_normalized_and_pool_preserves_shape():
    h=np.arange(1,18)
    idx,a=v1_joint(np.linspace(.09,.80,17),h)
    ids,b=v1_joint(np.linspace(.10,.70,17),h)
    assert idx.shape==(680,3) and np.array_equal(idx,ids)
    p=checked_pool([a,b],np.array([.75,.25]))
    assert np.isclose(p.sum(),1) and (p>0).all()
    assert not np.array_equal(p,a)
    with pytest.raises(ValueError):
        checked_pool([a,b],np.array([.8,.3]))
    with pytest.raises(ValueError):
        checked_pool([a,b[:-1]],np.array([.5,.5]))


def test_no_weight_fitted_on_2024_and_both_endpoints_included():
    q=[{
        "xgb":np.array([.75,.15,.06,.04]),
        "cat":np.array([.05,.15,.40,.40]),
        "actual_idx":0,
    } for _ in range(101)]
    x=weight_grid_two(q,"xgb","cat")
    assert x["year_labels_used"]==[2023]
    assert len(x["grid"])==21
    assert x["weight_left"]==1
    assert x["weight_right"]==0


def test_original_f3_marginals_not_automatically_preserved():
    h=np.arange(1,7)
    _,x=v1_joint(np.array([.9,.6,.4,.3,.2,.1]),h)
    _,y=v1_joint(np.array([.2,.2,.2,.4,.8,.9]),h)
    p=checked_pool([x,y],np.array([.5,.5]))
    assert np.max(np.abs(p-x))>.01
