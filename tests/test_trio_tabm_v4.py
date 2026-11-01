"""Official TabM contract tests on artificial rows; no postrace features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_official_tabm_member_probability_contract():
    tabm=pytest.importorskip("tabm")
    torch=pytest.importorskip("torch")
    m=tabm.TabM.make(n_num_features=4,cat_cardinalities=[3,4],
                     d_out=1,k=4,n_blocks=2,d_block=64,dropout=.1)
    m.eval()
    with torch.no_grad():
        out=m(torch.randn(6,4),torch.tensor([[0,0],[1,1],[2,3],
                                             [0,2],[1,3],[2,0]],dtype=torch.long))
        assert out.shape==(6,4,1)
        p=torch.sigmoid(out.squeeze(-1)).mean(1)
        assert p.shape==(6,) and torch.all(p>0) and torch.all(p<1)


def test_fit_preprocessor_train_only_rejects_future_unseen_categories():
    pytest.importorskip('tabm')
    pytest.importorskip('torch')
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
    from research_trio_v4_tabm import PreviousYearFeatures
    old=pd.DataFrame({"numeric":[1.,2.,np.nan],"racecourse":["TOKYO","TOKYO","KYOTO"]})
    future=pd.DataFrame({"numeric":[2.,100.],"racecourse":["NEW","TOKYO"]})
    p=PreviousYearFeatures().fit(old,["numeric"],["racecourse"])
    num,cat=p.transform(future)
    assert float(num[1,0])>5.
    assert int(cat[0,0])==0
    assert int(cat[1,0])>0
