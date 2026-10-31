"""Pre-year full-ticket ensemble cannot invent real 3-renpuku prices."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from research_trio_v4_multistructure import weights_on_2023


def test_mixture_meta_fit_only_requires_late_2023_prior_predictions():
    records=[]
    for rid in range(205):
        for k in range(4):
            records.append({
                "race_id":f"2023-{rid:04}",
                "horse_1":1,"horse_2":2,"horse_3":3+k,
                "actual":int(k==0),"incumbent":[.40,.30,.20,.10][k],
                "rank":[.35,.30,.20,.15][k],
                "tabm":[.30,.35,.20,.15][k],
            })
    train=pd.DataFrame(records)
    w=weights_on_2023(train)
    assert set(w)=={"f3_plus_rank","f3_plus_tabm","f3_plus_rank_plus_tabm"}
    for z in w.values():
        p=np.asarray(z["weights"],float)
        assert len(p)==3 and np.isclose(p.sum(),1)
        assert np.all(p>=0)
        assert z["2023_late_joint_nll"]>0
