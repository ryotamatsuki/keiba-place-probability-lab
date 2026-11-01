"""Synthetic V5 fit, chronological selection, ticket and persistence regressions.

These are NOT the genuine 2023/2024 JRA precision results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from keiba_place_lab.trio_feature_v5 import FEATURE_BLOCKS


@pytest.fixture(scope="module")
def v5():
    script_dir=str(Path(__file__).resolve().parents[1]/"scripts")
    if script_dir not in sys.path:
        sys.path.insert(0,script_dir)
    import research_trio_v5_feature_ablation
    return research_trio_v5_feature_ablation


@pytest.fixture()
def sample():
    """Eight dates/twelve full fields per year with finite and missing histories."""
    rows=[]
    all_features=FEATURE_BLOCKS["all"]
    for year in (2023,2024):
        for date_index in range(8):
            for horse in range(1,6):
                row={
                    "race_id":f"{year}{date_index:02d}",
                    "race_date":pd.Timestamp(year,1,1)+pd.Timedelta(days=7*date_index),
                    "horse_no":horse,"horse_id":f"horse-{horse}",
                    "field_size":5,"top3_label":int(horse in (1,2,3)),
                    "p_matched_f3":(.74,.68,.59,.51,.48)[horse-1],
                }
                for j,key in enumerate(all_features):
                    row[key]=float(horse*.13+date_index*.02+j*.025)
                rows.append(row)
    frame=pd.DataFrame(rows)
    # Deliberately non-contiguous, duplicate-free pandas labels.
    frame.index=pd.Index(np.arange(len(frame))*17+15305)
    return frame


def test_all_preregistered_candidates_fit_predict_without_mutating_cow_inputs(v5,sample):
    original=sample.copy(deep=True)
    for block in ("offset_only","agents","dynamics","pace","all"):
        training=sample.loc[sample.race_date.dt.year.eq(2023)].copy()
        future=sample.loc[sample.race_date.dt.year.eq(2024)].copy()
        relevant=FEATURE_BLOCKS[block]
        if relevant:
            training.loc[training.index[0],relevant[0]]=np.inf
            training.loc[training.index[1],relevant[0]]=-np.inf
            training.loc[training.index[2],relevant[0]]=np.nan
            training.loc[:,relevant[-1]]=7.0  # constant column
            if len(relevant)>1:
                training.loc[:,relevant[1]]=np.nan  # entire column absent
        train_snapshot=training.copy(deep=True)
        future_snapshot=future.copy(deep=True)
        with pd.option_context("mode.copy_on_write",True):
            model=v5.OffsetResidual(block,1.0).fit(training)
            medians=model.medians.copy() if relevant else None
            center=model.center.copy() if relevant else None
            scale=model.scale.copy() if relevant else None
            p=model.predict(future)
        assert p.shape==(len(future),)
        assert np.isfinite(p).all() and np.all((p>0)&(p<1))
        assert model.fit_info["success"] is True
        pd.testing.assert_frame_equal(training,train_snapshot)
        pd.testing.assert_frame_equal(future,future_snapshot)
        if relevant:
            np.testing.assert_array_equal(model.medians,medians)
            np.testing.assert_array_equal(model.center,center)
            np.testing.assert_array_equal(model.scale,scale)
    pd.testing.assert_frame_equal(sample,original)


def test_all_missing_feature_set_fails_closed(v5,sample):
    x=sample.iloc[:10].copy()
    x.loc[:,list(FEATURE_BLOCKS["agents"])]=np.nan
    with pytest.raises(ValueError,match="All feature values missing"):
        v5.OffsetResidual("agents",1.0).fit(x)


def test_optimizer_nonconvergence_is_never_scored(v5,sample):
    x=sample.iloc[:10].copy()
    result=SimpleNamespace(success=False,fun=.01,x=np.zeros(1),
                           nit=300,status=1,message="iteration limit")
    with patch.object(v5,"minimize",return_value=result):
        model=v5.OffsetResidual("offset_only",1.0)
        with pytest.raises(RuntimeError,match="optimization failed.*iteration limit"):
            model.fit(x)
        assert model.fit_info["success"] is False
        assert model.fit_info["status"]==1
        assert model.fit_info["iterations"]==300
        assert not hasattr(model,"beta")


def test_bootstrap_improvement_direction_for_loss_and_hit(v5):
    x=pd.DataFrame({
        "race_date":pd.date_range("2024-01-01",periods=12,freq="7D"),
        "base_loss":[4.0]*12,"better_loss":[3.0]*12,"worse_loss":[5.0]*12,
        "base_hit":[0.0]*12,"better_hit":[1.0]*12,"worse_hit":[-1.0]*12,
    })
    for higher,base,better,worse,direction in (
        (False,"base_loss","better_loss","worse_loss","difference_baseline_minus_candidate"),
        (True,"base_hit","better_hit","worse_hit","difference_candidate_minus_baseline"),
    ):
        plus=v5.paired_v5_ci(x,base,better,higher_is_better=higher,draws=150)
        minus=v5.paired_v5_ci(x,base,worse,higher_is_better=higher,draws=150)
        assert plus[direction]==pytest.approx(1.0)
        assert minus[direction]==pytest.approx(-1.0)
        assert plus["ci95"][0]>0 and plus["ci95"][1]>0
        assert minus["ci95"][0]<0 and minus["ci95"][1]<0
        assert plus["positive_improvement_bootstrap_share"]==1.0
        assert minus["positive_improvement_bootstrap_share"]==0.0


def test_full_synthetic_selection_refit_score_and_roundtrip(v5,sample,tmp_path):
    d23=sample.loc[sample.race_date.dt.year.eq(2023)]
    d24=sample.loc[sample.race_date.dt.year.eq(2024)]
    early=d23.loc[d23.race_date.lt(pd.Timestamp("2023-02-05"))]
    late=d23.loc[d23.race_date.ge(pd.Timestamp("2023-02-05"))]
    candidates=[("offset_only",1.0)]
    candidates.extend((block,c) for block in ("agents","dynamics","pace","all")
                      for c in v5.C_VALUES)
    selection=[]
    for block,c in candidates:
        model=v5.OffsetResidual(block,c).fit(early)
        pred=model.predict(late)
        metric,_=v5.score_cohort(late,{
            "f3":late.p_matched_f3.to_numpy(float),"candidate":pred,
        })
        selection.append((float(metric.candidate_joint_logloss.mean()),block,c))
    _,chosen,chosen_c=min(selection)
    assert chosen in v5.ALL_CANDIDATES
    models={"f3_v1":d24.p_matched_f3.to_numpy(float)}
    for block in v5.ALL_CANDIDATES:
        models[block]=v5.OffsetResidual(
            block,chosen_c if block==chosen else 1.0
        ).fit(d23).predict(d24)
    scored,tickets=v5.score_cohort(d24,models,write_tickets=True)
    assert len(scored)==8 and len(tickets)==80
    actual=pd.DataFrame(tickets)
    assert actual.actual.sum()==8
    assert actual.groupby("race_id").actual.sum().eq(1).all()
    for model in models:
        assert actual.groupby("race_id")[model].sum().sub(1).abs().max()<1e-10
    paired=v5.paired_v5_ci(scored,"f3_v1_joint_logloss",
                          "all_joint_logloss",higher_is_better=False,draws=30)
    out=tmp_path/"sha-run9-attempt1"
    out.mkdir(exist_ok=False)
    scored.to_csv(out/"racewise.csv",index=False)
    actual.to_csv(out/"tickets.csv.gz",index=False,compression="gzip")
    d24[["race_id","horse_no"]].assign(**models).to_csv(
        out/"runners.csv",index=False
    )
    (out/"report.json").write_text(json.dumps({"chosen":chosen,"paired":paired}))
    pd.testing.assert_frame_equal(pd.read_csv(out/"racewise.csv"),scored,
                                  check_exact=False,rtol=1e-10)
    pd.testing.assert_frame_equal(pd.read_csv(out/"tickets.csv.gz"),actual,
                                  check_exact=False,rtol=1e-10)
    assert len(pd.read_csv(out/"runners.csv"))==40
    assert json.loads((out/"report.json").read_text())["chosen"]==chosen
    with pytest.raises(FileExistsError):
        out.mkdir(exist_ok=False)
    different=tmp_path/"sha-run9-attempt2"
    different.mkdir(exist_ok=False)
    assert out.exists() and different.exists()
    assert (out/"tickets.csv.gz").exists()


def test_workflow_sha_pinned_and_result_location_is_attempt_versioned():
    workflow=(Path(__file__).resolve().parents[1]/
              ".github/workflows/trio-v5-feature-ablation.yml").read_text()
    assert "ref: ${{ github.sha }}" in workflow
    assert 'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"' in workflow
    assert "${GITHUB_SHA}-run${GITHUB_RUN_ID}-attempt${GITHUB_RUN_ATTEMPT}" in workflow
    assert 'git add -- "$V5_RUN_DIR"' in workflow
    assert "if: failure()" in workflow
    assert "force push" not in workflow
