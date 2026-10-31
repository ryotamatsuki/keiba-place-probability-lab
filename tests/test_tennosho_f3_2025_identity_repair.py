"""The seven pre-registered late-2025 corrupt frozen horse IDs are repaired only by two matching sources."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import tennosho_f3_2025_identity_repair as correction


def _fixture():
    rows=[]
    published={}
    names={}
    for rid, changes in correction.CORRECTIONS.items():
        published_rows=[]
        names[rid]={}
        for no, (old, new, old_name, rank) in changes.items():
            rows.append({"race_id":rid,"horse_id":old,"horse_name":old_name,
                         "horse_no":no,"finish_position":float(rank),
                         "race_date":"2025-12-27" if rid.endswith("07"+"06") else "2025-12-28"})
            published_rows.append({"race_id":rid,"horse_id":new,
                                   "horse_no":no,"rank":str(rank)})
            names[rid][no]="公開馬名"+str(no)
        published[rid]=pd.DataFrame(published_rows)
    return pd.DataFrame(rows),published,names


def _mock(monkeypatch,published,names,wrong_alt=None):
    monkeypatch.setattr(correction,"CachedFetcher",
                        lambda *args,**kwargs:SimpleNamespace(
                            fetch=lambda url:(url,SimpleNamespace(
                                url=url,sha256="e"*64))))
    monkeypatch.setattr(correction.live_umanity,"source_id",
                        lambda rid,date:rid)
    monkeypatch.setattr(correction,"_horse_names_from_yahoo",
                        lambda html,provider_id:names["20"+provider_id])
    monkeypatch.setattr(correction,"raw_result",
                        lambda html,provider_id:published["20"+provider_id].copy())
    def alt(html,rid,date):
        df=published[rid].copy()
        if wrong_alt and rid==wrong_alt:
            df.loc[df.index[0],"horse_id"]="2023100000"
        return df
    monkeypatch.setattr(correction,"umanity_raw_result",alt)


def test_repair_requires_two_independent_source_matches(monkeypatch,tmp_path:Path):
    frozen,published,names=_fixture()
    _mock(monkeypatch,published,names)
    repaired,audits=correction.repair_frozen_2025_identities(
        frozen,tmp_path)
    for audit in audits:
        assert audit["frozen_source_untouched"] is True
        assert audit["yahoo_sha256"] == "e"*64
        assert audit["umanity_sha256"] == "e"*64
        for item in audit["verified_corrections"]:
            subset=repaired.loc[repaired.race_id.eq(item["race_id"])
                & repaired.horse_no.eq(item["horse_no"])]
            assert len(subset)==1
            assert subset.horse_id.iloc[0] == item["published_id"]
            assert subset.horse_name.iloc[0] == item["published_name"]
    # Original frozen artifact was passed by value, never mutated.
    assert frozen.horse_id.tolist()!=repaired.horse_id.tolist()


def test_repair_refuses_divergent_independent_sources(monkeypatch,tmp_path:Path):
    frozen,published,names=_fixture()
    rid=next(iter(correction.CORRECTIONS))
    _mock(monkeypatch,published,names,wrong_alt=rid)
    with pytest.raises(ValueError,match="Two independent sources disagree"):
        correction.repair_frozen_2025_identities(frozen,tmp_path)
