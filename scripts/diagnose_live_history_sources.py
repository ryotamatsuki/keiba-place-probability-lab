"""Diagnose audited parsers on known completed 2026 races."""
from __future__ import annotations

from datetime import UTC, datetime

import requests

from keiba_place_lab.live_history import (
    DENMA_URL,
    RESULT_URL,
    parse_declared_entry_audit,
    parse_result_audited,
    verify_full_field,
)


def fetch(url: str) -> tuple[str, str]:
    r=requests.get(url,timeout=(15,45),headers={"User-Agent":"keiba-place-probability-lab research/0.1"})
    r.raise_for_status()
    return r.text, datetime.now(UTC).isoformat()


def main():
    for provider_id in ("2606010302",):
        print("\nRACE",provider_id,flush=True)
        try:
            ru=RESULT_URL.format(provider_id=provider_id)
            du=DENMA_URL.format(provider_id=provider_id)
            rh,rt=fetch(ru)
            dh,dt=fetch(du)
            from bs4 import BeautifulSoup
            soup=BeautifulSoup(rh,"html.parser")
            table=[t for t in soup.select("table") if all(s in t.get_text() for s in ("着順","馬名","通過順位","騎手名"))][0]
            first=[tr.find_all("td",recursive=False) for tr in table.select("tr")]
            first=[x for x in first if x][0]
            print("cells",[(i,repr(x.get_text(" ",strip=True))) for i,x in enumerate(first)],flush=True)
            starters,result_entries=parse_result_audited(rh,provider_id,source_url=ru,retrieved_at=rt)
            print("result",len(starters),len(result_entries),result_entries.finish_status.value_counts(dropna=False).to_dict(),flush=True)
            roster=parse_declared_entry_audit(dh,provider_id,source_url=du,retrieved_at=dt)
            print("roster",len(roster),roster.entry_status.value_counts(dropna=False).to_dict(),flush=True)
            print("qa",verify_full_field(result_entries,roster),flush=True)
        except Exception as exc:  # noqa: BLE001
            print("ERROR",type(exc).__name__,repr(str(exc)),flush=True)


if __name__=="__main__":
    main()
