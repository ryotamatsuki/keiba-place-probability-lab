"""Diagnose published source availability for a few 2026 JRA races."""
from __future__ import annotations

from pathlib import Path

import requests
from bs4 import BeautifulSoup

from keiba_place_lab.live_history import (
    CachedFetcher,
    build_expected_ledger,
    DENMA_URL,
    RESULT_URL,
)
import pandas as pd


def inspect(url: str) -> dict:
    r = requests.get(url, timeout=(15,45), headers={"User-Agent":"keiba-place-probability-lab research/0.1"})
    soup = BeautifulSoup(r.text, "html.parser")
    return {
        "url": url,
        "status": r.status_code,
        "final_url": r.url,
        "chars": len(r.text),
        "title": soup.title.get_text(" ", strip=True) if soup.title else None,
        "has_denma_list": soup.select_one("#denma_list table") is not None,
        "result_tables": len([t for t in soup.select("table") if all(s in t.get_text() for s in ("着順","馬名","通過順位","騎手名"))]),
        "race_info": soup.select_one(".hr-predictRaceInfo") is not None,
    }


def main():
    fetcher=CachedFetcher(Path("/tmp/live-history-diag-cache"), pause_seconds=0)
    ledger,_=build_expected_ledger(year=2026, through=pd.Timestamp("2026-10-03"), fetcher=fetcher, force_schedule=True)
    flat=ledger.loc[ledger.target_flat]
    print("ledger", len(ledger), "flat", len(flat), "days", flat.race_date.nunique(), flush=True)
    sample=pd.concat([flat.head(2), flat.tail(2)]).drop_duplicates("provider_id")
    for row in sample.itertuples(index=False):
        print("\nRACE", row.provider_id, row.race_date, flush=True)
        for url in (RESULT_URL.format(provider_id=row.provider_id), DENMA_URL.format(provider_id=row.provider_id)):
            try:
                print(inspect(url), flush=True)
            except Exception as exc:
                print({"url":url,"error":f"{type(exc).__name__}: {exc}"}, flush=True)


if __name__=="__main__":
    main()
