"""Diagnose result/entry source availability for known completed 2026 races."""
from __future__ import annotations

import requests
from bs4 import BeautifulSoup

from keiba_place_lab.live_history import DENMA_URL, RESULT_URL


def inspect(url: str) -> dict:
    r = requests.get(
        url,
        timeout=(15, 45),
        headers={"User-Agent": "keiba-place-probability-lab research/0.1"},
        allow_redirects=True,
    )
    soup = BeautifulSoup(r.text, "html.parser")
    return {
        "url": url,
        "status": r.status_code,
        "final_url": r.url,
        "chars": len(r.text),
        "title": soup.title.get_text(" ", strip=True) if soup.title else None,
        "has_denma_list": soup.select_one("#denma_list table") is not None,
        "result_tables": len(
            [
                t
                for t in soup.select("table")
                if all(s in t.get_text() for s in ("着順", "馬名", "通過順位", "騎手名"))
            ]
        ),
        "race_info": soup.select_one(".hr-predictRaceInfo") is not None,
        "body_prefix": soup.get_text(" ", strip=True)[:300],
    }


def main():
    for provider_id in ("2608040101", "2608040111", "2605040101", "2605040111"):
        print("\nRACE", provider_id, flush=True)
        for url in (
            RESULT_URL.format(provider_id=provider_id),
            DENMA_URL.format(provider_id=provider_id),
        ):
            try:
                print(inspect(url), flush=True)
            except Exception as exc:  # noqa: BLE001 - diagnostics print transport failures
                print({"url": url, "error": f"{type(exc).__name__}: {exc}"}, flush=True)


if __name__ == "__main__":
    main()
