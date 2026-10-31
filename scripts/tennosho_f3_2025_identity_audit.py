"""Independently diagnose frozen 2025 missing-source starter identity errors.

This audit does NOT change frozen data; it prints every id/rank discrepancy
against published full-field results for all races without original Kaggle V1
runner data. No target race or future-date result is consulted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from keiba_place_lab.historical_panel import build_historical_panel
from keiba_place_lab.information_expansion_v1 import raw_observations
from keiba_place_lab.live_history import CachedFetcher, RESULT_URL, load_frozen_history
from keiba_place_lab import live_umanity
from tennosho_f3_prior_builder import _jra_internal_id
from tennosho_f3_raw_audit import raw_result, umanity_raw_result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--standardized", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--races", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path("/tmp/f3_2025_identity_audit"))
    args = p.parse_args()
    frozen = load_frozen_history(args.standardized)
    frozen["horse_id"] = frozen.horse_id.map(_jra_internal_id)
    panel = build_historical_panel(frozen)
    raw, audit = raw_observations(panel, args.results, args.races,
                                  horse_id_normalizer=_jra_internal_id)
    missing = sorted(raw.loc[
        raw.race_date.dt.year.eq(2025) & raw.source_horse_no.isna(),
        "race_id",
    ].astype(str).unique())
    print("AUDIT_START", json.dumps({
        "missing_race_count": len(missing), "races": missing,
        "kaggle_matched": audit["matched_raw_runner_rows"],
    }), flush=True)
    if len(missing) > 70:
        raise ValueError("Unexpected missing source coverage")
    fetcher = CachedFetcher(args.cache, pause_seconds=.45, max_attempts=2)
    discrepancies = []
    failed = []
    for rid in missing:
        expected = frozen.loc[frozen.race_id.astype(str).eq(rid)].copy()
        expected["horse_no"] = expected.horse_no.astype(int)
        date = pd.Timestamp(expected.race_date.iloc[0])
        provider = None
        alternatives = [
            ("umanity", f"https://umanity.jp/racing/result.php?race_id={live_umanity.source_id(rid,date)}"),
            ("yahoo", RESULT_URL.format(provider_id=rid[2:])),
        ]
        for name, url in alternatives:
            try:
                html, rec = fetcher.fetch(url)
                observed = (umanity_raw_result(html, rid, date) if name == "umanity"
                            else raw_result(html, rid[2:]))
                provider = name
                break
            except Exception as exc:
                print("RETRY",rid,name,str(exc)[:90],flush=True)
        if provider is None:
            failed.append(rid)
            continue
        observed["horse_id"] = observed.horse_id.map(_jra_internal_id)
        expected_keys = dict(zip(expected.horse_no, expected.horse_id))
        observed_keys = dict(zip(observed.horse_no, observed.horse_id))
        if set(expected_keys) != set(observed_keys):
            failures = {"race_id":rid,"provider":provider,
                        "expected_numbers":sorted(expected_keys),
                        "observed_numbers":sorted(observed_keys)}
            discrepancies.append(failures)
            print("FIELD_DISAGREEMENT",json.dumps(failures,ensure_ascii=False),flush=True)
            continue
        substitutions=[]
        for no in sorted(expected_keys):
            if expected_keys[no] == observed_keys[no]:
                continue
            erow=expected.loc[expected.horse_no.eq(no)].iloc[0]
            orow=observed.loc[observed.horse_no.eq(no)].iloc[0]
            substitutions.append({
                "number":int(no),
                "old_id":str(erow.horse_id),
                "old_name":str(erow.horse_name),
                "published_id":str(orow.horse_id),
                "old_finish_position":str(erow.finish_position),
                "published_rank":str(orow["rank"]),
            })
        if substitutions:
            details={
                "race_id":rid,"race_date":date.date().isoformat(),
                "provider":provider, "url":rec.url,"source_sha256":rec.sha256,
                "expected_count":len(expected),"published_count":len(observed),
                "substitutions":substitutions
            }
            discrepancies.append(details)
            print("IDENTITY_DISCREPANCY",json.dumps(details,ensure_ascii=False),flush=True)
    report={
        "status":"DIAGNOSTIC_ONLY_NO_SOURCE_MUTATION",
        "missing_kaggle_races":len(missing),
        "discrepant_races":len(discrepancies),
        "failed_races":failed,
        "discrepancies":discrepancies,
    }
    print("AUDIT_RESULT",json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
