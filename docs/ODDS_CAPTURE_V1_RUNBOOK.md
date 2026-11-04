# Market odds V1 — licensed capture and operational handoff

Research preparation status: **code / schema implemented; JRA-VAN credentials and live historical export NOT acquired**.

## Authoritative sources (checked 2026-10-31 JST)

1. JRA-VAN Data Lab official service/pricing: https://jra-van.jp/dlb/ and https://jra-van.jp/buy/ . Data Lab subscription is **JPY 2,090 / month (tax included)**, with the JV-Link module and license key. It is not a free authenticated HTTP API.
2. Official data schedule: https://jra-van.jp/dlb/ddata.html . Historical intraday odds are recorded roughly every 5–10 minutes and cover the available products listed by the provider; time-series includes win, place, bracket, quinella.
3. Official Data Lab FAQ: https://jra-van.jp/dlb/sdv/faq.html . A time-specific query cannot be supplied to JVRead directly: read time-series records and filter timestamps. Coverage/freshness of actual archived dates must be confirmed in an authorized client; do not assume a given season's records are present.
4. Official JV-Link developer community (June 2026): https://developer.jra-van.jp/t/topic/872 . Community guidance notes a formal one-year guaranteed delivery window for time-series odds with observations of older records, potentially reaching 2003. This is **not** a guarantee of complete 2016–2024 sampling or free retrieval, and must be audited race by race.

## Runtime/authorization

JV-Link is a client library installed in a supported Windows PC environment, not available in GitHub-hosted Ubuntu runners without a licensed export. No subscription status, JV-Link credential, Windows host, market export or redistribution rights have been supplied to this project. **No historical intraday JRA-VAN odds have been downloaded in V1.** The proposed interface consumes CSV from an authorized exporter. Do not use authenticated scraping, share the Data Lab client key, or push licensed row-level exports to GitHub.

## Schema, identity, and capture

Required columns (all present, for place odds the `odds` single-value field may be empty; for win odds the low/high range may be empty):

`race_id,horse_id,bet_type,odds,place_odds_low,place_odds_high,provider_updated_at,captured_at,scheduled_off_at,source,snapshot_id`

Timestamps MUST be source-observed and timezone-aware. Keep provider and local acquisition times independently, both immutable. `snapshot_id` identifies a simultaneous full-field capture, not a horse-level invented timestamp. Capture append-only under gitignored `data/raw/market_odds_v1/`; write SHA256 provenance. At a checkpoint require **all active win-odds horses** from roster with one snapshot ID. Never build a field from multiple snapshots or substitute final odds.

- T−60 and T−30: most recent full snapshot with `provider_updated_at <= checkpoint` and `captured_at <= checkpoint`, maximum provider age 10 minutes.
- T−10: same rule, maximum provider age **5 minutes**. Capture/lock must precede the existing T−10 to T−9 minute pre-off lock. No T−9 or final-odds substitution.
- Any missing field, stale snapshot, cancellation differing from active roster, ambiguous timestamp, future update, duplicate IDs or bad value => fail, preserve missing checkpoint as unavailable. Report the number of complete checkpoint fields by timing domain.
- `place_odds_low/high` are displayed payouts, NOT independent probabilities; never use `1/place_odds` as calibrated top3 probability.

Commands:

```bash
# On a machine holding a legally exported, observed-time CSV:
python scripts/capture_market_odds_v1.py ingest \
  --source-csv /path/to/licensed_intraday_race.csv \
  --output-root data/raw/market_odds_v1

python scripts/capture_market_odds_v1.py select \
  --ledger data/raw/market_odds_v1/RACE_ID.csv \
  --roster /path/to/active_roster.csv \
  --minutes-before 10 \
  --output data/raw/market_odds_v1/locks/RACE_ID_tminus10.csv
```

For a publication lock, copy only **permitted derived probabilities and hashes** to the canonical forward lock pipeline after further data-license review. Do not treat any simulated fixture, final-odds reconstruction or delayed server record as a captured T−10 measurement.

## Future research partitions

Real intraday snapshots can be compared at T−60, T−30, T−10 only when every candidate has the same race, exact active horse roster, and exact target-time provider record. An alternative near-off forecast is a separate model and separate preregistration. DataLab coverage gaps are reported, not filled using 2025 labels or future odds. The 1,000 races / 60 dates / 90 days prospective adoption rule is unchanged.
