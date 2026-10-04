# 2026 JRA live-history database specification

Status: implementation specification for the fixed Stage-4 Scope V3 forward period.

## Purpose

Maintain a reusable, audited 2026 JRA flat-race history database for live feature generation.
This database updates prediction-time history only. It does **not** retrain or mutate the frozen
Scope V3 model, preprocessing, routing or model bundle.

## Completeness contract

A snapshot is publishable only when every expected JRA flat race through the requested cutoff is
confirmed and independently field-checked.

The manifest records:

- `schema_version`
- `snapshot_id`
- `requested_through`
- `complete_through`: calendar date through which no expected flat race is missing
- `latest_race_date`: latest race date physically present in the starter history
- `expected_races` / `confirmed_races`
- `missing_race_ids`
- `expected_race_days` / `confirmed_race_days`
- row counts and SHA-256 hashes for history, entry audit and race ledger.

`complete_through` and `latest_race_date` are intentionally distinct. A single newer race
cannot advance completeness past an earlier missing race. Non-racing days are handled by the
expected-race ledger rather than by a simple age-since-last-race heuristic.

## Snapshot layout

```
data/live_history/2026/
  current.json
  snapshots/
    SNAPSHOT_ID/
      jra_flat_history.parquet
      entry_audit.parquet
      race_ledger.csv
      manifest.json
```

`current.json` points to one immutable snapshot. Any corrected result creates a new snapshot ID
and history hash. Existing snapshots remain usable to reproduce prior predictions.

## Expected-race ledger

Monthly published schedules enumerate JRA meeting-days. Each meeting-day race list enumerates all
scheduled races. Obstacle races remain in the ledger but are marked outside the flat-history
target; new-maiden races, unsupported prediction distances and other flat races remain in the
history because they may contribute to later horse histories.

Abandoned races remain in the ledger as `race_status=abandoned`, with an explicit
race/date list, official JRA source URL and reason in `meeting_events.json`. They require
no result and contribute no horse starts. A missing or malformed result alone never
authorizes exclusion. On 2026-02-07, Tokyo 8R–12R were abandoned due to snow:
https://jra.jp/news/202602/020707.html . Rescheduled meeting-days are retained under
their actual published dates; they are not inferred from the original calendar.

Snapshot identity includes the history, entry-audit and ledger hashes. An existing
snapshot is never deleted or overwritten. Missing-race retries can reuse captured
HTML; deliberate recent-result reconciliation refreshes both published views.

## Independent inventory and alternative source

For the October 3 backfill, the official JRA 2026 results index and all 220 published
daily result PDFs provide an independently extracted race inventory. The inventory
includes the explicitly evidenced abandoned races and must account for race numbers
1–12 on each meeting-day. An obstacle abandonment on February 8 (Kokura 4R) is also
retained, outside the flat-starter population.

The PDF parser determines flat/obstacle status from the condition header, before the
prize/result rows. A horse name containing ジャンプ must not affect classification.

When Yahoo results return persistent HTTP 500, `--source umanity` uses separately
published Umanity result and declared-entry pages. Horse IDs, race/date identity,
condition metadata, starters and statuses are reconciled exactly as for Yahoo.
Source URLs, retrieval times and content hashes remain explicit per race. Already
captured and verified Yahoo pages can be reused; no failed acquisition authorizes
dropping an expected race. Neither odds nor proprietary prediction indices become
history features.

Commands used for the audited backfill:

```bash
PYTHONPATH=src python scripts/extract_official_jra.py --year 2026 --workers 3 --compact-excerpts
PYTHONPATH=src python scripts/build_live_history_ledger.py \
  --conditions data/derived/jra_official_conditions_2026.csv --through 2026-10-03
PYTHONPATH=src python scripts/backfill_live_history.py \
  --through 2026-10-03 --source umanity --max-workers 3 \
  --ledger analysis/live_history_2026/official_race_ledger.csv
```

The independent ledger must be rebuilt when advancing beyond its audited cutoff;
an older ledger must never certify a later date as complete.

## Result and full-field verification

For every flat race, collection uses two separately parsed published views:

1. confirmed result table;
2. declared-entry / roster table including cancellation status.

The result parser preserves:

- `finished`
- `dnf` (競走中止)
- `disqualified` (失格)
- `scratched` (取消)
- `excluded` (除外).

`jra_flat_history.parquet` contains actual starters only. `entry_audit.parquet` retains the
declared field, including scratches/exclusions, so actual-starter completeness is checked
independently from the result row count. The audit also stores declared field size, source URLs
and retrieval timestamps.

A result is accepted only when declared identities and cancellation/exclusion status reconcile and
the starter history contains exactly the verified active field.

## Update and correction policy

An update does not merely append after the latest stored date.

For every run:

1. rebuild the expected-race ledger through the requested cutoff;
2. refetch every missing race, even when the missing race is old;
3. refetch a configurable recent window (default 14 calendar days) to detect result corrections;
4. replace a corrected race atomically at race level;
5. perform whole-snapshot QA;
6. publish the new snapshot and advance `current.json` only after all QA passes.

Any fetch, parse or reconciliation failure aborts publication and leaves the previous current
snapshot unchanged.

## Prediction-time as-of barrier

When a fixed historical base and the live snapshot are combined, the caller explicitly applies:

```python
history = history.loc[history["race_date"] < target_date].copy()
```

before calling `build_live_context()`. The existing `build_live_context()` rejection of any
on/after-target history remains in place as a second information barrier.

Prediction manifests record the historical-base SHA, live `snapshot_id`, `schema_version`,
live-history SHA and `complete_through`. A later correction therefore cannot silently change the
input claimed for an earlier prediction.

## Forward-model versioning

Scope V3 remains trained through 2025 during its current prospective period. Updating confirmed
2026 history is permitted and required for current features. Retraining is not technically
forbidden, but any retrained model must be frozen as a new model/version and evaluated in a
separate prospective period rather than silently replacing V3.

## Reproduction gate

Before this database is considered operational, the common DB must reproduce the already captured
2026-10-04 Kyoto Daishoten and Mainichi Okan morning trial:

- 35 runners;
- prior feature context;
- frozen V3 non-market probabilities;
- unchanged model-bundle SHA.

`scripts/qa_live_history_reproduction.py` performs this check.
