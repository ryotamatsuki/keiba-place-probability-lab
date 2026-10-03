# 2026-10-04 live shadow trial

Targets chosen before racing: Kyoto Daishoten (202608040211, turf 2400,
15:30 Asia/Tokyo) and Mainichi Okan (202605040211, turf 1800, 15:45).
Official JRA entry pages confirm both schedules. Configuration is in
`config/live_trial_20261004.json`. The fixed V3 bundle remains unchanged;
Stage 5 production stays market-only. No betting or model promotion is included.

## Preparation

`prepare_live_trial.py` merges the frozen 2010–2025 standardized source with
all missing JRA flat races in every target horse's public complete history,
plus the most recent completed Tokyo/Kyoto cards. Every selected past race
retains all actual starters, including peers not running today. Target horse
identities must match the baseline; every known baseline start must appear in
the public inventory. Every target horse's final history must exactly match
its entire listed JRA flat inventory. Foreign/regional/obstacle races remain
outside the historical training definition, rather than being silently treated
as new JRA starts. The source is Sports Navi; raw pages, retrieval timestamps
and hashes are retained locally. Confirmed DNF is explicit.

This is an audited **target-scoped history**, not a complete 2026 JRA feed.
Its completeness claim covers these 35 horses only. Do not use it for a different
roster without repeating coverage checks. The context file includes full past
fields to compute relative times; peers' unrelated career histories are not needed
for today's target features. Strict prior-calendar-date and seven-day freshness
checks remain enabled. Market fields are never passed to historical features.

```bash
python scripts/prepare_live_trial.py --source /path/to/standardized_jra_flat_source_v1.parquet
python scripts/capture_live_trial.py preview --race-id 202608040211 --output analysis/live_trial_20261004/kyoto_preview_v1
python scripts/capture_live_trial.py preview --race-id 202605040211 --output analysis/live_trial_20261004/tokyo_preview_v1
```

Morning previews are operational checks, not prospective locks. The prediction
manifest verifies the immutable fitted bundle and input hashes. Market probability
uses the existing normalized-win-odds/Harville conversion on the complete active
field. Use the provider's explicit odds update time, not HTTP retrieval time.

## Timed execution

Invoke the lock command at 15:19:00 and 15:34:00 Japan time respectively. It fetches
a new roster and recomputes predictions for any pre-lock cancellation, then waits
up to a minute for T-minus-ten before fetching timestamped current win odds. All
horse IDs/numbers/names must match, odds must be at most five minutes old, and
the actual lock clock must be inside T-minus-ten to T-minus-nine.

```bash
python scripts/capture_live_trial.py lock --race-id 202608040211 --output analysis/live_trial_20261004/kyoto_lock_inputs_v1
python scripts/capture_live_trial.py lock --race-id 202605040211 --output analysis/live_trial_20261004/tokyo_lock_inputs_v1
```

Publish the two generated lock files under `prospective/locks/RACE_ID` to main
before off. Use the main push Actions run ID with `run_scope_prospective.py verify`
and publish the resulting separate `publication.json`. Git author time alone is
not proof. If a request/scheduler is late, odds are stale, a roster changed to an
uncovered horse, or publication is late, record the failure; never backdate or
replace the lock using final odds. New captures use unique filenames/directories.

## After off

Retrieve confirmed results and reconcile against JRA official results. Save
race_id, horse_id, top3_label for all locked starters, alongside source hashes.
Any post-lock scratch or race cancellation excludes the entire race for every
candidate and goes in the separate cancellation events ledger. Unverified
provider results remain provisional, not official evaluation outcomes.

Keep the forward protocol unchanged: no interim model/weight selection. These
two races can establish operational completion and collection counts. They do
not meet the 1,000-race / 60-date / 90-day Stage 5 review thresholds.
