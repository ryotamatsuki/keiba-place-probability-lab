# Scope V3 operations

Install the verified runtime, then reproduce comparisons in order:

```bash
python -m pip install -e '.[dev,historical,modeldev]' -c requirements-scope.txt
python scripts/run_scope_expansion_completion.py --panel /path/to/jra_flat_historical_panel_v1.parquet --stage features
python scripts/run_scope_expansion_completion.py --panel /path/to/jra_flat_historical_panel_v1.parquet --stage turf
python scripts/run_scope_expansion_completion.py --panel /path/to/jra_flat_historical_panel_v1.parquet --stage dirt
python scripts/run_stage4_scope_v3.py train --panel data/historical_processed/scope_augmented.parquet --surface all
```

Training artifacts are versioned in `models/stage4_scope_v3/`; manifest hashes
identify the fitted models and routing. Do not rerun production training as a
search against 2025. Reproduction of historical comparisons never uses these
full-history production weights. XGBoost 3.2.0 is required: 3.4.1 failed the
frozen A/B/C prediction check, and was rejected before any feature selection.

The frozen 2010–2025 source remains the model/refit base. Current-year results are
maintained separately as versioned live-history snapshots; see
`LIVE_HISTORY_2026_SPEC.md`.

Build or reconcile the common 2026 database through a confirmed cutoff:

```bash
python scripts/update_live_history.py --year 2026 --through 2026-10-03
```

The updater rebuilds the expected-race ledger, retries old gaps, rechecks the
recent correction window, verifies the complete starter field against the
published entry page, and advances `data/live_history/2026/current.json` only
after all expected flat races pass QA. A failed update leaves the previous
snapshot current.

Roster CSV has one row per active starter and the following entry columns:
race_id, race_date, horse_id, horse_no, horse_name, field_size,
declared_field_size, surface, distance_m, racecourse, race_class, age, sex,
assigned_weight_kg, turn_direction. It must contain no outcomes/market fields.

For the common database path, pass the frozen pre-2026 history and the live
snapshot separately:

```bash
python scripts/run_stage4_scope_v3.py predict \
  --roster /path/to/roster.csv \
  --historical-base /path/to/standardized_jra_flat_source_v1.parquet \
  --live-history-root data/live_history/2026 \
  --output /path/to/shadow_v1.csv
```

The caller concatenates the two sources and explicitly keeps
`race_date < target_date` before invoking `build_live_context()`; the latter's
strict on/after-target rejection remains active as a second barrier. The prediction
manifest records the base SHA plus live snapshot id/schema/SHA/completeness date.

The legacy `--history` argument remains available for a separately audited,
already-complete as-of history file. Prediction recomputes histories strictly
before race date, includes every active starter in LOO context and preserves
declared draw positions after cancellation. Ineligible runners retain missing
nonmarket probabilities. Unsupported scopes return explicit market-only routing.
The prediction CSV and adjacent `.manifest.json` are versioned; no overwrite.

Reproduce the captured 2026-10-04 35-runner morning trial from the common DB with:

```bash
python scripts/qa_live_history_reproduction.py
```

## Prospective collection

The forward protocol is committed in `SCOPE_PROSPECTIVE_V3_SPEC.md`. Market-only
remains canonical throughout collection. Review thresholds are 1,000 complete
races, 60 distinct dates and 90 elapsed days; no interim score inspection/adoption.
One fitted bundle hash must remain constant during a collection period.

Capture the existing canonical Stage 5 V2 market-only marginal probabilities
at the same lock time (not historical final odds). Market CSV columns are
race_id, horse_id, p_market. Its sidecar JSON has canonical_stage5_version equal
to `stage5_v2_market_only`, snapshot_at with timezone and snapshot_sha256 of CSV.
The fresh market snapshot may be at most five minutes old. Lock between 10 and
9 minutes before the scheduled off; CLI uses the actual system clock.

```bash
python scripts/run_scope_prospective.py lock --predictions /path/to/shadow_v1.csv \
  --prediction-manifest /path/to/shadow_v1.manifest.json \
  --market /path/to/market.csv --market-manifest /path/to/market.manifest.json \
  --off-at '2026-10-04T15:40:00+09:00' --output prospective/locks/RACE_ID
```

Publish both `predictions.csv` and `manifest.json` in a Git commit pushed to **main** before
off. Then use the corresponding GitHub Actions **push** run ID as publication
evidence (an author/committer timestamp alone can be backdated):

```bash
python scripts/run_scope_prospective.py verify --lock prospective/locks/RACE_ID \
  --repository-path prospective/locks/RACE_ID --run-id PUSH_RUN_ID
```

Verification fetches server-created push-run time and both immutable files at
that run's exact head SHA. It writes a separate publication record. Review
re-verifies this evidence, not a caller's unverified local timestamp. Keep one
lock per race; after-lock scratch/race cancellation is a separate CSV event
with race_id, event_at (timezone-aware), reason (`late_cancellation` or
`race_cancelled`). A changed roster excludes the entire race for every candidate.
Supported races with some ineligible runners stay included; mixed candidates use
the same market probability for each ineligible runner, and keep nonmarket missing.

Official outcome CSV: race_id, horse_id, top3_label. All starters' keys must match
the lock; unavailable official outcomes exclude the whole race and are reported.
Review checks the complete published-main lock inventory under the period's
folder, ends at the first complete day meeting all thresholds, and rejects
duplicate race locks, changed protocols, changed models and changed payloads.

```bash
python scripts/run_scope_prospective.py review --locks prospective/locks \
  --outcomes /path/to/official_outcomes.csv --events /path/to/cancellation_events.csv \
  --output analysis/prospective_scope_v3_review
```

Below thresholds, only collection counts are returned. The first eligible review
records scores and paired date-clustered SE versus market-only for fixed weights
0/.10/.25/.50/1. No automatic production promotion: adoption needs a new recorded
decision following that gate. No real forward locks or outcomes are manufactured
by this development run; live adoption remains pending.
