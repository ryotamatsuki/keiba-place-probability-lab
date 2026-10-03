# Live trial task runbook

Repository: ryotamatsuki/keiba-place-probability-lab. Work on current main.
The user authorized execution of the October 4 trial, including pre-off locks
and confirmed outcome recording. Production remains market-only.

## Pre-off task

1. Clone/fetch main, check Japan time with an aware system clock, read
   `LIVE_TRIAL_20261004.md`, configuration, history manifest and QA record.
   If needed install `.[dev,historical,modeldev]` constrained by
   `requirements-scope.txt`. The committed target history is sufficient;
   do not rerun collection or train a model during the lock window.
2. At T-minus-eleven invoke `capture_live_trial.py lock` for the designated
   race. Use a new output directory. It prepares roster/nonmarket predictions,
   waits for T-minus-ten, fetches fresh provider odds and creates a clock-checked
   lock. Check process output at intervals no longer than a minute. Provider
   update time must be within five minutes; never substitute the old morning
   replay snapshot. Cold setup or scheduling may run late; reject any late lock.
3. Publish `prospective/locks/RACE_ID/{predictions.csv,manifest.json}` and
   small input/provenance files to main **before scheduled off**. Connected
   GitHub Git-data tools can create a tree against latest main, create a
   commit with that main as parent, and update main without force. Read latest
   main immediately before publication and reject/rebase a non-fast-forward.
   No shell GitHub credentials are available; do not print/probe secrets.
4. Fetch Actions push runs for the exact resulting commit. Select a `push`
   run with `head_branch=main`, then run `run_scope_prospective.py verify`.
   Publish its separate `publication.json`; verification can happen later,
   but the server-created push event must precede off. Fetch through the GitHub
   connector if public API shell networking is unavailable, preserving the
   same run-time and immutable-byte verification. Author/committer dates are
   insufficient proof. Never create a fake push run or timestamp.
5. Retain and report failures. Network refusal, source unavailability, stale
   odds, changed uncovered horses and late execution must not be bypassed.
   A failed attempt is not a formal lock. Preserve unique attempt directories;
   record status with actual observation time. Do not edit an existing lock.

The morning acquisition succeeded for the 35 horses. Fresh market capture
subsequently failed because network approval was cancelled / the proxy refused
the connection. The morning nonmarket outputs are real; no fresh market pair
or formal forward lock was completed then. Check current access and conditions
at the designated time; report an unsuccessful attempt accurately.

## Results task

Read the entire main lock inventory. Retrieve JRA's official current result
URL for each target; do not invent CNAME checksums. Run
`record_live_trial_result.py --race-id ID --official-url URL --output UNIQUE_DIR`.
It reconciles all provider and official starter IDs/finishing ranks, saves binary
top-three outcomes and identifies changed post-lock fields. Publish outcome and
source records. If network/data is unavailable or results are not confirmed,
leave the result pending and record why. Race cancellation needs a separately
verified cancellation event. Event times are first observation times, not guessed
historical timestamps. No interim Brier, weight selection or promotion is allowed.

Report operational completion, genuine lock count, exclusions and pending
results. Two races do not satisfy the fixed Stage 5 review thresholds.
