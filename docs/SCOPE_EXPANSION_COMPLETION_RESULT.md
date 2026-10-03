# Scope expansion completion

Stage 1 existed at main afd5d5f and was preserved. Stages 2–4 and the independent dirt comparison are complete.

## Common 1200 m development comparison

| name     |   race_macro_brier |   race_macro_log_loss |   rows |   races |
|:---------|-------------------:|----------------------:|-------:|--------:|
| A_F0     |        0.148958593 |           0.459405336 |   6313 |     495 |
| B_F0     |        0.148774168 |           0.458557074 |   6313 |     495 |
| B_regime |        0.148425742 |           0.457755146 |   6313 |     495 |
| C_F0     |        0.149394066 |           0.460243990 |   6313 |     495 |

Sprint + regime versus original 1200: improvement 0.000532851, date-clustered SE 0.000319054; passes fixed gate.

## All-distance development results

| surface   | name     |   race_macro_brier |   rows |   races |
|:----------|:---------|-------------------:|-------:|--------:|
| turf      | G0       |        0.149311091 |  28926 |    2651 |
| turf      | G3       |        0.149315139 |  28926 |    2651 |
| turf      | R3       |        0.149301190 |  28926 |    2651 |
| turf      | baseline |        0.177907262 |  28926 |    2651 |
| turf      | routed   |        0.149016666 |  28926 |    2651 |
| dirt      | G0       |        0.149715516 |  34350 |    3080 |
| dirt      | G3       |        0.149687716 |  34350 |    3080 |
| dirt      | R3       |        0.149994619 |  34350 |    3080 |
| dirt      | baseline |        0.177702106 |  34350 |    3080 |
| dirt      | routed   |        0.149715516 |  34350 |    3080 |

The routed turf score includes selection on these same development folds. 2023–2024 and 2025 are not independent future evidence.

## Fitted production configurations

| model              |   training_rows |   training_races | through    | blocks                      |
|:-------------------|----------------:|-----------------:|:-----------|:----------------------------|
| turf_G0            |          148333 |            13471 | 2025-12-28 | F0                          |
| turf_R3_2001_2600  |           19918 |             1753 | 2025-12-28 | near,regime,course_distance |
| turf_1200_override |           49392 |             3960 | 2025-12-28 | regime                      |
| dirt_G0            |          176886 |            15508 | 2025-12-28 | F0                          |

All four models use frozen XGB01, fitted preprocessing, full-field LOO and training-side priors. Training is 2016–2025; every artifact passes serialization prediction parity. Source histories retain 2010 warmup. V2 code/results remain unchanged.

## Verification and handoff

Full 753,387-row/53,220-race context and history QA passes; new surface history matches every frozen turf history. Tests cover future/same-date mutations, all-peer LOO, known-zero versus missing, dirt replacement, live-history parity, draws after cancellation, partial eligibility, model-date leakage, pre-off timing, snapshot freshness and server publication time.

Pinned environment reproduces the frozen A/B/C predictions within 1e-7. XGBoost 3.4.1 failed this check; 3.2.0 is required. Manifests record local execution checkpoint SHAs, code fingerprints, data hashes and library versions; remote publication follows these local checkpoints via GitHub API.

**Stage 5 live adoption is pending.** No genuine forward locks/results are fabricated. Collection needs an updated confirmed JRA history feed (the supplied data ends 2025), real rosters/off times and time-matched canonical market snapshots. The pre-off lock, publication verification, entire-main ledger audit and threshold-based review CLIs are implemented. Until the actual forward gate is passed, canonical Stage 5 stays market-only.

See `docs/STAGE4_SCOPE_V3_FREEZE.md`, `docs/SCOPE_PROSPECTIVE_V3_SPEC.md`, and `docs/SCOPE_V3_OPERATIONS.md`.
