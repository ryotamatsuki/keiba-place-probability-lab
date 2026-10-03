# Historical Data Source Decision — Stage 3.6

Date: 2026-10-03 JST
Status: FROZEN V1 — exact primary source/version/license verified; official-date reconstruction required

## Decision

Stage 3.6 uses the exact Kaggle dataset version below as the row-level bootstrap source and
reconstructs calendar dates and race-type facts against official JRA result PDFs.

Primary source:

- dataset: `noriyukifurufuru/japan-horse-racing-2010-2025`
- Kaggle version: **1**
- license metadata: **CC0: Public Domain**
- lastUpdated metadata: **2025-12-28**
- required files: `keiba_races.csv`, `keiba_results.csv`, `keiba_payouts.csv`
- raw files: local/CI only; never committed

Authoritative verification source:

- JRA annual/daily result PDFs for 2010-2025:
  `https://www.jra.go.jp/datafile/seiseki/report/`

Secondary dataset `takamotoki/jra-horse-racing-dataset` remains reference-only and is not
needed to construct the frozen v1 panel.

No netkeiba scraper is added to this repository.

## Why the Kaggle race date is not trusted

The primary dataset's `keiba_races.csv.date` is not accepted as chronological truth. Stage 3.6
reconstructs `race_id -> actual_date` from official JRA result PDFs before any previous-race
linkage or rolling feature is calculated.

The final mapping is therefore fail-closed: ambiguous official dates, conflicting dates, duplicate
calendar race keys, or source race days absent from the official archive stop materialization.

The verified 2020 Nakayama continuation-racing exception is handled at race level: the first two
races of 3rd Nakayama meeting day 2 were held on 2020-03-29 and races 3-12 were continued on
2020-03-31.

## Authority by field

The v1 adapter deliberately does not treat every PDF-extracted token as equally reliable.

- `actual_date`: official JRA PDF, required and authoritative.
- flat/obstacle race kind: official JRA race markers/conditions are authoritative, with only
  explicitly verified race-level metadata overrides where PDF text extraction loses the marker.
- distance/class: primary row source remains the canonical value for v1 after conflict auditing.
  Generic four-digit PDF text extraction produced false distance candidates, so those official
  values are retained only as diagnostics rather than silently overriding the source.
- course layout / handicap flag: retained as audit fields where confidently extracted, but excluded
  from the historical v1 model-input allowlist when coverage is incomplete.
- odds / popularity / payout: prohibited from the Stage 4 non-market panel.

This separation prevents an unreliable PDF parser from replacing a usable source value merely
because the PDF is authoritative in principle.

## License / provenance gate

The freeze script queries Kaggle metadata at build time and requires both:

- `currentVersionNumber == 1`
- `licenseName == "CC0: Public Domain"`

A metadata change is a hard stop requiring a new source review. SHA-256 fingerprints of all three
downloaded source files are recorded in the Stage 3.6 QA/freeze outputs.

Raw historical files and official PDFs remain outside git. The public repository contains code,
schemas, aggregate QA, reconciliation tables, fingerprints and documentation. The complete
processed row-level panel is distributed as a GitHub Actions artifact for the freeze run rather
than committed to git.

## Result-file integrity gate

`keiba_results.csv` has an 18-field schema but contains malformed 19-field records in the full
source. Stage 3.6 audits malformed rows before standardization and fails if any malformed record
belongs to the JRA target population. Malformed rows are never silently skipped.

## Scope of the first model panel

Temporal partitions are fixed before Stage 4:

- 2010-2015: history warm-up only
- 2016-2022: train
- 2023-2024: validation / model and cohort selection
- 2025: untouched final historical test
- 2026+: future/live application only

The 2026-10-03 Kyoto 11R target outcome is not part of training, validation or test construction.

Phase A eligibility is also fixed in advance:

- flat turf
- official starter field size >= 8
- at least 3 strictly prior valid starts for the horse

Predeclared comparison cohorts are all eligible turf, turf 1000-1400m and exact turf 1200m.
Stage 4 may choose among them using validation results only; the 2025 test set and 2026 target race
must not influence that choice.

## Official references

- JRA historical-results FAQ: https://jra.jp/faq/pop02/1_6.html
- JRA race-result data: https://jra.jp/datafile/seiseki/index.html
- JRA annual result PDFs: https://www.jra.go.jp/datafile/seiseki/report/
- Kaggle primary source: https://www.kaggle.com/datasets/noriyukifurufuru/japan-horse-racing-2010-2025
- Kaggle secondary reference: https://www.kaggle.com/datasets/takamotoki/jra-horse-racing-dataset
