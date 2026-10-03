# Historical Data Source Decision — Stage 3.6

Date: 2026-10-03 JST
Status: FROZEN FOR INITIAL HISTORICAL PANEL

## Decision

Stage 3.6 will build the historical training panel from **public, locally acquired historical
race-result data**, while keeping raw source files out of this public repository.

The acquisition order is:

1. **Primary bootstrap candidate:** `noriyukifurufuru/japan-horse-racing-2010-2025`
   on Kaggle.
2. **Secondary legacy / cross-check candidate:** `takamotoki/jra-horse-racing-dataset`
   on Kaggle.
3. **Authoritative verification / gap checking:** JRA official historical race results and
   annual result PDFs.

No netkeiba scraper is added to this repository.

## Why this source strategy

JRA states that historical race results are available from 1986 onward, and annual race-result
PDFs are available for 2002 onward. These are authoritative, but a full multi-year training panel
requires thousands of race-level records plus reconstruction of each horse's prior history.

A public Kaggle dataset provides a practical bootstrap layer, while the official JRA source is
used for spot checks, date correction, schema validation and future extension.

## License / provenance gate

Do **not** infer redistribution permission merely because a dataset is publicly downloadable.

Public references currently report:

- `Japan Horse Racing Data 2010-2025`: CC0 / public-domain metadata;
- `JRA日本中央競馬会 Horse Racing Dataset`: CC BY 4.0 metadata.

The exact license shown on the dataset page / downloaded metadata must be checked at acquisition
time before any derived row-level dataset is committed.

Until that check is recorded:

- raw historical files stay local;
- `data/historical_raw/` is git-ignored;
- `data/historical_processed/` is git-ignored;
- only code, schema, source manifest, aggregate diagnostics and model artifacts are committed.

## Raw-source policy

Raw HTML, JRA PDFs, netkeiba pages, PAT data, cookies and authenticated-session data are not
committed.

If an external dataset includes odds, popularity or payout columns, those columns may remain in
the local raw source but are **excluded from Stage 4 non-market feature generation**.

## Official references

- JRA historical-results FAQ: https://jra.jp/faq/pop02/1_6.html
- JRA race-result data: https://jra.jp/datafile/seiseki/index.html
- JRA annual result PDFs: https://jra.jp/datafile/seiseki/report/2026.html
- Kaggle primary candidate: https://www.kaggle.com/datasets/noriyukifurufuru/japan-horse-racing-2010-2025
- Kaggle secondary candidate: https://www.kaggle.com/datasets/takamotoki/jra-horse-racing-dataset

## Scope of the first model panel

The builder accepts all standardized JRA flat-race rows, but the initial Phase A model cohort will
be selected without looking at 2026 target-race outcomes.

Planned temporal partitions:

- 2010-2015: history warm-up only
- 2016-2022: train
- 2023-2024: validation
- 2025: untouched test
- 2026+: future / live application

The 2026-10-03 Kyoto 11R target race is never part of model fitting or model selection.
