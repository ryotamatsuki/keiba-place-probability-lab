# Historical Panel Freeze v1

Status: PASS

- source: noriyukifurufuru/japan-horse-racing-2010-2025 version 1
- source license: CC0: Public Domain
- source lastUpdated: 2025-12-28T07:41:37.95Z
- adapter: jra-historical-adapter-v2
- feature spec: historical-v1.1
- panel-builder commit: 33add480a784ba59682d7367ed6aa5d7a90a8e66
- date-map SHA256: f75a44b593f5f9fff3fd49e9d7c1083b30ec550024f29b3b19b61689518a3ce3
- panel file: jra_flat_historical_panel_v1.parquet
- panel SHA256: cee9ae9a099f521f12b1bcdd371c25a3d7b9ba3555fbcf5a46f2c9098f59a33d
- races: 53,220
- runner rows: 753,387
- horses: 81,900
- actual date range: 2010-01-05 to 2025-12-28

Generation command: python -u scripts/freeze_historical_panel.py

Raw Kaggle CSVs and JRA PDFs are not committed. The Parquet panel is retained as
a GitHub Actions artifact; aggregate QA, mapping, schema, and fingerprints are committed.

Historical model exclusions: course_layout and handicap_indicator because official
coverage is incomplete. No synthetic/default fill is used.
