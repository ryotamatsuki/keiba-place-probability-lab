# Historical Panel QA Report Template

Dataset:
Source/version:
License verified:
Acquisition date:
Standardization code commit:
Panel-builder commit:

## Coverage

- min race date:
- max race date:
- runner rows:
- unique races:
- unique horses:
- surface coverage:
- field-size range:

## Temporal split

| split | years | races | runners | top3 prevalence |
|---|---|---:|---:|---:|
| warmup | 2010-2015 | | | |
| train | 2016-2022 | | | |
| validation | 2023-2024 | | | |
| test | 2025 | | | |

## Leakage checks

- duplicate race_id × horse_id:
- prev_race_date >= race_date:
- target-race odds columns:
- target-race popularity columns:
- target-race payout columns:
- same-race outcome used in own predictors:

## Missingness

Report missing fraction for every active canonical feature.

## Reconciliation

Compare yearly race counts with JRA official totals or another independent reference.
Document any exclusions: obstacles, cancellations, invalid rows, small fields, missing time/corner data.

## Decision

- [ ] training-ready
- [ ] blocked

Blocking reasons:
