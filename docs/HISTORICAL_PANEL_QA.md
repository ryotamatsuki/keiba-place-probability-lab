# Stage 3.6 Historical Training Panel QA

Status: **BLOCKED**

## Sources

- Primary historical race/results: Kaggle Japan Horse Racing Data 2010-2025, CC0.
- 2010-2020 actual dates: Kaggle legacy JRA dataset, CC BY 4.0.
- 2021-2024 date helper pinned at commit da8eb65883495b4ae37fa65dc3a0d9c9076a9e3a.
- 2025 date helper pinned at commit 3ca10446d8ae28a21ee67ceed6429bacc0ebba93.
- Helper repositories are used only to reconstruct factual race_id-to-date mappings;
  their row-level source data are not redistributed.

## Source integrity

- malformed raw result rows (all racing): 217
- malformed raw result rows belonging to JRA race IDs: 0
- date-map overlap rows cross-checked: 2040
- date-map overlap mismatches: 0
- date-map rows: 137175

## Standardized flat panel

- flat races: 53172
- starter rows: 752668
- Niigata straight-1000m races restored from blank course metadata: 402
- obstacle races excluded by documented winner-last3F<20 rule: 2028
- invalid core rows excluded: 0
- 2016-2025 flat races: 33230
- independent benchmark: 33290
- benchmark exact match: False

## Phase A cohort

- surface: turf
- field size: >= 8 starters
- previous starts: >= 3
- runner rows: 245651
- races: 22054

| split      |   runner_rows |   races |   top3_prevalence |
|:-----------|--------------:|--------:|------------------:|
| warmup     |         93219 |    8305 |          0.214248 |
| train      |        107688 |    9677 |          0.220879 |
| validation |         29795 |    2711 |          0.222151 |
| test       |         14949 |    1361 |          0.221955 |

## Leakage / structural checks

- duplicate race_id x horse_id: 0
- previous date >= current date: 0
- forbidden market columns in artifact: 0
- current-race odds, popularity and payout are absent.
- rolling/cumulative horse features are shifted to strictly prior starts.

## Missingness in Phase A artifact

~~~text
race_class         0.379921
finish_position    0.002223
~~~

## Blockers

- 2016-2025 flat race count 33230 != benchmark 33290

Stage 4 model fitting is unblocked only when Status is PASS.
