# Historical Source Probe — primary Kaggle candidate

Dataset: noriyukifurufuru/japan-horse-racing-2010-2025

## Kaggle metadata

~~~json
{
  "title": "Japan Horse Racing Data 2010-2025",
  "subtitle": "",
  "id": 9140476,
  "ref": "noriyukifurufuru/japan-horse-racing-2010-2025",
  "lastUpdated": "2025-12-28T07:41:37.95Z",
  "licenseName": "CC0: Public Domain",
  "totalBytes": 383238987,
  "usabilityRating": 0.29411766
}
~~~

## Download result

- files: 4
- total downloaded bytes: 383238987
- local CI path: ephemeral / not committed

## File schemas

| file | bytes | rows | encoding | columns | error |
|---|---:|---:|---|---|---|
| .complete/datasets/noriyukifurufuru/japan-horse-racing-2010-2025/1/bundle.complete | 0 |  |  |  |  |
| keiba_payouts.csv | 80809803 | 2359257 | utf-8-sig | race_id \| bet_type \| horse_num \| payout \| popularity |  |
| keiba_races.csv | 11722602 | 198044 | utf-8-sig | race_id \| date \| venue \| race_name \| race_number \| course_type \| distance \| turn \| weather \| track_condition \| race_class |  |
| keiba_results.csv | 290706582 | 2253541 | utf-8-sig | race_id \| rank \| frame \| number \| horse_id \| horse_name \| sex_age \| weight \| jockey_id \| jockey_name \| trainer \| time \| margin \| passing \| last_3f \| odds \| popularity \| horse_weight |  |

## Safety

- Raw files were downloaded only into the ephemeral Actions runner.
- No raw row-level source file is committed by this probe.
- The exact license field above must be verified before any row-level derived data is published.
