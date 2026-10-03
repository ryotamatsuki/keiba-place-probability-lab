# Stage 4 Successor v2 — Production Rehearsal

Status: **PASS — frozen successor refit through 2025 and target scoring completed**

This is a retrospective production rehearsal using the already frozen 2026-10-03
pre-race target matrix. It is not a genuine pre-start v2 lock and the target outcome is not
loaded.

## Frozen production specification

- model: **XGB01 + full-field relative ability**
- XGB01 hyperparameters are unchanged from the development freeze
- five full-field leave-one-out relative-ability features are included
- Phase-3B recent-trend features: **not included**
- Phase-3C explicit interactions: **not included**
- post-hoc calibration: **none**
- market information: **none**

## Production refit

- fit period: 2016-01-09 through 2025-12-28
- eligible rows: 33,069
- races: 2,565
- production prior mean: 0.205600411
- 2025 is used as historical training data, not as an untouched test

## Target coverage

- active starters: 18
- Stage-4 eligible starters: 18
- complete Stage-4 coverage: **True**
- declared field size: 18
- active field size: 18
- raw marginal probability sum: **3.479234353**
- canonical sum-to-three adjustment applied: **no**

## Canonical Stage-4 v2 probabilities

|   rank_eligible |   horse_no | horse_name   |   p_top3_stage4_v2 |
|----------------:|-----------:|:-------------|-------------------:|
|               1 |         18 | ディアナザール      |          0.404602  |
|               2 |         10 | ヒシアイラ        |          0.375584  |
|               3 |         13 | クラスペディア      |          0.373589  |
|               4 |          8 | レッドエヴァンス     |          0.298454  |
|               5 |          4 | メイショウヨゾラ     |          0.281141  |
|               6 |          6 | リリージョワ       |          0.271637  |
|               7 |         15 | タガノアラリア      |          0.247546  |
|               8 |          3 | タマモイカロス      |          0.234899  |
|               9 |          1 | カルチャーデイ      |          0.206193  |
|              10 |         14 | ヤマニンアルリフラ    |          0.182752  |
|              11 |         17 | ヨシノイースター     |          0.134884  |
|              12 |         16 | フロムダスク       |          0.0939756 |
|              13 |          2 | ショウナンアビアス    |          0.093228  |
|              14 |          5 | フィオライア       |          0.081674  |
|              15 |          7 | テーオーダヴィンチ    |          0.061342  |
|              16 |         11 | オタルエバー       |          0.0556867 |
|              17 |         12 | デイトナモード      |          0.0450126 |
|              18 |          9 | タマモブラックタイ    |          0.0370349 |

These are raw marginal probabilities. They are intentionally not transformed to sum to 3.

## Scratch policy

For a scratch before lock, remove the horse from the active relative-feature peer set and
reduce active field_size, but preserve official horse numbers and declared_field_size.
draw_pct remains based on the declared field. A scratch after Stage-6 lock does not mutate
the locked prediction file.

## Stage 5 boundary

This production refit is not used to reconstruct 2023-2024 Stage-5 history. Stage 5 v2 must
use out-of-time non-market predictions for each historical evaluation year and retain
market-only as the mandatory incumbent/reference.
