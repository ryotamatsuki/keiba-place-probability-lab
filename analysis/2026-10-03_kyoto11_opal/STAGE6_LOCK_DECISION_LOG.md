# Stage 6 — Retrospective Pre-race Lock Rehearsal

Status: **PASS — RETROSPECTIVE_DRY_RUN**

This is an operational rehearsal of the pre-race locking step. It was executed
after the scheduled start and must not be represented as a genuine pre-start lock.

## Locked source

- source Stage 5 commit: `086f7a2a20e20dc226db6535b4738905e2011f1b`
- frozen market timestamp: `2026-10-03T08:35:00+09:00`
- model version: `stage5-ensemble-v1`
- runners: 18
- sum P(top3): `3.000000000000`

## Information barriers

- target outcome loaded: **no**
- later/final target odds loaded: **no**
- payout loaded: **no**
- popularity loaded: **no**
- source inputs are only the already-frozen Stage 2 / Stage 4 / Stage 5 files

## Locked leading probabilities

| rank | horse_no | horse_name | p_ensemble | low | high |
|---:|---:|:---|---:|---:|---:|
| 1 | 10 | ヒシアイラ | 0.342056 | 0.340497 | 0.373911 |
| 2 | 6 | リリージョワ | 0.311323 | 0.248073 | 0.314479 |
| 3 | 18 | ディアナザール | 0.287488 | 0.284162 | 0.355232 |
| 4 | 3 | タマモイカロス | 0.277096 | 0.241292 | 0.278869 |
| 5 | 4 | メイショウヨゾラ | 0.243440 | 0.243381 | 0.244864 |
| 6 | 8 | レッドエヴァンス | 0.230565 | 0.224593 | 0.230860 |

## File fingerprints

- `stage5_ensemble.csv`: `560387ab264462af0421109f1f045f9335de3c6e030c91fba8837b0becad365f`
- `nonmarket_baseline.csv`: `a4340b194ef38daaf14d3e45f9a983f7bef498eddae7a726f15e3efaf46aa149`
- `market_baseline.csv`: `19462942f4493785c955512b486a60b0098dc65959a58339290fa22f2f45ca6e`
- `probability_estimates.csv`: `eefa16ad709c7568690e3f75a0df22491a31b3cb6aebdeebcf27fab274bc9e66`

## Decision

The values in probability_estimates.csv are treated as immutable rehearsal-lock
values from this point forward. Future result evaluation must compare against these
values and must not rewrite them.

For the next live race, this exact step must run and commit before scheduled start.
