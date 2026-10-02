# Stage 1 Race Snapshot — 2026-10-03 京都11R オパールS

## Freeze point

- static data accessed: **2026-10-03 08:47:27 JST**
- market snapshot: **2026-10-03 08:35 JST**
- scheduled start: **15:30 JST**
- race: Listed, turf 1200m, handicap, 18 runners
- conditions shown at snapshot: **sunny / good turf**

This is a **pre-race snapshot**, not a prediction result.

## What is stored

`features.csv` contains one row per runner with:

- frame / horse number / horse name
- sex-age / jockey / handicap / trainer
- JBIS speed index
- historical place rates derived from overall, turf, Kyoto-turf and turf-1200 records
- mean of the latest three *available* JBIS speed indices
- count of top-three finishes in the latest three starts
- the 08:35 win/place odds snapshot and market ranks

`market_snapshot_0835.csv` preserves the small factual odds snapshot separately so later market movement can be studied.

## Missingness rules

Missing values are intentional.

- A horse with no prior start under a condition is **missing**, not a 0% historical place rate.
- Race exclusion / a race with no comparable JBIS speed index is omitted from the recent-three speed mean.
- Historical place rates are descriptive frequencies, **not probability estimates**.

## Immediate data-quality observations

1. Several runners have small-sample 100% historical place rates under Kyoto or 1200m. These must not be treated as 100% future probabilities.
2. No. 6 リリージョワ has no prior turf-1200 start in the stored distance table; its Kyoto record comes from other distances.
3. No. 2 ショウナンアビアス has no turf-1200 or Kyoto-turf start in the referenced JRA-record tables; recent form is mainly dirt.
4. The JBIS speed index is a published third-party feature. It is not treated as ground truth and will be tested separately.
5. Odds are an early-morning snapshot, not final prices. Market movement must be timestamped rather than overwritten.

## Anti-anchoring check

No. 10 ヒシアイラ had already been mentioned before this systematic snapshot. That fact remains documented.
Stage 1 does not declare No. 10, or any other runner, the model favorite.

## Stage 1 status

**PASS for structure and pre-race timing.**

Next: Stage 2 constructs a market baseline for `P(top3)`, with an explicit method for converting/using the place-odds market without pretending that the displayed odds range is itself a probability.
