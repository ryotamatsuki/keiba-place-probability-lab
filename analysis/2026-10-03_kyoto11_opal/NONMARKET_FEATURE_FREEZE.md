# Stage 3 — Non-market feature freeze

Race: 2026-10-03 Kyoto 11R Opal Stakes  
Feature freeze: 2026-10-03 09:00 JST  
Scheduled start: 15:30 JST

## Purpose

Stage 3 creates an input set that is independent of betting-market prices.
No win odds, place odds, popularity rank, or Stage 2 market probability is used
to construct the features in `nonmarket_features.csv`.

The market baseline remains frozen separately for later comparison.

## Course / track context

The race is Kyoto turf 1200m, inner course, on the first day of the fourth Kyoto meeting.
JRA states that the first nine days of the fourth Kyoto meeting use the A course.
The inner-course home straight is 328.4m, and the turf inner-course elevation difference is 3.1m.

JRA's course description for Kyoto 1200m notes:

- approximately 300m from the start to the third turn;
- an uphill section early, followed by the third-turn descent;
- a relatively short, flat home straight;
- early positioning requires acceleration/power, but overly aggressive early competition can expose leaders late.

The race card showed sunny / good turf at the Stage 3 access point.
The pre-meeting track report described the turf as generally in good condition after renovation
and confirmed A-course use.

These are global context variables. Stage 3 does **not** mechanically assert
that A-course / good turf means an inside or front-running bias. That would require historical
evidence and interaction testing.

## Derived per-horse features

### Recent running position

For each usable recent race:

```text
early_position_pct = (early_position - 1) / (field_size - 1)
```

This normalizes a horse running 3rd in a 9-horse field and 3rd in an 18-horse field.

The mean and median over up to four recent starts are stored.
The categorical `pace_style_heuristic` is descriptive only:

- front: <= 0.20
- forward_mid: <= 0.45
- mid_rear: <= 0.70
- deep_rear: > 0.70

The numeric feature is preferred in modeling; the label exists for auditability.

### Pace-field diagnostic

Six runners have a recent normalized early-position mean <= 0.25:

- #1 カルチャーデイ
- #4 メイショウヨゾラ
- #6 リリージョワ
- #10 ヒシアイラ
- #13 クラスペディア
- #18 ディアナザール

That is 6 / 18 = 0.3333 of the field.

This supports a **competitive-front-group diagnostic**, not a deterministic "fast pace" claim.
Several horses are position-variable, and race tactics can change.

### Distance / course experience

The file stores:

- best recorded Kyoto turf-1200 time where available;
- difference from the current Kyoto 1200 course record of 66.4 seconds;
- best listed turf-1200 time at other tracks;
- an explicit missing-value flag for horses without a Kyoto-1200 time.

Examples of important missingness:

- #2 ショウナンアビアス has no recent turf starts in the four-race window and no stored turf-1200 best time.
- #6 リリージョワ has no stored turf-1200 time; its recent evidence is mainly 1400/1600m.
- #4 メイショウヨゾラ has recent 1200m turf form but no stored Kyoto-1200 best time.

Missing is not encoded as zero.

### Recent form / class exposure

Stored descriptors include:

- recent-three JBIS speed-index mean;
- recent-three top-three count;
- recent-four finish mean and top-three count;
- number of recent turf starts;
- number of recent turf-1200 starts;
- number of recent open/listed/graded starts;
- number of recent graded starts.

These are descriptive features only. Stage 3 does not assign a final horse score.

### Handicap change

`weight_delta_vs_last_kg = current_handicap - last-race_weight`.

Examples:

- #4 メイショウヨゾラ: -2.0kg
- #10 ヒシアイラ: -2.0kg
- #12 デイトナモード: -3.0kg
- #16 フロムダスク: +2.5kg

A lower assigned weight is not automatically treated as a positive coefficient; Stage 4 must
define or learn the relationship consistently.

## Model-input allowlist

For the first non-market baseline, the allowed horse-specific inputs are:

- frame
- sex / age
- handicap_kg
- weight_delta_vs_last_kg
- jbis_speed_index
- recent3_jbis_speed_mean
- recent3_top3_count
- overall_place_rate
- turf_place_rate
- kyoto_turf_place_rate
- turf1200_place_rate
- kyoto1200_best_sec
- kyoto1200_vs_track_record_sec
- other_turf1200_best_sec
- recent4_finish_mean
- recent4_top3_count
- recent4_turf_count
- recent4_turf1200_count
- recent4_open_plus_count
- recent4_graded_count
- recent4_early_pos_pct_mean
- recent4_early_pos_pct_median

Horse name/number are identifiers, not predictors.

Jockey and trainer identities are **not** included in the first baseline, because this repository
does not yet contain a properly historical, as-of-date rider/trainer performance panel. This avoids
replacing data with subjective reputation.

Market fields are explicitly forbidden in the Stage 4 non-market model.

## Anti-leakage / anti-anchoring

- JRA's current race-card page also displays odds, but Stage 3 ignores those fields.
- The Stage 2 market ranking is not used to select or change a feature for an individual horse.
- #10 ヒシアイラ remains documented as a pre-existing conversational anchor.
- No aggregate non-market ranking is produced in Stage 3.
- Final probabilities remain unlocked.

## Stage 3 status

COMPLETE.

Next: Stage 4 defines a transparent non-market P(top3) baseline using the frozen allowlist.
