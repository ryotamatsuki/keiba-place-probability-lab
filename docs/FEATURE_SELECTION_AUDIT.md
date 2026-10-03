# Feature Selection Audit v1 — pre-Stage-4

Date: 2026-10-03 JST  
Scope: Phase A non-market P(top3) model  
Status: **FROZEN FOR STAGE 4 AFTER MATERIALIZATION**

## Executive conclusion

The original Stage 3 feature set was directionally reasonable but **not yet suitable as the canonical Stage 4 input set**.

The audit found four main problems:

1. several features were redundant or almost redundant;
2. raw historical place rates were too sensitive to small denominators;
3. raw best times were not sufficiently normalized for course / distance / race environment;
4. several high-value, repeatedly supported pre-race variables were missing, especially days since previous race, field-normalized recent finishing performance, previous-race condition changes, and relative time / final-3F performance.

Therefore the original Stage 3 file remains a descriptive snapshot, while the Stage 4 canonical allowlist is replaced by Feature Spec v1 below.

## Evidence used

### 1. JRA-VAN operational data-mining specification

JRA-VAN describes an operational prediction system using objective pre-race variables including:

- racecourse, distance, course type, race grade, weather / going;
- field size and share of front / forward-running horses;
- gate, body weight, assigned weight and weight-to-body-weight ratio;
- jockey and trainer recent win / top-3 rates;
- sex and age;
- starts / wins / top-3 counts overall, same track and same distance;
- running style;
- previous-race condition, layoff, race level;
- relative race-time and final-3F ratios over the prior three starts;
- recent best relative time / final-3F;
- workout information.

Source: https://jra-van.jp/fun/dm/mining.html

The important design lesson is not to copy every variable. It is that **raw time should be normalized by race / course / distance context**, and that recency, race level, distance/track experience, running style and carried weight are first-class predictor groups.

### 2. 2026 JRA leakage-aware temporal-validation study

Sugiura (2026) used JRA flat-racing data with 2015–2022 training, 2023–2024 validation and a 2025–2026 future test set. The simpler 25-feature model based on basic entry information plus raw historical features outperformed a 110-feature augmented/domain-theory model on the place outcome.

Relevant simpler features included:

- racecourse / surface / distance / draw / sex / age / assigned weight;
- starts before target race;
- days since previous race;
- historical win / place rate;
- mean finish over recent starts;
- distance change and surface change from previous start.

The study explicitly recommends leakage audit, temporal validation and feature-block ablation before deployment.

Source: https://doi.org/10.3389/frai.2026.1896192

This is especially relevant because it is recent, JRA-specific and evaluates place prediction under future temporal validation.

### 3. Ranking-oriented horse-racing research

Chung et al. (2024) found pairwise learning-to-rank approaches stronger than pointwise approaches in Seoul racing and standardized race records by distance. Horse performance and previous race records were among the important variables.

Source: https://doi.org/10.5351/KJAS.2024.37.2.239

A 2025 SHAP-based racing study reported that sectional-speed variables improved NDCG and that early race development was important in short / middle-distance races.

Source: https://doi.org/10.7583/JKGS.2025.25.6.57

These findings support keeping a compact pace / early-position block and prioritizing normalized performance rather than raw finishing time.

### 4. Classical computer-handicapping design

Benter's computer-handicapping framework emphasizes current condition, recent performance, time since last race, normalized past times, strength of competition, weight carried, post position, jockey contribution and distance / surface / track preference, with unseen-race testing to control overfit.

Bibliographic source: https://doi.org/10.1142/9789812819192_0019

## Audit of the original Stage 3 variables

The current race snapshot contains 18 runners. A Python audit of the numeric Stage 3 file identified:

- `kyoto1200_best_sec` and `kyoto1200_vs_track_record_sec`: exact linear duplication (correlation 1.000);
- `recent4_early_pos_pct_mean` and `recent4_early_pos_pct_median`: correlation 0.954 in this race;
- `recent3_top3_count` and `recent4_top3_count`: correlation 0.861;
- `recent3_top3_count` and raw `recent4_finish_mean`: correlation -0.870.

These single-race correlations are **not feature-importance evidence**, but they are enough to flag unnecessary duplication before a hand-built Stage 4 score.

Missingness in the current snapshot was 3/18 for Kyoto-1200 best time and 2/18 for other-turf-1200 best time.

## Canonical Feature Spec v1

### Block A — stable current-entry variables

Included:

- `draw_pct`: normalized horse number / gate position within the field;
- `sex`;
- `age`;
- `assigned_weight_kg`;
- `assigned_weight_delta_from_prev_kg`;
- `days_since_prev`;
- `distance_change_from_prev_m`;
- `surface_changed_from_prev`.

Why: repeatedly supported by operational and leakage-aware models; available before race outcome; low subjectivity.

### Block B — experience and suitability as counts

Included as **counts**, not unsmoothed percentages:

- `career_starts`, `career_top3`;
- `turf_starts`, `turf_top3`;
- `same_distance_starts`, `same_distance_top3`;
- `same_course_starts`, `same_course_top3`.

Why: raw rates such as 1/1 = 100% are unstable. Stage 4 must either use counts directly or apply one common shrinkage rule. A horse with no prior start is missing / zero exposure, not a 0% latent ability horse.

### Block C — recent form

Included:

- `recent3_finish_pct_mean`: finish positions normalized by field size;
- `recent3_top3_count`;
- `recent3_open_plus_count`;
- `recent3_graded_count`;
- `days_since_prev` (also current-condition block).

The raw `recent4_finish_mean` is replaced by field-normalized finish percentile.

### Block D — normalized performance / speed

**Required high-priority representation**:

- `recent3_relative_time_mean`;
- `recent3_relative_last3f_mean`;
- optionally `recent1_relative_time` and `recent1_relative_last3f`.

Preferred definitions follow the JRA-VAN principle of comparing a horse's time / final-3F against a race-level reference so performances from different tracks and distances are not treated as directly comparable raw seconds.

If these features cannot be materialized rights-clean and pre-race, Stage 4 must not silently substitute raw best time. Instead it may run:

- Core model without speed block; and
- a separately labelled sensitivity model using pre-race JBIS speed indices.

### Block E — running style / pace

Included:

- `recent4_early_pos_pct_mean`;
- race-level `front_forward_share`.

Not included in the model:

- `recent4_early_pos_pct_median` (redundant for v1);
- `pace_style_heuristic` (diagnostic label only).

The numerical early-position feature is retained because short-distance racing evidence supports early-race development as informative, while the label is only a human-readable rendering of the same signal.

### Block F — race-level context

Stored for model portability / future historical training:

- `field_size`;
- `racecourse`;
- `surface`;
- `distance_m`;
- `course_layout`;
- `race_class`;
- `handicap_indicator`.

For a single current race these do not rank horses by themselves, but across a historical panel they are necessary context.

## Deferred blocks — not in Stage 4 canonical v1

### Jockey / trainer rolling performance

JRA-VAN uses rolling jockey / trainer rates and horse-racing studies commonly include them. However Stage 4 will not use raw identities, subjective reputation or current-season end-state summaries.

They move to a future optional block only after a genuine as-of-date rolling panel (for example last-100 starts) can be constructed and ablated temporally.

### Pedigree

Sire / damsire can carry signal, especially for inexperienced horses, but it is a high-cardinality block and is not required for the first open-class adult-horse baseline. Add only as an ablated future block.

### Training / workout / health

Operational systems and recent ranking research indicate value, but the current rights-clean public-data pipeline does not yet provide a complete, timestamp-stable training / health panel for all runners. Deferred.

### Same-day body weight / weather / going

Useful close to post time, but availability and stability are timestamp-dependent. They are excluded from the **early canonical model** and reserved for an explicitly separate late-update model.

### Market odds

Excluded from Stage 4 by design. Market-only remains Stage 2; blending is Stage 5.

## Explicit removals from canonical Stage 4

- raw `overall_place_rate`, `turf_place_rate`, `kyoto_turf_place_rate`, `turf1200_place_rate`;
- raw `kyoto1200_best_sec`, `kyoto1200_vs_track_record_sec`, `other_turf1200_best_sec`;
- duplicate `recent4_early_pos_pct_median`;
- categorical `pace_style_heuristic`;
- raw `recent4_finish_mean`;
- `jbis_speed_index` and `recent3_jbis_speed_mean` from the canonical core.

The JBIS variables are retained only in a **named sensitivity block** because they are useful pre-race external ratings but are not fully transparent raw features.

## What “optimal” means here

This audit does **not** claim that Feature Spec v1 is empirically optimal.

It is the strongest defensible pre-registered candidate set under the current constraints:
public / rights-clean sources, pre-race availability, no market information, low redundancy and literature-supported predictor families.

Empirical optimality can only be established after a historical panel exists and candidate blocks are tested with:

1. chronological train / validation / test splits;
2. race-grouped evaluation;
3. feature-block ablation;
4. matched model hyperparameters;
5. Brier score / log loss / calibration;
6. bootstrap uncertainty;
7. comparison against the market baseline.

Stage 4 may proceed only after the required v1 variables have been materialized or explicitly marked unavailable.
