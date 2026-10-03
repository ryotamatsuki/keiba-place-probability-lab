# Stage 4 Baseline Specification v2.1 — QA correction freeze

Race: 2026-10-03 Kyoto 11R Opal Stakes  
Date: 2026-10-03 JST  
Status: frozen before corrected rerun

This document supersedes v2 for execution while retaining v2 as an audit record.

## Why v2.1 is required

The first historical execution of v2 exposed two implementation-contract issues during QA.

First, the frozen Stage 3.6 Phase-A cohorts are **runner-filtered**: a row enters only when the horse had at least three prior career starts. Therefore a historical race group in those cohort Parquets can contain fewer rows than the race's actual `field_size`. Enforcing `sum P(top3)=3` on such a partial group is invalid because excluded runners can occupy top-three slots.

Second, the frozen target matrix uses English display labels such as `Kyoto` and `Listed_open`, while the historical adapter stores the matching categorical contexts as `京都` and `Open`. Without explicit canonicalization those target values are unseen one-hot categories.

Both issues are structural and were detected from QA diagnostics. The invalid first run is discarded. Its 2025 performance is not used to select a cohort, regularization value, feature block, or any other model choice.

## Corrected historical evaluation contract

Model fitting and selection remain unchanged:

- training: 2016–2022;
- selection: 2023–2024;
- candidate training cohorts: all turf, turf 1000–1400m, turf 1200m;
- every candidate is evaluated on the same eligible 2023–2024 turf-1200 runner rows;
- model: L2 logistic regression;
- candidate `C`: 0.1, 1, 10;
- primary selection metric: marginal Brier score;
- tie-breakers: marginal log loss, then smaller `C`.

Historical validation and 2025 test use the model's **raw marginal P(top3)** for every eligible runner row. They do not impose a three-slot constraint on incomplete race groups.

For diagnostic purposes only, race-level `sum P(top3)=3` is checked on the subset of historical races where:

`number of retained rows == recorded field_size`.

That completeness test uses only pre-race `field_size`, never the outcome.

## Target race consistency

The target matrix contains all 18 declared runners. Its raw marginal probabilities are therefore transformed by a common logit intercept, solved by bisection, until:

`sum_i P(top3_i) = 3`.

This preserves ranking.

Before model application, categorical display labels are deterministically mapped to the historical adapter vocabulary. For this race:

- `Kyoto -> 京都`
- `Listed_open -> Open`

No outcome or market information is involved in this mapping.

## 2025 handling

After cohort and `C` are selected using 2023–2024 only, the chosen specification is refit on the selected 2016–2024 cohort and evaluated on 2025. The target model also uses only 2016–2024. No 2025 outcome enters target fitting.

Because an invalid v2 diagnostic run was executed before this QA correction, the final report must disclose that run and state explicitly that its 2025 metric was discarded and did not alter model selection.

## Other frozen controls

All v2 controls remain in force:

- market-like columns hard-fail;
- common empirical-Bayes shrinkage, prior strength 6;
- fitting-sample median plus missing indicators for numeric missingness;
- categorical `UNKNOWN` and unknown-safe one-hot encoding;
- no `course_layout` or `handicap_indicator` historical model input;
- no Stage 2 blend;
- no horse-10 anchor variable, prior, constraint, or weight;
- leave-one-feature-block-out sensitivity analysis.


## Terminology

The final report calls 2025 a **held-out test** rather than claiming it remained literally unseen after the discarded QA run. The invariant required for scientific validity is preserved: 2025 outcomes never determine feature definitions, cohort choice, regularization, sensitivity choices, or target-model fitting.
