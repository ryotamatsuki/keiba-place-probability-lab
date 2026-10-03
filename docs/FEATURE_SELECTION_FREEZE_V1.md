# Feature Selection Freeze v1

Date: 2026-10-03 JST  
Applies to: Stage 4 non-market P(top3) baseline

## Frozen decisions

1. Use a compact, stable, pre-race core rather than the largest possible feature set.
2. Replace raw historical percentages with counts and common shrinkage / model handling.
3. Replace raw finishing rank with field-size-normalized recent finish.
4. Prefer race-relative time / final-3F measures over raw best times.
5. Keep one numeric running-position signal; do not double count its median / categorical derivative.
6. Add layoff, distance-change and surface-change variables.
7. Keep market variables completely outside Stage 4.
8. Treat JBIS speed indices as a separately labelled sensitivity block, not canonical raw evidence.
9. Defer jockey / trainer / pedigree / workout features until as-of-date historical panels exist and can be ablated.
10. Keep same-day body weight / going / weather in a separate late-update model because prediction-time availability is unstable.

## Change control

Any change to the canonical Stage 4 allowlist after this freeze requires:

- a written rationale;
- source / timing classification;
- explicit statement whether the change was made before observing the race outcome;
- a new version number;
- retention of v1 for audit.

No horse-specific feature may be added or removed because it helps or hurts a particular runner in the current race.

## Empirical-selection gate

Feature Spec v1 is not labelled “empirically optimal.” That label is reserved for a later historical walk-forward study with feature-block ablation and calibration evaluation.
