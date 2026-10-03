# Scope Expansion Stage 1 — Frozen Result

Status: **COMPLETE — CURRENT 1200m SUCCESSOR RETAINED**  
Comparison scope: training-population width only  
Common evaluation: 2023-2024 turf 1200m, 6,313 rows / 495 races  
Evaluation fingerprint: `a72fa10cc9adcd1ceee4c5b5eacd0dfdcdfc33fc984bf0b7866ec70b617bfd85`

## Frozen candidates

- A: current 1200m training population
- B: turf 1000-1400m training population
- C: turf 1000-2600m training population

All three use the same frozen XGB01 hyperparameters, base features, five relative-ability features,
EB strength 6, preprocessing rules, and common 1200m evaluation rows.

## Result

Race-macro Brier:

- B: `0.148774168`
- A: `0.148958593`
- C: `0.149394066`

Point-Brier-best is B.

A versus B:

- A minus B Brier difference: `0.000184425`
- paired race-date-clustered SE: `0.000324919`

The improvement is smaller than one paired clustered SE, so A remains inside the frozen
incumbent-retention band.

**Decision: retain A_current_1200.**

C does not improve 1200m prediction and is worse than A on the primary score.

## Year-level diagnostic

2023 race-macro Brier:

- A: `0.149100523`
- B: `0.149431676`
- C: `0.149655969`

2024:

- B: `0.148103242`
- A: `0.148813767`
- C: `0.149126818`

B therefore does not show a stable same-direction improvement across the two outer years:
it is worse in 2023 and better in 2024.

## Interpretation

The experiment does not support replacing the 1200m specialist merely by widening the training
population while holding the feature representation and model capacity fixed.

This does not show that nearby-distance data are useless. It shows that the current feature
representation does not extract a sufficiently stable incremental benefit from those rows to pass
the preregistered replacement gate.

The broad 1000-2600m model is especially informative: substantially more training data alone does
not improve 1200m performance. This motivates testing distance-suitability representations before
considering a global model or distance-regime deployment.

## Data audit boundary

- existing 1200m evaluation reproduced exactly;
- all full-field relative contexts complete;
- straight-course races excluded from primary A/B/C populations;
- 143 anomalous non-round source-distance rows across 10 races occur only in the separately
  excluded straight-course subset;
- primary A/B/C population contains no non-100m source distances;
- 2025 not used in selection;
- market data not used;
- no sum-to-three adjustment.

## Next stage

Proceed to Step 4 as a separate preregistered experiment.

Distance-suitability feature blocks should be introduced separately, without changing the
training-population comparison retrospectively:

1. same-surface +/-200m neighborhood history;
2. same-surface distance-regime history;
3. same-course x same-surface x exact-distance history.

Each block must be applied consistently to A/B/C candidates so feature improvement remains
separable from training-population expansion.
