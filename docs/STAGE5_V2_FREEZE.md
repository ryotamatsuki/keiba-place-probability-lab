# Stage 5 v2 Freeze

Status: **HISTORICAL-DOMAIN SELECTION COMPLETE / LIVE-MORNING ADOPTION NOT AUTHORIZED**  
Stage-4 successor: `XGB01 + full-field relative ability`  
Selection protocol: `docs/MODEL_SELECTION_PROTOCOL_V2.md`  
Stage-5 v2 spec: `docs/STAGE5_V2_SPEC.md`

## Frozen historical-domain result

Historical market domain:
- final win odds;
- reciprocal odds normalized across every starter;
- Harville/Plackett-Luce marginal P(top3).

### Market calibration

A logit intercept+slope calibrator was fitted on paired 2023 rows and evaluated on 2024.

2024 race-macro Brier:
- raw market: `0.139486`
- calibrated market: `0.138085`

Paired raw-minus-calibrated Brier improvement:
- mean: `0.001401`
- date-clustered SE: `0.000665`

Raw market lies outside the frozen one-SE retention band, so calibrated market is adopted as the
**historical final-odds market-only base**.

This does not authorize applying the same calibration to an early-morning live market as the
canonical forecast.

### Market / non-market blend

The frozen 2024 point-Brier-best weight is:

- market: 95%
- non-market: 5%

Its race-macro Brier is `0.138036`, versus `0.138085` for market-only.

The point improvement is therefore only `0.0000485`. The paired race-date-clustered SE versus
the point-best is approximately `0.0001113`.

The mandatory market-only incumbent remains inside the one-SE band.

Therefore the v2 selection decision is:

**retain market-only; selected non-market weight = 0.00**

The former Stage-5 v1 95/5 blend remains a historical v1 audit result only.

## 2025 known-outcome audit

2025 was already observed before this development cycle and is not an untouched test.

The Stage-4 v2 2025 probabilities were nevertheless reconstructed correctly out of time:
- Stage-4 fit through 2024 only;
- predict 2025;
- no production-through-2025 model used for backprediction.

Using the recipe frozen by 2024:

- market-only historical-domain base race-macro Brier: `0.142938`
- selected Stage-5 v2 result race-macro Brier: `0.142938`

Because the selected non-market weight is zero, these are identical.

The 2025 audit is descriptive and cannot alter selection.

## Live-morning production policy

Historical final odds and live morning odds are different domains.

Until time-matched market snapshots have been accumulated prospectively:

- canonical live Stage-5 v2 = **raw live market-only**;
- Stage-4 v2 non-market forecast = shadow challenger;
- historical-final-odds calibrated market = shadow only;
- historical-final-odds blend recipe = shadow only;
- no historical-final-odds result can authorize a morning blend.

For the frozen 2026-10-03 08:35 rehearsal:

- raw market probability sum: `3.000000`
- raw Stage-4 v2 probability sum: `3.479234`
- historical-domain shadow sum: `3.230663`
- no sum-to-three transform is applied to the shadow forecast.

## Next empirical decision

The next actual adoption decision is prospective.

For multiple future eligible turf-1200 races:

1. capture a market snapshot at the preregistered live lock time;
2. compute Stage-4 v2 from the same pre-race information boundary;
3. store market-only and one or more preregistered shadow challengers;
4. commit Stage 6 before scheduled post time;
5. evaluate only after the result is available;
6. wait until the preregistered review count/time;
7. apply race-macro Brier plus paired race-date-clustered one-SE again.

No single live race can change the frozen policy.

## Information boundary

- 2023 Stage-4 OOF trained through 2022.
- 2024 Stage-4 OOF trained through 2023.
- 2025 audit Stage-4 OOF trained through 2024.
- production Stage-4 fit through 2025 was never used to back-predict 2023-2025.
- 2025 did not select calibration policy or blend weight.
- target outcome was not loaded into Stage-5 v2 target rehearsal.
