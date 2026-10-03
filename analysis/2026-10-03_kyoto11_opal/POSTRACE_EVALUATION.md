# Stage 7 — Post-race Evaluation

Status: **PASS — official outcome joined to immutable Stage 6 rehearsal lock**

## Evaluation status

This is a retrospective pipeline test, not a genuine pre-start forecast audit,
because Stage 6 missed the scheduled-start timing gate. The locked probabilities
are nevertheless immutable and were not changed after the outcome was known.

- official JRA outcome source: https://www.jra.go.jp/JRADB/accessS.html?CNAME=pw01sde0108202604011120261003/2F
- locked probability SHA256: `eefa16ad709c7568690e3f75a0df22491a31b3cb6aebdeebcf27fab274bc9e66`
- locked source commit: `086f7a2a20e20dc226db6535b4738905e2011f1b`
- frozen market timestamp: `2026-10-03T08:35:00+09:00`
- later/final target odds substituted into forecast: **no**

## Official top 3

| finish_rank | horse_no | horse_name | p_stage2_market | p_stage4_nonmarket | p_ensemble_stage6 | stage6_forecast_rank |
|---:|---:|:---|---:|---:|---:|---:|
| 1 | 6 | リリージョワ | 0.349399 | 0.247155 | 0.311323 | 2 |
| 2 | 18 | ディアナザール | 0.309414 | 0.352212 | 0.287488 | 3 |
| 3 | 8 | レッドエヴァンス | 0.239937 | 0.224102 | 0.230565 | 6 |

## One-race probability metrics

| model | brier | log_loss | actual_top3_probability_mass | actual_top3_forecast_ranks |
|:---|---:|---:|---:|:---|
| stage2_market_0835 | 0.107207 | 0.335163 | 0.898750 | 2/3/6 |
| stage5_ensemble | 0.111097 | 0.350378 | 0.829376 | 2/3/6 |
| stage6_locked | 0.111097 | 0.350378 | 0.829376 | 2/3/6 |
| stage4_nonmarket | 0.113597 | 0.355176 | 0.823468 | 2/4/7 |
| uniform_3_over_18 | 0.138889 | 0.450561 | 0.500000 | all_tied_1-18 |

Lower Brier and log loss are better. Probability mass is the sum of each model's
three marginal probabilities on the horses that actually finished 1st-3rd.

## What this race says

- Stage 6/Stage 5 ranked all three actual top-3 horses inside its top six (2nd, 3rd, 6th).
- The raw 08:35 Stage 2 market had the same forecast ranks for the actual top three.
- On this single race, raw Stage 2 market beat the Stage 5/6 ensemble on both Brier and log loss.
- Stage 4 non-market alone was weaker than the raw market on this race.
- Therefore the historical 95/5 ensemble advantage did not reproduce in this one-race test.

This single race cannot overturn the 2025 held-out aggregate comparison; equally, the
historical aggregate result must not be used to claim that the blend improved this race.

## Largest ensemble-vs-market error contributions

| horse_no | horse_name | finish_rank | top3_label | p_stage2_market | p_ensemble_stage6 | market_squared_error | stage6_squared_error | ensemble_minus_market_sqerr |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 6 | リリージョワ | 1 | 1 | 0.349399 | 0.311323 | 0.423282 | 0.474276 | 0.050994 |
| 18 | ディアナザール | 2 | 1 | 0.309414 | 0.287488 | 0.476909 | 0.507673 | 0.030764 |
| 8 | レッドエヴァンス | 3 | 1 | 0.239937 | 0.230565 | 0.577696 | 0.592031 | 0.014335 |
| 13 | クラスペディア | 6 | 0 | 0.147483 | 0.162582 | 0.021751 | 0.026433 | 0.004682 |
| 15 | タガノアラリア | 12 | 0 | 0.101390 | 0.119821 | 0.010280 | 0.014357 | 0.004077 |

## Largest pre-race market/non-market disagreements

| horse_no | horse_name | finish_rank | top3_label | p_stage2_market | p_stage4_nonmarket | abs_component_gap |
|---:|:---|---:|---:|---:|---:|---:|
| 13 | クラスペディア | 6 | 0 | 0.147483 | 0.286066 | 0.138583 |
| 16 | フロムダスク | 17 | 0 | 0.197097 | 0.085672 | 0.111425 |
| 6 | リリージョワ | 1 | 1 | 0.349399 | 0.247155 | 0.102244 |
| 15 | タガノアラリア | 12 | 0 | 0.101390 | 0.188117 | 0.086727 |
| 3 | タマモイカロス | 16 | 0 | 0.302453 | 0.240499 | 0.061954 |

## Error analysis rule

No causal explanation is inferred from the finishing order alone. The error review is
restricted to discrepancies that were already observable in the frozen pre-race probabilities:
market/non-market disagreement, forecast rank, and probability error.

## Reproducibility

- Stage 7 workflow run #2: `37110151054` — SUCCESS
- evaluated head SHA: `56ddf0abff6e1991a1a2246a47f9e1e8d6912356`
- artifact: `stage7-postrace-evaluation`
- artifact ID: `11270045420`
- artifact digest: `sha256:c224ba078232365bcd4e5a360c43df87f2a0d03b29acce254374517a525ef252`
- locked Stage 6 probability file remained SHA256 `eefa16ad709c7568690e3f75a0df22491a31b3cb6aebdeebcf27fab274bc9e66`

## Stage conclusion

The full outcome/evaluation pipeline works. The next live experiment should repeat Stages
1-6 before post time, then use this unchanged Stage 7 procedure after the official result.
