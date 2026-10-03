# Evaluation Protocol

## Scope

This document defines the repository-wide evaluation principles for marginal `P(top3)`
probability forecasts.

The exact **future model-winner rule** is frozen in
[`MODEL_SELECTION_PROTOCOL_V2.md`](MODEL_SELECTION_PROTOCOL_V2.md).

Historical Stage 4/5 specifications remain immutable audit records. v2 governs future
model-improvement and ensemble-selection cycles.

## Primary probability metrics

### Race-macro Brier score — winner metric for v2

For runner `i` in race `r`:

```text
Brier_r = mean_i((p_ri - y_ri)^2)
Primary Brier = mean_races(Brier_r)
```

where `y ∈ {0,1}` and `p = P(top3)`.

Lower is better. Each race receives equal weight, and all candidates must be scored on exactly
the same eligible race/runner rows.

The runner-micro Brier used in the historical v1 reports is retained as a companion metric for
continuity.

### Log loss — mandatory guardrail

Log loss is always reported. It is a strictly proper scoring rule and more strongly penalizes
catastrophic overconfidence than Brier.

For v2 it is **not** independently optimized. It is used as the secondary safety gate in the
paired incumbent one-standard-error selection rule.

### Calibration — mandatory diagnostic

Report out-of-fold:

- calibration intercept / calibration-in-the-large;
- calibration slope;
- reliability curve;
- ECE for descriptive continuity.

Calibration alone does not choose the winner. In particular, ECE depends on binning and can look
good for low-resolution forecasts.

### Discrimination — diagnostic only

ROC-AUC may be reported to describe ranking/discrimination, but it cannot select the winner
because it does not assess the probability scale.

## v2 winner rule

Do not select the numerically smallest Brier score without accounting for validation noise.

1. Identify the candidate with the lowest race-macro Brier (`Brier-best`).
2. Compute paired per-race loss differences between the incumbent and `Brier-best`.
3. Estimate the standard error with race date as the cluster.
4. Retain the incumbent when it is within one paired clustered SE of `Brier-best` on both
   Brier and log loss.
5. Otherwise select `Brier-best`.

This is a one-SE model-selection regularizer, not a significance test.

For ensemble selection, `market-only` is the mandatory reference/default. A non-zero
non-market blend must clear the same gate to replace it.

Implementation:
`src/keiba_place_lab/model_selection.py`.

## Secondary / descriptive metrics

These may be reported but never override the v2 probability winner rule:

- 複勝圏的中率
- レースごとの最高 `p_place` 馬の複勝圏率
- actual-top3 probability mass
- top-k hit rate
- ROC-AUC
- ECE
- ROI / realized betting profit

ROI is a separate downstream utility question and is deliberately separated from probability
model selection.

## Validation design

- 時系列順に学習・検証を分離する
- 同一レースの馬をtrain/testへ分割しない
- 将来情報・確定結果由来の特徴を禁止する
- final oddsを予測時点で取得していない場合、事前モデルには使わない
- preprocessing / feature selection / tuning / calibrationもtraining側だけでfitする
- model selectionとfinal assessmentの役割を分離する
- compared candidates must use an identical evaluation-row fingerprint

Because the 2025 outcomes have already been inspected, 2025 is not reused as an untouched v2
test set. After a v2 winner is frozen, a production refit may use 2025 as legitimately historical
training data for later live races. The next genuinely untouched assessment is the pre-registered
live Stage 6/7 sequence.

## Prediction lock

レースごとに `probability_estimates.csv` を発走前にコミットし、そのコミットSHAを
`PRE_ANALYSIS_LOG.md` に記録します。以後、予測値の上書きは禁止し、訂正が必要な場合は
新しいファイルと理由を追加します。
