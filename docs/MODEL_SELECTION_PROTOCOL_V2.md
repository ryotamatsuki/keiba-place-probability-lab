# Model Selection Protocol v2

Status: **FROZEN FOR FUTURE MODEL-IMPROVEMENT CYCLES**  
Scope: Phase A marginal `P(top3)` model selection and market/non-market ensemble selection  
Supersedes: only future selection decisions. Historical Stage 4/5 freeze documents remain immutable audit records.

## 1. Decision target

The object being selected is a **probability forecaster**, not a threshold classifier.

For runner `i` in race `r`:

```text
y_ri in {0,1}
p_ri = forecast P(top3)
```

The selection loss must therefore reward honest probabilities. Accuracy, hit rate, F1, ROI,
top-k hit rate, and ROC-AUC do not determine the winner.

## 2. Why Brier is primary

Both Brier score and logarithmic score are strictly proper scoring rules. They are minimized in
expectation by reporting the true probability. Brier is the primary rule here because:

- it directly measures squared error of the marginal `P(top3)`;
- it is bounded for binary probabilities and is less dominated by a single extreme miss than log loss;
- it combines calibration/reliability and resolution in one proper score;
- it remains easy to interpret and compare in paired model experiments.

Log loss remains mandatory because it is also strictly proper and is more sensitive to
catastrophic overconfidence. It is a secondary diagnostic and deterministic tie-breaker, not a
co-equal optimization target.

Calibration diagnostics are mandatory, but calibration alone is not a winner rule: a nearly
constant forecast can be well calibrated while having little resolution. ECE is also sensitive to
binning choices.

ROC-AUC is diagnostic only. It measures ranking/discrimination and is invariant to monotone
recalibration, so a model can keep the same AUC while its probability scale becomes materially
wrong.

## 3. Analysis unit and weighting

### Primary score: race-macro Brier

For every race, compute Brier over the **common eligible runner set** shared by all candidates:

```text
Brier_r = mean_i_in_race((p_ri - y_ri)^2)
Primary Brier = mean_races(Brier_r)
```

Each race receives equal weight. This prevents large fields from receiving more influence merely
because they contain more runner rows.

All compared candidates must be evaluated on exactly the same race/runner rows.

### Mandatory companion scores

Record, but do not optimize independently:

- race-macro log loss;
- runner-micro Brier and log loss, for continuity with the v1 reports;
- calibration-in-the-large / calibration intercept;
- calibration slope;
- reliability diagram;
- ECE as a descriptive diagnostic only;
- ROC-AUC as a discrimination diagnostic only.

## 4. Chronology and information boundaries

### Historical v1 audit

The existing 2025 Stage 4/5 results remain valid historical audit results under their frozen
specifications. They must not be rewritten.

### Future v2 model selection

Because the 2025 outcomes and v1 2025 metrics have already been observed, **2025 is not an
untouched test set for v2 model development**.

For a v2 improvement cycle:

1. feature definitions, model families, tuning ranges, calibration family, and selection rule are
   frozen before candidate evaluation;
2. architecture / feature / hyperparameter selection uses data no later than 2024;
3. all validation is chronological; future races never train models that predict earlier races;
4. any preprocessing, feature selection, hyperparameter tuning, or calibration is fitted only on
   the training side of a temporal split;
5. after the winner is frozen, the production model may be refit using all legitimately historical
   data available before the live target, including 2025;
6. the next genuinely untouched assessment is the pre-registered live Stage 6/7 sequence.

If nested tuning is used, both inner and outer splits must preserve chronology.

## 5. Winner rule: paired incumbent one-standard-error gate

The repository does **not** select the candidate with the numerically lowest validation Brier
without accounting for score-estimation noise.

### Step A — point-estimate best

On the frozen chronological out-of-fold predictions, identify the candidate with the lowest
race-macro Brier. Call it `Brier-best`.

### Step B — paired uncertainty

For each race compute the Brier loss difference:

```text
d_r = Brier_r(incumbent) - Brier_r(Brier-best)
```

The mean is taken across races. Its standard error is estimated with **race date as the cluster**,
so races sharing the same date are not treated as fully independent.

Use the paired loss differences, not independent standard errors for two model scores. This
preserves the strong covariance created by evaluating all candidates on the same races.

### Step C — incumbent retention gate

Retain the incumbent when:

```text
mean(Brier_incumbent - Brier_best) <= 1 * paired_clustered_SE_Brier
```

In words: a challenger must improve the **primary race-macro Brier** by more than one paired
standard error to dislodge the incumbent.

This is a model-selection regularizer, not a null-hypothesis significance test. The one-SE rule is
used to reduce validation-set overfitting and unnecessary model churn.

Log loss remains mandatory and is reported beside Brier, but it does not override a clear Brier
decision. This avoids silently changing the objective when two strictly proper scoring rules rank
imperfect models differently.

If the incumbent fails the gate, select `Brier-best`. Exact Brier ties are resolved by lower log
loss, then a stable lexical candidate ID.

## 6. Non-market model selection

For Stage 4 successor models:

- incumbent = the currently frozen L2-logistic non-market model;
- candidate families may include logistic variants, Random Forest, gradient boosting, XGBoost /
  LightGBM, or other predeclared models;
- all candidate feature sets and tuning spaces must be frozen before the selection run;
- the winner is chosen only by the rule in Section 5;
- AUC, ECE, feature importance, SHAP, hit rate, and ROI cannot override the winner rule.

A new feature block is treated as a model change and must clear the same gate.

## 7. Ensemble selection

For a new market/non-market ensemble selection cycle:

- `market-only` is the mandatory reference and default;
- every non-zero non-market blend weight is a challenger;
- a non-zero blend is adopted only if market-only fails the Section 5 retention gate;
- endpoints must always be included, so `w_nonmarket = 0` is an allowed winner.

This explicitly prevents tiny point-estimate improvements from forcing a blend.

The historical Stage 5 v1 selection of 95% market / 5% non-market remains an audit result; it is
not retroactively relabeled under this v2 rule.

## 8. Market-timing domain rule

Historical final odds and live morning odds are different prediction domains.

A blend weight tuned against historical final odds must not be presented as an optimized weight
for an 08:35 live market snapshot. Until time-matched historical snapshots exist, live morning
ensemble performance is established by repeated pre-race locks and post-race evaluation, not by
retuning against final-odds history.

## 9. Calibration diagnostics

For out-of-fold predictions report:

- calibration intercept: ideal 0;
- calibration slope: ideal 1;
- reliability curve;
- ECE for continuity only.

No arbitrary ECE cutoff determines the winner.

A non-finite probability, probability outside [0,1], missing prediction, or calibration slope
that cannot be estimated is a QA failure. Calibration diagnostics otherwise describe *why* a
proper score differs; they do not replace the primary score.

## 10. Metrics explicitly forbidden as winner criteria

The following may be reported but cannot select a probability model:

- Accuracy;
- Precision / Recall / F1;
- top-1 or top-k hit rate;
- actual-top3 probability mass from a single race;
- ROC-AUC by itself;
- ECE by itself;
- ROI, payout, or realized betting profit.

ROI is a separate downstream decision/utility question and must not be allowed to tune the
probability forecaster.

## 11. Required selection artifact

Every future selection run must save:

- candidate registry and frozen parameter grids;
- exact training/validation dates;
- common evaluation-row fingerprint;
- race-macro Brier and log loss for every candidate;
- runner-micro Brier and log loss;
- paired race-level deltas versus `Brier-best`;
- date-clustered paired SEs;
- incumbent gate result;
- calibration intercept/slope and reliability table;
- ROC-AUC diagnostic;
- final deterministic winner and reason;
- source commit SHA and data artifact fingerprint.

## 12. References

- Brier GW. Verification of forecasts expressed in terms of probability. *Monthly Weather
  Review*. 1950;78:1-3.
- Murphy AH. A new vector partition of the probability score. *Journal of Applied Meteorology*.
  1973;12:595-600.
- Gneiting T, Raftery AE. Strictly proper scoring rules, prediction, and estimation. *JASA*.
  2007;102:359-378. doi:10.1198/016214506000001437.
- Yates LA, Aandahl Z, Richards SA, Brook BW. Cross validation for model selection: a review
  with examples from ecology. *Ecological Monographs*. 2023;93:e1557.
  doi:10.1002/ecm.1557.
- Hoessly L. On misconceptions about the Brier score in binary prediction models. *Global
  Epidemiology*. 2026; doi:10.1016/j.gloepi.2025.100242.

Implementation reference: `src/keiba_place_lab/model_selection.py`.
