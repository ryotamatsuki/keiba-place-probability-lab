# Scope Expansion Stages 2–5 pre-registration

Status: committed before new comparisons. Base: afd5d5f9fd245cb85a4592e5b32fc17087e9e4d8.
Stage 1 is complete; its A incumbent and frozen 2023–2024 predictions remain immutable.

## Data and common rules

Use Stage 3.6 full JRA flat panel, retain all distances/surfaces/straight races as
historical observations, but exclude straight courses from prediction cohorts.
Target: surface-specific 1000–2600 m, actual field >=8, prior career starts >=3.
Train 2016–2022 for 2023, 2016–2023 for 2024; 2010–2015 warms up histories.
History uses strictly earlier calendar dates. No market inputs, future labels,
2025 selection, probability sum-to-three adjustment, or hyperparameter tuning.
XGB01 and existing base + five full-field LOO relative features stay fixed.
EB strength 6; prior is each fold's training-label mean. Fit imputation and
categories exclusively on training rows. Log loss, calibration, row/race losses,
year/subgroup scores, complete-context checks and fingerprints are recorded.
Gate: improvement in race-macro Brier > paired date-clustered CR1 1-SE.
This is an engineering retention rule, not a significance/equivalence test;
2023–2024 are reused development data and require prospective confirmation.

## Stage 2: turf distance feature blocks

A=1200, B=1000–1400, C=1000–2600, common 1200 evaluation keys: 6,313 rows/495 races.
First rerun F0 and verify against Stage 1 OOF (float tolerance 1e-7).
Order: near, regime, course_distance. In each family compare its currently
retained feature set with that set plus the next block, retaining only on gate.
All three families receive the same proposed block definitions. At the end,
compare each retained family with original A; choose point-Brier best only if
it passes A's gate (exact ties: log loss then candidate name).

Each block has: EB top3 rate, log1p starts, zero-experience flag, missing-history
flag, and own EB rate minus full-field leave-one-out EB rate. Known zero counts
shrink to prior; unknown keys/unknown matching results stay missing.
- near: same surface and |past distance - target distance| <=200 m, inclusive.
- regime: same surface and same band: <=1400, (1400,2000], (2000,2600], >2600.
  Distances below 1000 may contribute to the <=1400 history band.
- course_distance: same racecourse, surface and exact distance. Existing
  same_course does not include distance and remains unchanged.

## Stage 3: all-distance transport and segmentation

For turf 1000–2600 compare G0 (Global F0), G3 (Global all three new blocks),
R3 (three separate band models, all three blocks), and baseline 3/field_size,
on identical eligible rows. Whole-range gate uses G0 incumbent and G3/R3.
Separately gate each predefined band's R3 against the selected global (G0/G3,
selected by whole-range gate). These band decisions define the shadow routing.
No post-hoc additional splits. Distance/course/class/age/field diagnostics are
hypotheses only. Retain the Stage-2 1200 winner at exactly 1200, unless the
proposed broader routing beats it by paired 1-SE on exactly the common keys.
Also report whole-range performance of the final routed development predictions.
No capacity tuning in this series; any tuning needs a new committed protocol.

## Stage 4: production and prospective Stage 5

Freeze routing, surface, scope and per-model feature sets; fit 2016–2025 only
(no later rows), maintain immutable V2 history. Store fitted preprocessing,
model, priors, training/code/data fingerprints and cutoff. New files use V3.
Production needs complete active-starter context; eligible starters get raw
nonmarket probabilities, ineligible starters and unsupported scope use explicit
market-only routing. Never invent nonmarket predictions for them. Preserve draw
positions from declared field; cancellation regenerates a new version before
lock. Locked predictions remain immutable; late cancellation is a separate
record and excludes the entire race from primary comparison for all candidates.
History for live prediction must be generated from input records strictly before
the target date; reject unresolved outcomes in prior history and stale datasets
unless the caller explicitly supplies a freshness bound that is met.

Prospective protocol: all supported turf/dirt scopes, lock 10 minutes before
scheduled off (market snapshot age <=5 minutes); same roster and timestamp.
Weights on nonmarket: 0, .10, .25, .50, 1.00, linear raw probabilities,
no fitted calibration or sum correction. Market probabilities must come from
canonical current Stage 5 and be validated, not inferred from historical final odds.
Review once after >=1,000 complete races AND >=60 distinct race dates AND
>=90 elapsed days from first eligible lock; no interim adoption. End date is
when all thresholds are met. All committed pre-off locks in that period enter,
excluding documented cancellations/missing official outcomes. Select point-best
with market-only incumbent and paired 1-SE; no automatic production promotion.
Require a verifiable Git commit published before off, not just a local timestamp.
Cannot complete future evaluation until actual pre-off locks and outcomes exist.

## Stage 5: dirt

After turf selection is frozen, repeat G0/G3/R3/baseline comparison and band gates
on dirt 1000–2600. Replace the base turf history rate/count input with same-surface
(dirt) rate/count under the same shrinkage. Other prior career, same-surface
same_distance/same_course and recent-form histories are defined as above.
No turf-to-dirt model reuse. New dirt is a shadow candidate; prospective protocol
covers it independently in diagnostics and jointly in the locked primary cohort.

## Required outputs

Audit by year/distance/course (rows, races, distinct dates, feature missingness),
OOF predictions, scores, paired and per-race differences, calibration bins and
intercept/slope, selected routing JSON, source/code/spec/data hashes, leak/LOO QA,
V3 trained artifacts and reproducible runner, prospective lock/evaluation CLI.
Each stage's results are committed before beginning the next dependent stage.
