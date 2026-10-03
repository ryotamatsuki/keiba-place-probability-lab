# Historical External Benchmarks

Purpose: secondary QA targets only. These are **not** authoritative replacements for JRA.

A recent independent public-data analysis that combined the same two Kaggle families and
cross-checked against JRA reports the following for JRA flat racing:

- official 2016-2025 flat-race count: 33,290;
- combined public-data coverage through 2025-12-21: 33,244 races;
- stated gap: 46 late-December 2025 races;
- one pre-race-history analysis reports 464,232 runner rows over the covered 2016-2025 races;
- it reports 416,419 rows for which a previous JRA flat start could be linked;
- it reports zero horse-ID × same-date duplicates and zero previous dates equal to or after the
  current date after correcting actual race dates.

The same analysis warns that the raw public-data date field should not be trusted for previous-race
linking and that an actual JRA meeting/date correspondence must be applied first.

It also reports that some obstacle races can be misclassified in the public source data, so
flat-race identification must be explicitly QA'd rather than inferred only from a turf/dirt flag.

Reference:
https://note.com/like_newt7346/n/n8eddc4f19e77

## How to use these numbers

They are only anomaly detectors.

A difference from these numbers does not automatically mean our panel is wrong because:

- our source version may be newer;
- our inclusion/exclusion rules differ;
- Stage 3.6 may intentionally exclude small fields, invalid finishes or rows without enough history;
- JRA official counts remain the reconciliation target.

The QA report must explain differences rather than force the data to match an external number.
