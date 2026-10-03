# October 4 morning nonmarket trial

Both fixed V3 routes scored all 35 declared starters: Kyoto Daishoten (18),
Mainichi Okan (17). Input histories include 7,037 rows across 515 complete prior
JRA flat races through 2026-10-03. Every target horse's entire listed JRA flat
career was reconciled; 35/35 are eligible. The fitted bundle is unchanged.

`morning_nonmarket_predictions.csv` contains the raw top-three probabilities.
`morning_feature_context.csv` contains the market-free current feature context.
`target_history.parquet` and `history_manifest.json` retain reproducible, audited,
target-scoped histories. Full context and declared draw positions are preserved.
Probabilities are not normalized to sum three.

Fresh roster acquisition succeeded for the Kyoto prediction. Subsequent current
market requests failed with proxy 403 / cancelled network approval. Tokyo was
scored from the prepared, source-hashed morning roster. There are no formal live
locks and no valid fresh market/nonmarket pairs yet. A saved 07:30:06 Tokyo odds
page was used solely for conversion QA: its Harville marginal sum is three.
This stale replay is not a current market snapshot or prospective lock.

Failed preview attempts are recorded in QA.json; their partial files do not
constitute lock artifacts. Future tasks must refresh both roster and market at
the fixed pre-off window. Source access and precise scheduling are prerequisites;
failed/late attempts must remain failures. Confirmed official results and
publication proof are pending. No interim outcome scores or adoption were made.
