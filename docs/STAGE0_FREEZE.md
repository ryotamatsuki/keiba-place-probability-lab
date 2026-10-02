# Stage 0 Freeze

Date: 2026-10-03 JST

Stage 0 fixes the following before the first full race analysis:

- target: pre-race probability of finishing in the place-paying positions, normally top 3
- primary question: which horse has the highest calibrated place probability
- market odds: benchmark and optional model input, not ground truth
- primary evaluation: Brier score, log loss, calibration
- leakage rule: only information available at the recorded as-of time may enter a pre-race prediction
- audit rule: source URL, access time, transformation and missingness must be recorded
- rights rule: do not commit third-party screenshots, raw HTML, PAT data or non-redistributable datasets
- prediction lock: pre-race estimates are immutable after the race starts; corrections require a new artifact and explanation

The initial mention of horse #10 in conversation is documented as potential anchoring and is not a model conclusion.
