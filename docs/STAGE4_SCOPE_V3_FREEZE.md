# Stage 4 scope V3 freeze

Turf selection completed under `SCOPE_EXPANSION_COMPLETION_SPEC.md`.
This adds V3 production/shadow artifacts; it does not replace Stage 5 market-only.
V2 results and code remain intact. All development estimates use 2023–2024 folds.

| Prediction scope | Training scope | Features |
|---|---|---|
| Turf exactly 1200 m | Turf 1000–1400 m | F0 + regime block |
| Turf 1000–1400 except 1200, and >1400–2000 m | Turf 1000–2600 m | F0 |
| Turf >2000–2600 m | Turf >2000–2600 m | F0 + near + regime + course_distance |

F0 is frozen XGB01 + original five full-field relative features. New block rates
use strength 6 and training-cohort prior. Zero/missing flags and log counts are
included; no trends/interactions or new search. Histories retain all prior starts.
Train final fitted artifacts on 2016–2025 only with 2010–2015 history warmup;
features at each historical row remain as-of that row. Live target histories
update from a complete standardized, confirmed-outcome feed strictly before
target date. The shipped 2025 feed is insufficient for a current 2026 race.

Target eligibility is active field >=8, career starts >=3; retain all active
starters for relative context. Declared-field draw positions survive scratches.
Ineligible starters get no fabricated nonmarket probability; the mixed candidate
uses market-only for those starters. Unsupported surface/distance/straight course
or field <8 returns market-only. Outcomes and market fields are forbidden in the
roster. Pre-lock cancellation creates a new input/version; post-lock cancellation
is a separate immutable event and excludes the whole race from primary evaluation.

Stage 2 Sprint+regime beats original A by 0.000532851 Brier (SE 0.000319).
Across all turf, G0 remains the global incumbent; aggregate R3 does not pass its
gate. The pre-defined long band independently passes R3 versus G0; 1200 retains
Stage 2. Final routed OOF Brier is about 0.149017, with exact metrics, calibration,
band gates and row-level predictions in `analysis/scope_expansion_turf/`.
This routed development score includes selection on the same development folds.

## Independent dirt freeze

Dirt 1000–2600 m uses one Global F0 trained only on dirt targets. The base
turf-history rate/count slots are replaced with same-surface (dirt) histories;
past turf races still contribute to career/recent form, not dirt-specific rates.
The added blocks and all three predefined band models fail the retention gates.
OOF Global F0 Brier is about 0.149716 versus 0.177702 for 3/field_size on identical
eligible rows. Exact scores, date-clustered differences, calibration and audits
are in `analysis/scope_expansion_dirt/`. This does not establish market advantage.
Final dirt fitting uses 2016–2025 under the same cutoff and live context rules.

Stage 5 prospective results and adoption remain pending real pre-off locks.
