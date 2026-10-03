# Prospective scope V3 — first forward collection period

Pre-registered before any real forward locks. Historical development is complete;
no historical score is used as a forward result. Stage 5 canonical remains
market-only. Frozen bundle is `models/stage4_scope_v3/bundle.joblib`; its SHA256
is checked against the model manifest and must remain constant for the period.
Routing is frozen in `STAGE4_SCOPE_V3_FREEZE.md` and `routing.json`.

Target all supported non-straight turf/dirt 1000–2600 races, active field >=8,
with at least one horse meeting prior career starts >=3. Every active starter
is included in the score denominator; ineligible runners use market-only for
every mixture and have missing raw nonmarket probability. No sum-to-three.

Lock at 10 minutes before the scheduled off, allowing 60 seconds to complete.
Use actual UTC clock and timezone-aware scheduled off/snapshot times; market
snapshot <=5 minutes old, canonical `stage5_v2_market_only`, full identical roster.
Commit both lock files to canonical repository **main** before off. A GitHub
Actions push run's server-created timestamp and exact committed lock bytes prove
publication; backdated author/committer time is insufficient. Review inventories
the entire current-main period folder and rejects omitted/extra lock manifests.
One immutable lock per race. Post-lock cancellation is a separate event and
excludes the whole race for all candidates. Missing official outcomes are whole
race exclusions reported explicitly; malformed labels abort rather than filter.

Fixed nonmarket weights: 0, .10, .25, .50, 1.00. Raw linear mixtures, no fitted
calibration, no tuning after outcomes. Market-only is incumbent. Review once when
the primary cohort reaches 1,000 complete races, 60 distinct calendar dates and
90 elapsed days from its first date. End at the first full calendar day meeting
all three thresholds, including all eligible locks that day; later collected
races cannot alter this period's decision. Below thresholds return counts only.
Do not select a partial set of locks or inspect interim scores for adoption.

Primary race-macro Brier; gate improvement > paired date-clustered CR1 1-SE.
Report log loss, calibration, per-year/surface/distance diagnostics and paired
race differences. No automatic promotion; a new recorded Stage 5 adoption
decision must cite the completed review and retain market-only if gate fails.
Changing models, protocol, scope, market conversion or roster timing starts a
separately pre-registered period. No real forward data exist at development finish.
