# Stage 2 — Market baseline for P(top3)

Race: 2026-10-03 Kyoto 11R Opal Stakes  
Frozen market snapshot: 2026-10-03 08:35 JST

## Decision

The primary market-only baseline is built in four steps:

1. convert win odds to reciprocal-odds strength;
2. normalize across all 18 runners to remove the common overround;
3. treat the normalized win vector as first-place probabilities;
4. derive top-three marginals with the Harville / Plackett-Luce sequential-ranking assumption.

The place-odds range is used as a diagnostic market signal, not directly inverted into an implied probability.

## Why not simply invert place odds?

JRA place betting normally has multiple winning horses. JRA's payout formula uses the amount bet on the winning selection, the amount bet on losing selections, the number of winning selections, and the payout-rate parameter. Win and place normally use an 80% payout rate.

Because displayed place odds are ranges and the realised payout depends on which other horses also finish in the paying positions, a displayed place-odds value does not map one-to-one to a marginal P(top3).

JRA-VAN documentation likewise notes that reverse-estimating vote share from place odds has larger error and depends on the combination of the other horses.

Therefore Stage 2 does not label reciprocal place odds as probability.

## Win-market normalization

For horse i with decimal win odds o_i:

q_i = 1 / o_i

p_i = q_i / sum_j(q_j)

For the 08:35 snapshot:

sum_i(1 / o_i) = 1.2588602636

1 / sum_i(1 / o_i) = 0.7943693426

The second number is close to the ordinary 80% JRA win payout fraction. This is a sanity check, not an exact identity; displayed odds are rounded and the pool is still moving.

## Harville / Plackett-Luce top-3 baseline

For an ordered top three (i,j,k):

P(i,j,k) = p_i * p_j/(1-p_i) * p_k/(1-p_i-p_j)

The market-only place probability for horse i is then the sum of all ordered top-three outcomes containing i.

Audit identity:

sum_i P_market(top3_i) = 3

The implementation enumerates all ordered triples exactly.

## 08:35 market-only ranking

| Rank | Horse | Win odds | Harville P(top3) | Place odds |
|---:|---|---:|---:|---:|
| 1 | 10 ヒシアイラ | 5.8 | 0.383753 | 2.1–2.4 |
| 2 | 6 リリージョワ | 6.5 | 0.349399 | 3.4–4.3 |
| 3 | 18 ディアナザール | 7.5 | 0.309414 | 2.9–3.6 |
| 4 | 3 タマモイカロス | 7.7 | 0.302453 | 2.7–3.3 |
| 5 | 4 メイショウヨゾラ | 9.3 | 0.256106 | 3.0–3.8 |
| 6 | 8 レッドエヴァンス | 10.0 | 0.239937 | 2.3–2.8 |

These are not the project's final probabilities. They are the win-market baseline under one explicit ranking assumption.

## Place-market disagreement diagnostic

The two most conspicuous rank disagreements are:

- No. 8 レッドエヴァンス: place rank 2 vs Harville rank 6.
- No. 6 リリージョワ: place rank 6 vs Harville rank 2.

This shows the place pool is not simply reproducing the win-pool ordering. Stage 2 records the disagreement but does not explain it. Stage 3 must check form, distance, pace, field composition and other non-market evidence before assigning a cause.

## Assumptions and limitations

The Harville construction assumes that the same latent market strength used for first place can be renormalized sequentially for second and third place. This may miss horses that are disproportionately win-or-bust or place-stable.

Therefore:

- p_top3_winmarket_harville is the primary transparent baseline;
- place_strength_proxy is diagnostic only;
- later stages compare market-only, non-market and blended estimates;
- the 08:35 snapshot is never overwritten by later or final odds.

## References

- JRA payout formula / payout rates: https://www.jra.go.jp/faq/pop03/1_17.html
- JRA betting rules: https://www.jra.go.jp/kouza/baken/index.html
- Harville, D. A. (1973), Assigning Probabilities to the Outcomes of Multi-Entry Competitions, Journal of the American Statistical Association, 68(342), 312–316. DOI: 10.1080/01621459.1973.10482425
- JRA-VAN TARGET odds-share note: https://targetfaq.jra-van.jp/faq/detail?category=103&id=258&site=SVKNEGBV

## Stage 2 status

COMPLETE

Next: Stage 3 adds non-market form / suitability / pace features while keeping this market baseline frozen.
