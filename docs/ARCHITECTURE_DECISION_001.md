# Architecture Decision 001 — Place first, ranking distribution later

Date: 2026-10-03 JST

## Decision

このプロジェクトは、短期的には `P(top3)` を推定する。
ただし最終アーキテクチャは「複勝専用AI」ではなく、
**レース全体の順位分布を推定し、各馬券種の確率をそこから派生させる構造**とする。

## Why

複勝は最初の実験として正例が比較的多く、確率校正を検証しやすい。
一方、単勝・ワイド・馬連・3連複・3連単を別々の分類器で予測すると、
同じレースについて互いに矛盾する確率を出しやすい。

共通の順位分布を持てば、

```text
P(win), P(place), P(wide), P(quinella), P(trio), P(trifecta)
```

を同一の確率構造から導出できる。

## Important distinction

各馬の周辺着順分布

```text
P(rank_i = r)
```

だけでは組合せ馬券には足りない。

例えば3連複は3頭が同時に上位3着を占める確率なので、
最終的には馬同士の順位依存を含むjoint ranking distributionが必要となる。

## Implementation consequence

Phase A:
- market baseline
- `P(top3)` model
- calibration
- pre-race lock
- post-race evaluation

Phase B:
- race-level ranking model
- joint ranking simulation/distribution
- bet-type probability derivation
- joint-probability validation

## Non-decision

現時点ではPlackett-Luce、latent-performance Monte Carlo、Set Transformer等の
どれを最終モデルにするかは決めない。

十分なレース数を蓄積し、walk-forward比較を行ってから選択する。
