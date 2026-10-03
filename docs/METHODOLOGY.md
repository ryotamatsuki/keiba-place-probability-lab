# Methodology

## Short-term target

Phase Aでは、1頭ごとに発走前時点の情報から

```text
p_place = P(finishing_position <= 3)
```

を推定します。通常の複勝が3着以内を対象とするレースを主対象とし、
対象条件が異なる場合はレース単位で定義を明示します。

## Long-term target: finishing-order distribution

最終目標は複勝専用分類器ではなく、各馬の着順分布とレース内の順位依存を扱うことです。

個別馬については

```text
P(rank_i = 1), P(rank_i = 2), ..., P(rank_i = N)
```

を考えます。ただし、馬連・ワイド・3連複・3連単を正しく扱うには、
各馬の周辺分布だけでは不十分です。同一レースでは着順が相互排他的なので、
最終的には**joint ranking distribution**またはそれと整合的な順位モデルが必要です。

候補は以下です。

- Plackett-Luce / exploded logit
- race-level softmax / set model
- latent performance + Monte Carlo simulation
- permutation / ranking model
- sufficiently validated pairwise-to-ranking construction

## Why start with P(top3)

`P(top3)` は初期実験として、

- 単勝より正例が多く、少数レースでも校正を検証しやすい
- 「能力」だけでなく安定性・適性・展開耐性を評価しやすい
- Brier score / log loss / calibrationで監査しやすい
- 今日の少額練習の問いと一致する

という利点があります。

一方、複勝市場の表示オッズから `P(top3)` を直接逆算するのは単勝より難しいため、
市場ベースラインの設計は明示的に検討します。

## Bet-type mapping

順位分布が得られた場合の関係は以下です。

### Win

```text
P_win(i) = P(rank_i = 1)
```

### Place

通常3着払いなら

```text
P_place(i) = P(rank_i <= 3)
```

### Quinella / Wide / Trio / Trifecta

これらは複数馬の**結合確率**が必要です。

例:

```text
P_wide(i,j) = P(rank_i <= 3 and rank_j <= 3)
```

```text
P_trio(i,j,k) = P({i,j,k} occupy ranks {1,2,3})
```

したがって、将来のモデル評価では「個別馬の周辺確率が良い」だけでなく、
順位依存まで含むjoint probabilityの妥当性を評価します。

## Model layers

### A. Market baseline

オッズは公開情報を集約した強いベースラインとして扱います。
「市場を無視したモデル」と「市場を含むモデル」を分けて比較します。

### B. Horse ability / recent form

候補特徴量:

- 直近成績と着差
- 距離・コース・馬場適性
- 同級・近似条件での実績
- 走破時計・上がり等の比較可能な指標
- 休養間隔と近走安定性

### C. Race fit

- 枠順
- 脚質
- 想定ペース
- 先行馬密度
- 頭数
- 斤量
- コース形状との適合

### D. Human / condition factors

- 騎手・調教師
- 乗り替わり
- 当日馬体重と増減
- 馬場状態

## Modeling roadmap

初期段階では人間が監査できるベースラインを優先します。

1. 市場ベースライン
2. `P(top3)` の説明可能なベースライン
3. 単純なロジスティック回帰
4. 木系モデル（十分な学習データが蓄積した後）
5. 確率校正（sigmoid / isotonic）
6. レース内相対モデル
7. 順位分布モデル
8. joint ranking distributionから各馬券種確率を派生

データ量が不足した段階で複雑なMLモデルを導入し、「AIらしさ」で精度を装わないことを原則とします。

## Separation of questions

以下を混同しません。

- **最も3着以内に入りやすい馬**: `argmax p_place`
- **複勝期待値が高い馬**: `argmax p_place * payout_expectation`
- **最も勝ちやすい馬**: `argmax P(rank=1)`
- **組合せ馬券の的中確率**: joint ranking probabilityから導出
- **実際に買うか**: 予算・不確実性・価格を含む別の意思決定

Phase Aの第一目的は `P(top3)` の確率推定です。
プロジェクト全体の最終目的は、馬券種横断で整合的な順位確率モデルです。
