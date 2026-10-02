# Methodology

## Target

1頭ごとに、発走前時点の情報から

```text
p_place = P(finishing_position <= 3)
```

を推定します。通常の複勝が3着以内を対象とするレースを主対象とし、
対象条件が異なる場合はレース単位で定義を明示します。

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

Stage 1では人間が監査できるベースラインを優先します。

1. 市場ベースライン
2. 単純なロジスティック回帰
3. 木系モデル（十分な学習データが蓄積した後）
4. 確率校正（sigmoid / isotonic）
5. 必要に応じてレース内相対モデル、Plackett-Luce系、Monte Carlo

データ量が不足した段階で複雑なMLモデルを導入し、「AIらしさ」で精度を装わないことを原則とします。

## Separation of questions

以下を混同しません。

- **最も3着以内に入りやすい馬**: `argmax p_place`
- **複勝期待値が高い馬**: `argmax p_place * payout_expectation`
- **実際に買うか**: 予算・不確実性・価格を含む別の意思決定

このリポジトリの第一目的は1つ目の確率推定です。
