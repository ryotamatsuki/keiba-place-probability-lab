# Evaluation Protocol

## Primary metrics

### Brier score

二値の複勝圏結果 `y ∈ {0,1}` と予測確率 `p` に対して

```text
Brier = mean((p - y)^2)
```

を記録します。小さいほどよい指標です。

### Log loss

過信した誤予測を強く罰するため、log lossも併記します。

### Calibration

予測確率をビン分けし、「40%と予測した馬群が長期的に約40%複勝圏へ入ったか」を確認します。
ECE等はサンプル数が十分になってから導入します。

## Secondary metrics

- 複勝圏的中率
- レースごとの最高 `p_place` 馬の複勝圏率
- 市場ベースラインとの差
- ROI（参考値。確率精度とは分離）

## Validation design

- 時系列順に学習・検証を分離する
- 同一レースの馬をtrain/testへ分割しない
- 将来情報・確定結果由来の特徴を禁止する
- final oddsを予測時点で取得していない場合、事前モデルには使わない
- モデル選択と最終評価の期間を分離する

## Prediction lock

レースごとに `probability_estimates.csv` を発走前にコミットし、そのコミットSHAを
`PRE_ANALYSIS_LOG.md` に記録します。以後、予測値の上書きは禁止し、訂正が必要な場合は
新しいファイルと理由を追加します。
