# Workflow

## Stage 0 — Governance / freeze

- [x] 目的変数を `P(top3)` に固定
- [x] Data Policy作成
- [x] 評価指標固定
- [x] 既存OSS候補の棚卸し
- [x] 最初のレース用pre-analysisログ作成

## Stage 1 — Race snapshot

- [x] 18頭全頭の出走表を同じ基準で記録
- [x] 出典URLとas-of時刻を固定
- [x] 市場情報、枠、斤量、騎手、基本戦績を記録
- [x] 欠損項目を明示
- [x] 権利・再配布ポリシーに従い、生HTML・画像・PAT情報を保存しない

**Status: COMPLETE — 2026-10-03 pre-race**

## Stage 2 — Market baseline

- [ ] 市場情報からベースラインを構築
- [ ] 単勝人気を複勝圏確率と混同しない
- [ ] 複勝オッズが利用可能なら、その利用方法と限界を明記
- [ ] オッズの時点差を上書きせず、snapshotとして保存
- [ ] 市場ベースラインの仮定と不確実性を文書化

## Stage 3 — Form / suitability features

- 距離
- コース
- 馬場
- 近走
- クラス
- 斤量
- 脚質 / ペース

## Stage 4 — Baseline model

- 説明可能な手法から開始
- 入力、欠損処理、重みを固定
- 18頭へ同一ルール適用

## Stage 5 — Calibration / ensemble

- 市場と独立モデルを比較
- 必要ならブレンド
- 不確実性レンジを付与

## Stage 6 — Pre-race lock

- probability_estimates.csvをコミット
- decision_logを記録
- 発走後は予測値を変更しない

## Stage 7 — Post-race evaluation

- 着順を別データとして追加
- Brier / log loss等を更新
- 予測失敗を「結果論」ではなく事前仮説との差として分析

## Stage 8 — Accumulation

複数レースが蓄積した後に、ML・校正・walk-forward評価を本格化します。
