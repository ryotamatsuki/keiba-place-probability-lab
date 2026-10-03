# Workflow

## Stage 0 — Governance / freeze

- [x] Phase Aの目的変数を P(top3) に固定
- [x] Data Policy作成
- [x] 評価指標固定
- [x] 既存OSS候補の棚卸し
- [x] 最初のレース用pre-analysisログ作成
- [x] 長期目標を「順位分布モデル → 馬券種横断確率」に拡張

## Stage 1 — Race snapshot

- [x] 18頭全頭の出走表を同じ基準で記録
- [x] 出典URLとas-of時刻を固定
- [x] 市場情報、枠、斤量、騎手、基本戦績を記録
- [x] 欠損項目を明示
- [x] 権利・再配布ポリシーに従い、生HTML・画像・PAT情報を保存しない

Status: COMPLETE — 2026-10-03 pre-race

## Stage 2 — Market baseline for P(top3)

- [x] 市場情報からベースラインを構築
- [x] 単勝人気を複勝圏確率と混同しない
- [x] 複勝オッズの利用方法と限界を明記
- [x] オッズの時点差を上書きせず、snapshotとして保存
- [x] 市場ベースラインの仮定と不確実性を文書化
- [x] normalized win market + Harville/Plackett-Luceでmarket-only P(top3)を算出
- [x] 複勝オッズはprobabilityではなくdiagnostic strengthとして分離

Status: COMPLETE — 08:35 JST market snapshot

## Stage 3 — Form / suitability features

- [x] 距離
- [x] コース
- [x] 馬場
- [x] 近走
- [x] クラス
- [x] 斤量
- [x] 脚質 / ペース
- [x] market情報から独立した特徴量セットをfreeze
- [x] model-input allowlistを固定
- [x] missingnessを0と混同しない

Status: REVISED — initial snapshot retained, canonical Feature Spec v1 frozen after evidence audit

## Stage 3.5 — Canonical feature materialization

- [x] JRA-VAN operational feature design reviewed
- [x] recent JRA leakage-aware temporal-validation evidence reviewed
- [x] ranking/SHAP literature reviewed
- [x] redundancy and missingness audit run on current 18-runner snapshot
- [x] canonical Feature Spec v1 frozen
- [x] materialize all required v1 fields for 18 runners
- [x] mark any unavailable required field explicitly
- [x] verify zero market fields in canonical matrix
- [x] relative final-3F block explicitly excluded for current v1 rather than raw-time substitution
- [x] canonical matrix validated: 18 rows / 0 missing / 0 market-column matches

Status: COMPLETE — Stage 4 UNBLOCKED

## Stage 4 — P(top3) baseline model

- [ ] 説明可能な手法から開始
- [ ] 入力、欠損処理、重みを固定
- [ ] 18頭へ同一ルール適用
- [ ] market fieldsが入力にないことを検証
- [ ] aggregate non-market rankingを生成

## Stage 5 — Calibration / ensemble

- [ ] 市場と独立モデルを比較
- [ ] 必要ならブレンド
- [ ] 不確実性レンジを付与

## Stage 6 — Pre-race lock

- [ ] probability_estimates.csvをコミット
- [ ] decision_logを記録
- [ ] 発走後は予測値を変更しない

## Stage 7 — Post-race evaluation

- [ ] 着順を別データとして追加
- [ ] Brier / log loss等を更新
- [ ] 予測失敗を「結果論」ではなく事前仮説との差として分析

## Stage 8 — Accumulation / walk-forward

- [ ] 複数レースを蓄積
- [ ] walk-forward評価
- [ ] calibrationをサンプル外で検証
- [ ] market-only / model-only / blendedを比較

## Stage 9 — Rank-distribution prototype

十分なデータが蓄積してから着手。

- [ ] P(rank_i = r) を整合的に推定
- [ ] レース内の順位制約を満たすモデルを比較
- [ ] Plackett-Luce / latent-performance simulation / race-level modelを候補比較
- [ ] top1 / top3周辺確率がPhase Aモデルと整合するか検証

## Stage 10 — Cross-bet probability engine

順位分布から馬券種別確率を派生。

- [ ] 単勝
- [ ] 複勝
- [ ] ワイド
- [ ] 馬連
- [ ] 3連複
- [ ] 3連単
- [ ] joint probabilityの校正・妥当性を検証

馬券種別の独立した「当て物モデル」を乱立させず、共通の順位分布を中核にします。
