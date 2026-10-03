# keiba-place-probability-lab

JRAの公開情報等を用いて、レース前の**確率分布を監査可能な形で推定・検証する**実験リポジトリです。

現在の短期ターゲットは、各馬が3着以内に入る確率

```text
P(top3)
```

です。ただし、長期的には複勝専用モデルに固定せず、**各馬の着順分布・レース全体の順位分布**を推定し、
そこから単勝・複勝・ワイド・馬連・3連複・3連単等の確率を派生させることを目標にします。

## 目的

このプロジェクトの主目的は「回収率最大化」ではありません。まず、

1. レース前に利用可能だった情報だけで確率を推定する
2. 市場（オッズ）を強いベースラインとして比較する
3. 予測をレース前に固定し、後知恵による修正を禁止する
4. Brier score / log loss / calibrationで長期的に評価する
5. 最終的には馬券種ごとに別々の予想器を作らず、共通の順位確率モデルから各馬券確率を導出する

ことを重視します。

## Research phases

### Phase A — Place probability

`P(top3)` を対象に、公開情報・市場情報・能力/適性特徴量から、
監査可能で校正された複勝圏確率を作ります。

これは最初の実験対象です。複勝を永久的な最終目的にはしません。

### Phase B — Full finishing-order distribution

十分なデータと検証基盤ができた段階で、

```text
P(rank_i = 1), P(rank_i = 2), ..., P(rank_i = N)
```

およびレース内の順位の依存構造を扱います。

この分布から、

- 単勝: `P(rank_i = 1)`
- 複勝: `P(rank_i <= 3)`
- ワイド: 2頭がともに3着以内
- 馬連: 2頭が1・2着を占める
- 3連複: 3頭が上位3着を占める
- 3連単: 3頭が指定順で1・2・3着

を派生させます。

## Principles

- **Pre-race only**: 各データ行に `as_of_time` を持たせ、発走後の情報混入を防ぐ。
- **Market-aware**: オッズを無視せず、市場予測とモデル予測を分けて評価する。
- **Probability first**: 的中率やROIだけでなく、確率予測の校正を主要評価対象にする。
- **Auditable**: 出典URL、取得時刻、加工内容、判断理由を残す。
- **Rights-clean**: 公開ページの画像・HTML・PAT画面・第三者データセットを無断転載しない。
- **No hindsight**: 予測ロック後は結果を見て予測値を書き換えない。
- **One latent race model**: 長期的には馬券種別ごとの独立予想器ではなく、共通の順位分布から確率を導出する。

## Repository layout

```text
analysis/   # レース単位の事前分析・予測・事後評価
data/       # 出典マニフェストと再配布可能な派生データ
docs/       # 方法論、データ利用方針、評価手順、既存OSSレビュー
scripts/    # ローカル取得・履歴パネル構築CLI
src/        # 再現可能な分析コード
templates/  # 予測・意思決定ログのテンプレート
tests/      # 自動テスト
```

最初のケースは `2026-10-03 京都11R オパールS` とし、18頭をゼロベースで比較します。

## Current status

**Stage 3.6 historical training panel — COMPLETE / QA PASS**

2010-2025のJRA履歴をactual-date基準で再構成し、公式JRA inventoryと55,268レースを
年次ごとに完全照合しました。Kaggle v1に未収録だった2025年末48レースは公式JRADB
結果ページから補完し、障害2,048レースを除いたflat 53,220レース / 753,387走 /
81,900頭のleakage-safe履歴パネルをfreezeしています。

固定splitは2010-2015 warmup / 2016-2022 train / 2023-2024 validation /
2025 untouched testです。chronology、race×horse重複、same-race contamination、
market列、future-year、未解決日付、JRA malformed selected rowはいずれも0です。

Freeze artifactは `jra_flat_historical_panel_v1.parquet`
(SHA256 `cee9ae9a099f521f12b1bcdd371c25a3d7b9ba3555fbcf5a46f2c9098f59a33d`)。
GitHub Actions materialize run #27 の `stage36-historical-training-panel-v1` artifact
(ID `11266998371`, archive digest `sha256:014b3e5d554fa24d8db042aaacc338a8a562db6bf93824e2197b84065aec9c85`) に保存しています。
`course_layout` と `handicap_indicator` は公式データcoverage不足のためhistorical-v1.1
model inputから除外し、監査列としてのみ保持します。

Stage 4はunblockedですが、まだモデルfitには進んでいません。

## License

現時点ではリポジトリ全体にOSSライセンスを付与していません。
外部履歴データは各配布元のライセンスを別途確認し、raw row-level dataはデフォルトでgit管理外に置きます。
