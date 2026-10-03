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

**Stage 5 calibration / ensemble — COMPLETE / QA PASS**

Stage 2の08:35 win-market baselineとStage 4のnon-market baselineを、履歴JRA芝1200mで
同一行に突合して比較しました。履歴市場はKaggle v1の単勝オッズを全starterで正規化し、
Stage 2と同じHarville / Plackett-LuceでP(top3)へ変換しています。

Stage 5の時間分割は、2023 calibration fit / 2024 blend selection / 2025 held-out testです。
2024で選択されたblendは **market 95% / non-market 5%**。2025 paired test
(245 races / 3,070 eligible rows)では、selected blendが Brier `0.139662`,
log loss `0.434544`、calibrated market-onlyが Brier `0.139753`,
log loss `0.434701`でした。改善幅は小さく、Stage 5は市場主導モデルです。

2026-10-03 京都11Rの18頭には、凍結済み08:35 market snapshotとStage 4出力だけを使用し、
final ensembleを `sum P(top3)=3` に整合化しています。上位は
10 ヒシアイラ `0.342056`, 6 リリージョワ `0.311323`,
18 ディアナザール `0.287488` です。

Results:
- `analysis/2026-10-03_kyoto11_opal/STAGE5_ENSEMBLE.md`
- `analysis/2026-10-03_kyoto11_opal/stage5_ensemble.csv`
- `analysis/2026-10-03_kyoto11_opal/stage5_metrics.json`
- reproducible runner: `scripts/run_stage5_ensemble.py`
- calibration/ensemble helpers: `src/keiba_place_lab/ensemble.py`
- historical market reconstruction: `src/keiba_place_lab/historical_market.py`

Final verified Stage 5 Actions run #2 (run id `37108131492`) was SUCCESS on
head `79a885bf46c1c398e6cabc60ddb22e4cbfc3847e`.
Artifact ID `11269180077`, digest
`sha256:ef7b6d93d20fa2b5762fec4329612b35f19ffe035d65230c832e98bea372ac44`.

Important transport limitation: historical Kaggle odds are effectively final win odds whereas the
target market input is the frozen **08:35** snapshot. The historical Stage 5 test is therefore not
perfectly time-matched to the target. Later/final target odds were not substituted.

Stage 5 execution occurred after the scheduled target-race start, so this remains a blind
retrospective reconstruction from frozen pre-race inputs, not a pre-start prediction lock.
Stage 6 has not been performed.

## License

現時点ではリポジトリ全体にOSSライセンスを付与していません。
外部履歴データは各配布元のライセンスを別途確認し、raw row-level dataはデフォルトでgit管理外に置きます。
