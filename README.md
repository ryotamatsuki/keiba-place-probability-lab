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

**Stage 4 non-market P(top3) baseline — COMPLETE / QA PASS**

Stage 3.6のleakage-safe historical panelを使い、market情報を一切入力しない透明な
L2 logistic baselineをfitしました。2016-2022をtrain、2023-2024をvalidationとして
predeclared cohortと正則化を選択し、選択された仕様は `turf_1200` / `C=0.1` です。

2025 held-out test（芝1200m、248 races / 3,100 eligible runner rows）は
Brier `0.150537`、log loss `0.467970`、10-bin ECE `0.016938`。
2025 outcomeはtarget-model fitには使用していません。

2026-10-03 京都11Rの18頭へ同一モデルを適用し、レース内共通logit interceptで
`sum P(top3)=3` を厳密に満たす非市場確率を生成しました。Stage 4コードは
odds / popularity / payout / Stage 2 market probability / target outcomeを読み込みません。

結果:
- `analysis/2026-10-03_kyoto11_opal/NONMARKET_BASELINE.md`
- `analysis/2026-10-03_kyoto11_opal/nonmarket_baseline.csv`
- reproducible runner: `scripts/run_stage4_nonmarket.py`
- model helpers: `src/keiba_place_lab/nonmarket.py`
- tests: `tests/test_nonmarket.py`

Final verified Actions execution: Stage 4 run #10 (run id `37106873451`) SUCCESS.
Artifact ID `11267173870`, digest
`sha256:72f395d8a5503cdc8a905053e331904133c33b6b3f9717f75f2ca795120cfcf7`.

Important: Stage 4 was generated after the scheduled target-race start, so it is documented as a
blind retrospective reconstruction from the frozen pre-race matrix, not as a pre-start prediction lock.
Stage 5 market comparison / calibration / ensemble has not started.

## License

現時点ではリポジトリ全体にOSSライセンスを付与していません。
外部履歴データは各配布元のライセンスを別途確認し、raw row-level dataはデフォルトでgit管理外に置きます。
