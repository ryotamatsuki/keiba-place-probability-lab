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
4. race-macro Brierを主指標、log lossをsecondary diagnostic / tie-breakerとして長期的に評価する
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

## Model-selection governance

Future model improvements follow
[`docs/MODEL_SELECTION_PROTOCOL_V2.md`](docs/MODEL_SELECTION_PROTOCOL_V2.md).

The frozen winner rule is:

- primary: equal-race-weighted (race-macro) Brier score;
- secondary diagnostic / tie-breaker: race-macro log loss;
- selection regularization: paired date-clustered one-standard-error incumbent gate;
- diagnostics only: calibration intercept/slope, reliability curve, ECE, ROC-AUC;
- forbidden as winner criteria: hit rate, F1, top-k hits, ROI;
- ensemble default/reference: market-only, with a non-zero blend required to clear the same gate.

Historical Stage 4/5 outputs remain immutable audit records; this rule governs future selection
cycles.

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

**Stage 7 post-race evaluation — RETROSPECTIVE PIPELINE TEST COMPLETE / QA PASS**

Stage 6で固定した18頭の確率を一切変更せず、JRA公式の2026-10-03京都11R
オパールステークス結果を別outcomeとして追加し、Stage 2 / 4 / 5 / 6を同じ18頭で
事後評価しました。

Official top 3:
- 1着 6 リリージョワ
- 2着 18 ディアナザール
- 3着 8 レッドエヴァンス

One-race metrics:
- Stage 2 raw market (08:35): Brier `0.107207`, log loss `0.335163`
- Stage 5/6 ensemble: Brier `0.111097`, log loss `0.350378`
- Stage 4 non-market: Brier `0.113597`, log loss `0.355176`
- uniform 3/18: Brier `0.138889`, log loss `0.450561`

Stage 5/6は実際の3着内3頭を予測順位2位・3位・6位に置きましたが、この1レースでは
08:35 raw marketの方がBrier/log lossとも良好でした。したがって、2025 held-out aggregateで
確認されたmarket 95% / non-market 5%の小幅改善は、今回の単発レースでは再現しませんでした。

Locked file controls:
- `probability_estimates.csv` SHA256:
  `eefa16ad709c7568690e3f75a0df22491a31b3cb6aebdeebcf27fab274bc9e66`
- Stage 6 values modified after outcome: **no**
- later/final target odds substituted: **no**
- market timestamp: `2026-10-03T08:35:00+09:00`

Files:
- `analysis/2026-10-03_kyoto11_opal/official_outcome.csv`
- `analysis/2026-10-03_kyoto11_opal/POSTRACE_EVALUATION.md`
- `analysis/2026-10-03_kyoto11_opal/postrace_metrics.csv`
- `analysis/2026-10-03_kyoto11_opal/stage7_evaluation_manifest.json`
- reproducible runner: `scripts/run_stage7_evaluation.py`
- tests: `tests/test_postrace.py`

Stage 7 verified workflow run #2: `37110151054` SUCCESS.
Artifact ID `11270045420`, digest
`sha256:c224ba078232365bcd4e5a360c43df87f2a0d03b29acce254374517a525ef252`.

This remains a **retrospective pipeline test**, not a genuine pre-start forecast audit,
because the Stage 6 timing gate was missed for this race. The next live target must execute
Stages 1-6 before post time, then use this same Stage 7 procedure after the official result.

## License

現時点ではリポジトリ全体にOSSライセンスを付与していません。
外部履歴データは各配布元のライセンスを別途確認し、raw row-level dataはデフォルトでgit管理外に置きます。
