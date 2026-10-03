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

**Stage 6 pre-race lock rehearsal — RETROSPECTIVE_DRY_RUN COMPLETE / QA PASS**

2026-10-03 京都11RはStage 5完成時点で発走予定時刻を過ぎていたため、Stage 6は
本番のpre-race lockではなく、運用手順を検証するretrospective dry runとして実施しました。

凍結対象はStage 5確定commit
`086f7a2a20e20dc226db6535b4738905e2011f1b` のみです。市場は引き続き08:35 snapshotを
使用し、結果・後刻/最終オッズ・払戻・人気はlock処理へ入力していません。

Rehearsal lock:
- 18 runners / unique horse numbers: PASS
- exact `sum P(top3)=3`: PASS
- model version: `stage5-ensemble-v1`
- `probability_estimates.csv`: populated and frozen
- decision log / SHA256 manifest: frozen
- lock mode: `RETROSPECTIVE_DRY_RUN`
- source Stage 5 commit: `086f7a2a20e20dc226db6535b4738905e2011f1b`
- Stage 6 rehearsal workflow run #2: SUCCESS
- artifact ID: `11268249828`
- artifact digest: `sha256:144854c3b0742fc62d54aa078ff68420d8068ad8491a9fe322e1792b277ffbdd`
- locked `probability_estimates.csv` SHA256:
  `eefa16ad709c7568690e3f75a0df22491a31b3cb6aebdeebcf27fab274bc9e66`

Files:
- `analysis/2026-10-03_kyoto11_opal/probability_estimates.csv`
- `analysis/2026-10-03_kyoto11_opal/STAGE6_LOCK_DECISION_LOG.md`
- `analysis/2026-10-03_kyoto11_opal/stage6_lock_manifest.json`

The rehearsal-locked values are immutable from this point forward and must be used unchanged
for any subsequent evaluation of this race.

**The genuine Stage 6 timing gate was not satisfied for this race.** A real live execution must
run and commit the same locking procedure before the scheduled start of the next target race.

## License

現時点ではリポジトリ全体にOSSライセンスを付与していません。
外部履歴データは各配布元のライセンスを別途確認し、raw row-level dataはデフォルトでgit管理外に置きます。
