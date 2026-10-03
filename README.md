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
src/        # 再現可能な分析コード
templates/  # 予測・意思決定ログのテンプレート
tests/      # 最小限の自動テスト
```

最初のケースは `2026-10-03 京都11R オパールS` とし、18頭をゼロベースで比較します。

## Current status

**Stage 3.5 complete / Stage 4 non-market baseline ready**

Stage 2で市場だけのP(top3)ベースラインを固定しました。Stage 3の初期特徴量はその後、
JRA-VANの運用仕様、2026年のJRA leakage-aware temporal-validation研究、ranking/SHAP研究、
および現在の18頭データの冗長性・欠損監査で再評価しました。

Canonical Stage 4 inputは `docs/FEATURE_SELECTION_AUDIT.md` と
`docs/FEATURE_REGISTRY_V1.csv` に固定し、18頭分を
`analysis/2026-10-03_kyoto11_opal/canonical_nonmarket_features_v1.csv`
として実データ化しました。初期Stage 3の `nonmarket_features.csv` は監査用スナップショットとして保持します。

初期候補として10番ヒシアイラが会話上で挙がっていますが、これはモデル結論ではありません。
全頭分析では先入観として固定せず、18頭を同一手順で評価します。

## License

現時点ではライセンスを付与していません。公開リポジトリであること自体は、
第三者への再利用許諾を意味しません。OSSライセンスは依存関係・データ利用条件を確認後に決定します。
