# 3連複モデル構造V4 事前登録仕様（新規結果を読む前に固定）

状態: **PR40現行先端 commit `33501c044b2fc27054bef36224fea9d4c9054533` を基点に、比較対象・学習予算・昇格規則を先行固定**。
研究ブランチ `research/trio-structures-and-pooling-v4-20261031`。凍結F3・既存V1・2026年天皇賞の推論/ロックは変更しない。

## 監査とデータ
- `AGENTS.md` はPR40のGit treeを再帰確認して**不在**。READMEと `docs/MODEL_SELECTION_PROTOCOL_V2.md` 確認。後者は馬別P(top3)のrace-macro Brierの選択規約。本研究3連複は**別の予測目的**のため、以下の事前定義joint log lossを一次指標とし既存の規約を書き換えない。
- 既存モデルの真正性：`models/tennosho_f3_20261101/f3_bundle.joblib` SHA256 `f8db9efe4245782165a0732945f5ab50c9c845d52fec25f20150d65e6ab096aa`。研究分岐元とのGit Blob SHAおよび終了時SHA一致を確認する。
- 予測入力：`analysis/trio_v3_matched_oof/2023_2024_matched_800_400_oof.csv`、PR40の`experiment_manifest.json`が示す同一構成XGB800+Cat400 OOF。新たなJRA生データが必要なrank/TabMはKaggle元V1/固定stage36、source SHAで監査。
- 研究用完全レース：**2023年1,355レースで混合重み/順位モデルの係数を推定し、2024年1,336レースで開発用比較**。PR40完全レース集合（2024年）。出走馬全員に過去3戦以上・芝ダート1000–2600m・8頭以上・直線コース除外・全馬オッズと上位3頭の正しい着順、identity/race_id一致が必要。途中候補の欠損で2024年評価対象を減らさず、失敗候補は「未評価」と記録する。
- 2025年1,253レースも2024年1,336レースも既に**結果を参照済み**。ここに「新規独立確認」とは書かず、翌日以降の前向きholdoutだけを最終的な独立証拠とする。
- 単勝最終オッズの履歴評価であり、実3連複オッズ／前日〜T-10同時点市場は未取得。利益／ROIの改善や実3連複市場に対する優位性は主張しない。

## フェーズ1 最優先・大規模追加学習なし
元の馬別`p_matched_xgb800, p_matched_cat400, p_matched_f3`から別々にV1 exact-3 max entropy分布を生成。現行75:25の馬別logit offset混合→V1をincumbent。算術 joint混合候補：
- XGB800単体V1、Cat400単体V1、現行統合F3+V1
- XGB/Cat equal(.50/.50)
- XGB/Cat 75/25
- XGB/Cat weight `w` : 2023年joint NLLから`w∈[0,1]` を**0.05刻み21点**で推定、同率時`w=1`に近い簡素な端点優先。
- incumbent F3V1との混合：`w∈{0,.25,.50,.75,1}`、追加単体候補が2023年joint NLLで最良のモデルを先に選択し、2023年だけで最適w選択。
- アンサンブルの確率は非負・総和1、確率周辺和3を満たすが**既存のF3馬別周辺確率を保持するとは限らない**。

## フェーズ2 共変量付き順位モデル
F3と同一予測時点の市場事前logit、XGB800、Cat400の予測強度を説明変数とする（非独立メタ順位モデル）。**2018–2022→2023年の学習外F3の2023年上位1–3着結果**でメタ係数推定、2024年未学習結果で評価。歴史的原本から1,2,3着を厳密に復元し、欠けていればモデル評価不可（標本の黙示削減禁止）。通常PLの候補およびstage2/3別指数調整候補。強さ`s_i=β_0+β_1 logit(p_xgb)+β_2 logit(p_cat)+β_3 logit(p_market)`、勝ち順位1,2,3でsoftmax(α_rank * s_i)を逐次選択。上位3頭の順列6通りを合計し、順不同確率の全組合せの総和1を断言テスト。
- βのL2 `{0.03,0.10}`、α=(1,1,1) or α=(1,0.8,0.6) fixed or 2023年学習で最大三パラメータまで; イベント2024結果を使って選択しない。
- 係数・順位補正選択は2023年内の前半fit・後半valid（日付境界）か固定事前候補の2023年スコアのみ。2024年調整不可。
- 演算予算: Python scipy L-BFGS-B、最大150反復、CPU2コア以下。

## フェーズ3 TabM 公式OSS
TabM ICLR2025 Gorishniy et al. 公式repo `https://github.com/yandex-research/tabm`、PyPI `tabm==0.0.3` (2025-08-14)、**Apache-2.0**。`torch` stable pip、版を実行ログ保存、独自変更時は明記。
- 各年までのF3数値47・カテゴリ4のみで2023/2024 chronological OOF。StandardScaler/中央値補完、カテゴリone-hot vocabは前年までにfit。
- 小型TabM `k=4,` `d_block=64`, `n_blocks=2`, dropout 0.1、batch 512、AdamW lr .001 weight-decay .001、seed 20261031、CPU2/epoch最大12、壁時計制限18分/評価fold。
- 本試験は馬別P(top3) BCE (オフセット logit p_market_oof にnet補正)→V1変換。旧47数値+4カテゴリがない場合や計算制約で実行不可なら明示「未実施」。
- 順不同集合NLLを直接最適化する小MLPは追加候補1つ：各馬1スカラーscore→`logP(S)=Σ_{i∈S}s_i-logsumexp_{|T|=3}Σ_{j∈T}s_j`、固定レース単位損失で学習。**680個の独立クラス分類はしない**。時間枠/入力問題で未完ならそのまま報告。

## フェーズ4 既存XGBoost最小探索
既存round比較はdepth=2、reg等固定なので新候補は`(max_depth,min_child_weight,reg_lambda)`を **(2,1,1) incumbent vs (3,5,10) vs (4,10,20)** 3設定のみ、rounds=400、prior cutoff/カテゴリ/offset/重み不変。2018–22→2023候補を比較、2023年NLL良い上位1つを2018–23→2024へ、追加の計算上限は各fold 3モデル、CPU2、合計30分。モデル選択に2024年結果は使用しない。

## フェーズ5 メタアンサンブル
固定候補は「V1+rank」「V1+TabM」「V1+rank+TabM」だけ。候補ごとの単体 joint P(S) が同一2023/2024の全レースで揃っている場合に限り、2023年joint logloss最小化、非負/総和1。2者grid 0.05、3者単体0.10格子、同点時にincumbent優先。2024年は係数を固定して1,336レースの同一標本で評価。欠けた候補について結果を創作しない。

## 指標と昇格判定
**一次:** 2024年1,336レース等重み unordered exact-3 joint log loss。
**二次:** Joint Brier、top1/top5/top10 hit率、各topK予測質量/経験率差、10分位ECE、馬別予測差とracewise損失相関、芝/ダート×距離の探索的層別。
同じレースが全候補共通で、組合せ確率和1、actual一意、重複馬番0、race ID完全一致。開催日clusterを使う対応あり2,500 bootstrap95% percentile CI。one-SE incumbent rule＝改善`>1 paired cluster SE`は選択正則化に過ぎず統計的有意性とは別、最終採用には**一次改善の95%CI下限>0、joint Brier非悪化**が必要。複数候補選択・既観測年の探索によりこのCIは事後選択バイアスの影響を受け、真の昇格根拠は前向きholdout。
**本番はNO_BET・凍結F3+V1維持。** 2026発走前の正式予測にこのブランチを適用しない。

## 追跡
source SHA・Git blob SHA・seed・依存と実行時間・GitHub Actions run ID・各評価コホートをJSON保存。新しいスコアを書き出す前に本仕様をcommit。
