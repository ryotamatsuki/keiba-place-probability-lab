# V5特徴量エンジニアリング研究（松尾研講義＋元論文・2026関連研究）

2026年10月31日。**GitHub Actions環境上のブロッカーにより、実JRAデータを使ったV5追加学習・評価は未完了であり、改善スコアは未計測。** 未実施を実測と表示しない。保存済みF3＋V1の2024年joint NLL 4.483655はPR #40評価原本からの**既存基準**であり、本研究新記録ではない。

## 1. 研究根拠

- Guyon & Elisseeff (2003) *An Introduction to Variable and Feature Selection*, JMLR. https://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf  ：「変数を増やす」より事前仮説別blockの逐次評価・除去（ablation）。
- Glickman (1999) *Parameter estimation in large dynamic paired comparison experiments*, JRSS C. https://www.glicko.net/research/glicko.pdf ：真の能力が時間とともに変化し、推定不確実性にも意味がある。今回の直近差分と分散は**Glickoの実装ではない**。本格状態空間化は別研究。
- Herbrich, Minka & Graepel (2006), *TrueSkill*, NeurIPS. https://proceedings.neurips.cc/paper/2006/hash/f44ee263952e65b3610b8ba51229d1f9-Abstract.html ：多者競合と強さの分散。今回の簡易分散がTrueSkillの事後分散だとは称さない。
- *Assessing the predictability of racing performance of Thoroughbreds using mixed-effects model* (2023), https://pubmed.ncbi.nlm.nih.gov/37635693/ ：日本の競走馬の平均速度・レース条件・騎手効果・距離の研究。今回の現役騎手と条件設計の参考。ただし研究の性能指標はJRA3連複ではない。
- Sugiura (2026-09-24), *Leakage-aware pre-event machine learning evaluation for horse-race prediction under temporal validation*, https://pmc.ncbi.nlm.nih.gov/articles/PMC13646064/ ：国内JRAデータで、理論的な追加特徴110変数が、元の簡素な25特徴量モデルより時間外予測性能を悪化させた。**今回も特徴を増やせば精度が上がるという前提を置かない**。予測時点監査・ブロックアブレーション・年代別外部評価が核心。
- Sugiura (2026-08-17), *Leakage-aware race-level upset-risk diagnostics*, https://pubmed.ncbi.nlm.nih.gov/42699352/ ：予測時点のレース構造と荒れやすさを馬別能力から分離して扱う動機付け。荒れ予測が3連複joint確率を改善したという実証ではない。
- Prokhorenkova et al. (2018), *CatBoost: unbiased boosting with categorical features*, NeurIPS. https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html ：ターゲット統計のリークを回避。現在の騎手/調教師の3着内率は**前日までの結果だけ**で集計する。
- ユーザーGoogle Drive保管：松尾・岩澤研究室 GCI 2024W 第7回「特徴量エンジニアリング」、第8回「時系列データとモデリング」、DL2026S第10回「表現学習と自己教師あり学習」。一般的技法を競馬問題へ翻訳したものであり講義内で競馬予測の精度は報告されていない。

## 2. 既存F3 47数値＋4カテゴリからの実際の差

既存：`NEW_HISTORY` 12本（上がり3F直近・過去3走平均、位置取り、馬体重、前走騎手の実績と継続）、`NEW_STRENGTH` 6本（先行date Elo、他馬との相対Elo）、相対能力5本、市場校正2本、市場分布5本。**2026予測への後付け追加禁止**。

V5研究用新規特徴量：`src/keiba_place_lab/trio_feature_v5.py`

1. **A 現在の騎手と調教師の前日時点実績**：現行`history_last_jockey_prior_rate`は各馬の前回騎手priorをさらにずらしているが、今回騎乗する騎手の前日時点top3率そのものではない。新特徴 `v5_jockey_prior_place`, `v5_trainer_prior_place`, 各過去スタート数log1p、他出走馬の事前能力平均との差。前日までの累積結果だけで平滑化 `(prior_successes+4)/(prior_starts+20)`。**同日の他レース結果まで事前に見ない**。
2. **B 近走能力の変動**：生データ `raw_last3f_rel`、着順百分位、`raw_passing_gain`の各馬**shift(1), shift(2), shift(3)**から、直近2回の差分・3回の分散・観測数。現行単純平均や`history_early_stability3`と異なる、直近の上昇/下降と不安定さを表す**追加情報候補**。ただし条件補正済みタイム・厳密状態空間フィルタではない。
3. **C レース先行構成**：各馬の`history_early_pct3`を使い、レース全体の先行率・頭数・市場単勝確率で加重した先行馬シェア・散らばり・他馬との相対早期位置・観測カバレッジ。前もって確定した**全頭同一レース構成**で算出（欠けた馬はfail-closed）。旧V3の3頭interaction係数ではなく馬別モデルへ入れる。

今回**未採用の候補**：条件補正の馬別走破速度・latent state filter、芝ダート/コース別Bayes階層モデル、当日の馬場変化、騎手調教師の複雑なcross-fit hierarchical effects。過去タイムの基準条件を未来のレースから推定してしまう危険や、新規データの予測時刻不確実性を解消するまで実装しない。

## 3. 実験設計・再現手順

[先行コミット済み事前登録仕様](https://github.com/ryotamatsuki/keiba-place-probability-lab/blob/research/trio-feature-ablation-v5-20261031/docs/TRIO_V5_FEATURE_ABLATION_PREREG_20261031.md)、commit `bf5a3f82a73a40ca93fbfcfdfaeade488c1a6144`。

- 2018–2022学習で作られた2023 F3 800/400 OOFと、2018–2023学習で作られた2024 F3 800/400 OOF（**既存PR40の原本**）を固定。
- `scripts/research_trio_v5_feature_ablation.py`：2023前半75%開催日ラベルで市場＋F3 offsetへの正則化ロジスティック残差を学習、同年後半25%開催日における**順不同3頭joint NLL**で feature block A/B/C/A+B+C と regularization `C={0.01,0.1,1}`、較正対照intercept-onlyを選択。2023年全体で再fit後、2024年**既存1336レース**の各全券種を同じexact-3 V1で算出。
- 二次：Joint Brier、top1/5/10 hit、予測mass、対応付き開催日bootstrap 2500回CI。評価年に新モデルの予測が無ければ標本を減らさず**失敗停止**。
- 全頭まで一致する元F3+V1レース別joint loglossをPR40原本と突合し、誤差許容`5e-6`を超えれば失敗。新モデルの馬別周辺値は元F3とは異なってよいが元bundleは不変。
- **これはF3 800/400の再学習ではない**。凍結済み2023/24 F3 OOFをベースに、追加特徴量が残差的に有用かを小規模学習で検出する最初のablation。実証が出ればXGB800+Cat400の新特徴量入りモデルを次工程で時系列再学習する必要がある。
- `tests/test_trio_feature_v5.py`：同日ラベルを変えてもその日の事前特徴量が変化しない、同日確定タイムを変えても事前ダイナミクスに影響しない、出走馬欠損時の停止、特徴量登録数と馬ID重複検査。
- GitHub Actions `.github/workflows/trio-v5-feature-ablation.yml`：`stage36-historical-training-panel-v1` original artifact run `37104212189`、Kaggle元JRA V1 `noriyukifurufuru/japan-horse-racing-2010-2025/versions/1`、当repoのPR40 OOF CSVを入力。

## 4. GitHub Actions現状：モデルの良否とは別の計算基盤ブロッカー

- 追加学習ジョブ https://github.com/ryotamatsuki/keiba-place-probability-lab/actions/runs/41204663373 は**2回ともfail、job.steps=null、jobログ取得BlobNotFound（404）**。
- この状態では`pip install`、`pytest`、実データ学習が**一度も起動していない**。Pythonコード実行時に現れた失敗ではなく、GitHub-hosted jobの起動・runner割当て以前の障害が疑われる。
- GitHub公式のActions課金・予算ガイダンス：https://docs.github.com/en/actions/concepts/billing-and-usage 。予算・支払・GitHub側の割当障害など複数原因があり、**契約画面を確認できず原因は未確定**。
- 計算不能につき`analysis/trio_v5_feature_ablation/v5_features_experiment.json`、予測CSVは**未生成**。実測改善値・95%CIの捏造はせず、現行F3+V1の実運用維持。

## 5. 現行F3・T10予測ファイル非改変の独立確認

親V4とV5末尾のGit blob SHAが完全一致：
- `models/tennosho_f3_20261101/f3_bundle.joblib`: `8836764acfc27f39c7a06ddafc7f923f64ab67b0` （元ファイルSHA256 `f8db9efe4245782165a0732945f5ab50c9c845d52fec25f20150d65e6ab096aa`）。
- `models/tennosho_f3_20261101/manifest.json`: `5ab3b11cd26d7237ce19b0b6134dfc13b473bbe8`
- `analysis/tennosho_preoff_20261101/prior_f3_features.csv`: `50093aedfbf194ab0d794889b0cb6b21ae34fc52`
- `analysis/tennosho_preoff_20261101/previews/f3_preview_20261031T173406.csv`: `a38796572b50fce476f2dcae0e51e218c02e1714`
- `.github/workflows/tennosho-preoff-freeze-v1.yml`: `2c6a342f6a5e205adfb493e2fc5b4ef15cc4d283`

## 6. 判定と次の研究工程

**最良の統計的に実証された構成は既存F3+V1のまま**。V5は新規モデル実測ができていないため、2024の追加joint NLLは未判定。上位10点の有意差・校正/買い目順位の変化も未判定。

Actionsのjob起動が復旧した場合、上記の事前登録済みワークフローだけを再実行し、2023固定選択・2024全頭同一標本スコア・paired day CIを保存する。データの前向き独立確認とT-10市場スナップショット、実3連複オッズはさらに別の合格条件。**モデル本番自動昇格しない。**
