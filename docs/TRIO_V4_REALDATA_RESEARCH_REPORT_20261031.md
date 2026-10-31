# 3連複予測構造V4：事前登録済み実験・実測報告（2026-10-31）

研究ブランチ：`research/trio-structures-and-pooling-v4-20261031`、親PR #40最新commit `33501c044b2fc27054bef36224fea9d4c9054533`。**本番モデルを変更しない研究Draft PR。**

## 1. 実施状況と評価上の結論

| 事前指定工程 | 研究実装 | 合成/既存テスト | 実際のJRA 2023/24実測 | 判断 |
|---|---|---|---|---|
| 変換後の算術joint混合・端点・50:50・75:25・2023年学習重み | **完了** | **合格** | **完了：2024年1,336レース** | 既存F3+V1維持 |
| 共変量付き順位PLおよび2着/3着補正GPL | 実装済み | 初回静的検査未通過（Python lint） | **未完了** | 採否保留 |
| 公式 TabM 0.0.3 / 47+4 F3入力→V1 | 実装済み | 公式forwardテスト成功、pandas3の書込み不可配列問題を修正 | **未完了** | 採否保留 |
| XGBoost深さ2/3/4、固定400木 | 実装済み | 検査・実データ評価のActionsジョブが実行ステップ前に失敗 | **未完了** | 採否保留 |
| 2023年後半の未学習rank、TabM、F3 joint三者混合 | 実装済み、欠損時停止 | 予測生成待ち | **未完了**（前段データなし） | 採否保留 |
| TabMで集合尤度を直接学習するモデル | **未実装** | — | 未実施 | 次期研究 |
| 実3連複市場オッズ・ROI | PR40での厳密な未取得ゲート維持 | — | 実市場価格履歴なし | NO_BET |

**重要：完了していない候補を「学習・検証した」とは記載しない。** 本報告はその時点で成果を保存した部分完了報告。Draft PRも完成/マージ可とは扱わない。

## 2. 事前登録・時系列

変更された候補を評価する前に`docs/TRIO_V4_PREREGISTRATION_20261031.md`をcommit `215f18aac25b6185ad26ee392e85dabaab3a68c1`で先行固定。古い深さ比較設定のchild/正則化値に誤記があり、木深さ実験開始前に既存`FIXED_MODEL`を確認、同文書追記コミット`dfc34da237e964b2e526b992845c51b6d5467706`で訂正。元の記載を消さず記録。2024年結果を見て木探索候補を選び直していない。

2023年1,355完全レース（F3は2018–22で学習済みOOF）で混合重み学習／順位モデルの内側時系列分割に利用。2024年1,336完全レースで評価（F3は2018–23学習済みOOF）。2024年・2025年とも過去のPRで参照済みであり、**新しい独立標本ではない**。最終単勝オッズは発走前T−10で観測できた情報ではない。

### 正規のベースライン

`analysis/trio_v3_matched_oof/experiment_manifest.json`で照合済み：
- 2024年、市場maxent：joint logloss `4.511338507937051`
- 同構成XGB800＋Cat400 75:25馬別F3→exact3 V1：`4.483654926419768`
- matched V2：`4.483063300811052`
- conditional V3：`4.482789955405274`
V2・V3上積みは統計的に未確証、既存F3+V1を研究上のincumbentと固定。

## 3. 実証済み：V1変換後のアンサンブル（初回最優先）

[実データ実験成功Actions](https://github.com/ryotamatsuki/keiba-place-probability-lab/actions/runs/41199651269) — **追加の木/ニューラル学習ゼロ**。既存の年別OOF（XGB800、CatBoost400、75:25 F3）を全レース・全馬の順序付きkeysで一致させ、各馬の複勝確率からexact-three maximum entropyへ個別に変換し、確率分布同士を算術混合。2023年1,355レースで混合重みをjoint NLL最小化した後、2024年1,336レースで固定評価。

| 順不同3連複joint構造 | 2024レース平均joint logloss ↓ | Joint Brier ↓ | top1的中 | top5的中 | top10的中 |
|---|---:|---:|---:|---:|---:|
| **現行F3馬別75:25→V1** | **4.483654927** | 0.973033350 | 7.934% | 24.626% | 36.826% |
| XGBoost800単体→V1 | **4.483075880** | **0.973023505** | 7.934% | 24.401% | 36.976% |
| CatBoost400単体→V1 | 4.487834195 | 0.973114385 | 7.859% | 24.925% | 36.751% |
| XGB/Cat変換後50:50 | 4.484678691 | 0.973050770 | 7.859% | 24.401% | 36.751% |
| XGB/Cat変換後75:25 | 4.483672437 | 0.973032593 | 7.934% | 24.551% | 36.901% |
| **2023年OOFで学習したXGB/Cat混合** | **4.483075880** | **0.973023505** | 7.934% | 24.401% | 36.976% |
| 現行V1＋選択した別構造（2023年の係数） | 4.483075880 | 0.973023505 | 7.934% | 24.401% | 36.976% |

21点（5%刻み）の事前重み候補により、2023年の最良XGB重みは **1.0**（CatBoost重み0.0）。2023年のXGB対Cat学習済み結果のみで決めている。現行V1＋最良第2構造（XGB）という二者重みも2023年5点グリッドで**全XGB重み1.0**。混合の意味ある補完効果を確認できなかった。

**現行V1対XGB単体の改善**（正がXGB優位）：`+0.0005790472`（レースあたりjoint NLL）、2024年106開催日・2,500回paired cluster bootstrap95%区間`[-0.00034455,+0.00149297]`。**有意でない**。Joint Brier改善`+0.0000098445`。

Top10実測的中差：XGB 36.976% 対 現行36.826%、**+0.1497 percentage points**、paired cluster 95%区間`[-0.3657,+0.6712]` percentage points（有意差なし）。Top1では同一。XGBのTop5的中率は現行より低下。

分布を混合するときの各馬上位3頭への周辺確率は、元の凍結F3馬別P(top3)と一致するとは限らない。独立した確率研究用の結果であり元の凍結本体は不変。

保存済み：
- `analysis/trio_v4_postmix/2024_post_v1_experiment.json`（2023重み、2024統計・出典SHA・73.6秒の本処理時間）
- `analysis/trio_v4_postmix/2024_post_v1_racewise_scores.csv`（完全な同一1,336レース、全候補の一次/二次指標）
- `analysis/trio_v4_postmix/2024_post_v1_all_trio_predictions.csv.gz`（全候補の各3頭組合せ、actual flag付き）

**この時点での最良実測単体はXGB800→V1、最良実測学習アンサンブルも重みがXGB100%に収束した同等モデル。** ただしCIがゼロを含むため、本番・研究正式候補とも既存V1を維持。

## 4. 追加予測構造の実装と実測未完の理由

### 共変量付きPL/GPL
- `src/keiba_place_lab/trio_covariate_rank_v4.py`：市場・XGB・CatBoostの年別OOF確率から事前logit特徴量、上位1–3着の順序付き尤度で共変量の強さを学習。通常PLと、2着・3着段階の強さ倍率を変えるfixed/learned GPL。各順不同3頭の**6通りの順列確率を足す**ことで3連複分布を生成。 `tests/test_trio_covariate_rank_v4.py`が組合せ全確率和1と6順列一致を検査。
- `scripts/research_trio_v4_rank.py`：実JRA着順をOriginal Kaggle V1に結び、2023年前半の順位モデル学習・同年後半のNLLモデル選択→2023年全体再学習→2024年のPR40全標本。内側の時系列外順位予測のうち後半のみを混合重み学習に使用。
- 最初のRank Actionsジョブ `41199936499` は静的lint段階で失敗し**実順位学習を実行できていない**。その後は新規ジョブがrunnerを開始せず終了する現象を確認、最終実結果は未取得。

### 公式TabM
- 論文：Gorishniy, Kotelnikov, Babenko *TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling*, ICLR2025、[公式リポジトリ](https://github.com/yandex-research/tabm)。
- `tabm==0.0.3`（2025年8月14日版、Apache-2.0）、PyTorch CPU。47数値・4カテゴリ、前年までに標準化・欠損補完・カテゴリ符号化。`k=4`、隠れ64、2 blocks、12 epoch上限。モデルのメンバー4本それぞれの**BCEを平均して最適化**し、推論時に4本の確率を平均。単独F3と同一予測時点の市場logitへ残差補正。
- `scripts/research_trio_v4_tabm.py`、`tests/test_trio_tabm_v4.py`、Actions `trio-v4-tabm.yml`。
- 初回Actions `41200226464` は依存導入・公式forwardテストを通過した後、Pandas 3の書込み不可配列で特徴量前処理が失敗。コードを`.copy()`へ修正したが、新たな実データ学習成功を確認できていない。**TabM結果は未実測**。TabMをMLPにすり替えて「TabM試験完了」と表現しない。

### XGBoost depth
- `scripts/research_trio_v4_xgb_depth.py`：400木固定、PR40と同じ学習済みF3情報、事前3設定、2023年でモデル選択・2024年で検証。元のXGBの実設定はdepth2/child60/lambda20/alpha2。候補A depth3/child5/lambda10、B depth4/child10/lambda20（alpha2共通）。
- Actions `41200394398` が実行ステップ前に終了し、失敗時ログの実体が取得できなかった。**XGB再学習の実測スコア未取得**。

### 複数構造アンサンブル
- `scripts/research_trio_v4_multistructure.py`：事前3候補、2023年後半の完全な未学習rankとTabMの全券種OOFで凸重みを学習し、2024年1,336レースの同一3頭組合せで公平評価。馬別周辺確率相関、上位1点の買い目変化、NLL/Brier/topKを保存。
- `.github/workflows/trio-v4-multistructure.yml` は必要な実験ファイル5種が**全部揃うまでは明示的にfail-closed**。実市場オッズや暫定値で補完しない。従って実測は未完了。

## 5. 前向き検証と昇格判定

Git Blob SHAを分岐元・本研究の最終状態で突合：
- `models/tennosho_f3_20261101/f3_bundle.joblib`: `8836764acfc27f39c7a06ddafc7f923f64ab67b0`
- `models/tennosho_f3_20261101/manifest.json`: `5ab3b11cd26d7237ce19b0b6134dfc13b473bbe8`
- `analysis/tennosho_preoff_20261101/prior_f3_features.csv`: `50093aedfbf194ab0d794889b0cb6b21ae34fc52`
- `analysis/tennosho_preoff_20261101/previews/f3_preview_20261031T173406.csv`: `a38796572b50fce476f2dcae0e51e218c02e1714`
- `.github/workflows/tennosho-preoff-freeze-v1.yml`: `2c6a342f6a5e205adfb493e2fc5b4ef15cc4d283`

すべて**親ブランチとのSHA一致**を確認。Git Blob SHAはGitオブジェクトSHAであり、凍結JoblibのファイルバイトSHA256ではない。JoblibのバイトSHA256は`f8db9efe4245782165a0732945f5ab50c9c845d52fec25f20150d65e6ab096aa`。

モデル選択は予測対象2024年の結果を使って重み最適化していない。ただし、多候補を2024年に照合したあとで新しい研究方向を決めている以上、95%CIを無調整で多数候補の真の優位性の証明とみなさない。one-SEは選択正則化、95%CI有意性とは別。

2026年11月1日天皇賞（秋）のT−10既存予測はこのブランチから書き換えない。実3連複の全券種オッズと的中払戻未取得、実損益は未検証。**合格していないモデルの自動的な本番昇格・購入推奨は禁止（NO_BET）。**

## 6. 再現手順

1. PR40ヘッドから本研究ブランチへcheckoutし、`python -m pip install -e '.[dev,historical,modeldev]' -c requirements-scope.txt`。
2. `pytest -q tests/test_trio_pool_v4.py`。コード/固定結果:`.github/workflows/trio-v4-postmix.yml` を実行。
3. 元Kaggle V1 `noriyukifurufuru/japan-horse-racing-2010-2025/versions/1`、2010–2024 stage36固定パネルはPR40と同一run `37104212189`から復元（プロビナンスはPR40 manifest）。
4. `.github/workflows/trio-v4-rank.yml` を成功するまで検証し、`trio-v4-tabm.yml`、`trio-v4-xgb-depth.yml` を成功させる。
5. 2, 3のすべての実モデルの全券種予測が実在する場合にのみ `trio-v4-multistructure.yml` を実行する。実装のみの候補を最良モデルのランキングへ混在させない。
6. 全ラン成功・CIテスト合格と未参照の前向き評価まで、本Draft PRは**本番採用不可**。

使用元データは公開レース成績と市場**最終単勝オッズ**。過去の券種固有の実オッズではない。
