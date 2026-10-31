# 3連複 F3 特徴量V5・実験事前登録（**新スコアを読む前に固定**）

研究ブランチ: `research/trio-feature-ablation-v5-20261031`, 親 `549a11726f0f85976d380fde86d38318a7faf402`。既存凍結F3、2026年T-10ロック、本番予測は不変。2024・2025の結果は既に参照済みで**新しいholdoutではない**。

## 論文による設計根拠（適用は仮説・JRA効果を保証しない）
- Guyon & Elisseeff (2003), *An Introduction to Variable and Feature Selection*, JMLR 3:1157–1182, https://jmlr.org/papers/v3/guyon03a.html ：追加特徴と特徴選択/ablationの必要性。
- Glickman (1999), *Parameter estimation in large dynamic paired comparison experiments*, JRSS C 48:377–394, https://www.glicko.net/research/glicko.pdf ：能力状態の時間変化と不確実性。今回の簡易近走トレンド・分散はGlicko自体ではない。
- Herbrich, Minka & Graepel (2006), *TrueSkill*, NeurIPS, https://proceedings.neurips.cc/paper/2006/hash/f44ee263952e65b3610b8ba51229d1f9-Abstract.html ：多者競合の強さと不確実性。今回の近走分散はTrueSkillの厳密なBayes事後ではない。
- *Assessing the predictability of racing performance of Thoroughbreds using mixed-effects model* (2023), PMID 37635693, https://pubmed.ncbi.nlm.nih.gov/37635693/ ：日本の競走馬、レース・年齢・騎手・距離が走行能力に寄与しうる。
- Sugiura (2026-09-24), *Leakage-aware pre-event machine learning evaluation for horse-race prediction under temporal validation*, Front. Artif. Intell. DOI:10.3389/frai.2026.1896192, https://pmc.ncbi.nlm.nih.gov/articles/PMC13646064/ ：予測時点整合、時系列ablation、同一ハイパーパラメータ。単純な歴史特徴集合が追加理論ブロックより優位だった事実を考慮。
- Prokhorenkova et al. (2018), *CatBoost: unbiased boosting with categorical features*, NeurIPS, https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html ：同日の成績や本人ラベルを使って平均を作るtarget encodingは危険。
- 松尾・岩澤研究室 GCI2024W 第7回 特徴量エンジニアリング（ユーザーGoogle Drive）：欠損情報、交差項、ラグとリーク防止の講義を参照。

## 対象・対照
親PR40で確認済み 2023年1,355完全レース・2024年1,336完全レース。元の `analysis/trio_v3_matched_oof/2023_2024_matched_800_400_oof.csv` の全頭2023/24 OOF（XGBoost800+Cat400 F3）を固定。JRA実結果Kaggle V1、元のstage36固定過去特性パネルを使用。全出走馬の馬番/horse_id/race_idと実1–3着が揃うレース以外を都合良く捨てず、PR40と完全同一population。新特徴の欠損は欠損状態フラグを使用／モデル内で訓練期間だけでmedianを推定。全面の欠落・参照不一致は**検証失敗**。
本研究は **凍結F3自体のXGB/Cat再学習ではなく**、年別時間外F3予測をoffsetとして新特徴の追加情報だけを検査する正則化ロジスティック残差学習（臨床的/比較実験の簡易・低予算前段）。元の800+400パラメータを保持。これによって新特徴の追加価値が確認できた場合に、後段で800/400モデル自体を再学習する。**この段階を本格的F3再学習と混同しない**。

## 説明変数（構造を事前固定）
**block A current_agent**: 出走当日の`jockey_name`/ `trainer`の、過去**日付**までの3着以内率（Bayes擬似試行数=20、prior=0.20）と有効過去出走数log1p、両者の同レース各出走馬平均との差。各日`t`のラベルが当日他レースに入らないよう、同日をまとめて累積更新。騎手・調教師のID自体をモデルに投入しない。注意：最終単勝オッズが同日の発走前固定時刻より後に決定されるデータであるため、厳密T−10適合性は別途必要。
**block B recent_dynamics**: 過去3〜4走の`raw_last3f_rel`, `raw_finish_position/field_size`, `raw_passing_gain`の**過去レースのみ**の変化（近走2値の差）、分散、観測数。対象レースの当該結果は決して用いない。Glickman/TrueSkillは理論的動機であり、本特徴は簡易なmomentで厳密推定ではない。
**block C race_pace_composition**: 同一レース出走全馬の、過去3戦の`history_early_pct3`を対象とした逃げ候補割合・分散・順位、および事前の市場単勝シェアで加重した逃げ馬割合、強い逃げ馬頭数。すべて同じレースの他馬情報は予測時点に既知。ここで市場とは履歴の**最終単勝**シェア（T-10ではない）。このblockは先行V3の3頭相互作用補正と異なり、馬別モデルへのレース構造特徴。
欠損状態は各blockで`*_missing`として明示。補完、推定変換器は必ず過去学習行でfit。

## 完全に先行固定する比較
- 既存PR40 `F3+V1`、残差追加の `A`、`B`、`C`、`A+B+C`、校正対照`INTERCEPT_ONLY`を2023年後半のinner validationで比較、2023年以前に学習済みOOF F3の馬別予測をoffset logitとする。
- Train 2023前半75%開催日、inner validation 2023後半25%開催日→block選択／L2強度 `C∈{0.01,0.1,1.0}` を選ぶ。条件数は5候補×3 + offset-only。各候補の早期スコアは3連複joint log lossのみ（馬別二値lossで重み調整しない）。
- 選ばれたblock/Cを2023年1,355レース全部で再fitし、2024年1,336レースの予測を作る。2024年結果で係数・blockを調整しない。**学習者の最大予算：scikit-learn LogisticRegression/L-BFGS, maxiter=300, 2 CPU, fixed seed=20261031**。
- 3連複変換は既存`trio_bridge_v1`完全同一のexact-3最大エントロピー（2024全頭を残す）。同レースで成績順が確定した3頭のunordered joint log loss、同時Brier、top1,5,10 hit, mass、日付cluster 2500 bootstrap95%CI、full field一致・sum=1・馬別確率未改変検査。
- 改善がCI下限正であっても2024年は既に観察済みのため暫定開発評価；one-SE・多数候補探索によるバイアスに留意。前向きholdout・T-10時刻の市場入力監査・実3連複オッズとROI検証までは**本番NO_BET**。

## データと未検証事項
過去歴のmarket_finalはT-10ではなく、実3連複市場の価格ではない。出走当日の`jockey_name` `trainer`の履歴は当日以降の結果を**shift by day**で除外。実レースの当日天候・途中展開・確定時計・オッズ変化は一切入力禁止。原本SHA、Git Blob SHA、seed、Python/依存、実行時間を記録。2026予測ロックは完全不変。
