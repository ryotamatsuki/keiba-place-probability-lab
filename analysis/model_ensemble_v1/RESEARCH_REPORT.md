# Model Ensembles V1 — 既存OOF予測の追加精度探索（2026-10-31）

**結論：X0・X1・X2の単純平均は既存の最良X2を上回らなかった。一方、2023年のOOFに限って学習した市場logit補正係数を2024年のレースへ適用すると、X2単体をさらに改善した。発走10分前の予測・未見将来データへの効果は未判定。**

## 1. 再現性・データの厳密な範囲

本実験はPR #32（`research/information-expansion-v1-20261031`、親コミット `1b276b695db600753007d565ea52d670cd2d2917`）に保存済みの2023年・2024年の時系列OOF `analysis/information_expansion_v1/outer_oof_predictions.csv` のみを使う。**実験仕様と7つの候補をスコアを見る前にコミット**：`docs/MODEL_ENSEMBLE_V1_SPEC.md`、`1a1227f4ee838c63ec03c49d8ca7d1845b84bcd8`。

- 基本モデルM0＝過去の**最終**単勝オッズをHarvilleとlogit校正で3着内確率へ変換したもの。**T−10の実際のオッズは存在せず、そこへの効果は報告しない。**
- X0＝従来履歴だけを使った非線形XGBoost市場補正、X1＝新たな過去履歴、X2＝過去の対戦相手に基づくEloを追加した同系統XGBoost。
- X0/X1/X2はPR #32で事前に定めた`max_depth=2, n_estimators=120`等のパラメータのまま。今回は再fitしない。
- 2023年OOF（2,887レース）で**E5のスカラー係数だけ**学習。2024年の結果で重み学習しない。2024年は2,844レース・31,225頭、同一評価行の完全照合。評価行SHA256 `2291807521499019cb2a0c62f416c1e9e4f43f0b421711371d5f41aa97c8941a`。
- **2024年の結果自体は過去のPR #32で既に観察された開発標本**であり、完全な独立未見テストではない。
- race-macro Brier第一、log loss第二、開催日クラスタ化paired one-SE。市場M0・既存X2の**両方**を比較対象とする。ROI、的中率、単発G1をモデル選択に使わない。

## 2. 事前固定した7候補

- M0：市場のみ。
- X2：既存の最良XGBoost。
- E1：`0.5 * X0 + 0.5 * X2`（確率平均）。
- E2：`sigmoid(0.5 * logit(X0) + 0.5 * logit(X2))`（logit平均）。
- E3：`0.5 * X1 + 0.5 * X2`（確率平均）。
- E4：`0.1 * M0 + 0.9 * X2`（市場への縮小）。
- E5：`sigmoid(logit(M0) + alpha * (logit(X2)-logit(M0)))`。2023年のOOFだけで `0 <= alpha <= 1.5`、Brier＋事前固定正則化 `0.0001*(alpha-1)^2` を最小化。**alpha=0でM0、alpha=1でX2**へ戻ることを確認。学習結果 **alpha=1.4782533520459664** を2024年に固定適用。

## 3. 同一2024年評価標本の結果

| 候補 | race-macro Brier↓ | race-macro log loss↓ | X2基準のpaired one-SE |
|---|---:|---:|---|
| M0（最終単勝オッズのみ） | 0.136376519 | 0.422136248 | 下回る |
| **X2** | **0.135939400** | **0.420907085** | 比較基準 |
| E1：X0/X2確率平均 | 0.135982257 | 0.421005680 | 悪化 |
| E2：X0/X2 logit平均 | 0.135982135 | 0.421004481 | 悪化 |
| E3：X1/X2確率平均 | 0.135972232 | 0.421009712 | 悪化 |
| E4：市場10%/X2 90% | 0.135969238 | 0.420991363 | 悪化 |
| **E5：2023年のみで学習した補正幅** | **0.135838654** | **0.420632775** | **通過** |

E5はX2よりBrier `0.000100746` 改善（開催日paired SE `0.000038889`、改善率約0.0741%）、市場M0より `0.000537865` 改善（paired SE `0.000119097`、相対改善率約0.3944%）。既存 `select_with_incumbent_one_se` で、市場をincumbentにしても、X2をincumbentにしても、**点推定最良のE5が選択**された。ただしone-SEは統計的有意性の保証ではない。

## 4. 含意

**単純なアンサンブルを重ねることでは精度は改善しなかった。** 木の学習器が同じ・特徴量も大きく重複するX0/X1/X2を混ぜると、最良X2の補正が弱まる可能性がある。保存済み結果からは、CatBoostやLightGBMを混ぜるメリットは判断できない（未学習）。

E5で改善が確認されたのは、「X2の補正が弱めに推定されている可能性」を示唆する。**E5は平均アンサンブルではなく補正の強さを学ぶスカラー後処理**である。alphaが探索上限1.5に近いこともあり、最適係数の安定性は未確認。別のデータ・年・実際のT−10市場で過信しない。

2024年全体の校正切片・傾きはX2が `-0.012754, 0.996653`、E5が `-0.014015, 1.000732`。E5の確率調整で改善した分と校正の違いを別の期間でも比較する必要がある。これらの診断は学習用ではない。

## 5. 学術研究・技術文書による次の改善仮説

1. **CatBoostの市場オフセット：優先度高。** カテゴリ変数をネイティブに扱え、XGBoostと予測誤差の多様性を生み得る。CatBoost `Pool(baseline=logit(M0))` または `fit(baseline=...)` による加算logit入力を利用し、推論時のbaseline合算が正しいことをテストする。カテゴリのordered statisticsは時間順と整合させる必要があり、`has_time=True`だけでは**同日他レースのラベル混入**を完全には防止しない。生の騎手/調教師カテゴリは現在の確定出走者情報を示していないため、安易にそのまま使わない。参照：https://catboost.ai/docs/en/concepts/python-reference_pool , https://catboost.ai/docs/en/references/training-parameters/common .
2. **LightGBMの市場オフセット：優先度高。** XGBと違うleaf-wise学習の帰納バイアスが補完的か実測する。少葉数・強い正則化・学習側のみのearly stoppingとし、`init_score`が**推論時も含めて市場logitを正しく復元するか**を回帰テスト。参照：https://lightgbm.readthedocs.io/en/latest/Parameters.html .
3. **XGB自体のバイアス・分散比較：優先度高。** depth2/120 trees固定の現行と、depth3でより強い正則化、より小さい学習率等を少数の事前登録候補で比較する。ただし全候補2024年を見てから追加することは再選択バイアスになる。次期探索は新しい時系列OOFを別登録する。
4. **異種学習器の低自由度アンサンブル：優先度高。** XGB/CatBoost/LightGBMの残差 `f_k=logit(p_k)-logit(p_market)` をOOFのみで推定した小数の非負重みで足す。正則化で`f=0`（市場）へ戻せること、既存X2を単独候補として残すことが必須。単純平均が既にX2を下回った点を踏まえ、過学習しやすい条件別重み/大規模stackingは後回し。参照：Wang et al. (2023), *Forecast combinations: An over 50-year review*, https://arxiv.org/abs/2205.04216 ; Gneiting & Ranjan (2010), *Combining probability forecasts*, https://doi.org/10.1111/j.1467-9868.2009.00726.x .
5. **確率の構造的制約・レース全体の順位モデル：中期。** listwise等の順位精度の改善は3着内確率のBrierを直接最小にするものではない。先行研究：https://doi.org/10.9708/jksci.2025.30.11.311 。全出走馬のOOFと同着/取消の規約を整えてから構造化確率や`sum P(top3)=3`射影を比較し、部分評価時に非悪化保証がない点を明記。
6. **EBM・深層学習：後順位。** EBMは非線形を解釈可能な形で表現し、タブラーデータに強い。複数の大規模ベンチマークではGBDTが依然優勢で、深層学習の無条件優位を期待しない。参照：https://interpret.ml/docs/ebm.html ; https://papers.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html .

## 6. 次の実験を設計する際の厳格な条件

1. **新しい事前登録**で3学習器の候補数・探索幅を小さくし、内側2019–2022 OOFでアンサンブルをfitし、外側2023を予測。次に2019–2023 OOFでfitし、外側2024を予測。すべての外側ラベルを特徴量生成、校正、時系列重み探索から遮断。
2. モデル間の「予測差とレース損失差の相関」「両者のどちらも誤るレース」を分析し、異種モデルがX2と**独立の追加情報**を持つことを確認する。
3. 市場のみ・X2・E5・異種モデルを同じ最終オッズ／同じ出走馬集合で競わせ、レースmacro Brierと日付クラスタpaired SE、2023/24別・芝/ダート・距離帯/クラスの校正も保存。
4. **2025年・2026年は既知結果**である。過去のモデル改良のデータを完全未見テストと呼ばず、新規採用判断は本番と同時刻の市場オッズを使う真の前向き検証だけに行う。
5. 既存の本番Stage5市場モデル、発走10〜9分前の予測ロック、最低1000レース・60開催日・90日間、途中採否変更禁止を維持。PRの研究モデルを自動採用しない。

## 7. 生成物と再現コマンド

```bash
python -m pip install -e '.[dev,historical,modeldev]' -c requirements-scope.txt
ruff check .
pytest -q
python scripts/run_model_ensemble_v1.py \
  --oof analysis/information_expansion_v1/outer_oof_predictions.csv \
  --output-dir /tmp/model-ensemble-v1
```

実行したGitHub Actions： https://github.com/ryotamatsuki/keiba-place-probability-lab/actions/runs/41130443457 （**167 tests PASS**、score実行PASS）。

保存成果物：`analysis/model_ensemble_v1/manifest.json`（ソースhash、評価行hash、E5係数）、`joint_market_gate.csv`、`joint_x2_gate.csv`、`paired_vs_incumbents.csv`、`2024_predictions.csv`、`race_losses.csv`、`race_differences.csv`、`calibration.csv`、`reliability.csv`。作成コード `scripts/run_model_ensemble_v1.py` と追加の回帰テスト `tests/test_model_ensemble_v1.py`。

**最終判断：2024年の既知の開発標本では、2023年学習のE5 logit補正強度調整がX2をさらに上回った。単純平均アンサンブルは上回らなかった。CatBoost/LightGBM異種混合はまだ試験しておらず、次段階の探索余地として残る。**
