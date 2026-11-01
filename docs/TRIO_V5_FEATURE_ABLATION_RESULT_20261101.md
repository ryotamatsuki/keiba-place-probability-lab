# V5特徴量ablation 実測結果（2026-11-01）

## 結論

事前登録の残差ロジスティック一次スクリーニングは、2023年1,355レースで候補選択・全2023年再学習を行い、**開発データとして既知の**2024年1,336レースにおける3連複joint確率評価まで完了した。成功したGitHub Actions実行IDは [41256745528](https://github.com/ryotamatsuki/keiba-place-probability-lab/actions/runs/41256745528)。起動時・実checkout SHAは共に `d40e6ab1e8249b462dfa7f5976e63ee9f93d62eb`。

2023年後半だけで選択したのは **all (騎手・調教師、直近変化、先行構成) / C=1.0**。同内部検証joint log lossは4.4899943153、同標本F3対照は4.4936653785だった。しかし2024年joint log lossはF3+V1 4.4836549265に対してall 4.4838327450であり、差は悪化方向。**再現性のある実行成功とモデル精度の改善を混同しない。F3+V1維持、NO_BET、Draft。**

これは凍結XGBoost800本・CatBoost400本の再学習ではなく、過去に生成済みの年別OOF周辺確率へロジスティック残差を足すablation。

## 評価条件と結果

- 2023: 1,355レース、2023年内前半985レースで学習・後半370レースで候補選択、後に1,355レース全体で再fit。
- 2024: 1,336レース・18,258頭・506,219の順不同3頭組合せ。同じ正解と同じF3 OOF母集団を使用し、対象の間引きなし。
- 2024検証は既に参照した開発評価であり、新規独立holdoutではない。

| 指標 | 基準 F3+V1 | 選択 all / C1 |
|---|---:|---:|
| Joint log loss ↓ | 4.483654926488 | 4.483832744982 |
| Joint Brier ↓ | 0.973033349629 | 0.972977912780 |
| top1 hit ↑ | 0.079341317365 | 0.080089820359 |
| top5 hit ↑ | 0.246257485030 | 0.239520958084 |
| top10 hit ↑ | 0.368263473054 | 0.375000000000 |

開催日106日のペア日付クラスターbootstrap 2,500回。**損失の改善量は（基準−候補）、的中率は（候補−基準）**、正値が改善。hit率の数値差は割合単位であり、ポイント換算は100倍。

| 指標 | 改善量 | 95% CI | 改善割合 |
|---|---:|---|---:|
| Joint log loss | -0.000177818494 | [-0.004722280269, +0.004234794682] | 46.24% |
| Joint Brier | +0.000055436849 | [-0.000193132794, +0.000308534126] | 64.96% |
| top1 hit | +0.000748502994 | [-0.002263427831, +0.003908010907] | 59.76% |
| top5 hit | -0.006736526946 | [-0.013508698120, 0.000000000000] | 2.28% |
| top10 hit | +0.006736526946 | [-0.001533184073, +0.015184938683] | 93.80% |

いずれも信頼区間下限が正の改善を示していない。top10 hitの点推定上昇は、事前登録済みjoint log lossを上書きするモデル選択理由にはならない。

## 品質・再現性・完全性

- [CI #41256745576](https://github.com/ryotamatsuki/keiba-place-probability-lab/actions/runs/41256745576): ruff pass、pytest 319 passed / 6 skipped（ソースコード修正コミットに対して）。
- V5 workflow #41256745528: test、履歴・結果取得、2023選択・再fit、2024評価、基準一致、結果保存、pushの全工程success。
- 基準PR40 F3+V1 joint lossのレース別最大絶対差 **4.3231895174855595e-7**、許容 **5e-6** 内。
- 組合せ数 C(n,3)、レースごとの各モデル確率総和1、正解チケットの一意性、CSV/gzip/JSONの読み戻しを検証。
- 保護対象5ファイル（F3モデル本体、manifest、2026事前17頭特徴量、preview、T-10 workflow）の実行前後SHA256ファイルは完全一致。出力成果物以外の過去評価を置換していない。
- pandas 3.0.6 CoW配列のコピー、未収束opt.success=False不採用、改善方向、非連続index、全頭検証、版管理、起動SHA固定、合成E2E、読み戻し回帰テストを追加。
- 失敗診断・途中checkpointは公開結果とは別のActions failure artifactに保存する設計。
- 実データ版は `analysis/trio_v5_feature_ablation/d40e6ab1e8249b462dfa7f5976e63ee9f93d62eb-run41256745528-attempt1` にSHA/実行ID/試行番号付きで保存。元コード、依存ファイル、原本SHA256、開催日bootstrap、各種full-ticketを追跡可能。

成果物: [完全保存ディレクトリ](https://github.com/ryotamatsuki/keiba-place-probability-lab/tree/07c1109339432cfda7c7c8b83e5d8c94d8598d79/analysis/trio_v5_feature_ablation/d40e6ab1e8249b462dfa7f5976e63ee9f93d62eb-run41256745528-attempt1)。
主なファイルは `v5_features_experiment.json`、`2024_v5_features_full_trio.csv.gz`、`2024_v5_features_racewise.csv`、`2024_v5_features_runner_probabilities.csv`、`protected_before.sha256`、`protected_after.sha256`、`execution_identity.txt`、`execution_log.txt`、`dependencies.txt`。

## 再現手順

事前登録コミット、入力データ、正しいcheckout SHAを揃える。workflow [.github/workflows/trio-v5-feature-ablation.yml](https://github.com/ryotamatsuki/keiba-place-probability-lab/blob/d40e6ab1e8249b462dfa7f5976e63ee9f93d62eb/.github/workflows/trio-v5-feature-ablation.yml) が完全再現の入口。手動再現ではまずcommit `d40e6ab1e8249b462dfa7f5976e63ee9f93d62eb` にcheckoutし、Python 3.11環境で依存をインストールする。

```bash
python -m pip install -e '.[dev,historical,modeldev]' -c requirements-scope.txt
python -m pip install 'catboost==1.2.10' 'lightgbm==4.7.0'
ruff check .
pytest -q
```

過去のActions #37104212189 のartifact `stage36-historical-training-panel-v1` を `/tmp/stage36` に展開し、Kaggle JRA V1 `noriyukifurufuru/japan-horse-racing-2010-2025/versions/1` を `/tmp/kaggle_trio_v5` に取得した後:

```bash
RESULT=$(find /tmp/kaggle_trio_v5 -name keiba_results.csv -print -quit)
RACES=$(find /tmp/kaggle_trio_v5 -name keiba_races.csv -print -quit)
python -u scripts/research_trio_v5_feature_ablation.py \
  --historical-dir /tmp/stage36/data/historical_processed/stage36_freeze_v1 \
  --results "$RESULT" --races "$RACES" \
  --matched-oof analysis/trio_v3_matched_oof/2023_2024_matched_800_400_oof.csv \
  --pr40-2023 analysis/trio_v3_matched_oof/2023_training_year_diagnostics.csv \
  --pr40-2024 analysis/trio_v3_matched_oof/2024_comparable_complete_trio_scores.csv \
  --out analysis/trio_v5_feature_ablation/manual-NEW-UNIQUE-RUN
```

保存先は存在しない新ディレクトリを指定し、既存の実験成果物を上書きしない。ActionsはGitHubのSHA・run ID・attemptで固有ディレクトリを構築する。

## 残る制約

独立前向きholdout、同時刻T-10オッズ、実3連複市場価格およびROIは未検証。2024年の評価を新規の独立証拠として扱えず、同年結果を見た探索条件変更は行わない。現行凍結F3と2026年天皇賞の正式予測ロックは保護。
