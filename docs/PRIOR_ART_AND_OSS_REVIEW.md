# Prior Art / OSS Review

Stage 0では、方法論を学ぶための候補リポジトリを列挙します。
**コードを再利用する前に各リポジトリのLICENSEを個別確認します。LICENSE不明の場合は実装をコピーしません。**

| Project | URL | Stage 0 relevance |
|---|---|---|
| asayao-fp/keiba | https://github.com/asayao-fp/keiba | 複勝圏確率を直接扱う国内例 |
| leo-miura-robot/keiba_prediction_ai | https://github.com/leo-miura-robot/keiba_prediction_ai | 時系列検証・単勝/複勝の分離 |
| umazkym/KeibaAI_v2 | https://github.com/umazkym/KeibaAI_v2 | 不確実性とシミュレーションの参考候補 |
| kosei-matsuzaki/keiba-ai | https://github.com/kosei-matsuzaki/keiba-ai | レース内の相対関係を扱うモデル候補 |
| analeonescu/horse-racing-bets-and-evals | https://github.com/analeonescu/horse-racing-bets-and-evals | 市場ベースライン・確率評価 |
| Xand4r/Horse_Racing_Modelling | https://github.com/Xand4r/Horse_Racing_Modelling | レース単位モデリング・校正 |
| catowabisabi/horse-racing-model-training | https://github.com/catowabisabi/horse-racing-model-training | 市場とのブレンド |
| davidklan-png/keibamon | https://github.com/davidklan-png/keibamon | as-of時点管理・データ基盤の考え方 |
| hturner/PlackettLuce | https://github.com/hturner/PlackettLuce | 順位モデルの理論・実装参照候補 |

## Review rule

各候補について後続Stageで以下を確認します。

1. LICENSE
2. 対象国・競馬制度
3. 目的変数
4. 特徴量
5. train/test分割
6. leakage対策
7. calibration
8. market oddsの扱い
9. 公開データの権利処理
10. 本プロジェクトへ採用する考え方 / 採用しない考え方

この文書は現時点では**候補リスト**であり、各リポジトリの品質認証ではありません。
