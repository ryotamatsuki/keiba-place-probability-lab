# Pre-analysis log — 2026-10-03 Kyoto 11R

## Status

Stage 1 snapshot completed before the 15:30 start.

- static-data access time: 2026-10-03T08:47:27+09:00
- market snapshot time: 2026-10-03T08:35:00+09:00

## Pre-existing information / potential anchoring

会話上、簡易な候補として **10番 ヒシアイラ** が一度挙げられています。
これは体系的な18頭比較の結論ではなく、Stage 1以降のランキングへ固定しません。

この情報を残す目的は、後から「最初からその馬を評価していた」と書き換えることを防ぐためです。

## Ticket status

このログ更新時点で、このリポジトリ上では購入確定を記録していません。

## Stage 1 controls

- 18頭全頭を同一の列定義で記録
- 市場オッズは08:35スナップショットとして固定
- 公開ページ自体のHTML・画像・PAT情報は保存しない
- 条件別出走歴なしは0%ではなく欠損扱い
- JBIS speed indexは出典付き第三者特徴量として明示
- 最終的な複勝確率はまだ作成しない

## Analysis rule

- 全18頭を同一の特徴量定義で処理する
- 事前候補に有利な変数選択をしない
- 出典と取得時刻を記録する
- 最終予測は発走前にロックする
