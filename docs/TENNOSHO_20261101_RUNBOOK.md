# 天皇賞（秋）2026 — 実オッズ前向き検証・運用手順

対象：2026-11-01 東京11R 天皇賞（秋）（`race_id=202605041111`、Yahoo `2605041111`）。**発走15:40 JST、17頭**。研究仕様は `docs/TENNOSHO_20261101_PROSPECTIVE_SPEC.md`、開始コミット `8e0cf2013a90af410c388d0859f91bd41f99e044`。研究起点PR #36 head `1c6033e3e1d18f0b1c98ae64ce2462a25dac04ad`。本番予測モデルは依然としてmarket-only。

## 現在の実装境界

- F3向け研究予測の候補：市場のみM0_live（17頭の最新同時刻単勝オッズの正規化＋Harville）、market logit offsetで学習したXGB400、XGB800、CAT400、事前凍結logit残差75:25混合。2024年の保存済みOOFは2026年出走馬の予測に転用禁止。
- `scripts/freeze_tennosho_f3_20261101.py` は2018〜2024年の歴史ラベルだけでF3学習器と2016〜2024年の過去最終オッズ校正器をfitし、`models/tennosho_f3_20261101/{f3_bundle.joblib,manifest.json}` にSHA256付き保存。**新たな2026年前向き予測用bundle**であり、PR #36の2024評価済みtrained-on-through-2023 weightそのものではない。2016〜24 final oddsで学習した校正器を実際のT−10市場に適用するのは分布外利用であり、将来実証までは仮説の域を出ない。
- `src/keiba_place_lab/tennosho_forward.py` の確率推論には上記のbundleに加え、**予測日の前日以前に確定している全17頭のJRA出走履歴**と同じF3 source ruleの出走時点特徴量を保証する必要がある。既存 `data/live_history/2026/current.json` の確定日が2026-10-03のままなら、10月4〜31日の漏れがないことは主張できない。実際の各馬最新出走を照合し、F3のElo / 3F / 前走通過順位 / 馬体重 / 騎手・厩舎歴 / レース内全頭relative contextまで再構成してからフラグを立てる。現状その17頭の監査済み `prior_f3_features.csv` はリポジトリ内に**ない**。名前や馬番を埋めるだけの合成F3は禁止。
- 市場だけであれば、対象17頭の同時刻単勝オッズ、出典提供者更新時刻、当日の出走表を厳密に確保したときに限り「発走前市場観測」が可能。これは **B_rawとXGB800の比較完了ではない**。

## 今日10/31の事前準備

```bash
git fetch origin
git switch research/tennosho-akiten-preoff-20261101
python -m pip install -e '.[dev,historical,modeldev]' -c requirements-scope.txt
python -m pip install 'catboost==1.2.10' 'lightgbm==4.7.0'
ruff check .
pytest -q tests/test_tennosho_preoff_20261101.py
```

事前学習を再現する場合は、過去のStage3.6の合法的な2010〜2024年度パーティション3ファイルと、SHAが確定したKaggle V1の2ファイルが必要。GitHub Actions `.github/workflows/tennosho-preoff-freeze-v1.yml` は既存の原本をrunnerの /tmp へ復元し、研究branchへ **モデル本体＋SHA manifestのみ** をコミットする。再学習してSHAをすり替えず、ロック日の前にbundleを確定する。

```bash
python scripts/freeze_tennosho_f3_20261101.py \
  --historical-dir /path/to/stage36_freeze_v1 \
  --results-csv /path/to/kaggle_v1/keiba_results.csv \
  --races-csv /path/to/kaggle_v1/keiba_races.csv \
  --out models/tennosho_f3_20261101
```

無料公開サイトの現在の運用応答だけを確かめるpreflight。**このコマンドは今日のオッズを取得することがあっても、T−10ロックでも将来予測でもない**。

```bash
python scripts/tennosho_preoff_capture.py preflight \
 --config config/tennosho_preoff_20261101.json \
 --output /tmp/tennosho_20261101_preflight_YYYYMMDDTHHMM
```

HTTP障害、YahooのDOM、`time.hr-horseResult__update[datetime]` の明示 timezone、JRA17頭に相当する馬番・horse_idと完全な単勝オッズが一致することを確認する。JRA-VAN認証を無断回避しない。

## 明日11/1のT−10（発走15:40）

1. JST **15:29** 頃に実行ホスト・ネットワーク・出走取消・コード・履歴とF3 SHAを再確認する。GitHub Actionsのスケジュール/queueは実行分を保証しないので、`date` 等の実際の時計で確認。15:30:00前の朝データをT−10と記録しない。
2. **15:30:00〜15:31:00の間だけ** `capture`。実オッズはページが表示する更新日時を用い、`provider_updated_at <= captured_at <= locked_at < off_at`、`locked_at - provider_updated_at <= 5min`、取得からlockまで60秒以内、全17頭が完全一致。相違・欠測・日付が欠損し時間帯の解釈も未監査・接続失敗の場合は即座に失敗記録。
3. 17頭全員の過去履歴を監査しF3計算済みでない限り、**市場だけの観測にする**（下記最小コマンド）。歴史が事前に監査されていてモデルSHA一致の場合に限り、フルモデルコマンドを使い、候補を実行後に変更しない。

市場観測だけの最小手順（前向きB_raw検証の得点には入らない）：

```bash
python scripts/tennosho_preoff_capture.py capture \
  --config config/tennosho_preoff_20261101.json \
  --output analysis/tennosho_preoff_20261101/attempts/YYYYMMDDTHHMMSS
```

**全条件を満たす場合に限る**全候補予測（`prior_f3_features.csv` とmanifestは監査済みの外部入力、推測で作成禁止）：

```bash
python scripts/tennosho_preoff_capture.py capture \
  --config config/tennosho_preoff_20261101.json \
  --bundle models/tennosho_f3_20261101/f3_bundle.joblib \
  --bundle-manifest models/tennosho_f3_20261101/manifest.json \
  --prior-features /verified/prospective/prior_f3_features.csv \
  --prior-manifest /verified/prospective/prior_f3_features.manifest.json \
  --output analysis/tennosho_preoff_20261101/attempts/YYYYMMDDTHHMMSS \
  --formal-out prospective/tennosho_20261101/locks/202605041111
```

4. 作成直後に（15:40の発走**前に**）、`prospective/tennosho_20261101/locks/202605041111/{predictions.csv,manifest.json}` を**main**へ公開する。PR上のコミットや手元の作成時刻だけでは、mainへの発走前公開が確実とは認めない。GitHub `push`イベントを引き起こしたActionsの**サーバー作成日時が発走より前**、かつそのrun `head_sha` で公開済み2ファイルのSHA/bytesが一致することが必要。後日の確認は許されるが、**公開自身は発走前**。
5. 結果確定後に別outcome台帳へJRA公式着順と取消を追記し、ロックされた確率やモデルは一切変更しない。1,000完全レース、60異なる開催日、90経過日までは候補採択・再学習・中間Brier選択をしない。

## FAIL_CLOSED（本日の後知恵防止）

- 15:31:00 JSTを超えて計算／オッズ取得完了したら、T−10正式ロックを作らない。
- providerが「15:24」等の単なる時計だけで完全な年月日を明示しない、更新時刻が5分より古い、発走の時計変更が起きた場合、データをT−10として使わない。
- 出走取消などにより17頭の事前固定rosterと一致しない場合、**別の出走馬集合としての新規固定と完全監査まで**正式候補ロックしない。
- 市場P(top3)は全出走馬でHarvilleの合計が3となるが、XGBの個別3着内補正確率を恣意的に合計3へ再正規化しない。
- 限定的な朝のpreflightが成功しても明日の実際の15:30にアクセス可能とは限らない。逆も同様。
- ユーザーに報告する際は、(1) 事前学習bundleの保存、(2) 実市場のT−10観測、(3) モデル比較の正式ロック、(4) mainへの発走前公開証明の4点を**それぞれ別の真偽値**で報告する。全て完了していない時は完了を装わない。

## 成果物

- `docs/TENNOSHO_20261101_PROSPECTIVE_SPEC.md` 事前仕様
- `config/tennosho_preoff_20261101.json` race/候補/時刻
- `scripts/freeze_tennosho_f3_20261101.py` 固定モデル生成（過去だけ）
- `src/keiba_place_lab/tennosho_forward.py` strict same-market inference
- `scripts/tennosho_preoff_capture.py` preflight/capture
- `tests/test_tennosho_preoff_20261101.py` 未来ラベル/時刻/行キー対策
- `.github/workflows/tennosho-preoff-freeze-v1.yml` 再現可能なCPUモデル凍結とアクセス確認


### 2026-10-31 午前の公開データpreflight所見

GitHub Actions 実行 `41149387677` では公開Yahoo出馬表とオッズの取得、全17頭への参加馬の照合、provider ISO日付時刻までは到達したが、`datetime` 属性に `+09:00` がないため最初のソース形式検査で停止した。運用上、日本国内のJRA情報サイトの**完全なISO年月日時刻**についてだけ、元の属性リテラルを記録し `Asia/Tokyo` と解釈する。sourceがUTCオフセットを明記しているかどうかはmanifestに別フィールドで残す。年／月／日が欠ける表示は引き続き不合格。5分の鮮度・取得時刻・実時計・ロックの条件は緩めない。preflightを再実行してこのソース表記に適合するか確認する。
