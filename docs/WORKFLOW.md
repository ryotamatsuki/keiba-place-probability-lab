# Workflow

## Stage 0 — Governance / freeze

- [x] Phase Aの目的変数を P(top3) に固定
- [x] Data Policy作成
- [x] 評価指標固定
- [x] future winner rule v2固定（race-macro Brier + paired date-clustered one-SE incumbent gate）
- [x] 既存OSS候補の棚卸し
- [x] 最初のレース用pre-analysisログ作成
- [x] 長期目標を「順位分布モデル → 馬券種横断確率」に拡張

## Stage 1 — Race snapshot

- [x] 18頭全頭の出走表を同じ基準で記録
- [x] 出典URLとas-of時刻を固定
- [x] 市場情報、枠、斤量、騎手、基本戦績を記録
- [x] 欠損項目を明示
- [x] 権利・再配布ポリシーに従い、生HTML・画像・PAT情報を保存しない

Status: COMPLETE — 2026-10-03 pre-race

## Stage 2 — Market baseline for P(top3)

- [x] 市場情報からベースラインを構築
- [x] 単勝人気を複勝圏確率と混同しない
- [x] 複勝オッズの利用方法と限界を明記
- [x] オッズの時点差を上書きせず、snapshotとして保存
- [x] 市場ベースラインの仮定と不確実性を文書化
- [x] normalized win market + Harville/Plackett-Luceでmarket-only P(top3)を算出
- [x] 複勝オッズはprobabilityではなくdiagnostic strengthとして分離

Status: COMPLETE — 08:35 JST market snapshot

## Stage 3 — Form / suitability features

- [x] 距離
- [x] コース
- [x] 馬場
- [x] 近走
- [x] クラス
- [x] 斤量
- [x] 脚質 / ペース
- [x] market情報から独立した特徴量セットをfreeze
- [x] model-input allowlistを固定
- [x] missingnessを0と混同しない

Status: REVISED — initial snapshot retained, canonical Feature Spec v1 frozen after evidence audit

## Stage 3.5 — Canonical feature materialization

- [x] JRA-VAN operational feature design reviewed
- [x] recent JRA leakage-aware temporal-validation evidence reviewed
- [x] ranking/SHAP literature reviewed
- [x] redundancy and missingness audit run on current 18-runner snapshot
- [x] canonical Feature Spec v1 frozen
- [x] materialize all required v1 fields for 18 runners
- [x] mark any unavailable required field explicitly
- [x] verify zero market fields in canonical matrix
- [x] relative final-3F block explicitly excluded for current v1 rather than raw-time substitution
- [x] canonical matrix validated: 18 rows / 0 missing / 0 market-column matches

Status: COMPLETE

## Stage 3.6 — Historical training panel

- [x] public-data source strategy fixed
- [x] raw-source / redistribution gate documented
- [x] standardized historical row schema fixed
- [x] leakage-safe panel builder implemented
- [x] chronological split fixed: 2010-15 warmup / 2016-22 train / 2023-24 validation / 2025 test
- [x] target-race market fields prohibited at builder boundary
- [x] unit tests added for shift / cumulative-history leakage controls
- [x] local acquisition and build CLIs added
- [x] acquire and verify exact dataset version/license
- [x] standardize the historical source into canonical event rows
- [x] materialize the full historical panel locally
- [x] produce QA counts, missingness and leakage report
- [x] reconcile race counts against JRA official references
- [x] freeze the training-ready panel fingerprint

Status: COMPLETE / QA PASS — 55,268 official JRA races reconciled; 53,220 flat races / 753,387 runner rows frozen

## Stage 4 — P(top3) historical baseline model

- [x] fit transparent historical model(s)
- [x] compare predeclared cohorts on validation only
- [x] keep 2025 out of model/cohort/regularization selection and target-model fitting
- [x] evaluate Brier score / log loss / calibration
- [x] apply selected model to the frozen 2026-10-03 target matrix
- [x] verify market fields are absent from model inputs
- [x] enforce exact target-race sum P(top3)=3
- [x] run leave-one-feature-block-out sensitivity analysis
- [x] freeze NONMARKET_BASELINE.md and nonmarket_baseline.csv
- [x] reproduce on GitHub Actions against the frozen Stage 3.6 artifact

Status: COMPLETE / QA PASS — turf_1200, L2 logistic C=0.1; 2025 Brier 0.150537 / log loss 0.467970 / ECE 0.016938; target 18-runner sum P(top3)=3.0

Timing note: execution occurred after the scheduled target-race start. The Stage 4 output is therefore a blind retrospective reconstruction from the frozen pre-race feature matrix, not a pre-start probability lock.

## Stage 5 — Calibration / ensemble

- [x] compare frozen market-only and non-market-only probabilities on paired historical rows
- [x] reconstruct historical market P(top3) on complete starter fields
- [x] fit transparent market/non-market logit calibration on 2023 only
- [x] select convex blend weight on 2024 only
- [x] keep 2025 out of all Stage 5 selection and evaluate once after freeze
- [x] compare Brier score / log loss for raw, calibrated, and blended variants
- [x] retain the frozen 08:35 target market snapshot; do not substitute final target odds
- [x] enforce exact target-race sum P(top3)=3
- [x] add model-sensitivity uncertainty range
- [x] document historical-final-odds vs 08:35 target timing mismatch
- [x] freeze STAGE5_ENSEMBLE.md / stage5_ensemble.csv / metrics
- [x] reproduce in GitHub Actions

Status: COMPLETE / QA PASS — selected market/non-market weights 0.95 / 0.05; 2025 paired test Brier 0.139662 / log loss 0.434544; target 18-runner sum P(top3)=3.0

Interpretation: the non-market component adds only a small incremental improvement; Stage 5 remains overwhelmingly market-led. Historical market odds are effectively final odds, while the target market input is the frozen 08:35 snapshot, so historical performance is not perfectly time-matched.

Timing note: Stage 5 was executed after the target race's scheduled start. It is a blind retrospective reconstruction, not a pre-start lock.

## Stage 6 — Pre-race lock

2026-10-03 Kyoto 11R status: **RETROSPECTIVE_DRY_RUN COMPLETE / operational QA PASS**.

- [x] `probability_estimates.csv`を18頭分コミット
- [x] decision log / lock manifest / SHA256 fingerprintsを記録
- [x] 08:35 market snapshotとStage 5確定commitだけをlock sourceに使用
- [x] result / later-final odds / payout / popularityをlock処理から遮断
- [x] rehearsal lock以後は当該確率値を変更しない
- [ ] scheduled start前にlock commitを作る本来のtiming gate — **this raceでは未達**

Locked source commit: `086f7a2a20e20dc226db6535b4738905e2011f1b`  
Lock mode: `RETROSPECTIVE_DRY_RUN`  
Locked probability file SHA256: `eefa16ad709c7568690e3f75a0df22491a31b3cb6aebdeebcf27fab274bc9e66`

Interpretation: the operational procedure is verified, but this must not be described as a
genuine pre-race lock. For the next live target, execute the identical procedure before start.

## Stage 7 — Post-race evaluation

2026-10-03 Kyoto 11R status: **RETROSPECTIVE PIPELINE TEST COMPLETE / QA PASS**.

- [x] JRA公式着順を予測ファイルとは別のoutcomeデータとして追加
- [x] immutable Stage 6 lock SHA256を検証してから突合
- [x] 18頭すべてにtop3 labelを付与
- [x] Stage 2 raw market / Stage 4 non-market / Stage 5 ensemble / Stage 6 lockを同一18頭で評価
- [x] Brier / log loss / actual-top3 probability massを記録
- [x] 実際のtop3が予測順位の何位だったかを記録
- [x] uniform 3/18 baselineと比較
- [x] 結果論の因果説明を避け、事前のmarket/non-market乖離と確率誤差だけで失敗分析
- [x] official outcome / metrics / evaluation report / manifest / code / testsをfreeze
- [x] GitHub Actionsで再現

Official top 3: 6 リリージョワ / 18 ディアナザール / 8 レッドエヴァンス.

One-race metrics:
- Stage 2 market 08:35: Brier `0.107207`, log loss `0.335163`, actual-top3 mass `0.898750`
- Stage 5/6: Brier `0.111097`, log loss `0.350378`, actual-top3 mass `0.829376`
- Stage 4: Brier `0.113597`, log loss `0.355176`, actual-top3 mass `0.823468`
- uniform: Brier `0.138889`, log loss `0.450561`

Conclusion: all actual top-3 runners were inside the Stage 5/6 forecast top six, but the
95/5 ensemble did not beat the frozen 08:35 raw market in this single race. This does not
reverse the 2025 aggregate test, and the 2025 aggregate result does not imply improvement here.

Timing caveat: this remains retrospective because Stage 6 was not committed before scheduled start.

## Stage 8 — Accumulation / walk-forward

- [x] future model-selection protocol v2をfreeze
- [x] winner metricをrace-macro Brierに固定
- [x] log lossをsecondary diagnostic / tie-breakerに固定
- [x] ROC-AUC / ECE / hit rate / ROIをwinner criterionから除外
- [x] market-onlyをensemble selectionのmandatory default/referenceに固定
- [x] 2025をv2のuntouched testとして再利用しないことを固定
- [x] Stage 4 successor Phase 1をpre-register
- [x] incumbent Logisticを2023/2024 outer walk-forwardで再構築・診断
- [x] Phase 1 common OOF: 495 races / 6,313 rows; race-macro Brier 0.153096
- [x] field-size baseline比 Brier skill 10.68%; 2023/2024 performance stabilityを確認
- [x] Phase 1 subgroup diagnosticsをfreeze（Class3 / career 11-20 / Kokura等は低skill仮説）
- [x] Phase 2 challenger registry / hyperparameter gridsを最終freeze
- [x] Phase 2 inner chronological tuning + outer 2023/2024 candidate predictionsを生成
- [x] Phase 2 model-family winner: XGBoost (XGB01 in both outer folds)
- [x] XGBoost race-macro Brier 0.151494 vs incumbent 0.153096; delta -0.001602
- [x] incumbentはpaired date-clustered 1-SE gate外（delta / SE = 2.39）となり、Phase 2ではXGBoostへ交代
- [x] Phase 2 evaluation fingerprintがPhase 1と完全一致（495 races / 6,313 rows）
- [x] Phase 3A relative-ability blockをpre-registerして個別比較
- [x] Phase 3A winner: XGB01 + full-field relative ability
- [x] race-macro Brier 0.148959 vs Phase 2 XGB01 0.151494; delta -0.002535
- [x] Phase 2 incumbentはpaired date-clustered 1-SE gate外（delta / SE = 2.87）
- [x] Phase 3Aでは全スターター文脈を使うleave-one-out相対特徴5本を採用
- [x] Phase 3B recent-trend blockをpre-registerしてPhase 3A incumbentへ追加比較
- [x] recent-trend追加版はpoint-Brier-best（0.148674）だがPhase 3A incumbent 0.148959との差は -0.000285
- [x] paired date-clustered SE 0.000322の範囲内のためPhase 3A incumbentを維持（trend blockは不採用）
- [x] 2023/2024とも点推定は同方向に改善したが、固定1-SE gateを満たさないことを記録
- [x] Phase 3C interaction blockをpre-registerしてPhase 3A incumbentへ追加比較
- [x] interaction追加版はpoint-Brier-best（0.148768）だがPhase 3A incumbent 0.148959との差は -0.000191
- [x] paired date-clustered SE 0.000216の範囲内のためPhase 3A incumbentを維持（interaction blockは不採用）
- [x] Phase 3 development COMPLETE — final Stage 4 successor = XGB01 + full-field relative ability
- [x] Stage 4 successor production contract v2をfreeze
- [x] frozen successorを2016-2025の33,069 eligible rows / 2,565 racesでproduction refit
- [x] target scoringはraw marginal P(top3)をcanonical出力とし、sum-to-three補正を廃止
- [x] relative abilityはlock時点のactive全スターターで計算
- [x] scratch policyをfreeze（horse_no/declared_field_size維持、active field_size/peer集合を更新）
- [x] 2026-10-03 frozen targetでproduction rehearsal PASS（18/18 eligible、relative features 100% coverage）
- [x] Stage 5 v2: out-of-time successor predictions + market-only incumbent + race-macro Brier / paired 1-SE
- [x] 2023でlogit calibration familyをfitし、2024 market gateでcalibrated marketをhistorical final-odds baseとして採用
- [x] 2024 blend point-bestは5% non-marketだが改善 0.0000485 < paired clustered SE 0.0001113
- [x] market-onlyを維持し、Stage 5 v2 historical-domain selected non-market weight = 0.00
- [x] 2025は2016-2024 fitによるOOFを再生成し、KNOWN_OUTCOME_AUDIT_NOT_TESTとして分離
- [x] live morning canonical policy = raw market-only; Stage 4 v2 / historical-domain blendはshadow
- [x] time-matched market snapshotが得られるまではmorning-odds blendをhistorical final oddsで再最適化しない
- [ ] 複数liveレースを発走前lockで蓄積
- [ ] live walk-forward評価
- [x] Scope Expansion Stage 1 audit: turf 1000-2600 primary 239,015 rows / 21,586 races; 1200m evaluation 6,313 rows / 495 races reproduced exactly
- [x] Scope Expansion Stage 1 A/B/C comparison frozen before scoring
- [x] A current 1200 Brier 0.148959 / B 1000-1400 Brier 0.148774 / C 1000-2600 Brier 0.149394
- [x] B is point-best but A-B improvement 0.000184 < paired clustered SE 0.000325; current 1200 successor retained
- [x] C worsens 1200m primary score; no cross-distance production authorization from Stage 1
- [ ] Scope Expansion Stage 2: preregister distance-suitability feature blocks
- [ ] calibrationをサンプル外で検証
- [ ] market-only / model-only / blendedをpaired proper scoresで比較

## Stage 9 — Rank-distribution prototype

十分なデータが蓄積してから着手。

- [ ] P(rank_i = r) を整合的に推定
- [ ] レース内の順位制約を満たすモデルを比較
- [ ] Plackett-Luce / latent-performance simulation / race-level modelを候補比較
- [ ] top1 / top3周辺確率がPhase Aモデルと整合するか検証

## Stage 10 — Cross-bet probability engine

順位分布から馬券種別確率を派生。

- [ ] 単勝
- [ ] 複勝
- [ ] ワイド
- [ ] 馬連
- [ ] 3連複
- [ ] 3連単
- [ ] joint probabilityの校正・妥当性を検証

馬券種別の独立した「当て物モデル」を乱立させず、共通の順位分布を中核にします。
