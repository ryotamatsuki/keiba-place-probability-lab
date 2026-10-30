# B_rawの400本上限・公平な3学習器比較：実データ研究報告

## 最終判断

**分類：A。** 時系列で選択した学習回数によりB_rawを上回り、開催日paired one-SEを通過。

**採用判断：研究上の勝者は XC_75_25_logit。** 2024年のBrierは 0.135621648、既存B_rawは 0.135721276。
B_rawとの差は 0.000099627、開催日クラスタpaired SEは 0.000047572。 事前のone-SEゲートを通過した。
**単体XGBoostの学習回数変更：** 800本の2024年Brier 0.135626758、B_raw比改善 0.000094518、paired SE 0.000059490。
アンサンブルの補完効果と単体800本への単純な補正強度変更の効果を混同せず、同じ2024年レースで別々に比較した。

評価は既知の2023・2024年であり、未見の将来テストではない。過去の最終単勝オッズを発走10分前のオッズと扱わない。

## 実験条件と研究の系譜

PR #35のコミット00fd15638548285a8c8e4e27f4aee0860cb24750を起点に、PR #35→#34→#33→#32→#31へ依存。モデル選択規約V2を維持した。
登録仕様コミット183717e5a875b8e9a00508989cfaed57a2320e2dは新スコア算出前に作成。後から学習率や正則化を変更していない。

F3の既存市場・能力特徴量と完全な出走頭数を固定。XGBoost、LightGBM、CatBoostの元パラメータを変えず、木数60、120、240、400、800、1600だけを比較。学習器ごとの上限は100分、CPU最大2スレッド、年別成果物を随時保存。

外側2023の学習2018〜2022、選択／補正係数の学習には内側2021〜2022のみ。外側2024の学習2018〜2023、選択／補正係数には内側2021〜2023のみ。年ごとに厳密に過去だけで学習し、Brierはレースmacro平均で選択。

## 各モデルの内側学習曲線（全6チェックポイント）

| family   |   year |   rounds |   races |   race_macro_brier |   race_macro_log_loss |
|:---------|-------:|---------:|--------:|-------------------:|----------------------:|
| xgb      |   2021 |       60 |    2893 |           0.136677 |              0.423625 |
| xgb      |   2021 |      120 |    2893 |           0.13657  |              0.423366 |
| xgb      |   2021 |      240 |    2893 |           0.136497 |              0.423193 |
| xgb      |   2021 |      400 |    2893 |           0.136501 |              0.423193 |
| xgb      |   2021 |      800 |    2893 |           0.13655  |              0.423339 |
| xgb      |   2021 |     1600 |    2893 |           0.136727 |              0.42385  |
| xgb      |   2022 |       60 |    2889 |           0.135579 |              0.420399 |
| xgb      |   2022 |      120 |    2889 |           0.135535 |              0.420231 |
| xgb      |   2022 |      240 |    2889 |           0.135505 |              0.420112 |
| xgb      |   2022 |      400 |    2889 |           0.135485 |              0.420049 |
| xgb      |   2022 |      800 |    2889 |           0.135476 |              0.42004  |
| xgb      |   2022 |     1600 |    2889 |           0.135605 |              0.420483 |
| xgb      |   2023 |       60 |    2887 |           0.137726 |              0.425372 |
| xgb      |   2023 |      120 |    2887 |           0.137579 |              0.424966 |
| xgb      |   2023 |      240 |    2887 |           0.137434 |              0.424589 |
| xgb      |   2023 |      400 |    2887 |           0.137336 |              0.424368 |
| xgb      |   2023 |      800 |    2887 |           0.13724  |              0.424191 |
| xgb      |   2023 |     1600 |    2887 |           0.137264 |              0.424298 |
| lgb      |   2021 |       60 |    2893 |           0.136591 |              0.42338  |
| lgb      |   2021 |      120 |    2893 |           0.136522 |              0.423264 |
| lgb      |   2021 |      240 |    2893 |           0.136596 |              0.423452 |
| lgb      |   2021 |      400 |    2893 |           0.136645 |              0.42358  |
| lgb      |   2021 |      800 |    2893 |           0.136903 |              0.424284 |
| lgb      |   2021 |     1600 |    2893 |           0.13753  |              0.426032 |
| lgb      |   2022 |       60 |    2889 |           0.135525 |              0.420211 |
| lgb      |   2022 |      120 |    2889 |           0.135507 |              0.420119 |
| lgb      |   2022 |      240 |    2889 |           0.135503 |              0.420064 |
| lgb      |   2022 |      400 |    2889 |           0.135544 |              0.420264 |
| lgb      |   2022 |      800 |    2889 |           0.135765 |              0.420952 |
| lgb      |   2022 |     1600 |    2889 |           0.136396 |              0.422866 |
| lgb      |   2023 |       60 |    2887 |           0.137523 |              0.424847 |
| lgb      |   2023 |      120 |    2887 |           0.137352 |              0.424441 |
| lgb      |   2023 |      240 |    2887 |           0.137299 |              0.424378 |
| lgb      |   2023 |      400 |    2887 |           0.137287 |              0.424383 |
| lgb      |   2023 |      800 |    2887 |           0.137341 |              0.42466  |
| lgb      |   2023 |     1600 |    2887 |           0.13767  |              0.425767 |
| cat      |   2021 |       60 |    2893 |           0.136635 |              0.423748 |
| cat      |   2021 |      120 |    2893 |           0.136563 |              0.423497 |
| cat      |   2021 |      240 |    2893 |           0.136515 |              0.423307 |
| cat      |   2021 |      400 |    2893 |           0.136545 |              0.423307 |
| cat      |   2021 |      800 |    2893 |           0.13674  |              0.423749 |
| cat      |   2021 |     1600 |    2893 |           0.136996 |              0.424365 |
| cat      |   2022 |       60 |    2889 |           0.135563 |              0.420488 |
| cat      |   2022 |      120 |    2889 |           0.135536 |              0.420339 |
| cat      |   2022 |      240 |    2889 |           0.135515 |              0.420182 |
| cat      |   2022 |      400 |    2889 |           0.135533 |              0.42018  |
| cat      |   2022 |      800 |    2889 |           0.135628 |              0.420426 |
| cat      |   2022 |     1600 |    2889 |           0.135759 |              0.420811 |
| cat      |   2023 |       60 |    2887 |           0.137624 |              0.425257 |
| cat      |   2023 |      120 |    2887 |           0.137459 |              0.424802 |
| cat      |   2023 |      240 |    2887 |           0.137326 |              0.424446 |
| cat      |   2023 |      400 |    2887 |           0.137237 |              0.424225 |
| cat      |   2023 |      800 |    2887 |           0.137252 |              0.424237 |
| cat      |   2023 |     1600 |    2887 |           0.137299 |              0.424355 |

## 外側年に選んだ木本数、補正係数と学習時間

| family   |   outer_year |   selected_rounds |    alpha | alpha_fit_years   | train_max_date   | eval_min_date   |   selected_model_fit_seconds |   baseline_fit_seconds |   original_fixed_parity_max_error | upper_bound_selected   |
|:---------|-------------:|------------------:|---------:|:------------------|:-----------------|:----------------|-----------------------------:|-----------------------:|----------------------------------:|:-----------------------|
| xgb      |         2023 |               400 | 0.841689 | 2021,2022         | 2022-12-28       | 2023-01-05      |                      6.59916 |                0       |                             3e-08 | False                  |
| xgb      |         2024 |               800 | 0.814247 | 2021,2022,2023    | 2023-12-28       | 2024-01-06      |                     13.7571  |                7.90048 |                             3e-08 | False                  |
| lgb      |         2023 |               120 | 0.934224 | 2021,2022         | 2022-12-28       | 2023-01-05      |                      4.11724 |                0       |                             0     | False                  |
| lgb      |         2024 |               120 | 1.06731  | 2021,2022,2023    | 2023-12-28       | 2024-01-06      |                      2.9487  |                0       |                             0     | False                  |
| cat      |         2023 |               240 | 0.918399 | 2021,2022         | 2022-12-28       | 2023-01-05      |                     15.4996  |                5.66471 |                             0     | False                  |
| cat      |         2024 |               400 | 0.927283 | 2021,2022,2023    | 2023-12-28       | 2024-01-06      |                     30.2078  |                6.67225 |                             0     | False                  |

## 2024年同一レース・同一頭のBrierとlog loss

| name                |   race_macro_brier |   race_macro_log_loss |   races |   rows | point_brier_best   | within_one_se_brier   |
|:--------------------|-------------------:|----------------------:|--------:|-------:|:-------------------|:----------------------|
| XC_75_25_logit      |           0.135622 |              0.420005 |    2844 |  31225 | True               | True                  |
| XLC_ridge_2023      |           0.135626 |              0.420096 |    2844 |  31225 | False              | True                  |
| xgb_selected_raw    |           0.135627 |              0.420034 |    2844 |  31225 | False              | True                  |
| XL_75_25_logit      |           0.135645 |              0.420062 |    2844 |  31225 | False              | False                 |
| xgb_selected_scaled |           0.135669 |              0.420102 |    2844 |  31225 | False              | False                 |
| cat_selected_raw    |           0.135674 |              0.420202 |    2844 |  31225 | False              | True                  |
| cat_selected_scaled |           0.135694 |              0.420253 |    2844 |  31225 | False              | False                 |
| B_raw               |           0.135721 |              0.420229 |    2844 |  31225 | False              | False                 |
| lgb_selected_scaled |           0.135788 |              0.420421 |    2844 |  31225 | False              | False                 |
| LightGBM120         |           0.135804 |              0.420457 |    2844 |  31225 | False              | False                 |
| lgb_selected_raw    |           0.135804 |              0.420457 |    2844 |  31225 | False              | False                 |
| E5                  |           0.135839 |              0.420633 |    2844 |  31225 | False              | False                 |
| CatBoost120         |           0.135867 |              0.420694 |    2844 |  31225 | False              | False                 |
| X2                  |           0.135939 |              0.420907 |    2844 |  31225 | False              | False                 |
| M0                  |           0.136377 |              0.422136 |    2844 |  31225 | False              | False                 |

## B_raw対候補のpaired差と開催日クラスタSE

| candidate           |   B_raw_minus_candidate_brier |   date_cluster_SE_brier |   B_raw_minus_candidate_log_loss | beats_B_raw_pairwise_one_se   |
|:--------------------|------------------------------:|------------------------:|---------------------------------:|:------------------------------|
| M0                  |                  -0.000655243 |              0.00015107 |                     -0.00190702  | False                         |
| X2                  |                  -0.000218125 |              8.4878e-05 |                     -0.000677855 | False                         |
| E5                  |                  -0.000117378 |              6.4555e-05 |                     -0.000403545 | False                         |
| B_raw               |                   0           |              0          |                      0           | False                         |
| LightGBM120         |                  -8.2958e-05  |              4.1379e-05 |                     -0.000228147 | False                         |
| CatBoost120         |                  -0.000145824 |              6.4208e-05 |                     -0.000464301 | False                         |
| xgb_selected_raw    |                   9.4518e-05  |              5.949e-05  |                      0.000194919 | True                          |
| xgb_selected_scaled |                   5.1941e-05  |              3.4286e-05 |                      0.000127713 | True                          |
| lgb_selected_raw    |                  -8.2958e-05  |              4.1379e-05 |                     -0.000228147 | False                         |
| lgb_selected_scaled |                  -6.661e-05   |              3.7722e-05 |                     -0.00019203  | False                         |
| cat_selected_raw    |                   4.6902e-05  |              5.5858e-05 |                      2.7526e-05  | False                         |
| cat_selected_scaled |                   2.7431e-05  |              5.2339e-05 |                     -2.402e-05   | False                         |
| XL_75_25_logit      |                   7.621e-05   |              3.9817e-05 |                      0.000167588 | True                          |
| XC_75_25_logit      |                   9.9627e-05  |              4.7572e-05 |                      0.00022436  | True                          |
| XLC_ridge_2023      |                   9.5746e-05  |              9.4315e-05 |                      0.000133411 | True                          |

## 校正切片と傾き

| candidate           |   intercept |    slope |
|:--------------------|------------:|---------:|
| M0                  | -0.0168476  | 0.982911 |
| X2                  | -0.0127539  | 0.996653 |
| E5                  | -0.0140148  | 1.00073  |
| B_raw               | -0.0117863  | 0.99244  |
| LightGBM120         | -0.0113532  | 0.99391  |
| CatBoost120         | -0.00580578 | 0.999499 |
| xgb_selected_raw    | -0.0134964  | 0.988012 |
| xgb_selected_scaled | -0.0114058  | 0.988927 |
| lgb_selected_raw    | -0.0113532  | 0.99391  |
| lgb_selected_scaled | -0.0115742  | 0.994274 |
| cat_selected_raw    | -0.00527883 | 0.997773 |
| cat_selected_scaled | -0.00527372 | 0.99739  |
| XL_75_25_logit      | -0.0122031  | 0.990072 |
| XC_75_25_logit      | -0.0109803  | 0.990659 |
| XLC_ridge_2023      | -0.0161433  | 0.987659 |

## 芝・ダート、距離、レースクラス別診断

| candidate           | group         | value     |   race_macro_brier |   race_macro_log_loss |   races |   rows |
|:--------------------|:--------------|:----------|-------------------:|----------------------:|--------:|-------:|
| M0                  | surface       | dirt      |           0.135462 |              0.419516 |    1539 |  16992 |
| M0                  | surface       | turf      |           0.137455 |              0.425227 |    1305 |  14233 |
| M0                  | distance_band | 1000_1400 |           0.134259 |              0.417103 |    1042 |  12274 |
| M0                  | distance_band | 1401_2000 |           0.138136 |              0.426813 |    1570 |  16252 |
| M0                  | distance_band | 2001_2600 |           0.133981 |              0.413091 |     232 |   2699 |
| M0                  | race_class    | Class1    |           0.139475 |              0.431642 |     895 |  11147 |
| M0                  | race_class    | Class2    |           0.14077  |              0.435445 |     449 |   5937 |
| M0                  | race_class    | Class3    |           0.13563  |              0.424311 |     206 |   2924 |
| M0                  | race_class    | Maiden    |           0.131894 |              0.407506 |    1050 |   7966 |
| M0                  | race_class    | Open      |           0.136845 |              0.423904 |     244 |   3251 |
| X2                  | surface       | dirt      |           0.134973 |              0.417996 |    1539 |  16992 |
| X2                  | surface       | turf      |           0.137079 |              0.424341 |    1305 |  14233 |
| X2                  | distance_band | 1000_1400 |           0.13373  |              0.415515 |    1042 |  12274 |
| X2                  | distance_band | 1401_2000 |           0.137759 |              0.425825 |    1570 |  16252 |
| X2                  | distance_band | 2001_2600 |           0.13355  |              0.411841 |     232 |   2699 |
| X2                  | race_class    | Class1    |           0.139069 |              0.430608 |     895 |  11147 |
| X2                  | race_class    | Class2    |           0.140223 |              0.433866 |     449 |   5937 |
| X2                  | race_class    | Class3    |           0.135435 |              0.423918 |     206 |   2924 |
| X2                  | race_class    | Maiden    |           0.131387 |              0.406038 |    1050 |   7966 |
| X2                  | race_class    | Open      |           0.136592 |              0.42292  |     244 |   3251 |
| E5                  | surface       | dirt      |           0.13485  |              0.417592 |    1539 |  16992 |
| E5                  | surface       | turf      |           0.137005 |              0.424218 |    1305 |  14233 |
| E5                  | distance_band | 1000_1400 |           0.133588 |              0.415076 |    1042 |  12274 |
| E5                  | distance_band | 1401_2000 |           0.137686 |              0.425667 |    1570 |  16252 |
| E5                  | distance_band | 2001_2600 |           0.133445 |              0.411525 |     232 |   2699 |
| E5                  | race_class    | Class1    |           0.138986 |              0.43043  |     895 |  11147 |
| E5                  | race_class    | Class2    |           0.14006  |              0.433382 |     449 |   5937 |
| E5                  | race_class    | Class3    |           0.135443 |              0.424008 |     206 |   2924 |
| E5                  | race_class    | Maiden    |           0.131256 |              0.405677 |    1050 |   7966 |
| E5                  | race_class    | Open      |           0.136582 |              0.422745 |     244 |   3251 |
| B_raw               | surface       | dirt      |           0.134707 |              0.41713  |    1539 |  16992 |
| B_raw               | surface       | turf      |           0.136917 |              0.423884 |    1305 |  14233 |
| B_raw               | distance_band | 1000_1400 |           0.133358 |              0.414446 |    1042 |  12274 |
| B_raw               | distance_band | 1401_2000 |           0.137623 |              0.425419 |    1570 |  16252 |
| B_raw               | distance_band | 2001_2600 |           0.133467 |              0.411085 |     232 |   2699 |
| B_raw               | race_class    | Class1    |           0.138928 |              0.430178 |     895 |  11147 |
| B_raw               | race_class    | Class2    |           0.139976 |              0.433108 |     449 |   5937 |
| B_raw               | race_class    | Class3    |           0.135247 |              0.423543 |     206 |   2924 |
| B_raw               | race_class    | Maiden    |           0.131117 |              0.405229 |    1050 |   7966 |
| B_raw               | race_class    | Open      |           0.136344 |              0.421794 |     244 |   3251 |
| LightGBM120         | surface       | dirt      |           0.134839 |              0.417576 |    1539 |  16992 |
| LightGBM120         | surface       | turf      |           0.136943 |              0.423855 |    1305 |  14233 |
| LightGBM120         | distance_band | 1000_1400 |           0.133467 |              0.414685 |    1042 |  12274 |
| LightGBM120         | distance_band | 1401_2000 |           0.137699 |              0.425666 |    1570 |  16252 |
| LightGBM120         | distance_band | 2001_2600 |           0.133478 |              0.411135 |     232 |   2699 |
| LightGBM120         | race_class    | Class1    |           0.138919 |              0.430158 |     895 |  11147 |
| LightGBM120         | race_class    | Class2    |           0.140028 |              0.433342 |     449 |   5937 |
| LightGBM120         | race_class    | Class3    |           0.135366 |              0.4238   |     206 |   2924 |
| LightGBM120         | race_class    | Maiden    |           0.131281 |              0.405633 |    1050 |   7966 |
| LightGBM120         | race_class    | Open      |           0.136442 |              0.422136 |     244 |   3251 |
| CatBoost120         | surface       | dirt      |           0.13494  |              0.417884 |    1539 |  16992 |
| CatBoost120         | surface       | turf      |           0.13696  |              0.424007 |    1305 |  14233 |
| CatBoost120         | distance_band | 1000_1400 |           0.133573 |              0.415136 |    1042 |  12274 |
| CatBoost120         | distance_band | 1401_2000 |           0.137734 |              0.42572  |    1570 |  16252 |
| CatBoost120         | distance_band | 2001_2600 |           0.133539 |              0.411641 |     232 |   2699 |
| CatBoost120         | race_class    | Class1    |           0.139033 |              0.430433 |     895 |  11147 |
| CatBoost120         | race_class    | Class2    |           0.140093 |              0.43358  |     449 |   5937 |
| CatBoost120         | race_class    | Class3    |           0.13535  |              0.4238   |     206 |   2924 |
| CatBoost120         | race_class    | Maiden    |           0.131322 |              0.405855 |    1050 |   7966 |
| CatBoost120         | race_class    | Open      |           0.136472 |              0.422489 |     244 |   3251 |
| xgb_selected_raw    | surface       | dirt      |           0.134535 |              0.41667  |    1539 |  16992 |
| xgb_selected_raw    | surface       | turf      |           0.136914 |              0.424002 |    1305 |  14233 |
| xgb_selected_raw    | distance_band | 1000_1400 |           0.13321  |              0.414091 |    1042 |  12274 |
| xgb_selected_raw    | distance_band | 1401_2000 |           0.137564 |              0.425369 |    1570 |  16252 |
| xgb_selected_raw    | distance_band | 2001_2600 |           0.133368 |              0.410628 |     232 |   2699 |
| xgb_selected_raw    | race_class    | Class1    |           0.13877  |              0.42983  |     895 |  11147 |
| xgb_selected_raw    | race_class    | Class2    |           0.139882 |              0.432879 |     449 |   5937 |
| xgb_selected_raw    | race_class    | Class3    |           0.135057 |              0.422987 |     206 |   2924 |
| xgb_selected_raw    | race_class    | Maiden    |           0.131071 |              0.405257 |    1050 |   7966 |
| xgb_selected_raw    | race_class    | Open      |           0.136353 |              0.421569 |     244 |   3251 |
| xgb_selected_scaled | surface       | dirt      |           0.134608 |              0.416862 |    1539 |  16992 |
| xgb_selected_scaled | surface       | turf      |           0.136921 |              0.423922 |    1305 |  14233 |
| xgb_selected_scaled | distance_band | 1000_1400 |           0.133301 |              0.414309 |    1042 |  12274 |
| xgb_selected_scaled | distance_band | 1401_2000 |           0.137577 |              0.425321 |    1570 |  16252 |
| xgb_selected_scaled | distance_band | 2001_2600 |           0.133395 |              0.410795 |     232 |   2699 |
| xgb_selected_scaled | race_class    | Class1    |           0.138804 |              0.429858 |     895 |  11147 |
| xgb_selected_scaled | race_class    | Class2    |           0.139964 |              0.433097 |     449 |   5937 |
| xgb_selected_scaled | race_class    | Class3    |           0.135079 |              0.422971 |     206 |   2924 |
| xgb_selected_scaled | race_class    | Maiden    |           0.131116 |              0.405281 |    1050 |   7966 |
| xgb_selected_scaled | race_class    | Open      |           0.136362 |              0.421752 |     244 |   3251 |
| lgb_selected_raw    | surface       | dirt      |           0.134839 |              0.417576 |    1539 |  16992 |
| lgb_selected_raw    | surface       | turf      |           0.136943 |              0.423855 |    1305 |  14233 |
| lgb_selected_raw    | distance_band | 1000_1400 |           0.133467 |              0.414685 |    1042 |  12274 |
| lgb_selected_raw    | distance_band | 1401_2000 |           0.137699 |              0.425666 |    1570 |  16252 |
| lgb_selected_raw    | distance_band | 2001_2600 |           0.133478 |              0.411135 |     232 |   2699 |
| lgb_selected_raw    | race_class    | Class1    |           0.138919 |              0.430158 |     895 |  11147 |
| lgb_selected_raw    | race_class    | Class2    |           0.140028 |              0.433342 |     449 |   5937 |
| lgb_selected_raw    | race_class    | Class3    |           0.135366 |              0.4238   |     206 |   2924 |
| lgb_selected_raw    | race_class    | Maiden    |           0.131281 |              0.405633 |    1050 |   7966 |
| lgb_selected_raw    | race_class    | Open      |           0.136442 |              0.422136 |     244 |   3251 |
| lgb_selected_scaled | surface       | dirt      |           0.13482  |              0.417526 |    1539 |  16992 |
| lgb_selected_scaled | surface       | turf      |           0.13693  |              0.423835 |    1305 |  14233 |
| lgb_selected_scaled | distance_band | 1000_1400 |           0.133436 |              0.414601 |    1042 |  12274 |
| lgb_selected_scaled | distance_band | 1401_2000 |           0.137692 |              0.425666 |    1570 |  16252 |
| lgb_selected_scaled | distance_band | 2001_2600 |           0.133464 |              0.411067 |     232 |   2699 |
| lgb_selected_scaled | race_class    | Class1    |           0.138903 |              0.430128 |     895 |  11147 |
| lgb_selected_scaled | race_class    | Class2    |           0.139997 |              0.433259 |     449 |   5937 |
| lgb_selected_scaled | race_class    | Class3    |           0.135369 |              0.423827 |     206 |   2924 |
| lgb_selected_scaled | race_class    | Maiden    |           0.131265 |              0.405605 |    1050 |   7966 |
| lgb_selected_scaled | race_class    | Open      |           0.136435 |              0.422077 |     244 |   3251 |
| cat_selected_raw    | surface       | dirt      |           0.134718 |              0.41726  |    1539 |  16992 |
| cat_selected_raw    | surface       | turf      |           0.136803 |              0.423671 |    1305 |  14233 |
| cat_selected_raw    | distance_band | 1000_1400 |           0.133276 |              0.414234 |    1042 |  12274 |
| cat_selected_raw    | distance_band | 1401_2000 |           0.137614 |              0.425545 |    1570 |  16252 |
| cat_selected_raw    | distance_band | 2001_2600 |           0.133324 |              0.41085  |     232 |   2699 |
| cat_selected_raw    | race_class    | Class1    |           0.138858 |              0.430031 |     895 |  11147 |
| cat_selected_raw    | race_class    | Class2    |           0.139786 |              0.43275  |     449 |   5937 |
| cat_selected_raw    | race_class    | Class3    |           0.135134 |              0.423142 |     206 |   2924 |
| cat_selected_raw    | race_class    | Maiden    |           0.131145 |              0.405448 |    1050 |   7966 |
| cat_selected_raw    | race_class    | Open      |           0.136379 |              0.422062 |     244 |   3251 |
| cat_selected_scaled | surface       | dirt      |           0.13474  |              0.417331 |    1539 |  16992 |
| cat_selected_scaled | surface       | turf      |           0.136819 |              0.423699 |    1305 |  14233 |
| cat_selected_scaled | distance_band | 1000_1400 |           0.133312 |              0.414346 |    1042 |  12274 |
| cat_selected_scaled | distance_band | 1401_2000 |           0.137621 |              0.425551 |    1570 |  16252 |
| cat_selected_scaled | distance_band | 2001_2600 |           0.133346 |              0.410938 |     232 |   2699 |
| cat_selected_scaled | race_class    | Class1    |           0.138872 |              0.430063 |     895 |  11147 |
| cat_selected_scaled | race_class    | Class2    |           0.13983  |              0.432871 |     449 |   5937 |
| cat_selected_scaled | race_class    | Class3    |           0.135139 |              0.423144 |     206 |   2924 |
| cat_selected_scaled | race_class    | Maiden    |           0.131165 |              0.405497 |    1050 |   7966 |
| cat_selected_scaled | race_class    | Open      |           0.136381 |              0.422112 |     244 |   3251 |
| XL_75_25_logit      | surface       | dirt      |           0.134585 |              0.416818 |    1539 |  16992 |
| XL_75_25_logit      | surface       | turf      |           0.136896 |              0.423887 |    1305 |  14233 |
| XL_75_25_logit      | distance_band | 1000_1400 |           0.133247 |              0.414157 |    1042 |  12274 |
| XL_75_25_logit      | distance_band | 1401_2000 |           0.137574 |              0.425369 |    1570 |  16252 |
| XL_75_25_logit      | distance_band | 2001_2600 |           0.133367 |              0.410664 |     232 |   2699 |
| XL_75_25_logit      | race_class    | Class1    |           0.13878  |              0.429833 |     895 |  11147 |
| XL_75_25_logit      | race_class    | Class2    |           0.139897 |              0.432929 |     449 |   5937 |
| XL_75_25_logit      | race_class    | Class3    |           0.135116 |              0.423129 |     206 |   2924 |
| XL_75_25_logit      | race_class    | Maiden    |           0.131093 |              0.40526  |    1050 |   7966 |
| XL_75_25_logit      | race_class    | Open      |           0.136355 |              0.421647 |     244 |   3251 |
| XC_75_25_logit      | surface       | dirt      |           0.134563 |              0.416744 |    1539 |  16992 |
| XC_75_25_logit      | surface       | turf      |           0.13687  |              0.42385  |    1305 |  14233 |
| XC_75_25_logit      | distance_band | 1000_1400 |           0.133208 |              0.41405  |    1042 |  12274 |
| XC_75_25_logit      | distance_band | 1401_2000 |           0.137561 |              0.425345 |    1570 |  16252 |
| XC_75_25_logit      | distance_band | 2001_2600 |           0.133339 |              0.41061  |     232 |   2699 |
| XC_75_25_logit      | race_class    | Class1    |           0.138776 |              0.429816 |     895 |  11147 |
| XC_75_25_logit      | race_class    | Class2    |           0.139845 |              0.432795 |     449 |   5937 |
| XC_75_25_logit      | race_class    | Class3    |           0.135067 |              0.422983 |     206 |   2924 |
| XC_75_25_logit      | race_class    | Maiden    |           0.131067 |              0.405207 |    1050 |   7966 |
| XC_75_25_logit      | race_class    | Open      |           0.136349 |              0.421648 |     244 |   3251 |
| XLC_ridge_2023      | surface       | dirt      |           0.134521 |              0.416662 |    1539 |  16992 |
| XLC_ridge_2023      | surface       | turf      |           0.136928 |              0.424146 |    1305 |  14233 |
| XLC_ridge_2023      | distance_band | 1000_1400 |           0.133153 |              0.413958 |    1042 |  12274 |
| XLC_ridge_2023      | distance_band | 1401_2000 |           0.137601 |              0.425583 |    1570 |  16252 |
| XLC_ridge_2023      | distance_band | 2001_2600 |           0.13336  |              0.410531 |     232 |   2699 |
| XLC_ridge_2023      | race_class    | Class1    |           0.138783 |              0.429937 |     895 |  11147 |
| XLC_ridge_2023      | race_class    | Class2    |           0.139802 |              0.432683 |     449 |   5937 |
| XLC_ridge_2023      | race_class    | Class3    |           0.135106 |              0.423203 |     206 |   2924 |
| XLC_ridge_2023      | race_class    | Maiden    |           0.131073 |              0.405385 |    1050 |   7966 |
| XLC_ridge_2023      | race_class    | Open      |           0.136388 |              0.421516 |     244 |   3251 |

## 完了・不足と再開

- xgb：COMPLETE
- lgb：COMPLETE
- cat：COMPLETE

内側で1600本が最良なら探索上限外の最適本数は不明であり、1600本を真の最適本数と断定しない。計算未完了なら分類Dとし、制約や欠けた年次を提示する。

## アンサンブル実行条件

{
  "2023_xgb_brier": 0.13733607802224054,
  "2023_candidate_brier": {
    "lgb": 0.13735187916498523,
    "cat": 0.13732588482419572
  },
  "2023_market_residual_correlations": {
    "lgb": 0.9726233607390312,
    "cat": 0.8962837832420245
  },
  "individual_trigger": {
    "lgb": false,
    "cat": true
  },
  "run_three_preregistered_ensembles": true
}

実行するかどうかは2023年のOOF条件のみで判断し、2024年の結果に基づく重み・候補の追加入れ替えは禁止した。

## 検証、データ権利と本番運用

旧400本B_rawと旧120本LightGBM/CatBoostを同じ前処理と年で再fit。LightGBMとCatBoostは最大絶対差1e−8以内、XGBoostは保存予測がfloat32のため各馬で最大2 ULPかつ全体絶対差1e−7以下の厳格な再現条件を適用（実測最大差は約2.98e−8）。古いM0、正解ラベル、全レース・全馬を照合。
AGENTS.mdは親ブランチの再帰ツリーで見つからなかった。README・モデル選択規約・PR #34/#35の仕様とログを確認した。
古いモデルと本番Stage5の市場のみモデルは変更していない。
元の出走結果CSVやJRA履歴の原本はコミットしていない。
予測の各行・レース差分・reliability・ソースSHAと学習時間を保存。

前向き採用は本番予測時刻T−10〜T−9のロック、同時刻市場情報、モデル固定、最低1000レース・60開催日・90日の規約を通過するまで不可。
中断からのコマンドとチェックポイント構造は docs/ROUND_SWEEP_1600_V1_RUNBOOK.md に記録。
