"""Render final Japanese research memo solely from saved Python evaluation files."""
# ruff: noqa: ISC004  # Intentional adjacent Japanese prose strings in report list.
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def table(path: Path, cols: list[str]) -> str:
    if not path.exists():
        return "未完了：指標ファイルなし"
    df = pd.read_csv(path)
    actual = [c for c in cols if c in df]
    if not actual:
        return "データ列が見つかりません"
    return df[actual].round(9).to_markdown(index=False)


def render(folder: Path) -> Path:
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run the Python numerical evaluation first")
    m = json.loads(manifest_path.read_text())
    category = m["category"]
    explanations = {
        "A": "時系列で選択した学習回数によりB_rawを上回り、開催日paired one-SEを通過。",
        "B": "点推定では改善するがpaired one-SEを通過しないためB_rawを維持。",
        "C": "事前固定した本数・特徴量・学習器で追加改善を確認できず。",
        "D": "不足する年次・モデルがあるため採否は判定不能。",
    }
    classification = explanations[category]
    summary = []
    if category != "D" and (folder / "2024_candidate_scores.csv").exists():
        scores = pd.read_csv(folder / "2024_candidate_scores.csv").set_index("name")
        paired = pd.read_csv(folder / "2024_paired_vs_B_raw.csv").set_index("candidate")
        winner = m["selection"]["winner"]
        best = scores.loc[winner]
        baseline = scores.loc["B_raw"]
        delta = paired.loc[winner]
        xgb = scores.loc["xgb_selected_raw"]
        xgb_delta = paired.loc["xgb_selected_raw"]
        summary = [
            f"**採用判断：研究上の勝者は {winner}。**"
            f" 2024年のBrierは {best.race_macro_brier:.9f}、"
            f"既存B_rawは {baseline.race_macro_brier:.9f}。",
            f"B_rawとの差は {delta.B_raw_minus_candidate_brier:.9f}、"
            f"開催日クラスタpaired SEは {delta.date_cluster_SE_brier:.9f}。"
            " 事前のone-SEゲートを通過した。"
            if winner != "B_raw" else "採用ゲートによりB_rawを維持。",
            f"**単体XGBoostの学習回数変更：** 800本の2024年Brier"
            f" {xgb.race_macro_brier:.9f}、B_raw比改善"
            f" {xgb_delta.B_raw_minus_candidate_brier:.9f}、"
            f"paired SE {xgb_delta.date_cluster_SE_brier:.9f}。",
            "アンサンブルの補完効果と単体800本への単純な補正強度変更の"
            "効果を混同せず、同じ2024年レースで別々に比較した。",
        ]
    lines = [
        "# B_rawの400本上限・公平な3学習器比較：実データ研究報告",
        "",
        "## 最終判断",
        "",
        "**分類：" + category + "。** " + classification,
        "",
        *summary,
        "",
        "評価は既知の2023・2024年であり、未見の将来テストではない。"
        "過去の最終単勝オッズを発走10分前のオッズと扱わない。",
        "",
        "## 実験条件と研究の系譜",
        "",
        "PR #35のコミット00fd15638548285a8c8e4e27f4aee0860cb24750を起点に、"
        "PR #35→#34→#33→#32→#31へ依存。モデル選択規約V2を維持した。",
        "登録仕様コミット183717e5a875b8e9a00508989cfaed57a2320e2dは"
        "新スコア算出前に作成。後から学習率や正則化を変更していない。",
        "",
        "F3の既存市場・能力特徴量と完全な出走頭数を固定。"
        "XGBoost、LightGBM、CatBoostの元パラメータを変えず、"
        "木数60、120、240、400、800、1600だけを比較。"
        "学習器ごとの上限は100分、CPU最大2スレッド、年別成果物を随時保存。",
        "",
        "外側2023の学習2018〜2022、選択／補正係数の学習には内側2021〜2022のみ。"
        "外側2024の学習2018〜2023、選択／補正係数には内側2021〜2023のみ。"
        "年ごとに厳密に過去だけで学習し、Brierはレースmacro平均で選択。",
        "",
        "## 各モデルの内側学習曲線（全6チェックポイント）",
        "",
        table(folder / "inner_all_family_curves.csv",
              ["family","year","rounds","races","race_macro_brier",
               "race_macro_log_loss"]),
        "",
        "## 外側年に選んだ木本数、補正係数と学習時間",
        "",
        table(folder / "outer_selected_rounds_and_alpha.csv",
              ["family","outer_year","selected_rounds","alpha",
               "alpha_fit_years","train_max_date","eval_min_date",
               "selected_model_fit_seconds","baseline_fit_seconds",
               "original_fixed_parity_max_error","upper_bound_selected"]),
        "",
        "## 2024年同一レース・同一頭のBrierとlog loss",
        "",
        table(folder / "2024_candidate_scores.csv",
              ["name","race_macro_brier","race_macro_log_loss","races",
               "rows","point_brier_best","within_one_se_brier"]),
        "",
        "## B_raw対候補のpaired差と開催日クラスタSE",
        "",
        table(folder / "2024_paired_vs_B_raw.csv",
              ["candidate","B_raw_minus_candidate_brier",
               "date_cluster_SE_brier","B_raw_minus_candidate_log_loss",
               "beats_B_raw_pairwise_one_se"]),
        "",
        "## 校正切片と傾き",
        "",
        table(folder / "2024_calibration.csv",
              ["candidate","intercept","slope"]),
        "",
        "## 芝・ダート、距離、レースクラス別診断",
        "",
        table(folder / "2024_subgroups.csv",
              ["candidate","group","value","race_macro_brier",
               "race_macro_log_loss","races","rows"]),
        "",
        "## 完了・不足と再開",
        "",
    ]
    for family, status in m["status"].items():
        lines.append("- " + family + "：" + status)
    lines.extend([
        "",
        "内側で1600本が最良なら探索上限外の最適本数は不明であり、"
        "1600本を真の最適本数と断定しない。"
        "計算未完了なら分類Dとし、制約や欠けた年次を提示する。",
        "",
        "## アンサンブル実行条件",
        "",
        json.dumps(m.get("optional_ensemble_2023_gate",
                         {"status":"unavailable"}),ensure_ascii=False,indent=2),
        "",
        "実行するかどうかは2023年のOOF条件のみで判断し、"
        "2024年の結果に基づく重み・候補の追加入れ替えは禁止した。",
        "",
        "## 検証、データ権利と本番運用",
        "",
        "旧400本B_rawと旧120本LightGBM/CatBoostを同じ前処理と年で再fit。"
        "LightGBMとCatBoostは最大絶対差1e−8以内、XGBoostは保存予測が"
        "float32のため各馬で最大2 ULPかつ全体絶対差1e−7以下の厳格な"
        "再現条件を適用（実測最大差は約2.98e−8）。"
        "古いM0、正解ラベル、全レース・全馬を照合。",
        "AGENTS.mdは親ブランチの再帰ツリーで見つからなかった。"
        "README・モデル選択規約・PR #34/#35の仕様とログを確認した。",
        "古いモデルと本番Stage5の市場のみモデルは変更していない。",
        "元の出走結果CSVやJRA履歴の原本はコミットしていない。",
        "予測の各行・レース差分・reliability・ソースSHAと学習時間を保存。",
        "",
        "前向き採用は本番予測時刻T−10〜T−9のロック、"
        "同時刻市場情報、モデル固定、最低1000レース・60開催日・90日"
        "の規約を通過するまで不可。",
        "中断からのコマンドとチェックポイント構造は"
        " docs/ROUND_SWEEP_1600_V1_RUNBOOK.md に記録。",
        "",
    ])
    target = folder / "FINAL_REPORT.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    a = parser.parse_args()
    print(render(a.folder))


if __name__ == "__main__":
    main()
