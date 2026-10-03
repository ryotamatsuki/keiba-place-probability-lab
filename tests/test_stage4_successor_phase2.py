import numpy as np
import pandas as pd

from keiba_place_lab.stage4_successor_phase2 import (
    FAMILY_CONFIGS,
    RF_CONFIGS,
    registry_frame,
    select_inner_config,
)


def _frame() -> pd.DataFrame:
    rows = []
    for race_no in range(12):
        date = f"2022-01-{1 + race_no // 2:02d}"
        for y in [1, 1, 1, 0, 0, 0]:
            rows.append(
                {
                    "race_id": f"R{race_no}",
                    "race_date": date,
                    "top3_label": y,
                }
            )
    return pd.DataFrame(rows)


def test_registry_has_exact_preregistered_families_and_six_configs_each():
    registry = registry_frame()
    assert set(registry["family"]) == set(FAMILY_CONFIGS)
    counts = registry.groupby("family").size().to_dict()
    assert counts == {
        "random_forest": 6,
        "hist_gradient_boosting": 6,
        "xgboost": 6,
    }
    assert registry["config_id"].is_unique


def test_inner_one_se_can_prefer_simpler_config():
    frame = _frame()
    best = np.tile([0.72, 0.72, 0.72, 0.28, 0.28, 0.28], 12)
    nearly_same = best.copy()
    nearly_same[0] -= 0.01

    predictions = {
        "RF01": nearly_same,
        "RF02": best,
        "RF03": np.full(len(frame), 0.5),
        "RF04": np.full(len(frame), 0.5),
        "RF05": np.full(len(frame), 0.5),
        "RF06": np.full(len(frame), 0.5),
    }
    selected, table = select_inner_config(frame, predictions, RF_CONFIGS)

    assert selected.config_id == "RF01"
    assert table.loc[table["config_id"] == "RF02", "inner_point_best"].item()
    assert table.loc[table["config_id"] == "RF01", "within_one_se_of_inner_best"].item()


def test_inner_selection_chooses_clear_best_when_simpler_is_outside_one_se():
    frame = _frame()
    good = np.tile([0.9, 0.9, 0.9, 0.1, 0.1, 0.1], 12)
    weak = np.full(len(frame), 0.5)
    predictions = {
        "RF01": weak,
        "RF02": good,
        "RF03": weak,
        "RF04": weak,
        "RF05": weak,
        "RF06": weak,
    }
    selected, _ = select_inner_config(frame, predictions, RF_CONFIGS)
    assert selected.config_id == "RF02"
