import numpy as np
import pandas as pd
import pytest

from keiba_place_lab.model_selection import (
    candidate_score,
    paired_delta,
    race_loss_table,
    select_with_incumbent_one_se,
)


def _selection_frame() -> pd.DataFrame:
    rows = []
    for race_no in range(8):
        date = f"2024-01-{1 + race_no // 2:02d}"
        for runner, y in enumerate([1, 1, 1, 0], start=1):
            rows.append(
                {
                    "race_id": f"R{race_no}",
                    "race_date": date,
                    "runner": runner,
                    "top3_label": y,
                }
            )
    return pd.DataFrame(rows)


def test_race_macro_gives_each_race_equal_weight():
    frame = pd.DataFrame(
        {
            "race_id": ["A", "A", "B", "B", "B", "B"],
            "race_date": ["2024-01-01"] * 2 + ["2024-01-02"] * 4,
            "top3_label": [1, 0, 1, 1, 1, 0],
        }
    )
    p = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    score = candidate_score(frame, p, name="m")

    assert score.runner_micro_brier == pytest.approx(2.0 / 6.0)
    assert score.race_macro_brier == pytest.approx(0.5)


def test_identical_challenger_does_not_displace_incumbent():
    frame = _selection_frame()
    base = np.tile([0.70, 0.70, 0.70, 0.30], 8)

    decision, table = select_with_incumbent_one_se(
        frame,
        {"a_challenger": base.copy(), "z_incumbent": base.copy()},
        incumbent="z_incumbent",
    )

    assert decision.point_brier_best == "a_challenger"
    assert decision.winner == "z_incumbent"
    assert decision.incumbent_retained is True
    row = table.loc[table["name"] == "z_incumbent"].iloc[0]
    assert bool(row["within_one_se_brier"])
    assert bool(row["within_one_se_log_loss"])


def test_clear_brier_challenger_displaces_incumbent():
    frame = _selection_frame()
    incumbent = np.tile([0.55, 0.55, 0.55, 0.45], 8)
    challenger = np.tile([0.90, 0.90, 0.90, 0.10], 8)

    decision, table = select_with_incumbent_one_se(
        frame,
        {"incumbent": incumbent, "challenger": challenger},
        incumbent="incumbent",
    )

    assert decision.point_brier_best == "challenger"
    assert decision.winner == "challenger"
    assert decision.incumbent_retained is False
    row = table.loc[table["name"] == "incumbent"].iloc[0]
    assert not bool(row["within_one_se_brier"])


def test_paired_delta_requires_identical_race_coverage():
    frame = _selection_frame()
    p = np.tile([0.70, 0.70, 0.70, 0.30], 8)
    out = paired_delta(
        frame,
        p,
        p.copy(),
        candidate_name="a",
        reference_name="b",
    )
    assert out.mean_brier_delta == pytest.approx(0.0)
    assert out.mean_log_loss_delta == pytest.approx(0.0)


def test_probability_validation_rejects_out_of_range_values():
    frame = _selection_frame()
    p = np.full(len(frame), 0.5)
    p[0] = 1.1
    with pytest.raises(ValueError, match="probability must lie"):
        race_loss_table(frame, p)
