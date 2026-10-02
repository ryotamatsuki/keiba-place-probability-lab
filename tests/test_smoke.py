import numpy as np

from keiba_place_lab.evaluation import brier_score


def test_brier_score_perfect_forecast() -> None:
    y = np.array([1, 0, 1], dtype=float)
    p = np.array([1, 0, 1], dtype=float)
    assert brier_score(y, p) == 0.0
