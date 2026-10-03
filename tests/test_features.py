import numpy as np
import pytest

from keiba_place_lab.features import normalized_early_position, pace_style_from_mean_position


def test_normalized_early_position_boundaries() -> None:
    assert normalized_early_position(1, 18) == 0.0
    assert normalized_early_position(18, 18) == 1.0
    assert np.isclose(normalized_early_position(9, 17), 0.5)


def test_invalid_position_is_rejected() -> None:
    with pytest.raises(ValueError):
        normalized_early_position(0, 18)


def test_pace_style_labels() -> None:
    assert pace_style_from_mean_position(0.10) == "front"
    assert pace_style_from_mean_position(0.30) == "forward_mid"
    assert pace_style_from_mean_position(0.55) == "mid_rear"
    assert pace_style_from_mean_position(0.90) == "deep_rear"
