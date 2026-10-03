import numpy as np

from keiba_place_lab.market import (
    harville_top_k_probabilities,
    normalized_win_probabilities,
    place_odds_strength_proxy,
)


def test_normalized_win_probabilities_sum_to_one() -> None:
    p, overround = normalized_win_probabilities(np.array([2.0, 3.0, 6.0]))
    assert np.isclose(p.sum(), 1.0)
    assert np.isclose(overround, 1.0)


def test_harville_top3_marginals_sum_to_three() -> None:
    p = np.array([0.40, 0.30, 0.20, 0.10])
    top3 = harville_top_k_probabilities(p, k=3)
    assert np.isclose(top3.sum(), 3.0)
    assert np.all((top3 > 0.0) & (top3 < 1.0))


def test_place_proxy_is_only_scaled_strength() -> None:
    proxy = place_odds_strength_proxy(
        np.array([2.0, 3.0, 4.0]),
        np.array([2.4, 3.4, 4.4]),
        target_sum=3.0,
    )
    assert np.isclose(proxy.sum(), 3.0)
    assert proxy[0] > proxy[1] > proxy[2]
