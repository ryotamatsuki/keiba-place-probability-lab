import pandas as pd
import pytest

from keiba_place_lab.postrace import evaluate_race, validate_outcome


def test_outcome_contract():
    outcome = pd.DataFrame(
        {
            "finish_rank": range(1, 19),
            "horse_no": range(1, 19),
            "horse_name": [f"H{i}" for i in range(1, 19)],
            "top3_label": [1, 1, 1] + [0] * 15,
            "official_source_url": ["https://example.invalid"] * 18,
        }
    )
    validate_outcome(outcome)


def test_outcome_rejects_wrong_top3_label():
    outcome = pd.DataFrame(
        {
            "finish_rank": range(1, 19),
            "horse_no": range(1, 19),
            "horse_name": [f"H{i}" for i in range(1, 19)],
            "top3_label": [1, 1, 0] + [0] * 15,
            "official_source_url": ["https://example.invalid"] * 18,
        }
    )
    with pytest.raises(ValueError, match="inconsistent"):
        validate_outcome(outcome)


def test_evaluation_metrics_and_forecast_ranks():
    frame = pd.DataFrame(
        {
            "top3_label": [1, 1, 1] + [0] * 15,
            "p": [0.8, 0.7, 0.6] + [0.06] * 15,
        }
    )
    result = evaluate_race(frame, "p")
    assert result.brier < 0.05
    assert result.log_loss < 0.2
    assert result.actual_top3_probability_mass == pytest.approx(2.1)
    assert result.actual_top3_forecast_ranks == (1, 2, 3)
