"""Operationalization gates fail shut on missing/stale/mixed trio tote quotes."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from keiba_place_lab.trio_operational_gate import (
    evaluate_readiness, validate_complete_trio_quotes,
)


def sample():
    picks = list(combinations(range(1, 6), 3))
    p = pd.DataFrame(picks, columns=["horse_1", "horse_2", "horse_3"])
    p["probability"] = .1
    q = p.drop(columns="probability").copy()
    q["odds"] = 12.0
    q["race_id"] = "202605041111"
    q["source_updated_at"] = "2026-11-01T15:29:00+09:00"
    q["capture_at"] = "2026-11-01T15:30:00+09:00"
    return p, q


def test_all_quotations_are_required_and_one_snapshot():
    p, q = sample()
    x = validate_complete_trio_quotes(
        p, q, expected_horse_numbers=set(range(1, 6)),
        expected_race_id="202605041111",
    )
    assert len(x) == 10
    assert np.allclose(x.model_ev_per_yen_at_snapshot, .2)
    with pytest.raises(ValueError, match="all exact"):
        validate_complete_trio_quotes(
            p, q.iloc[:-1], expected_horse_numbers=set(range(1, 6)),
            expected_race_id="202605041111",
        )
    q.loc[1, "source_updated_at"] = "2026-11-01T15:28:00+09:00"
    with pytest.raises(ValueError, match="one identical"):
        validate_complete_trio_quotes(
            p, q, expected_horse_numbers=set(range(1, 6)),
            expected_race_id="202605041111",
        )


def test_stale_and_wrong_race_odds_are_rejected():
    p, q = sample()
    q["source_updated_at"] = "2026-11-01T15:20:00+09:00"
    with pytest.raises(ValueError, match="three minutes"):
        validate_complete_trio_quotes(
            p, q, expected_horse_numbers=set(range(1, 6)),
            expected_race_id="202605041111",
        )
    q["source_updated_at"] = "2026-11-01T15:29:00+09:00"
    q["race_id"] = "WRONG"
    with pytest.raises(ValueError, match="another race"):
        validate_complete_trio_quotes(
            p, q, expected_horse_numbers=set(range(1, 6)),
            expected_race_id="202605041111",
        )


def test_false_evidence_blocks_a_bet():
    gate = evaluate_readiness(
        matching_frozen_ensemble_holdout=False,
        independent_prospective_t10_validation=False,
        empirical_trio_calibration_approved=False,
        full_same_snapshot_trio_quotes=False,
        price_slippage_assessed=False,
        approved_risk_limits=False,
    )
    assert gate["decision"] == "NO_BET"
    assert len(gate["blockers"]) == 6
    gate2 = evaluate_readiness(
        matching_frozen_ensemble_holdout=True,
        independent_prospective_t10_validation=True,
        empirical_trio_calibration_approved=True,
        full_same_snapshot_trio_quotes=True,
        price_slippage_assessed=True,
        approved_risk_limits=True,
    )
    assert gate2["decision"] == "ELIGIBLE_FOR_MANUAL_REVIEW"
    assert gate2["never_auto_place_wagers"]
