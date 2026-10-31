"""Regression tests for anomalous published Umanity prior-race passage values."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tennosho_f3_raw_audit import _parse_umanity_passage  # noqa: E402


@pytest.mark.parametrize(
    ("published", "expected_passing"),
    [
        ("8 8 9 11 106.9", "8-8-9-11"),
        ("5 7 13 13 100.2", "5-7-13-13"),
        ("10 10 10 9 100.9", "10-10-10-9"),
    ],
)
def test_nonphysical_three_digit_elapsed_time_not_synthetic_last3f(
    published: str, expected_passing: str,
) -> None:
    passing, last_3f = _parse_umanity_passage(published, "right")
    assert passing == expected_passing
    assert last_3f is None


def test_normal_final_three_furlong_and_corner_parsing() -> None:
    assert _parse_umanity_passage("1 2 3 4 35.7", "left") == (
        "1-2-3-4", "35.7"
    )
    assert _parse_umanity_passage("35.7", "straight") == (None, "35.7")


def test_genuine_five_corner_source_is_rejected() -> None:
    with pytest.raises(ValueError, match=">4 corners"):
        _parse_umanity_passage("1 2 3 4 5 35.7", "right")
