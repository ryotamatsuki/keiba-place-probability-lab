"""The 2010-2025 frozen JRA prefix must survive all 2026 joins."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tennosho_f3_prior_builder import _jra_external_id, _jra_internal_id


def test_normalize_2025_frozen_and_2026_public_horse_to_same_identity():
    frozen = "jra:2023105344"
    public = "2023105344"
    assert _jra_internal_id(frozen) == _jra_internal_id(public)
    assert _jra_external_id(frozen) == public
    assert _jra_external_id(public) == public


@pytest.mark.parametrize("value", ["", "not-a-horse", "jra:123", "202605041111", None])
def test_invalid_jra_identity_refused(value):
    with pytest.raises(ValueError, match="Invalid verified JRA horse identity"):
        _jra_internal_id(value)
