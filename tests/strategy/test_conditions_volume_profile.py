"""Tests for src.strategy.conditions_volume_profile.

Uses inline mocks — no DB connection or network required.
All price values are in paisa (int).
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.strategy.conditions_volume_profile import (
    spot_above_poc,
    spot_above_vah,
    spot_at_va_extreme,
    spot_below_poc,
    spot_below_val,
    spot_in_value_area,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KEY = "NSE_EQ|INE002A01018"
TD = date(2026, 5, 8)

# A synthetic profile: POC=10_100, VAH=10_199, VAL=9_950
_PROFILE = {
    "poc_paisa": 10_100,
    "vah_paisa": 10_199,
    "val_paisa": 9_950,
    "total_volume": 50_000,
    "bin_size_paisa": 50,
    "bins": [],
}


def _mock_engine_with_profile(profile: dict | None):
    """Return (mock_engine, mock_builder) where get_profile returns *profile*."""
    engine = MagicMock()
    return engine, profile


def _patch_fetch(profile: dict | None):
    """Context manager that patches _fetch_profile to return *profile*."""
    return patch(
        "src.strategy.conditions_volume_profile._fetch_profile",
        return_value=profile,
    )


# ---------------------------------------------------------------------------
# spot_above_poc
# ---------------------------------------------------------------------------


class TestSpotAbovePoc:
    def test_true_when_above(self):
        with _patch_fetch(_PROFILE):
            assert spot_above_poc(KEY, 10_200, TD) is True

    def test_false_when_equal(self):
        with _patch_fetch(_PROFILE):
            assert spot_above_poc(KEY, 10_100, TD) is False

    def test_false_when_below(self):
        with _patch_fetch(_PROFILE):
            assert spot_above_poc(KEY, 9_500, TD) is False

    def test_false_when_no_profile(self):
        with _patch_fetch(None):
            assert spot_above_poc(KEY, 99_999, TD) is False


# ---------------------------------------------------------------------------
# spot_below_poc
# ---------------------------------------------------------------------------


class TestSpotBelowPoc:
    def test_true_when_below(self):
        with _patch_fetch(_PROFILE):
            assert spot_below_poc(KEY, 10_000, TD) is True

    def test_false_when_equal(self):
        with _patch_fetch(_PROFILE):
            assert spot_below_poc(KEY, 10_100, TD) is False

    def test_false_when_above(self):
        with _patch_fetch(_PROFILE):
            assert spot_below_poc(KEY, 10_500, TD) is False

    def test_false_when_no_profile(self):
        with _patch_fetch(None):
            assert spot_below_poc(KEY, 1, TD) is False


# ---------------------------------------------------------------------------
# spot_above_vah
# ---------------------------------------------------------------------------


class TestSpotAboveVah:
    def test_true_when_above(self):
        with _patch_fetch(_PROFILE):
            # VAH = 10_199; spot = 10_200 → above
            assert spot_above_vah(KEY, 10_200, TD) is True

    def test_false_when_at_vah(self):
        with _patch_fetch(_PROFILE):
            assert spot_above_vah(KEY, 10_199, TD) is False

    def test_false_when_below(self):
        with _patch_fetch(_PROFILE):
            assert spot_above_vah(KEY, 10_100, TD) is False

    def test_false_when_no_profile(self):
        with _patch_fetch(None):
            assert spot_above_vah(KEY, 99_999, TD) is False


# ---------------------------------------------------------------------------
# spot_below_val
# ---------------------------------------------------------------------------


class TestSpotBelowVal:
    def test_true_when_below(self):
        with _patch_fetch(_PROFILE):
            # VAL = 9_950; spot = 9_800 → below
            assert spot_below_val(KEY, 9_800, TD) is True

    def test_false_when_at_val(self):
        with _patch_fetch(_PROFILE):
            assert spot_below_val(KEY, 9_950, TD) is False

    def test_false_when_above(self):
        with _patch_fetch(_PROFILE):
            assert spot_below_val(KEY, 10_000, TD) is False

    def test_false_when_no_profile(self):
        with _patch_fetch(None):
            assert spot_below_val(KEY, 1, TD) is False


# ---------------------------------------------------------------------------
# spot_in_value_area
# ---------------------------------------------------------------------------


class TestSpotInValueArea:
    def test_true_when_between_vah_val(self):
        with _patch_fetch(_PROFILE):
            # Inside [9_950, 10_199]
            assert spot_in_value_area(KEY, 10_050, TD) is True

    def test_true_at_val_boundary(self):
        with _patch_fetch(_PROFILE):
            assert spot_in_value_area(KEY, 9_950, TD) is True

    def test_true_at_vah_boundary(self):
        with _patch_fetch(_PROFILE):
            assert spot_in_value_area(KEY, 10_199, TD) is True

    def test_false_below_val(self):
        with _patch_fetch(_PROFILE):
            assert spot_in_value_area(KEY, 9_900, TD) is False

    def test_false_above_vah(self):
        with _patch_fetch(_PROFILE):
            assert spot_in_value_area(KEY, 10_300, TD) is False

    def test_false_when_no_profile(self):
        with _patch_fetch(None):
            assert spot_in_value_area(KEY, 10_050, TD) is False


# ---------------------------------------------------------------------------
# spot_at_va_extreme
# ---------------------------------------------------------------------------


class TestSpotAtVaExtreme:
    def test_returns_vah_when_near_high(self):
        with _patch_fetch(_PROFILE):
            # VAH = 10_199; within 500 paisa → 10_199 ± 500
            result = spot_at_va_extreme(KEY, 10_100, TD, tolerance_paisa=500)
            # 10_100 is within 500 of VAH (10_199)? abs(10_100-10_199)=99 ✓
            # 10_100 is within 500 of VAL (9_950)?  abs(10_100-9_950)=150 ✓
            # Both in tolerance → tie-break: spot=10_100 >= mid=(10_199+9_950)//2=10_074 → VAH
            assert result == "VAH"

    def test_returns_val_when_near_low(self):
        with _patch_fetch(_PROFILE):
            # Near VAL only: spot = 9_950; abs(9_950 - 10_199)=249 < 500? Yes
            # abs(9_950 - 9_950)=0 ✓
            # Both in tolerance; mid=10_074; spot=9_950 < mid → VAL
            result = spot_at_va_extreme(KEY, 9_950, TD, tolerance_paisa=500)
            assert result == "VAL"

    def test_returns_vah_when_only_near_vah(self):
        with _patch_fetch(_PROFILE):
            # spot = 10_150 → abs(10_150-10_199)=49 ≤ 100; abs(10_150-9_950)=200 > 100
            result = spot_at_va_extreme(KEY, 10_150, TD, tolerance_paisa=100)
            assert result == "VAH"

    def test_returns_val_when_only_near_val(self):
        with _patch_fetch(_PROFILE):
            # spot = 9_970 → abs(9_970-9_950)=20 ≤ 100; abs(9_970-10_199)=229 > 100
            result = spot_at_va_extreme(KEY, 9_970, TD, tolerance_paisa=100)
            assert result == "VAL"

    def test_returns_none_when_far_from_both(self):
        with _patch_fetch(_PROFILE):
            # spot = 10_100; abs(10_100-10_199)=99 > 50; abs(10_100-9_950)=150 > 50
            result = spot_at_va_extreme(KEY, 10_100, TD, tolerance_paisa=50)
            assert result is None

    def test_returns_none_when_no_profile(self):
        with _patch_fetch(None):
            result = spot_at_va_extreme(KEY, 10_100, TD)
            assert result is None

    def test_default_tolerance_is_500(self):
        """Default tolerance_paisa is 500; spot exactly at VAH should return 'VAH'."""
        with _patch_fetch(_PROFILE):
            result = spot_at_va_extreme(KEY, 10_199, TD)  # no tolerance_paisa arg
            assert result == "VAH"


# ---------------------------------------------------------------------------
# Integration-style: fetch_profile delegates to VolumeProfileBuilder
# ---------------------------------------------------------------------------


class TestFetchProfileIntegration:
    """Ensure _fetch_profile calls VolumeProfileBuilder.get_profile correctly.

    VolumeProfileBuilder is imported *inside* _fetch_profile via a local
    import from src.indicators.volume_profile, so we patch it there.
    """

    def test_fetch_profile_calls_builder(self):
        mock_engine = MagicMock()
        mock_profile = {**_PROFILE}

        with patch(
            "src.indicators.volume_profile.VolumeProfileBuilder"
        ) as MockBuilder:
            instance = MockBuilder.return_value
            instance.get_profile.return_value = mock_profile

            result = spot_above_poc(KEY, 10_200, TD, db_engine=mock_engine)

        assert result is True
        MockBuilder.assert_called_once_with(db_engine=mock_engine)
        instance.get_profile.assert_called_once_with(KEY, TD)

    def test_no_engine_returns_false(self):
        """Without a db_engine, all conditions return False (profile = None)."""
        assert spot_above_poc(KEY, 99_999, TD, db_engine=None) is False
        assert spot_below_val(KEY, 1, TD, db_engine=None) is False
        assert spot_in_value_area(KEY, 10_000, TD, db_engine=None) is False
        assert spot_at_va_extreme(KEY, 10_000, TD, db_engine=None) is None
