"""Tests for src/strategy/conditions_corporate.py.

All CorporateCalendar interactions are mocked — no real DB.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.strategy.conditions_corporate import (
    has_bonus_in,
    has_dividend_in,
    has_split_in,
    in_corporate_blackout,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_calendar(blackout=False, reason=None, imminent=False):
    """Return a MagicMock CorporateCalendar with preset return values."""
    cal = MagicMock()
    cal.is_blackout.return_value = (blackout, reason)
    cal.has_imminent_action.return_value = imminent
    return cal


# ---------------------------------------------------------------------------
# in_corporate_blackout
# ---------------------------------------------------------------------------

class TestInCorporateBlackout:
    """Tests for in_corporate_blackout()."""

    def test_in_corporate_blackout_false_when_no_engine(self):
        """Returns False when db_engine is None (safe default)."""
        result = in_corporate_blackout("RELIANCE", db_engine=None)
        assert result is False

    def test_in_corporate_blackout_true_when_dividend_tomorrow(self):
        """Returns True when a dividend ex_date is T+1 (within window)."""
        cal = _mock_calendar(blackout=True, reason="DIVIDEND ex-date 2026-05-09")
        with patch(
            "src.strategy.conditions_corporate.CorporateCalendar", return_value=cal
        ):
            result = in_corporate_blackout("RELIANCE", days_before=1, days_after=1, db_engine=MagicMock())
        assert result is True

    def test_in_corporate_blackout_false_outside_window(self):
        """Returns False when no action is near the trade date."""
        cal = _mock_calendar(blackout=False, reason=None)
        with patch(
            "src.strategy.conditions_corporate.CorporateCalendar", return_value=cal
        ):
            result = in_corporate_blackout("TCS", db_engine=MagicMock())
        assert result is False

    def test_in_corporate_blackout_delegates_to_calendar(self):
        """is_blackout is called with today's date and correct window."""
        cal = _mock_calendar(blackout=False)
        with patch(
            "src.strategy.conditions_corporate.CorporateCalendar", return_value=cal
        ):
            in_corporate_blackout("INFY", days_before=2, days_after=2, db_engine=MagicMock())

        cal.is_blackout.assert_called_once_with(
            "INFY",
            trade_date=date.today(),
            days_before=2,
            days_after=2,
        )


# ---------------------------------------------------------------------------
# has_dividend_in
# ---------------------------------------------------------------------------

class TestHasDividendIn:
    """Tests for has_dividend_in()."""

    def test_has_dividend_in_false_when_no_engine(self):
        """Returns False when db_engine is None."""
        result = has_dividend_in("RELIANCE", db_engine=None)
        assert result is False

    def test_has_dividend_in_7_days_true(self):
        """Returns True when a dividend ex_date is within 7 days."""
        cal = _mock_calendar(imminent=True)
        with patch(
            "src.strategy.conditions_corporate.CorporateCalendar", return_value=cal
        ):
            result = has_dividend_in("RELIANCE", days=7, db_engine=MagicMock())
        assert result is True
        cal.has_imminent_action.assert_called_once_with(
            "RELIANCE",
            trade_date=date.today(),
            action_types=["DIVIDEND"],
            days=7,
        )

    def test_has_dividend_in_false_when_no_dividend(self):
        """Returns False when no dividend is upcoming."""
        cal = _mock_calendar(imminent=False)
        with patch(
            "src.strategy.conditions_corporate.CorporateCalendar", return_value=cal
        ):
            result = has_dividend_in("TCS", days=7, db_engine=MagicMock())
        assert result is False


# ---------------------------------------------------------------------------
# has_split_in
# ---------------------------------------------------------------------------

class TestHasSplitIn:
    """Tests for has_split_in()."""

    def test_has_split_in_false_when_no_engine(self):
        """Returns False when db_engine is None."""
        result = has_split_in("INFY", db_engine=None)
        assert result is False

    def test_has_split_in_30_days_true(self):
        """Returns True when a split ex_date is within 30 days."""
        cal = _mock_calendar(imminent=True)
        with patch(
            "src.strategy.conditions_corporate.CorporateCalendar", return_value=cal
        ):
            result = has_split_in("INFY", days=30, db_engine=MagicMock())
        assert result is True
        cal.has_imminent_action.assert_called_once_with(
            "INFY",
            trade_date=date.today(),
            action_types=["SPLIT"],
            days=30,
        )

    def test_has_split_in_false_when_no_split(self):
        """Returns False when no split is scheduled."""
        cal = _mock_calendar(imminent=False)
        with patch(
            "src.strategy.conditions_corporate.CorporateCalendar", return_value=cal
        ):
            result = has_split_in("HDFCBANK", days=30, db_engine=MagicMock())
        assert result is False


# ---------------------------------------------------------------------------
# has_bonus_in
# ---------------------------------------------------------------------------

class TestHasBonusIn:
    """Tests for has_bonus_in()."""

    def test_has_bonus_in_false_when_no_engine(self):
        """Returns False when db_engine is None."""
        result = has_bonus_in("SBIN", db_engine=None)
        assert result is False

    def test_has_bonus_in_true(self):
        """Returns True when a bonus ex_date is within 30 days."""
        cal = _mock_calendar(imminent=True)
        with patch(
            "src.strategy.conditions_corporate.CorporateCalendar", return_value=cal
        ):
            result = has_bonus_in("SBIN", days=30, db_engine=MagicMock())
        assert result is True
        cal.has_imminent_action.assert_called_once_with(
            "SBIN",
            trade_date=date.today(),
            action_types=["BONUS"],
            days=30,
        )

    def test_has_bonus_in_false_when_no_bonus(self):
        """Returns False when no bonus is scheduled."""
        cal = _mock_calendar(imminent=False)
        with patch(
            "src.strategy.conditions_corporate.CorporateCalendar", return_value=cal
        ):
            result = has_bonus_in("ICICIBANK", days=30, db_engine=MagicMock())
        assert result is False
