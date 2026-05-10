"""Tests for src.strategy.conditions_earnings — earnings condition helpers.

All tests use inline mocks (unittest.mock) and do NOT require a live database.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.strategy.conditions_earnings import (
    earnings_within,
    in_earnings_blackout,
    negative_earnings_surprise,
    positive_earnings_surprise,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SYMBOL = "RELIANCE"
_TODAY = date(2026, 5, 8)
_EARNINGS_DATE = date(2026, 5, 9)  # T+1 from _TODAY → within blackout (days_before=2)


def _engine_with_blackout_row(blocked: bool) -> MagicMock:
    """Build a mock engine whose is_blackout-style query returns a row or not."""
    row = (_EARNINGS_DATE, "Q4 FY26") if blocked else None

    result = MagicMock()
    result.fetchone.return_value = row

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value = result

    engine = MagicMock()
    engine.connect.return_value = mock_conn
    return engine


def _engine_with_upcoming_row(has_upcoming: bool) -> MagicMock:
    """Build a mock engine for upcoming_earnings queries."""
    row = (
        SYMBOL, _EARNINGS_DATE, "Q4 FY26", "BMO", 25.0, None, None, "MONEYCONTROL"
    ) if has_upcoming else None

    result = MagicMock()
    result.fetchone.return_value = row

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value = result

    engine = MagicMock()
    engine.connect.return_value = mock_conn
    return engine


def _engine_with_surprise_row(has_surprise: bool) -> MagicMock:
    """Build a mock engine for positive/negative surprise queries."""
    row = (1,) if has_surprise else None

    result = MagicMock()
    result.fetchone.return_value = row

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value = result

    engine = MagicMock()
    engine.connect.return_value = mock_conn
    return engine


# ---------------------------------------------------------------------------
# in_earnings_blackout tests
# ---------------------------------------------------------------------------

class TestInEarningsBlackout:
    """Tests for in_earnings_blackout()."""

    def test_in_earnings_blackout_returns_false_when_no_engine(self):
        """Should return False gracefully when db_engine is None."""
        assert in_earnings_blackout(SYMBOL, db_engine=None) is False

    @patch(
        "src.research.earnings_calendar._ist_today",
        return_value=_TODAY,
    )
    def test_in_earnings_blackout_true_for_T_minus_1(self, mock_today):
        """T-1: today is one day before earnings → must be blocked."""
        # Earnings on _TODAY + 1 day; T-2 window should cover today
        engine = _engine_with_blackout_row(blocked=True)
        result = in_earnings_blackout(SYMBOL, days_before=2, days_after=1, db_engine=engine)
        assert result is True

    @patch(
        "src.research.earnings_calendar._ist_today",
        return_value=_TODAY,
    )
    def test_in_earnings_blackout_false_when_no_upcoming_earnings(self, mock_today):
        """Should return False when no earnings fall within the blackout window."""
        engine = _engine_with_blackout_row(blocked=False)
        result = in_earnings_blackout(SYMBOL, days_before=2, days_after=1, db_engine=engine)
        assert result is False

    def test_in_earnings_blackout_returns_false_on_exception(self):
        """Should return False (not raise) when EarningsCalendar raises."""
        engine = MagicMock()
        engine.connect.side_effect = Exception("DB error")
        result = in_earnings_blackout(SYMBOL, db_engine=engine)
        assert result is False


# ---------------------------------------------------------------------------
# earnings_within tests
# ---------------------------------------------------------------------------

class TestEarningsWithin:
    """Tests for earnings_within()."""

    def test_earnings_within_returns_false_when_no_engine(self):
        """Should return False gracefully when db_engine is None."""
        assert earnings_within(SYMBOL, days=3, db_engine=None) is False

    @patch(
        "src.research.earnings_calendar._ist_today",
        return_value=_TODAY,
    )
    def test_earnings_within_3_days_true(self, mock_today):
        """Should return True when earnings are within the specified window."""
        engine = _engine_with_upcoming_row(has_upcoming=True)
        result = earnings_within(SYMBOL, days=3, db_engine=engine)
        assert result is True

    @patch(
        "src.research.earnings_calendar._ist_today",
        return_value=_TODAY,
    )
    def test_earnings_within_returns_false_when_no_upcoming(self, mock_today):
        """Should return False when no earnings within the window."""
        engine = _engine_with_upcoming_row(has_upcoming=False)
        result = earnings_within(SYMBOL, days=3, db_engine=engine)
        assert result is False

    def test_earnings_within_returns_false_on_exception(self):
        """Should return False (not raise) on DB error."""
        engine = MagicMock()
        engine.connect.side_effect = Exception("DB error")
        result = earnings_within(SYMBOL, db_engine=engine)
        assert result is False


# ---------------------------------------------------------------------------
# positive_earnings_surprise tests
# ---------------------------------------------------------------------------

class TestPositiveEarningsSurprise:
    """Tests for positive_earnings_surprise()."""

    def test_positive_earnings_surprise_returns_false_when_no_engine(self):
        """Should return False gracefully when db_engine is None."""
        assert positive_earnings_surprise(SYMBOL, db_engine=None) is False

    @patch(
        "src.strategy.conditions_earnings._ist_today",
        return_value=_TODAY,
    )
    def test_positive_earnings_surprise_threshold(self, mock_today):
        """Should return True when a surprise row exceeds the threshold."""
        engine = _engine_with_surprise_row(has_surprise=True)
        result = positive_earnings_surprise(
            SYMBOL, threshold_pct=5.0, days=30, db_engine=engine
        )
        assert result is True

    @patch(
        "src.strategy.conditions_earnings._ist_today",
        return_value=_TODAY,
    )
    def test_positive_earnings_surprise_false_when_no_qualifying_row(self, mock_today):
        """Should return False when no qualifying surprise found."""
        engine = _engine_with_surprise_row(has_surprise=False)
        result = positive_earnings_surprise(
            SYMBOL, threshold_pct=5.0, days=30, db_engine=engine
        )
        assert result is False

    def test_positive_earnings_surprise_returns_false_on_exception(self):
        """Should return False (not raise) on DB error."""
        engine = MagicMock()
        engine.connect.side_effect = Exception("DB error")
        result = positive_earnings_surprise(SYMBOL, db_engine=engine)
        assert result is False


# ---------------------------------------------------------------------------
# negative_earnings_surprise tests
# ---------------------------------------------------------------------------

class TestNegativeEarningsSurprise:
    """Tests for negative_earnings_surprise()."""

    def test_negative_earnings_surprise_returns_false_when_no_engine(self):
        """Should return False gracefully when db_engine is None."""
        assert negative_earnings_surprise(SYMBOL, db_engine=None) is False

    @patch(
        "src.strategy.conditions_earnings._ist_today",
        return_value=_TODAY,
    )
    def test_negative_earnings_surprise_true_when_below_threshold(self, mock_today):
        """Should return True when a surprise row is below the threshold."""
        engine = _engine_with_surprise_row(has_surprise=True)
        result = negative_earnings_surprise(
            SYMBOL, threshold_pct=-5.0, days=30, db_engine=engine
        )
        assert result is True

    @patch(
        "src.strategy.conditions_earnings._ist_today",
        return_value=_TODAY,
    )
    def test_negative_earnings_surprise_false_when_no_qualifying_row(self, mock_today):
        """Should return False when no qualifying negative surprise found."""
        engine = _engine_with_surprise_row(has_surprise=False)
        result = negative_earnings_surprise(
            SYMBOL, threshold_pct=-5.0, days=30, db_engine=engine
        )
        assert result is False

    def test_negative_earnings_surprise_returns_false_on_exception(self):
        """Should return False (not raise) on DB error."""
        engine = MagicMock()
        engine.connect.side_effect = Exception("DB error")
        result = negative_earnings_surprise(SYMBOL, db_engine=engine)
        assert result is False
