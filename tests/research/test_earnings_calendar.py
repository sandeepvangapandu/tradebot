"""Tests for src.research.earnings_calendar — EarningsCalendar query interface.

All tests use inline mocks (unittest.mock) and do NOT require a live database.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.research.earnings_calendar import EarningsCalendar


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SYMBOL = "RELIANCE"
_EARNINGS_DATE = date(2026, 5, 10)
_TODAY = date(2026, 5, 8)  # Simulate today = T (earnings on T+2)


def _engine_returning(rows) -> MagicMock:
    """Build a mock SQLAlchemy engine that returns ``rows`` from fetchall/fetchone."""
    result = MagicMock()
    if isinstance(rows, list):
        result.fetchall.return_value = rows
        result.fetchone.return_value = rows[0] if rows else None
    else:
        result.fetchone.return_value = rows
        result.fetchall.return_value = [rows] if rows is not None else []

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value = result

    engine = MagicMock()
    engine.connect.return_value = mock_conn
    return engine


# Rows match the SELECT column order:
# symbol, earnings_date, fiscal_quarter, reporting_time,
# expected_eps, actual_eps, surprise_pct, source
_ROW_FUTURE = (
    SYMBOL, _EARNINGS_DATE, "Q4 FY26", "BMO", 25.0, None, None, "MONEYCONTROL"
)
_ROW_PAST_WITH_ACTUAL = (
    SYMBOL, date(2026, 2, 8), "Q3 FY26", "AMC", 22.0, 25.0, 13.64, "MONEYCONTROL"
)
_ROW_PAST_BIG_POSITIVE = (
    SYMBOL, date(2026, 4, 20), "Q4 FY26", "BMO", 25.0, 30.0, 20.0, "MONEYCONTROL"
)
_ROW_PAST_NEGATIVE = (
    SYMBOL, date(2026, 3, 15), "Q3 FY26", "AMC", 22.0, 18.0, -18.18, "MONEYCONTROL"
)


# is_blackout query returns (earnings_date, fiscal_quarter) tuples
_BLACKOUT_ROW = (_EARNINGS_DATE, "Q4 FY26")

# last_surprise query returns (earnings_date, surprise_pct, expected_eps, actual_eps)
_SURPRISE_ROW = (date(2026, 2, 8), 13.64, 22.0, 25.0)


# ---------------------------------------------------------------------------
# get_earnings tests
# ---------------------------------------------------------------------------

class TestGetEarnings:
    """Tests for EarningsCalendar.get_earnings."""

    def test_get_earnings_returns_empty_when_no_engine(self):
        """Should return [] gracefully when db_engine is None."""
        cal = EarningsCalendar(db_engine=None)
        assert cal.get_earnings(SYMBOL) == []

    def test_get_earnings_filters_by_date_range(self):
        """Should include rows within [from_date, to_date] and map to dicts."""
        engine = _engine_returning([_ROW_FUTURE])
        cal = EarningsCalendar(db_engine=engine)

        results = cal.get_earnings(
            SYMBOL,
            from_date=date(2026, 5, 1),
            to_date=date(2026, 5, 31),
        )

        assert len(results) == 1
        r = results[0]
        assert r["symbol"] == SYMBOL
        assert r["earnings_date"] == _EARNINGS_DATE
        assert r["fiscal_quarter"] == "Q4 FY26"
        assert r["reporting_time"] == "BMO"

    def test_get_earnings_returns_empty_on_db_error(self):
        """Should return [] (not raise) when DB throws."""
        engine = MagicMock()
        engine.connect.side_effect = Exception("DB down")
        cal = EarningsCalendar(db_engine=engine)
        assert cal.get_earnings(SYMBOL) == []


# ---------------------------------------------------------------------------
# is_blackout tests
# ---------------------------------------------------------------------------

class TestIsBlackout:
    """Tests for EarningsCalendar.is_blackout."""

    def test_is_blackout_returns_false_when_no_engine(self):
        """Should return (False, None) when db_engine is None."""
        cal = EarningsCalendar(db_engine=None)
        blocked, reason = cal.is_blackout(SYMBOL)
        assert blocked is False
        assert reason is None

    def test_is_blackout_true_one_day_before(self):
        """T-1: trade date one day before earnings should be blocked."""
        engine = _engine_returning(_BLACKOUT_ROW)
        cal = EarningsCalendar(db_engine=engine)

        trade_date = _EARNINGS_DATE - timedelta(days=1)  # T-1
        blocked, reason = cal.is_blackout(SYMBOL, trade_date=trade_date)

        assert blocked is True
        assert reason is not None
        assert "2026-05-10" in reason

    def test_is_blackout_true_on_earnings_date(self):
        """T+0: trade date equal to earnings date should be blocked."""
        engine = _engine_returning(_BLACKOUT_ROW)
        cal = EarningsCalendar(db_engine=engine)

        blocked, reason = cal.is_blackout(SYMBOL, trade_date=_EARNINGS_DATE)
        assert blocked is True
        assert reason is not None

    def test_is_blackout_true_one_day_after(self):
        """T+1: one day after earnings should still be blocked."""
        engine = _engine_returning(_BLACKOUT_ROW)
        cal = EarningsCalendar(db_engine=engine)

        trade_date = _EARNINGS_DATE + timedelta(days=1)  # T+1
        blocked, reason = cal.is_blackout(SYMBOL, trade_date=trade_date)
        assert blocked is True

    def test_is_blackout_false_three_days_after(self):
        """T+3: three days after earnings should NOT be blocked (outside window)."""
        # The DB query would return no row for a date outside the window
        engine = _engine_returning(None)  # no row found
        cal = EarningsCalendar(db_engine=engine)

        trade_date = _EARNINGS_DATE + timedelta(days=3)
        blocked, reason = cal.is_blackout(SYMBOL, trade_date=trade_date)
        assert blocked is False
        assert reason is None

    def test_is_blackout_reason_contains_fiscal_quarter(self):
        """Reason string should mention the fiscal quarter when available."""
        engine = _engine_returning(_BLACKOUT_ROW)
        cal = EarningsCalendar(db_engine=engine)

        blocked, reason = cal.is_blackout(SYMBOL, trade_date=_EARNINGS_DATE)
        assert blocked is True
        assert "Q4 FY26" in reason

    def test_is_blackout_returns_false_on_db_error(self):
        """Should return (False, None) (not raise) when DB throws."""
        engine = MagicMock()
        engine.connect.side_effect = Exception("DB down")
        cal = EarningsCalendar(db_engine=engine)
        blocked, reason = cal.is_blackout(SYMBOL)
        assert blocked is False
        assert reason is None


# ---------------------------------------------------------------------------
# upcoming_earnings tests
# ---------------------------------------------------------------------------

class TestUpcomingEarnings:
    """Tests for EarningsCalendar.upcoming_earnings."""

    def test_upcoming_earnings_returns_none_when_no_engine(self):
        """Should return None gracefully when db_engine is None."""
        cal = EarningsCalendar(db_engine=None)
        assert cal.upcoming_earnings(SYMBOL) is None

    @patch("src.research.earnings_calendar._ist_today", return_value=_TODAY)
    def test_upcoming_earnings_returns_nearest_future(self, mock_today):
        """Should return the nearest future earnings row."""
        engine = _engine_returning(_ROW_FUTURE)
        cal = EarningsCalendar(db_engine=engine)

        result = cal.upcoming_earnings(SYMBOL, days_ahead=7)

        assert result is not None
        assert result["symbol"] == SYMBOL
        assert result["earnings_date"] == _EARNINGS_DATE
        assert result["fiscal_quarter"] == "Q4 FY26"

    @patch("src.research.earnings_calendar._ist_today", return_value=_TODAY)
    def test_upcoming_earnings_returns_none_when_no_upcoming(self, mock_today):
        """Should return None when no earnings found within window."""
        engine = _engine_returning(None)
        cal = EarningsCalendar(db_engine=engine)

        result = cal.upcoming_earnings(SYMBOL, days_ahead=7)
        assert result is None


# ---------------------------------------------------------------------------
# last_surprise tests
# ---------------------------------------------------------------------------

class TestLastSurprise:
    """Tests for EarningsCalendar.last_surprise."""

    def test_last_surprise_returns_none_when_no_engine(self):
        """Should return None gracefully when db_engine is None."""
        cal = EarningsCalendar(db_engine=None)
        assert cal.last_surprise(SYMBOL) is None

    @patch("src.research.earnings_calendar._ist_today", return_value=_TODAY)
    def test_last_surprise_returns_most_recent(self, mock_today):
        """Should return the most recent past earnings with actual_eps."""
        engine = _engine_returning(_SURPRISE_ROW)
        cal = EarningsCalendar(db_engine=engine)

        result = cal.last_surprise(SYMBOL)

        assert result is not None
        assert result["date"] == date(2026, 2, 8)
        assert abs(result["surprise_pct"] - 13.64) < 0.01
        assert result["expected_eps"] == 22.0
        assert result["actual_eps"] == 25.0

    @patch("src.research.earnings_calendar._ist_today", return_value=_TODAY)
    def test_last_surprise_returns_none_when_no_past_results(self, mock_today):
        """Should return None when there are no past results with actual_eps."""
        engine = _engine_returning(None)
        cal = EarningsCalendar(db_engine=engine)
        assert cal.last_surprise(SYMBOL) is None


# ---------------------------------------------------------------------------
# positive_surprise_recent tests
# ---------------------------------------------------------------------------

class TestPositiveSurpriseRecent:
    """Tests for EarningsCalendar.positive_surprise_recent."""

    def test_positive_surprise_recent_returns_false_when_no_engine(self):
        """Should return False gracefully when db_engine is None."""
        cal = EarningsCalendar(db_engine=None)
        assert cal.positive_surprise_recent(SYMBOL) is False

    @patch("src.research.earnings_calendar._ist_today", return_value=_TODAY)
    def test_positive_surprise_recent_true_when_above_threshold(self, mock_today):
        """Should return True when a recent positive surprise exceeds threshold."""
        # fetchone returns a truthy row → positive surprise found
        engine = _engine_returning((1,))
        cal = EarningsCalendar(db_engine=engine)

        result = cal.positive_surprise_recent(
            SYMBOL, threshold_pct=5.0, days=30
        )
        assert result is True

    @patch("src.research.earnings_calendar._ist_today", return_value=_TODAY)
    def test_positive_surprise_recent_false_when_no_row(self, mock_today):
        """Should return False when no qualifying surprise found."""
        engine = _engine_returning(None)
        cal = EarningsCalendar(db_engine=engine)

        result = cal.positive_surprise_recent(
            SYMBOL, threshold_pct=5.0, days=30
        )
        assert result is False

    @patch("src.research.earnings_calendar._ist_today", return_value=_TODAY)
    def test_positive_surprise_recent_returns_false_on_db_error(self, mock_today):
        """Should return False (not raise) when DB throws."""
        engine = MagicMock()
        engine.connect.side_effect = Exception("DB down")
        cal = EarningsCalendar(db_engine=engine)
        assert cal.positive_surprise_recent(SYMBOL) is False
