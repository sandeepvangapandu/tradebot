"""Tests for src/research/corporate_calendar.py.

All database interactions are mocked — no real DB connection required.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.research.corporate_calendar import CorporateCalendar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_action(
    symbol: str = "RELIANCE",
    action_type: str = "DIVIDEND",
    ex_date: date | None = None,
) -> dict:
    """Build a minimal corporate action dict as returned by get_actions_for_symbol."""
    if ex_date is None:
        ex_date = date.today() + timedelta(days=5)
    return {
        "id": 1,
        "symbol": symbol,
        "action_type": action_type,
        "ex_date": ex_date,
        "record_date": ex_date + timedelta(days=1),
        "details": f"{action_type} event",
        "ratio": 5.0 if action_type == "DIVIDEND" else 1.0,
        "source": "NSE",
    }


class MockedCalendar(CorporateCalendar):
    """Subclass that lets us inject a list of actions for testing."""

    def __init__(self, actions: list[dict]):
        super().__init__(db_engine=MagicMock())
        self._actions = actions

    @staticmethod
    def _coerce_date(value) -> date:
        """Coerce string or date to date object for comparison."""
        if isinstance(value, date):
            return value
        from datetime import datetime
        return datetime.strptime(value, "%Y-%m-%d").date()

    def get_actions_for_symbol(self, symbol, from_date=None, to_date=None):
        results = [a for a in self._actions if a["symbol"] == symbol]
        if from_date:
            results = [a for a in results if self._coerce_date(a["ex_date"]) >= from_date]
        if to_date:
            results = [a for a in results if self._coerce_date(a["ex_date"]) <= to_date]
        return results


# ---------------------------------------------------------------------------
# get_actions_for_symbol
# ---------------------------------------------------------------------------

class TestGetActionsForSymbol:
    """Tests for CorporateCalendar.get_actions_for_symbol."""

    def test_returns_empty_when_no_engine(self):
        """Returns empty list when db_engine is None."""
        cal = CorporateCalendar(db_engine=None)
        result = cal.get_actions_for_symbol("RELIANCE")
        assert result == []

    def test_get_actions_for_symbol_filters_by_date_range(self):
        """Only actions within [from_date, to_date] are returned."""
        today = date.today()
        actions = [
            _make_action("RELIANCE", ex_date=today - timedelta(days=10)),  # too old
            _make_action("RELIANCE", ex_date=today + timedelta(days=5)),   # in window
            _make_action("RELIANCE", ex_date=today + timedelta(days=20)),  # too far
        ]
        cal = MockedCalendar(actions)
        result = cal.get_actions_for_symbol(
            "RELIANCE",
            from_date=today - timedelta(days=1),
            to_date=today + timedelta(days=10),
        )
        assert len(result) == 1
        assert result[0]["ex_date"] == today + timedelta(days=5)

    def test_get_actions_for_symbol_no_date_filters_returns_all(self):
        """No date filters → all actions for the symbol are returned."""
        actions = [
            _make_action("RELIANCE", ex_date=date.today() + timedelta(days=5)),
            _make_action("RELIANCE", ex_date=date.today() + timedelta(days=15)),
        ]
        cal = MockedCalendar(actions)
        result = cal.get_actions_for_symbol("RELIANCE")
        assert len(result) == 2

    def test_get_actions_for_symbol_filters_by_symbol(self):
        """Only actions matching the given symbol are returned."""
        actions = [
            _make_action("RELIANCE"),
            _make_action("TCS"),
        ]
        cal = MockedCalendar(actions)
        result = cal.get_actions_for_symbol("TCS")
        assert all(a["symbol"] == "TCS" for a in result)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# is_blackout
# ---------------------------------------------------------------------------

class TestIsBlackout:
    """Tests for CorporateCalendar.is_blackout."""

    def test_is_blackout_returns_false_when_no_engine(self):
        """Returns (False, None) when db_engine is None."""
        cal = CorporateCalendar(db_engine=None)
        result, reason = cal.is_blackout("RELIANCE", date.today())
        assert result is False
        assert reason is None

    def test_is_blackout_true_within_window(self):
        """Returns True when trade_date is within (ex_date ± window)."""
        today = date.today()
        ex_date = today + timedelta(days=1)  # ex_date is tomorrow
        actions = [_make_action("RELIANCE", ex_date=ex_date)]
        cal = MockedCalendar(actions)

        # trade_date = today, days_before=1 → window is [ex_date-1, ex_date+1]
        # today = ex_date - 1 → inside window
        blackout, reason = cal.is_blackout(
            "RELIANCE", today, days_before=1, days_after=1
        )
        assert blackout is True
        assert reason is not None

    def test_is_blackout_false_outside_window(self):
        """Returns False when trade_date is clearly outside the blackout window."""
        today = date.today()
        ex_date = today + timedelta(days=10)  # far future
        actions = [_make_action("RELIANCE", ex_date=ex_date)]
        cal = MockedCalendar(actions)

        blackout, reason = cal.is_blackout(
            "RELIANCE", today, days_before=1, days_after=1
        )
        assert blackout is False
        assert reason is None

    def test_is_blackout_returns_reason_with_action_type(self):
        """Reason string includes the action type and ex_date."""
        today = date.today()
        ex_date = today  # trade on ex_date itself
        actions = [_make_action("RELIANCE", action_type="DIVIDEND", ex_date=ex_date)]
        cal = MockedCalendar(actions)

        blackout, reason = cal.is_blackout(
            "RELIANCE", today, days_before=1, days_after=1
        )
        assert blackout is True
        assert "DIVIDEND" in reason
        assert str(ex_date) in reason

    def test_is_blackout_true_day_after_ex_date(self):
        """day after ex_date is also blocked when days_after=1."""
        today = date.today()
        ex_date = today - timedelta(days=1)  # ex_date was yesterday
        actions = [_make_action("RELIANCE", ex_date=ex_date)]
        cal = MockedCalendar(actions)

        blackout, reason = cal.is_blackout(
            "RELIANCE", today, days_before=1, days_after=1
        )
        assert blackout is True

    def test_is_blackout_string_ex_date_handled(self):
        """ex_date as ISO string is parsed and compared correctly."""
        today = date.today()
        ex_date_str = (today + timedelta(days=1)).isoformat()
        action = _make_action("RELIANCE")
        action["ex_date"] = ex_date_str  # store as string
        cal = MockedCalendar([action])

        blackout, reason = cal.is_blackout(
            "RELIANCE", today, days_before=1, days_after=1
        )
        assert blackout is True


# ---------------------------------------------------------------------------
# upcoming_actions
# ---------------------------------------------------------------------------

class TestUpcomingActions:
    """Tests for CorporateCalendar.upcoming_actions."""

    def test_upcoming_actions_returns_within_lookahead(self):
        """Actions within days_ahead are returned; older ones are excluded."""
        today = date.today()
        actions = [
            _make_action("RELIANCE", ex_date=today + timedelta(days=3)),   # within 7
            _make_action("RELIANCE", ex_date=today + timedelta(days=30)),  # beyond 7
            _make_action("RELIANCE", ex_date=today - timedelta(days=1)),   # past
        ]
        cal = MockedCalendar(actions)
        result = cal.upcoming_actions("RELIANCE", days_ahead=7)
        assert len(result) == 1
        assert result[0]["ex_date"] == today + timedelta(days=3)

    def test_upcoming_actions_empty_when_none_due(self):
        """Returns empty list when no actions are due within the window."""
        actions = [
            _make_action("RELIANCE", ex_date=date.today() + timedelta(days=30)),
        ]
        cal = MockedCalendar(actions)
        result = cal.upcoming_actions("RELIANCE", days_ahead=7)
        assert result == []


# ---------------------------------------------------------------------------
# has_imminent_action
# ---------------------------------------------------------------------------

class TestHasImminentAction:
    """Tests for CorporateCalendar.has_imminent_action."""

    def test_has_imminent_action_returns_false_when_no_engine(self):
        """Returns False when db_engine is None."""
        cal = CorporateCalendar(db_engine=None)
        result = cal.has_imminent_action("RELIANCE", date.today())
        assert result is False

    def test_has_imminent_action_true_when_in_window(self):
        """Returns True when an action is within the look-ahead window."""
        today = date.today()
        actions = [_make_action("RELIANCE", ex_date=today + timedelta(days=2))]
        cal = MockedCalendar(actions)
        assert cal.has_imminent_action("RELIANCE", today, days=3) is True

    def test_has_imminent_action_false_when_beyond_window(self):
        """Returns False when all actions are beyond the look-ahead window."""
        today = date.today()
        actions = [_make_action("RELIANCE", ex_date=today + timedelta(days=10))]
        cal = MockedCalendar(actions)
        assert cal.has_imminent_action("RELIANCE", today, days=3) is False

    def test_has_imminent_action_filtered_by_action_type(self):
        """Only the specified action_types are considered."""
        today = date.today()
        actions = [
            _make_action("RELIANCE", action_type="DIVIDEND", ex_date=today + timedelta(days=1)),
            _make_action("RELIANCE", action_type="SPLIT", ex_date=today + timedelta(days=2)),
        ]
        cal = MockedCalendar(actions)

        # Looking for SPLIT only — should find the split action
        assert cal.has_imminent_action("RELIANCE", today, action_types=["SPLIT"], days=5) is True

        # Looking for BONUS only — no bonus action exists
        assert cal.has_imminent_action("RELIANCE", today, action_types=["BONUS"], days=5) is False

    def test_has_imminent_action_none_type_matches_all(self):
        """action_types=None matches any action type."""
        today = date.today()
        actions = [_make_action("RELIANCE", action_type="MERGER", ex_date=today + timedelta(days=1))]
        cal = MockedCalendar(actions)
        assert cal.has_imminent_action("RELIANCE", today, action_types=None, days=3) is True
