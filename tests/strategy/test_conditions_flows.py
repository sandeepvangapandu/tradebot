"""Tests for src.strategy.conditions_flows — FII/DII + yield condition helpers.

All tests use inline mocks.  No live DB connection required.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from src.strategy.conditions_flows import (
    dii_buying,
    fii_buying,
    fii_selling_streak,
    flow_regime_headwind,
    flow_regime_tailwind,
    yield_falling,
    yield_rising,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRADE_DATE = date(2026, 5, 8)


def _flow_engine(fii_net: float | None, dii_net: float | None) -> MagicMock:
    """Build a mock engine returning a single latest-flow row."""
    row = (TRADE_DATE, fii_net, dii_net)
    result = MagicMock()
    result.fetchone.return_value = row

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value = result

    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_conn
    return mock_engine


def _regime_engine(
    fii_streak: int = 0,
    dii_streak: int = 0,
    fii_regime: str = "NEUTRAL",
    dii_regime: str = "NEUTRAL",
    combined_signal: str = "MIXED",
) -> MagicMock:
    """Build a mock engine returning a single latest-regime row."""
    row = (TRADE_DATE, fii_streak, dii_streak, fii_regime, dii_regime, combined_signal)
    result = MagicMock()
    result.fetchone.return_value = row

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value = result

    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_conn
    return mock_engine


def _yield_engine(yield_pct: float | None, trend_5d: str | None) -> MagicMock:
    """Build a mock engine returning a single latest-yield row."""
    row = (TRADE_DATE, yield_pct, trend_5d)
    result = MagicMock()
    result.fetchone.return_value = row

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value = result

    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_conn
    return mock_engine


def _no_data_engine() -> MagicMock:
    """Build a mock engine that returns no rows."""
    result = MagicMock()
    result.fetchone.return_value = None
    result.fetchall.return_value = []

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value = result

    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_conn
    return mock_engine


# ---------------------------------------------------------------------------
# fii_buying tests
# ---------------------------------------------------------------------------

class TestFiiBuying:
    """Tests for fii_buying condition helper."""

    def test_fii_buying_true_when_net_positive(self):
        """fii_buying returns True when FII net > 0."""
        engine = _flow_engine(fii_net=1500.0, dii_net=500.0)
        assert fii_buying(min_net_crore=0, db_engine=engine) is True

    def test_fii_buying_true_when_above_threshold(self):
        """fii_buying returns True when FII net >= min_net_crore."""
        engine = _flow_engine(fii_net=2500.0, dii_net=0.0)
        assert fii_buying(min_net_crore=2000.0, db_engine=engine) is True

    def test_fii_buying_false_when_net_negative(self):
        """fii_buying returns False when FII net < 0."""
        engine = _flow_engine(fii_net=-500.0, dii_net=0.0)
        assert fii_buying(min_net_crore=0, db_engine=engine) is False

    def test_fii_buying_false_when_below_threshold(self):
        """fii_buying returns False when FII net < min_net_crore."""
        engine = _flow_engine(fii_net=1000.0, dii_net=0.0)
        assert fii_buying(min_net_crore=2000.0, db_engine=engine) is False

    def test_fii_buying_false_when_no_engine(self):
        """fii_buying returns False when no db_engine provided."""
        assert fii_buying(db_engine=None) is False

    def test_fii_buying_false_when_no_data(self):
        """fii_buying returns False when DB has no data."""
        assert fii_buying(db_engine=_no_data_engine()) is False


# ---------------------------------------------------------------------------
# fii_selling_streak tests
# ---------------------------------------------------------------------------

class TestFiiSellingStreak:
    """Tests for fii_selling_streak condition helper."""

    def test_fii_selling_streak_true_after_3_consecutive_sell_days(self):
        """fii_selling_streak True when streak is -3 or lower."""
        engine = _regime_engine(fii_streak=-3, combined_signal="MIXED")
        assert fii_selling_streak(min_days=3, db_engine=engine) is True

    def test_fii_selling_streak_true_after_5_consecutive_sell_days(self):
        """Streak of -5 satisfies min_days=3."""
        engine = _regime_engine(fii_streak=-5, combined_signal="HEADWIND")
        assert fii_selling_streak(min_days=3, db_engine=engine) is True

    def test_fii_selling_streak_false_when_streak_is_2(self):
        """Streak of -2 does not satisfy min_days=3."""
        engine = _regime_engine(fii_streak=-2, combined_signal="MIXED")
        assert fii_selling_streak(min_days=3, db_engine=engine) is False

    def test_fii_selling_streak_false_when_fii_buying(self):
        """Positive streak (FII buying) should return False."""
        engine = _regime_engine(fii_streak=3, combined_signal="TAILWIND")
        assert fii_selling_streak(min_days=3, db_engine=engine) is False

    def test_fii_selling_streak_false_when_no_engine(self):
        assert fii_selling_streak(db_engine=None) is False


# ---------------------------------------------------------------------------
# dii_buying tests
# ---------------------------------------------------------------------------

class TestDiiBuying:
    """Tests for dii_buying condition helper."""

    def test_dii_buying_true_when_net_positive(self):
        engine = _flow_engine(fii_net=0.0, dii_net=1200.0)
        assert dii_buying(min_net_crore=0, db_engine=engine) is True

    def test_dii_buying_false_when_net_negative(self):
        engine = _flow_engine(fii_net=0.0, dii_net=-300.0)
        assert dii_buying(min_net_crore=0, db_engine=engine) is False

    def test_dii_buying_false_when_no_engine(self):
        assert dii_buying(db_engine=None) is False


# ---------------------------------------------------------------------------
# flow_regime_tailwind tests
# ---------------------------------------------------------------------------

class TestFlowRegimeTailwind:
    """Tests for flow_regime_tailwind condition helper."""

    def test_flow_regime_tailwind_true_when_both_buy(self):
        """TAILWIND signal should return True."""
        engine = _regime_engine(
            fii_regime="BUY", dii_regime="STRONG_BUY", combined_signal="TAILWIND"
        )
        assert flow_regime_tailwind(db_engine=engine) is True

    def test_flow_regime_tailwind_false_when_headwind(self):
        engine = _regime_engine(
            fii_regime="SELL", dii_regime="SELL", combined_signal="HEADWIND"
        )
        assert flow_regime_tailwind(db_engine=engine) is False

    def test_flow_regime_tailwind_false_when_mixed(self):
        engine = _regime_engine(
            fii_regime="BUY", dii_regime="SELL", combined_signal="MIXED"
        )
        assert flow_regime_tailwind(db_engine=engine) is False

    def test_flow_regime_tailwind_false_when_no_engine(self):
        assert flow_regime_tailwind(db_engine=None) is False


# ---------------------------------------------------------------------------
# flow_regime_headwind tests
# ---------------------------------------------------------------------------

class TestFlowRegimeHeadwind:
    """Tests for flow_regime_headwind condition helper."""

    def test_flow_regime_headwind_true_when_both_sell(self):
        engine = _regime_engine(
            fii_regime="STRONG_SELL", dii_regime="SELL", combined_signal="HEADWIND"
        )
        assert flow_regime_headwind(db_engine=engine) is True

    def test_flow_regime_headwind_false_when_tailwind(self):
        engine = _regime_engine(
            fii_regime="BUY", dii_regime="BUY", combined_signal="TAILWIND"
        )
        assert flow_regime_headwind(db_engine=engine) is False

    def test_flow_regime_headwind_false_when_no_engine(self):
        assert flow_regime_headwind(db_engine=None) is False


# ---------------------------------------------------------------------------
# yield_rising tests
# ---------------------------------------------------------------------------

class TestYieldRising:
    """Tests for yield_rising condition helper."""

    def test_yield_rising_true_when_5d_trend_rising(self):
        """yield_rising returns True when trend_5d is RISING."""
        engine = _yield_engine(yield_pct=7.10, trend_5d="RISING")
        assert yield_rising(db_engine=engine) is True

    def test_yield_rising_false_when_5d_trend_falling(self):
        engine = _yield_engine(yield_pct=6.90, trend_5d="FALLING")
        assert yield_rising(db_engine=engine) is False

    def test_yield_rising_false_when_5d_trend_flat(self):
        engine = _yield_engine(yield_pct=7.00, trend_5d="FLAT")
        assert yield_rising(db_engine=engine) is False

    def test_yield_rising_false_when_no_engine(self):
        assert yield_rising(db_engine=None) is False

    def test_yield_rising_false_when_no_data(self):
        assert yield_rising(db_engine=_no_data_engine()) is False


# ---------------------------------------------------------------------------
# yield_falling tests
# ---------------------------------------------------------------------------

class TestYieldFalling:
    """Tests for yield_falling condition helper."""

    def test_yield_falling_true_when_5d_trend_falling(self):
        engine = _yield_engine(yield_pct=6.90, trend_5d="FALLING")
        assert yield_falling(db_engine=engine) is True

    def test_yield_falling_false_when_5d_trend_rising(self):
        engine = _yield_engine(yield_pct=7.10, trend_5d="RISING")
        assert yield_falling(db_engine=engine) is False

    def test_yield_falling_false_when_no_engine(self):
        assert yield_falling(db_engine=None) is False
