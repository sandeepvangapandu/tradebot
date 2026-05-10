"""Tests for src.strategy.conditions_vix.

Uses inline mocks — no external DB or network required.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.strategy.conditions_vix import (
    vix_above,
    vix_below,
    vix_in_regime,
    vix_percentile_above,
    vix_spike_detected_today,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db_engine_with_daily_row(
    vix_close: float = 15.0,
    regime: str = "NORMAL",
    percentile_60d: float | None = 42.0,
) -> MagicMock:
    """Return a mock engine whose connect() returns the given vix_regime_daily row."""
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_engine.connect.return_value = mock_conn

    # Row layout: trade_date, vix_close, regime, percentile_60d
    mock_conn.execute.return_value.fetchone.return_value = (
        date(2026, 5, 8),
        str(vix_close),
        regime,
        str(percentile_60d) if percentile_60d is not None else None,
    )
    return mock_engine


def _make_db_engine_no_row() -> MagicMock:
    """Return a mock engine whose connect() returns None for fetchone."""
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_engine.connect.return_value = mock_conn
    mock_conn.execute.return_value.fetchone.return_value = None
    return mock_engine


def _make_db_engine_with_spike_row(has_spike: bool) -> MagicMock:
    """Return a mock engine for intraday spike query."""
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_engine.connect.return_value = mock_conn
    mock_conn.execute.return_value.fetchone.return_value = (1,) if has_spike else None
    return mock_engine


# ---------------------------------------------------------------------------
# vix_in_regime
# ---------------------------------------------------------------------------

class TestVixInRegime:
    def test_vix_in_regime_normal_returns_true(self):
        """Returns True when stored regime matches query."""
        engine = _make_db_engine_with_daily_row(regime="NORMAL")
        assert vix_in_regime("NORMAL", db_engine=engine) is True

    def test_vix_in_regime_mismatch_returns_false(self):
        """Returns False when stored regime does not match query."""
        engine = _make_db_engine_with_daily_row(regime="HIGH")
        assert vix_in_regime("NORMAL", db_engine=engine) is False

    def test_vix_in_regime_no_row_returns_false(self):
        """Returns False when no daily row exists for today."""
        engine = _make_db_engine_no_row()
        assert vix_in_regime("NORMAL", db_engine=engine) is False

    def test_vix_in_regime_no_engine_returns_false(self):
        """Returns False when no DB engine is provided."""
        assert vix_in_regime("NORMAL", db_engine=None) is False


# ---------------------------------------------------------------------------
# vix_below
# ---------------------------------------------------------------------------

class TestVixBelow:
    def test_vix_below_threshold_returns_true(self):
        """Returns True when vix_close is strictly below threshold."""
        engine = _make_db_engine_with_daily_row(vix_close=14.0)
        assert vix_below(20.0, db_engine=engine) is True

    def test_vix_below_threshold_at_boundary_returns_false(self):
        """Returns False when vix_close equals threshold (strict <)."""
        engine = _make_db_engine_with_daily_row(vix_close=20.0)
        assert vix_below(20.0, db_engine=engine) is False

    def test_vix_below_threshold_above_returns_false(self):
        """Returns False when vix_close is above threshold."""
        engine = _make_db_engine_with_daily_row(vix_close=25.0)
        assert vix_below(20.0, db_engine=engine) is False

    def test_vix_below_no_data_returns_false(self):
        """Returns False when no DB data is available."""
        assert vix_below(20.0, db_engine=None) is False


# ---------------------------------------------------------------------------
# vix_above
# ---------------------------------------------------------------------------

class TestVixAbove:
    def test_vix_above_threshold_returns_true(self):
        """Returns True when vix_close is strictly above threshold."""
        engine = _make_db_engine_with_daily_row(vix_close=26.0)
        assert vix_above(25.0, db_engine=engine) is True

    def test_vix_above_threshold_at_boundary_returns_false(self):
        """Returns False when vix_close equals threshold (strict >)."""
        engine = _make_db_engine_with_daily_row(vix_close=25.0)
        assert vix_above(25.0, db_engine=engine) is False

    def test_vix_above_below_returns_false(self):
        """Returns False when vix_close is below threshold."""
        engine = _make_db_engine_with_daily_row(vix_close=10.0)
        assert vix_above(15.0, db_engine=engine) is False

    def test_vix_above_no_data_returns_false(self):
        """Returns False when no DB data is available."""
        assert vix_above(15.0, db_engine=None) is False


# ---------------------------------------------------------------------------
# vix_spike_detected_today
# ---------------------------------------------------------------------------

class TestVixSpikeDetectedToday:
    def test_vix_spike_detected_today_true_after_intraday_jump(self):
        """Returns True when a spike row exists for today."""
        engine = _make_db_engine_with_spike_row(has_spike=True)
        assert vix_spike_detected_today(db_engine=engine) is True

    def test_vix_spike_detected_today_false_when_no_spike(self):
        """Returns False when no spike row exists."""
        engine = _make_db_engine_with_spike_row(has_spike=False)
        assert vix_spike_detected_today(db_engine=engine) is False

    def test_vix_spike_detected_today_false_without_engine(self):
        """Returns False when no DB engine is provided."""
        assert vix_spike_detected_today(db_engine=None) is False


# ---------------------------------------------------------------------------
# vix_percentile_above
# ---------------------------------------------------------------------------

class TestVixPercentileAbove:
    def test_vix_percentile_above_threshold(self):
        """Returns True when percentile_60d > threshold."""
        engine = _make_db_engine_with_daily_row(percentile_60d=75.0)
        assert vix_percentile_above(70.0, db_engine=engine) is True

    def test_vix_percentile_at_boundary_returns_false(self):
        """Returns False when percentile_60d == threshold (strict >)."""
        engine = _make_db_engine_with_daily_row(percentile_60d=70.0)
        assert vix_percentile_above(70.0, db_engine=engine) is False

    def test_vix_percentile_below_returns_false(self):
        """Returns False when percentile_60d < threshold."""
        engine = _make_db_engine_with_daily_row(percentile_60d=30.0)
        assert vix_percentile_above(50.0, db_engine=engine) is False

    def test_vix_percentile_none_returns_false(self):
        """Returns False when percentile_60d is None."""
        engine = _make_db_engine_with_daily_row(percentile_60d=None)
        assert vix_percentile_above(50.0, db_engine=engine) is False

    def test_vix_percentile_no_engine_returns_false(self):
        """Returns False when no DB engine is provided."""
        assert vix_percentile_above(50.0, db_engine=None) is False
