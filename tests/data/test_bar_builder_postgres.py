"""Tests for BarBuilder Postgres persistence.

Verifies that closed bars are written to the ``bars`` table via the injected
SQLAlchemy engine, and that all failure modes are handled gracefully.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, call, patch
from zoneinfo import ZoneInfo

import pytest

from src.data.bar_builder import BarBuilder, OHLCVBar

IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(
    ts: datetime | None = None,
    open_: int = 100_00,
    high: int = 110_00,
    low: int = 95_00,
    close: int = 105_00,
    volume: int = 1000,
) -> OHLCVBar:
    if ts is None:
        ts = datetime(2026, 5, 14, 9, 15, 0, tzinfo=IST)
    return OHLCVBar(timestamp=ts, open=open_, high=high, low=low, close=close, volume=volume)


def _make_mock_engine() -> MagicMock:
    """Return a mock SQLAlchemy engine with a context-manager-aware begin()."""
    engine = MagicMock()
    conn = MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=conn)
    cm.__exit__ = MagicMock(return_value=False)
    engine.begin.return_value = cm
    return engine, conn


# ---------------------------------------------------------------------------
# Test: bar_builder_passes_engine_through_init
# ---------------------------------------------------------------------------

def test_bar_builder_passes_engine_through_init():
    """BarBuilder stores db_engine passed via __init__."""
    mock_engine, _ = _make_mock_engine()
    bb = BarBuilder(db_engine=mock_engine)
    assert bb._db_engine is mock_engine


def test_bar_builder_default_engine_is_none():
    """BarBuilder._db_engine is None when not provided."""
    bb = BarBuilder()
    assert bb._db_engine is None


# ---------------------------------------------------------------------------
# Test: persist_bar_no_engine_silently_skips
# ---------------------------------------------------------------------------

def test_persist_bar_no_engine_silently_skips():
    """_persist_bar_to_postgres is a no-op when _db_engine is None."""
    bb = BarBuilder()
    bar = _make_bar()
    # Must not raise
    bb._persist_bar_to_postgres("NSE_EQ|RELIANCE", 1, bar)


# ---------------------------------------------------------------------------
# Test: persist_bar_writes_to_engine_when_provided
# ---------------------------------------------------------------------------

def test_persist_bar_writes_to_engine_when_provided():
    """_persist_bar_to_postgres executes INSERT with correct params."""
    mock_engine, mock_conn = _make_mock_engine()
    bb = BarBuilder(db_engine=mock_engine)

    ts = datetime(2026, 5, 14, 9, 15, 0, tzinfo=IST)
    bar = _make_bar(ts=ts, open_=500_00, high=510_00, low=495_00, close=505_00, volume=2500)

    bb._persist_bar_to_postgres("NSE_EQ|RELIANCE", 1, bar)

    # engine.begin() must have been called once (context manager)
    mock_engine.begin.assert_called_once()
    # conn.execute must have been called once
    mock_conn.execute.assert_called_once()

    # Inspect the params dict passed to execute
    _sql_arg, params = mock_conn.execute.call_args.args
    assert params["ik"] == "NSE_EQ|RELIANCE"
    assert params["tf"] == "1m"
    assert params["ts"] == ts
    assert params["o"] == 500_00
    assert params["h"] == 510_00
    assert params["l"] == 495_00
    assert params["c"] == 505_00
    assert params["v"] == 2500


# ---------------------------------------------------------------------------
# Test: persist_bar_db_error_does_not_raise
# ---------------------------------------------------------------------------

def test_persist_bar_db_error_does_not_raise():
    """DB failure must not propagate — bot must keep running."""
    mock_engine, mock_conn = _make_mock_engine()
    mock_conn.execute.side_effect = Exception("connection refused")
    bb = BarBuilder(db_engine=mock_engine)

    bar = _make_bar()
    # Must not raise despite DB error
    bb._persist_bar_to_postgres("NSE_EQ|TCS", 5, bar)


# ---------------------------------------------------------------------------
# Test: persist_bar_timeframe_mapping_minutes_to_string
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tf_int, expected_str",
    [
        (1, "1m"),
        (5, "5m"),
        (15, "15m"),
        (30, "30m"),
        (60, "1h"),
        (3, "3m"),   # unknown → f"{tf}m" fallback
        (45, "45m"),
    ],
)
def test_persist_bar_timeframe_mapping_minutes_to_string(tf_int, expected_str):
    """Timeframe int → string mapping is correct for known and unknown values."""
    mock_engine, mock_conn = _make_mock_engine()
    bb = BarBuilder(db_engine=mock_engine)
    bar = _make_bar()

    bb._persist_bar_to_postgres("NSE_EQ|INFY", tf_int, bar)

    _sql_arg, params = mock_conn.execute.call_args.args
    assert params["tf"] == expected_str


# ---------------------------------------------------------------------------
# Test: bar close in _update_bar triggers persist call
# ---------------------------------------------------------------------------

def test_update_bar_calls_persist_on_bar_close():
    """When a bar closes, _persist_bar_to_postgres is invoked."""
    mock_engine, mock_conn = _make_mock_engine()
    bb = BarBuilder(db_engine=mock_engine)

    ts1 = datetime(2026, 5, 14, 9, 15, 30, tzinfo=IST)
    ts2 = datetime(2026, 5, 14, 9, 16, 30, tzinfo=IST)  # crosses 1m boundary

    # First tick — opens bar
    bb._update_bar("NSE_EQ|RELIANCE", 1, 500_00, 100, ts1)
    # Second tick — new minute, closes previous bar
    bb._update_bar("NSE_EQ|RELIANCE", 1, 502_00, 50, ts2)

    # After the second call, one INSERT should have been executed
    assert mock_conn.execute.call_count == 1
    _sql_arg, params = mock_conn.execute.call_args.args
    assert params["ik"] == "NSE_EQ|RELIANCE"
    assert params["tf"] == "1m"
