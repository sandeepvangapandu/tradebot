"""Tests for src.strategy.conditions_options.

All tests use inline mocks — no external fixtures or real DB connections required.
SQLAlchemy engines are replaced with MagicMocks that return controlled row data.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call
from zoneinfo import ZoneInfo

import pytest

from src.strategy.conditions_options import (
    pcr_above,
    pcr_below,
    iv_percentile,
    max_oi_strike_near_spot,
    call_writers_dominant,
)

IST = ZoneInfo("Asia/Kolkata")

KEY = "NSE_INDEX|Nifty 50"
EXPIRY = "2026-05-15"


# ---------------------------------------------------------------------------
# Helpers — DB mock builders
# ---------------------------------------------------------------------------

def _mock_engine_with_snapshot(snap: dict | None) -> MagicMock:
    """Return a mock SQLAlchemy engine whose connect() yields one snapshot row."""
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    if snap is None:
        conn.execute.return_value.fetchone.return_value = None
    else:
        # _latest_snapshot fetches these columns in this order:
        # pcr, total_call_oi, total_put_oi, atm_iv_call, atm_iv_put,
        # iv_skew, spot_paisa, max_pain_strike, ts
        row = (
            snap.get("pcr"),
            snap.get("total_call_oi", 0),
            snap.get("total_put_oi", 0),
            snap.get("atm_iv_call"),
            snap.get("atm_iv_put"),
            snap.get("iv_skew"),
            snap.get("spot_paisa", 0),
            snap.get("max_pain_strike"),
            snap.get("ts", datetime.now(IST)),
        )
        conn.execute.return_value.fetchone.return_value = row

    return engine


def _mock_engine_with_iv_history(iv_values: list[float]) -> MagicMock:
    """Return a mock engine whose fetchall() yields (iv, ts) tuples."""
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    now = datetime.now(IST)
    rows = [
        (iv, now - timedelta(days=len(iv_values) - i))
        for i, iv in enumerate(iv_values)
    ]
    conn.execute.return_value.fetchall.return_value = rows
    return engine


def _mock_engine_with_oi_build(ce_build: float, pe_build: float) -> MagicMock:
    """Return a mock engine that simulates OI-build rows for CE and PE."""
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    # call_writers_dominant queries GROUP BY option_type returning (option_type, net_oi_build)
    rows = [("CE", ce_build), ("PE", pe_build)]
    conn.execute.return_value.fetchall.return_value = rows
    return engine


# ---------------------------------------------------------------------------
# pcr_above
# ---------------------------------------------------------------------------

class TestPcrAbove:
    """Tests for pcr_above()."""

    def test_pcr_above_returns_true_when_pcr_exceeds(self):
        """pcr_above returns True when the latest snapshot PCR > threshold."""
        engine = _mock_engine_with_snapshot({"pcr": 1.35})
        assert pcr_above(KEY, threshold=1.20, db_engine=engine) is True

    def test_pcr_above_returns_false_when_pcr_below(self):
        """pcr_above returns False when PCR is below the threshold."""
        engine = _mock_engine_with_snapshot({"pcr": 0.90})
        assert pcr_above(KEY, threshold=1.20, db_engine=engine) is False

    def test_pcr_above_returns_false_when_no_snapshot(self):
        """pcr_above returns False (safe default) when no snapshot exists."""
        engine = _mock_engine_with_snapshot(None)
        assert pcr_above(KEY, threshold=1.20, db_engine=engine) is False

    def test_pcr_above_returns_false_on_db_error(self):
        """pcr_above returns False when the DB raises an exception."""
        engine = MagicMock()
        engine.connect.side_effect = RuntimeError("DB down")
        assert pcr_above(KEY, threshold=1.20, db_engine=engine) is False


# ---------------------------------------------------------------------------
# pcr_below
# ---------------------------------------------------------------------------

class TestPcrBelow:
    """Tests for pcr_below()."""

    def test_pcr_below_returns_true_when_pcr_under_threshold(self):
        """pcr_below returns True when PCR < threshold."""
        engine = _mock_engine_with_snapshot({"pcr": 0.65})
        assert pcr_below(KEY, threshold=0.80, db_engine=engine) is True

    def test_pcr_below_returns_false_when_pcr_above(self):
        """pcr_below returns False when PCR exceeds threshold."""
        engine = _mock_engine_with_snapshot({"pcr": 1.10})
        assert pcr_below(KEY, threshold=0.80, db_engine=engine) is False


# ---------------------------------------------------------------------------
# iv_percentile
# ---------------------------------------------------------------------------

class TestIvPercentile:
    """Tests for iv_percentile()."""

    def test_iv_percentile_low_when_recent_iv_low(self):
        """iv_percentile returns a low value when the most recent IV is the lowest."""
        # IV sequence: high values in the past, low value at end
        iv_values = [0.35, 0.33, 0.30, 0.28, 0.20, 0.15]
        engine = _mock_engine_with_iv_history(iv_values)

        pct = iv_percentile(KEY, lookback_days=60, db_engine=engine)

        # latest IV is 0.15, min is 0.15, max is 0.35 => 0th percentile
        assert math.isfinite(pct), "IV percentile should be a finite number"
        assert pct < 20.0, f"Expected low percentile, got {pct}"

    def test_iv_percentile_high_when_recent_iv_high(self):
        """iv_percentile returns a high value when the most recent IV is the highest."""
        iv_values = [0.15, 0.18, 0.20, 0.25, 0.30, 0.40]
        engine = _mock_engine_with_iv_history(iv_values)

        pct = iv_percentile(KEY, lookback_days=60, db_engine=engine)

        assert pct > 80.0, f"Expected high percentile, got {pct}"

    def test_iv_percentile_returns_nan_on_insufficient_data(self):
        """iv_percentile returns NaN when there is only one data point."""
        engine = _mock_engine_with_iv_history([0.20])
        pct = iv_percentile(KEY, db_engine=engine)
        assert math.isnan(pct)

    def test_iv_percentile_returns_50_when_all_iv_equal(self):
        """iv_percentile returns 50.0 when min == max (no range)."""
        engine = _mock_engine_with_iv_history([0.20, 0.20, 0.20])
        pct = iv_percentile(KEY, db_engine=engine)
        assert pct == 50.0

    def test_iv_percentile_returns_nan_on_db_error(self):
        """iv_percentile returns NaN when the DB raises an exception."""
        engine = MagicMock()
        engine.connect.side_effect = RuntimeError("DB down")
        pct = iv_percentile(KEY, db_engine=engine)
        assert math.isnan(pct)


# ---------------------------------------------------------------------------
# max_oi_strike_near_spot
# ---------------------------------------------------------------------------

class TestMaxOiStrikeNearSpot:
    """Tests for max_oi_strike_near_spot()."""

    def test_max_oi_strike_near_spot_true_within_tolerance(self):
        """Returns True when max-pain strike is within tolerance of spot."""
        # spot = 25000.00, max_pain = 25050 => 0.2% deviation < 0.5% tolerance
        engine = _mock_engine_with_snapshot(
            {"max_pain_strike": 25050.0, "spot_paisa": 2500000}
        )
        # spot_paisa passed directly; snapshot uses max_pain_strike from DB
        result = max_oi_strike_near_spot(
            KEY, spot_paisa=2500000, tolerance_pct=0.5, db_engine=engine
        )
        assert result is True

    def test_max_oi_strike_near_spot_false_outside_tolerance(self):
        """Returns False when max-pain strike is far from spot."""
        # spot = 25000, max_pain = 25200 => 0.8% > 0.5% tolerance
        engine = _mock_engine_with_snapshot(
            {"max_pain_strike": 25200.0, "spot_paisa": 2500000}
        )
        result = max_oi_strike_near_spot(
            KEY, spot_paisa=2500000, tolerance_pct=0.5, db_engine=engine
        )
        assert result is False

    def test_max_oi_strike_near_spot_false_when_no_snapshot(self):
        """Returns False when no snapshot exists."""
        engine = _mock_engine_with_snapshot(None)
        result = max_oi_strike_near_spot(KEY, spot_paisa=2500000, db_engine=engine)
        assert result is False

    def test_max_oi_strike_near_spot_false_when_spot_zero(self):
        """Returns False when spot_paisa is 0 (division guard)."""
        engine = _mock_engine_with_snapshot({"max_pain_strike": 25000.0})
        result = max_oi_strike_near_spot(KEY, spot_paisa=0, db_engine=engine)
        assert result is False


# ---------------------------------------------------------------------------
# call_writers_dominant
# ---------------------------------------------------------------------------

class TestCallWritersDominant:
    """Tests for call_writers_dominant()."""

    def test_call_writers_dominant_true_on_oi_buildup(self):
        """Returns True when net call OI build exceeds net put OI build."""
        # More calls written (bearish) than puts
        engine = _mock_engine_with_oi_build(ce_build=50000, pe_build=20000)
        result = call_writers_dominant(KEY, EXPIRY, lookback_minutes=30, db_engine=engine)
        assert result is True

    def test_call_writers_dominant_false_when_puts_dominant(self):
        """Returns False when net put OI build exceeds net call OI build."""
        engine = _mock_engine_with_oi_build(ce_build=10000, pe_build=40000)
        result = call_writers_dominant(KEY, EXPIRY, lookback_minutes=30, db_engine=engine)
        assert result is False

    def test_call_writers_dominant_false_on_equal_build(self):
        """Returns False when CE and PE build are equal (not strictly greater)."""
        engine = _mock_engine_with_oi_build(ce_build=30000, pe_build=30000)
        result = call_writers_dominant(KEY, EXPIRY, db_engine=engine)
        assert result is False

    def test_call_writers_dominant_false_on_db_error(self):
        """Returns False (safe default) when DB raises an exception."""
        engine = MagicMock()
        engine.connect.side_effect = RuntimeError("DB down")
        result = call_writers_dominant(KEY, EXPIRY, db_engine=engine)
        assert result is False

    def test_call_writers_dominant_false_on_no_rows(self):
        """Returns False when no OI-change rows exist in the lookback window."""
        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        conn.execute.return_value.fetchall.return_value = []

        result = call_writers_dominant(KEY, EXPIRY, db_engine=engine)
        assert result is False
