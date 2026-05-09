"""Tests for src.strategy.conditions_macro.

All tests use inline mocks — no external DB or network required.

Coverage targets:
  - test_macro_trend_up_true_when_classified
  - test_macro_zscore_above_threshold
  - test_cross_asset_bullish_for_TCS_with_USDINR_UP
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.strategy.conditions_macro import (
    cross_asset_bearish,
    cross_asset_bullish,
    macro_trend_down,
    macro_trend_up,
    macro_zscore_above,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_engine_with_trend(symbol: str, trend: str) -> MagicMock:
    """Return a mock SQLAlchemy engine whose query returns *trend* for *symbol*."""
    engine = MagicMock()
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchone.return_value = (trend,)

    mock_conn.execute.return_value = mock_result
    engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    return engine


def _mock_engine_with_zscore(zscore: float) -> MagicMock:
    """Return a mock SQLAlchemy engine whose zscore query returns *zscore*."""
    engine = MagicMock()
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchone.return_value = (zscore,)

    mock_conn.execute.return_value = mock_result
    engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    return engine


def _mock_engine_returning_none() -> MagicMock:
    """Return a mock engine where fetchone returns None (no data)."""
    engine = MagicMock()
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchone.return_value = None

    mock_conn.execute.return_value = mock_result
    engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    return engine


# ---------------------------------------------------------------------------
# macro_trend_up
# ---------------------------------------------------------------------------

class TestMacroTrendUp:
    def test_macro_trend_up_true_when_classified(self):
        """macro_trend_up returns True when DB returns 'UP' for the symbol."""
        engine = _mock_engine_with_trend("USDINR", "UP")
        assert macro_trend_up("USDINR", db_engine=engine) is True

    def test_macro_trend_up_false_when_trend_is_down(self):
        """macro_trend_up returns False when DB returns 'DOWN'."""
        engine = _mock_engine_with_trend("USDINR", "DOWN")
        assert macro_trend_up("USDINR", db_engine=engine) is False

    def test_macro_trend_up_false_when_trend_is_range(self):
        """macro_trend_up returns False when DB returns 'RANGE'."""
        engine = _mock_engine_with_trend("CRUDE", "RANGE")
        assert macro_trend_up("CRUDE", db_engine=engine) is False

    def test_macro_trend_up_false_when_no_data(self):
        """macro_trend_up returns False when DB has no row for the symbol."""
        engine = _mock_engine_returning_none()
        assert macro_trend_up("GOLD", db_engine=engine) is False

    def test_macro_trend_up_false_when_no_engine(self):
        """macro_trend_up returns False when db_engine is None."""
        assert macro_trend_up("USDINR", db_engine=None) is False

    def test_macro_trend_up_for_crude(self):
        """macro_trend_up works for CRUDE symbol."""
        engine = _mock_engine_with_trend("CRUDE", "UP")
        assert macro_trend_up("CRUDE", db_engine=engine) is True

    def test_macro_trend_up_for_gold(self):
        """macro_trend_up works for GOLD symbol."""
        engine = _mock_engine_with_trend("GOLD", "UP")
        assert macro_trend_up("GOLD", db_engine=engine) is True

    def test_macro_trend_up_for_silver(self):
        """macro_trend_up works for SILVER symbol."""
        engine = _mock_engine_with_trend("SILVER", "UP")
        assert macro_trend_up("SILVER", db_engine=engine) is True


# ---------------------------------------------------------------------------
# macro_trend_down
# ---------------------------------------------------------------------------

class TestMacroTrendDown:
    def test_macro_trend_down_true_when_classified(self):
        """macro_trend_down returns True when DB returns 'DOWN'."""
        engine = _mock_engine_with_trend("CRUDE", "DOWN")
        assert macro_trend_down("CRUDE", db_engine=engine) is True

    def test_macro_trend_down_false_when_trend_is_up(self):
        """macro_trend_down returns False when trend is 'UP'."""
        engine = _mock_engine_with_trend("USDINR", "UP")
        assert macro_trend_down("USDINR", db_engine=engine) is False

    def test_macro_trend_down_false_when_no_engine(self):
        """macro_trend_down returns False when db_engine is None."""
        assert macro_trend_down("CRUDE", db_engine=None) is False


# ---------------------------------------------------------------------------
# macro_zscore_above
# ---------------------------------------------------------------------------

class TestMacroZscoreAbove:
    def test_macro_zscore_above_threshold(self):
        """macro_zscore_above returns True when zscore > threshold."""
        engine = _mock_engine_with_zscore(1.8)
        assert macro_zscore_above("USDINR", threshold=1.5, db_engine=engine) is True

    def test_macro_zscore_above_false_when_below_threshold(self):
        """macro_zscore_above returns False when zscore < threshold."""
        engine = _mock_engine_with_zscore(0.3)
        assert macro_zscore_above("USDINR", threshold=1.0, db_engine=engine) is False

    def test_macro_zscore_above_false_at_exact_threshold(self):
        """macro_zscore_above returns False when zscore == threshold (strict >)."""
        engine = _mock_engine_with_zscore(1.0)
        assert macro_zscore_above("CRUDE", threshold=1.0, db_engine=engine) is False

    def test_macro_zscore_above_false_when_no_data(self):
        """macro_zscore_above returns False when DB has no zscore data."""
        engine = _mock_engine_returning_none()
        assert macro_zscore_above("GOLD", threshold=0.5, db_engine=engine) is False

    def test_macro_zscore_above_false_when_no_engine(self):
        """macro_zscore_above returns False when db_engine is None."""
        assert macro_zscore_above("USDINR", threshold=0.5, db_engine=None) is False

    def test_macro_zscore_above_negative_threshold(self):
        """macro_zscore_above with negative threshold (test zscore > -0.5 e.g.)."""
        engine = _mock_engine_with_zscore(0.2)
        assert macro_zscore_above("CRUDE", threshold=-0.5, db_engine=engine) is True


# ---------------------------------------------------------------------------
# cross_asset_bullish / cross_asset_bearish
# ---------------------------------------------------------------------------

class TestCrossAssetBullish:
    def test_cross_asset_bullish_for_TCS_with_USDINR_UP(self):
        """cross_asset_bullish returns True for TCS when USDINR regime is UP."""
        import pandas as pd
        from src.research.macro_overlay import MacroOverlay

        regime_df = pd.DataFrame({
            "symbol": ["USDINR", "CRUDE", "GOLD", "SILVER"],
            "trend": ["UP", "RANGE", "RANGE", "RANGE"],
            "zscore_20d": [1.2, 0.0, 0.0, 0.0],
            "return_pct_5d": [2.5, 0.0, 0.0, 0.0],
        })

        with patch.object(MacroOverlay, "get_today_regime", return_value=regime_df):
            result = cross_asset_bullish("TCS", db_engine=MagicMock())
        assert result is True

    def test_cross_asset_bullish_false_for_TCS_when_USDINR_DOWN(self):
        """cross_asset_bullish returns False for TCS when USDINR is DOWN (not UP)."""
        import pandas as pd
        from src.research.macro_overlay import MacroOverlay

        regime_df = pd.DataFrame({
            "symbol": ["USDINR", "CRUDE", "GOLD", "SILVER"],
            "trend": ["DOWN", "RANGE", "RANGE", "RANGE"],
            "zscore_20d": [-1.0, 0.0, 0.0, 0.0],
            "return_pct_5d": [-2.0, 0.0, 0.0, 0.0],
        })

        with patch.object(MacroOverlay, "get_today_regime", return_value=regime_df):
            result = cross_asset_bullish("TCS", db_engine=MagicMock())
        assert result is False

    def test_cross_asset_bullish_false_for_no_engine(self):
        """cross_asset_bullish returns False when no engine — regime not available."""
        # With no engine, get_today_regime returns empty df → neutral → False
        result = cross_asset_bullish("TCS", db_engine=None)
        assert result is False

    def test_cross_asset_bullish_for_RELIANCE_with_CRUDE_UP(self):
        """cross_asset_bullish returns True for RELIANCE when CRUDE is UP."""
        import pandas as pd
        from src.research.macro_overlay import MacroOverlay

        regime_df = pd.DataFrame({
            "symbol": ["USDINR", "CRUDE", "GOLD", "SILVER"],
            "trend": ["RANGE", "UP", "RANGE", "RANGE"],
            "zscore_20d": [0.0, 1.5, 0.0, 0.0],
            "return_pct_5d": [0.0, 3.0, 0.0, 0.0],
        })

        with patch.object(MacroOverlay, "get_today_regime", return_value=regime_df):
            result = cross_asset_bullish("RELIANCE", db_engine=MagicMock())
        assert result is True


class TestCrossAssetBearish:
    def test_cross_asset_bearish_for_HDFCBANK_when_USDINR_UP(self):
        """cross_asset_bearish returns True for HDFCBANK when USDINR is UP."""
        import pandas as pd
        from src.research.macro_overlay import MacroOverlay

        regime_df = pd.DataFrame({
            "symbol": ["USDINR", "CRUDE", "GOLD", "SILVER"],
            "trend": ["UP", "RANGE", "RANGE", "RANGE"],
            "zscore_20d": [1.0, 0.0, 0.0, 0.0],
            "return_pct_5d": [2.0, 0.0, 0.0, 0.0],
        })

        with patch.object(MacroOverlay, "get_today_regime", return_value=regime_df):
            result = cross_asset_bearish("HDFCBANK", db_engine=MagicMock())
        assert result is True

    def test_cross_asset_bearish_for_HINDUNILVR_when_CRUDE_UP(self):
        """cross_asset_bearish returns True for HINDUNILVR when CRUDE is UP."""
        import pandas as pd
        from src.research.macro_overlay import MacroOverlay

        regime_df = pd.DataFrame({
            "symbol": ["USDINR", "CRUDE", "GOLD", "SILVER"],
            "trend": ["RANGE", "UP", "RANGE", "RANGE"],
            "zscore_20d": [0.0, 1.2, 0.0, 0.0],
            "return_pct_5d": [0.0, 2.5, 0.0, 0.0],
        })

        with patch.object(MacroOverlay, "get_today_regime", return_value=regime_df):
            result = cross_asset_bearish("HINDUNILVR", db_engine=MagicMock())
        assert result is True

    def test_cross_asset_bearish_false_when_no_engine(self):
        """cross_asset_bearish returns False when no engine."""
        result = cross_asset_bearish("HDFCBANK", db_engine=None)
        assert result is False

    def test_cross_asset_bearish_false_for_unknown_stock(self):
        """cross_asset_bearish returns False for an unknown stock symbol."""
        import pandas as pd
        from src.research.macro_overlay import MacroOverlay

        regime_df = pd.DataFrame({
            "symbol": ["USDINR", "CRUDE", "GOLD", "SILVER"],
            "trend": ["UP", "UP", "UP", "UP"],
            "zscore_20d": [1.0, 1.0, 1.0, 1.0],
            "return_pct_5d": [2.0, 2.0, 2.0, 2.0],
        })

        with patch.object(MacroOverlay, "get_today_regime", return_value=regime_df):
            result = cross_asset_bearish("UNKNOWNCORP", db_engine=MagicMock())
        assert result is False
