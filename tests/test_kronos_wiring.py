"""Smoke tests for Phase G Kronos wiring into src/main.py and src/strategy/engine.py.

Verifies:
1. TradingBot declares kronos_forecaster and kronos_validator attributes after __init__.
2. _get_recent_bars returns None gracefully when db_engine is None.
3. StrategyEngine._build_kronos_dimension returns None when no forecaster is wired.
4. set_wave5_modules accepts new kronos_forecaster and db_engine kwargs.

No real DB / HuggingFace model / WebSocket connections used.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_settings() -> MagicMock:
    s = MagicMock()
    s.capital = 50_000_000  # ₹5L in paisa
    s.max_daily_loss = 0.03
    s.max_open_positions = 5
    s.max_position_size_pct = 0.20
    s.max_capital_deployment_pct = 0.80
    s.consecutive_loss_pause = 3
    s.pause_minutes = 30
    s.trading_mode = "paper"
    s.slippage_pct = 0.0005
    s.log_level = "DEBUG"
    s.log_file = "logs/test.log"
    s.agent_pipeline_enabled = False
    return s


# ---------------------------------------------------------------------------
# Test 1 — TradingBot declares Kronos attributes at __init__
# ---------------------------------------------------------------------------


def test_main_imports_with_kronos_modules() -> None:
    """TradingBot must declare kronos_forecaster and kronos_validator after __init__.

    Both attributes are allowed to be None (init doesn't load weights).
    """
    from src.main import TradingBot

    mock_settings = _make_mock_settings()

    with patch("src.main.get_settings", return_value=mock_settings):
        tb = TradingBot()

    assert hasattr(tb, "kronos_forecaster"), "TradingBot missing attribute: kronos_forecaster"
    assert hasattr(tb, "kronos_validator"), "TradingBot missing attribute: kronos_validator"
    # Both None before startup() is called
    assert tb.kronos_forecaster is None
    assert tb.kronos_validator is None


# ---------------------------------------------------------------------------
# Test 2 — _get_recent_bars returns None without db_engine
# ---------------------------------------------------------------------------


def test_get_recent_bars_returns_none_without_engine() -> None:
    """_get_recent_bars must return None gracefully when db_engine is None."""
    from src.main import TradingBot

    mock_settings = _make_mock_settings()

    with patch("src.main.get_settings", return_value=mock_settings):
        tb = TradingBot()

    # db_engine is None by default (no startup)
    assert tb.db_engine is None

    result = tb._get_recent_bars("NSE_INDEX|Nifty Bank", timeframe="5m", count=400)
    assert result is None


# ---------------------------------------------------------------------------
# Test 3 — _build_kronos_dimension returns None when no forecaster
# ---------------------------------------------------------------------------


def test_strategy_engine_build_kronos_dimension_returns_none_when_no_engine() -> None:
    """_build_kronos_dimension must return None when neither forecaster nor db_engine is wired."""
    from src.strategy.engine import StrategyEngine, Signal, SignalType

    engine = StrategyEngine(
        strategies_dir="config/strategies",
        bars_provider=MagicMock(return_value={}),
    )

    # No wave5 injection — _kronos_forecaster and _db_engine do not exist yet
    mock_signal = MagicMock(spec=Signal)
    mock_signal.instrument_key = "NSE_INDEX|Nifty Bank"
    mock_signal.signal_type = SignalType.BUY_CE

    result = engine._build_kronos_dimension(mock_signal)
    assert result is None


# ---------------------------------------------------------------------------
# Test 4 — set_wave5_modules accepts kronos_forecaster and db_engine
# ---------------------------------------------------------------------------


def test_strategy_engine_set_wave5_accepts_kronos_kwargs() -> None:
    """set_wave5_modules must accept and store kronos_forecaster and db_engine."""
    from src.strategy.engine import StrategyEngine

    engine = StrategyEngine(
        strategies_dir="config/strategies",
        bars_provider=MagicMock(return_value={}),
    )

    mock_forecaster = MagicMock()
    mock_db = MagicMock()

    engine.set_wave5_modules(
        confluence_engine=None,
        rejection_filter=None,
        regime_router=None,
        kronos_forecaster=mock_forecaster,
        db_engine=mock_db,
    )

    assert engine._kronos_forecaster is mock_forecaster
    assert engine._db_engine is mock_db
