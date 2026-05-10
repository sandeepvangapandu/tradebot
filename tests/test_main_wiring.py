"""Integration smoke tests for src/main.py Bloomberg-build wiring.

These tests verify:
1. TradingBot can be imported without error.
2. All phase-module attributes are declared on TradingBot instances.
3. The bot still initialises (attribute presence) even when DB is unavailable.
4. Strategy engine receives Wave-5 module injection without error.

No real broker / WebSocket / Postgres connections are used.
"""

from __future__ import annotations

import queue
import threading
from datetime import date, datetime
from typing import Any
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Test 1 — import guard
# ---------------------------------------------------------------------------


def test_main_imports_without_error() -> None:
    """src.main must import cleanly — no top-level exceptions."""
    import src.main  # noqa: F401

    assert hasattr(src.main, "TradingBot")
    assert hasattr(src.main, "main")


# ---------------------------------------------------------------------------
# Test 2 — attribute presence on fresh TradingBot instance
# ---------------------------------------------------------------------------

EXPECTED_PHASE_ATTRS = [
    "db_engine",
    "universe_scanner",
    "sector_rotation",
    "vix_regime",
    "macro_overlay",
    "flow_regime",
    "corporate_calendar",
    "earnings_calendar",
    "news_query",
    "insider_signals",
    "sentiment_classifier",
    "depth_feed",
    "tick_metrics",
    "options_chain",
    "flow_scraper",
    "corp_actions_scraper",
    "earnings_scraper",
    "news_scraper",
    "block_deals_scraper",
    "insider_scraper",
    "volume_profile",
    "confluence_engine",
    "rejection_filter",
    "regime_router",
    "kelly_sizer",
    "portfolio_risk",
    "greek_aggregator",
    "smart_router",
    "order_validator",
    "reconciler",
    "slippage_monitor",
    "archiver",
]


def test_phase_module_attributes_declared_on_init() -> None:
    """All bloomberg-build module attributes must exist after __init__ (None is OK)."""
    from src.main import TradingBot

    # Patch Settings so __init__ doesn't try to read .env
    mock_settings = MagicMock()
    mock_settings.capital = 50000000  # ₹5L in paisa
    mock_settings.max_daily_loss = 0.03
    mock_settings.max_open_positions = 5
    mock_settings.max_position_size_pct = 0.20
    mock_settings.max_capital_deployment_pct = 0.80
    mock_settings.consecutive_loss_pause = 3
    mock_settings.pause_minutes = 30
    mock_settings.trading_mode = "paper"
    mock_settings.slippage_pct = 0.0005
    mock_settings.log_level = "DEBUG"
    mock_settings.log_file = "logs/test.log"
    mock_settings.agent_pipeline_enabled = False

    with patch("src.main.get_settings", return_value=mock_settings):
        bot = TradingBot()

    for attr in EXPECTED_PHASE_ATTRS:
        assert hasattr(bot, attr), f"TradingBot missing attribute: {attr}"
        # All are None at __init__ time (before startup)
        assert getattr(bot, attr) is None, (
            f"Expected {attr} to be None before startup, got {getattr(bot, attr)!r}"
        )


# ---------------------------------------------------------------------------
# Test 3 — _init_phase_modules graceful degrade when DB unavailable
# ---------------------------------------------------------------------------


def test_init_phase_modules_handles_db_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """When storage DB is unreachable, _init_phase_modules logs warnings and continues.

    All phase module attributes must still exist (set to None or a module
    instance, depending on whether the DB-less path succeeds).
    """
    from src.main import TradingBot

    mock_settings = MagicMock()
    mock_settings.capital = 50000000
    mock_settings.trading_mode = "paper"
    mock_settings.slippage_pct = 0.0005

    with patch("src.main.get_settings", return_value=mock_settings):
        bot = TradingBot()

    # Patch get_sync_engine to raise so DB is unavailable
    def _raise(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("DB unavailable (test)")

    monkeypatch.setattr("src.main._get_sync_engine", _raise, raising=False)
    monkeypatch.setattr("src.main._STORAGE_DB_OK", True, raising=False)

    # Provide minimal stubs so the method doesn't error on attribute access
    bot.instrument_manager = MagicMock()
    bot.paper_broker = MagicMock()
    bot.strategy_engine = None
    bot.market_feed = None
    bot.bar_builder = None
    bot.settings = mock_settings

    # Should NOT raise — all exceptions caught internally
    bot._init_phase_modules(access_token="FAKE_TOKEN_FOR_TEST")

    # After the call, attributes must still exist
    for attr in EXPECTED_PHASE_ATTRS:
        assert hasattr(bot, attr), f"Attribute missing after _init_phase_modules: {attr}"


# ---------------------------------------------------------------------------
# Test 4 — strategy engine wave-5 injection
# ---------------------------------------------------------------------------


def test_strategy_engine_wave5_injection() -> None:
    """set_wave5_modules must succeed and store references."""
    from src.strategy.engine import StrategyEngine

    mock_bars = MagicMock(return_value={})
    engine = StrategyEngine(
        strategies_dir="config/strategies",
        bars_provider=mock_bars,
    )

    mock_confluence = MagicMock()
    mock_rejection = MagicMock()
    mock_regime = MagicMock()

    engine.set_wave5_modules(
        confluence_engine=mock_confluence,
        rejection_filter=mock_rejection,
        regime_router=mock_regime,
    )

    assert engine._wave5_confluence is mock_confluence
    assert engine._wave5_rejection is mock_rejection
    assert engine._wave5_regime is mock_regime


# ---------------------------------------------------------------------------
# Test 5 — _apply_wave5_gates pass-through when modules absent
# ---------------------------------------------------------------------------


def test_apply_wave5_gates_pass_through_without_modules() -> None:
    """_apply_wave5_gates returns True when no Wave-5 modules are set."""
    from src.strategy.engine import StrategyEngine, Signal
    from src.strategy.builder import SignalType

    mock_bars = MagicMock(return_value={})
    engine = StrategyEngine(
        strategies_dir="config/strategies",
        bars_provider=mock_bars,
    )

    # No wave5 modules injected — should still return True (allow through)
    sig = Signal(
        strategy_name="Test_Strategy",
        set_name="entry_1",
        instrument_key="NSE_EQ|RELIANCE",
        signal_type=SignalType.BUY,
    )

    result = engine._apply_wave5_gates(sig)
    assert result is True, "_apply_wave5_gates should pass-through without modules"


# ---------------------------------------------------------------------------
# Test 6 — _apply_wave5_gates blocks when regime router rejects
# ---------------------------------------------------------------------------


def test_apply_wave5_gates_blocks_on_regime_rejection() -> None:
    """_apply_wave5_gates returns False when regime router blocks the strategy type."""
    from src.strategy.engine import StrategyEngine, Signal
    from src.strategy.builder import SignalType

    mock_bars = MagicMock(return_value={})
    engine = StrategyEngine(
        strategies_dir="config/strategies",
        bars_provider=mock_bars,
    )

    mock_regime = MagicMock()
    mock_regime.is_strategy_type_allowed.return_value = False
    mock_regime.get_blocked_reason.return_value = "VIX_SPIKE: trend_following blocked"

    engine.set_wave5_modules(regime_router=mock_regime)

    sig = Signal(
        strategy_name="Test_Strategy",
        set_name="entry_1",
        instrument_key="NSE_EQ|RELIANCE",
        signal_type=SignalType.BUY,
    )

    result = engine._apply_wave5_gates(sig)
    assert result is False, "_apply_wave5_gates should block when regime router rejects"


# ---------------------------------------------------------------------------
# Test 7 — _apply_wave5_gates passes when regime router allows
# ---------------------------------------------------------------------------


def test_apply_wave5_gates_allows_on_regime_permit() -> None:
    """_apply_wave5_gates returns True when regime router allows and no other gates block."""
    from src.strategy.engine import StrategyEngine, Signal
    from src.strategy.builder import SignalType

    mock_bars = MagicMock(return_value={})
    engine = StrategyEngine(
        strategies_dir="config/strategies",
        bars_provider=mock_bars,
    )

    mock_regime = MagicMock()
    mock_regime.is_strategy_type_allowed.return_value = True

    mock_rejection = MagicMock()
    mock_rejection.evaluate.return_value = (True, None, {})

    engine.set_wave5_modules(regime_router=mock_regime, rejection_filter=mock_rejection)

    sig = Signal(
        strategy_name="Test_Strategy",
        set_name="entry_1",
        instrument_key="NSE_EQ|RELIANCE",
        signal_type=SignalType.BUY,
    )

    result = engine._apply_wave5_gates(sig)
    assert result is True, "_apply_wave5_gates should allow when regime permits and rejection passes"


# ---------------------------------------------------------------------------
# Test 8 — _apply_wave5_gates exception safety
# ---------------------------------------------------------------------------


def test_apply_wave5_gates_exception_safe() -> None:
    """Exceptions inside gates must be caught; signal must still flow through."""
    from src.strategy.engine import StrategyEngine, Signal
    from src.strategy.builder import SignalType

    mock_bars = MagicMock(return_value={})
    engine = StrategyEngine(
        strategies_dir="config/strategies",
        bars_provider=mock_bars,
    )

    # Regime gate raises
    mock_regime = MagicMock()
    mock_regime.is_strategy_type_allowed.side_effect = RuntimeError("Boom")

    engine.set_wave5_modules(regime_router=mock_regime)

    sig = Signal(
        strategy_name="Test_Strategy",
        set_name="entry_1",
        instrument_key="NSE_EQ|RELIANCE",
        signal_type=SignalType.BUY,
    )

    # Should NOT raise; exception is caught and signal passes through
    result = engine._apply_wave5_gates(sig)
    assert result is True, "Gate exception must degrade gracefully (signal allowed through)"
