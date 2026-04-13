"""Tests for StrategyEngine.evaluate_bar_sync (synchronous backtest path)."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.strategy.engine import StrategyEngine

IST = ZoneInfo("Asia/Kolkata")

SIMPLE_STRATEGY = {
    "name": "Test_EMA_Cross",
    "enabled": True,
    "underlying": {"instrument_key": "NSE_INDEX|Nifty Bank"},
    "trading_hours": {
        "start_time": "09:15:00",
        "end_time": "15:15:00",
        "days": [0, 1, 2, 3, 4],
    },
    "timeframes": {"primary": "5min"},
    "entry_sets": [
        {
            "name": "long",
            "signal": "BUY",
            "conditions": [
                {
                    "indicator": "EMA",
                    "comparison": ">",
                    "against": "EMA",
                    "parameters": {"fast_period": 9, "slow_period": 21},
                    "timeframe": "5min",
                }
            ],
        }
    ],
    "exit_rules": {"stop_loss_pct": 1.0, "target_pct": 2.0},
    "position_sizing": {"method": "fixed_quantity", "quantity": 1},
    "risk_management": {},
    "params": {"use_enhanced_filters": False},
}


@pytest.fixture
def strategy_dir(tmp_path):
    strat_file = tmp_path / "test_strategy.json"
    strat_file.write_text(json.dumps(SIMPLE_STRATEGY))
    return tmp_path


@pytest.fixture
def flat_bars() -> dict[str, pd.DataFrame]:
    """40 bars of flat price — EMA9 == EMA21, no signal expected."""
    idx = pd.date_range("2026-01-06 09:15", periods=40, freq="5min", tz=IST)
    df = pd.DataFrame(
        {"open": [5000]*40, "high": [5010]*40, "low": [4990]*40,
         "close": [5000]*40, "volume": [1000]*40},
        index=idx,
    )
    return {"NSE_INDEX|Nifty Bank": df}


def test_evaluate_bar_sync_returns_list(strategy_dir, flat_bars):
    """evaluate_bar_sync must return a list (possibly empty)."""
    engine = StrategyEngine(
        strategies_dir=str(strategy_dir),
        bars_provider=lambda: flat_bars,
    )
    engine.load_strategies()

    bar_time = datetime(2026, 1, 6, 10, 0, tzinfo=IST)
    signals = engine.evaluate_bar_sync(flat_bars, bar_time)
    assert isinstance(signals, list)


def test_evaluate_bar_sync_respects_trading_hours(strategy_dir, flat_bars):
    """No signals before market open."""
    engine = StrategyEngine(
        strategies_dir=str(strategy_dir),
        bars_provider=lambda: flat_bars,
    )
    engine.load_strategies()

    # 08:00 — before market open
    bar_time = datetime(2026, 1, 6, 8, 0, tzinfo=IST)
    signals = engine.evaluate_bar_sync(flat_bars, bar_time)
    assert signals == [], "No signals expected before market open"


def test_evaluate_bar_sync_skips_disabled_strategy(tmp_path, flat_bars):
    """Disabled strategy must not generate signals."""
    disabled = {**SIMPLE_STRATEGY, "enabled": False}
    (tmp_path / "disabled.json").write_text(json.dumps(disabled))
    engine = StrategyEngine(
        strategies_dir=str(tmp_path),
        bars_provider=lambda: flat_bars,
    )
    engine.load_strategies()
    bar_time = datetime(2026, 1, 6, 10, 0, tzinfo=IST)
    signals = engine.evaluate_bar_sync(flat_bars, bar_time)
    assert signals == []
