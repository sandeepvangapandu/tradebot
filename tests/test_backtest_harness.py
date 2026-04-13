"""Integration tests for BacktestHarness using real CSV data.

If nifty50_1min_30days.csv is present, these run as full end-to-end tests.
If not, they are skipped and the init tests (test_backtest_harness_init.py) cover
the basic smoke tests.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

REAL_CSV = Path("nifty50_1min_30days.csv")

MINIMAL_STRATEGY = {
    "name": "Integration_Test",
    "enabled": True,
    "underlying": {"instrument_key": "NSE_INDEX|Nifty 50"},
    "trading_hours": {
        "start_time": "09:15:00",
        "end_time": "15:00:00",
        "days": [0, 1, 2, 3, 4],
    },
    "timeframes": {"primary": "5min"},
    "entry_sets": [
        {
            "name": "long",
            "signal": "BUY",
            "conditions": [
                {
                    "indicator": "RSI",
                    "comparison": "<",
                    "value": 70,
                    "timeframe": "5min",
                    "parameters": {"length": 14},
                }
            ],
        }
    ],
    "exit_rules": {"stop_loss_pct": 1.0, "target_pct": 2.0},
    "position_sizing": {"method": "fixed_quantity", "quantity": 1},
    "risk_management": {},
    "params": {},
}


@pytest.fixture
def strategy_dir(tmp_path):
    (tmp_path / "integration.json").write_text(json.dumps(MINIMAL_STRATEGY))
    return tmp_path


@pytest.mark.skipif(not REAL_CSV.exists(), reason="nifty50_1min_30days.csv not found")
def test_harness_runs_without_crash(strategy_dir):
    """Full backtest run with real data completes without exceptions."""
    from src.backtest.harness import BacktestHarness

    harness = BacktestHarness(
        data_file=str(REAL_CSV),
        instrument_key="NSE_INDEX|Nifty 50",
        strategy_dir=str(strategy_dir),
        capital=100_000_000,
        prices_in_rupees=True,
    )
    results = harness.run()
    assert results is not None
    assert results.bars_processed > 0


@pytest.mark.skipif(not REAL_CSV.exists(), reason="nifty50_1min_30days.csv not found")
def test_harness_trades_have_valid_fields(strategy_dir):
    """All closed trades must have positive entry_price, quantity, and a side."""
    from src.backtest.harness import BacktestHarness

    harness = BacktestHarness(
        data_file=str(REAL_CSV),
        instrument_key="NSE_INDEX|Nifty 50",
        strategy_dir=str(strategy_dir),
        capital=100_000_000,
        prices_in_rupees=True,
    )
    results = harness.run()

    for trade in results.trades:
        assert trade["entry_price"] > 0, f"entry_price must be > 0: {trade}"
        assert trade["quantity"] > 0, f"quantity must be > 0: {trade}"
        assert trade["side"] in ("BUY", "SELL"), f"unexpected side: {trade}"


@pytest.mark.skipif(not REAL_CSV.exists(), reason="nifty50_1min_30days.csv not found")
def test_harness_equity_never_negative(strategy_dir):
    """Equity curve must never go negative (capital floor of 0)."""
    from src.backtest.harness import BacktestHarness

    harness = BacktestHarness(
        data_file=str(REAL_CSV),
        instrument_key="NSE_INDEX|Nifty 50",
        strategy_dir=str(strategy_dir),
        capital=100_000_000,
        prices_in_rupees=True,
    )
    results = harness.run()

    negatives = [e for e in results.equity_curve if e < 0]
    assert not negatives, f"Equity went negative: min={min(results.equity_curve)}"


@pytest.mark.skipif(not REAL_CSV.exists(), reason="nifty50_1min_30days.csv not found")
def test_harness_metrics_keys_present(strategy_dir):
    """HarnessResults.metrics must contain all required keys."""
    from src.backtest.harness import BacktestHarness

    harness = BacktestHarness(
        data_file=str(REAL_CSV),
        instrument_key="NSE_INDEX|Nifty 50",
        strategy_dir=str(strategy_dir),
        capital=100_000_000,
        prices_in_rupees=True,
    )
    results = harness.run()

    required = {
        "total_trades", "win_rate", "profit_factor",
        "net_pnl_rupees", "max_drawdown_rupees", "max_drawdown_pct",
        "return_pct", "sharpe_ratio",
    }
    missing = required - set(results.metrics.keys())
    assert not missing, f"Missing metric keys: {missing}"
