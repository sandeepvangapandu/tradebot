"""Tests that BacktestHarness can be instantiated and wires components correctly."""
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from src.backtest.harness import BacktestHarness, HarnessResults

IST = ZoneInfo("Asia/Kolkata")

MINIMAL_STRATEGY = {
    "name": "Minimal_Test",
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
                    "indicator": "RSI",
                    "comparison": "<",
                    "value": 80,
                    "timeframe": "5min",
                    "parameters": {"length": 14},
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
    (tmp_path / "minimal.json").write_text(json.dumps(MINIMAL_STRATEGY))
    return tmp_path


def _write_minimal_csv(path: Path, periods: int = 10) -> None:
    idx = pd.date_range("2026-01-06 09:15", periods=periods, freq="1min", tz=IST)
    df = pd.DataFrame(
        {"open": [5000000]*periods, "high": [5010000]*periods,
         "low": [4990000]*periods, "close": [5005000]*periods, "volume": [1000]*periods},
        index=idx,
    )
    df.index.name = "timestamp"
    df.to_csv(path)


def test_harness_instantiates(strategy_dir, tmp_path):
    """BacktestHarness can be created without errors."""
    csv_path = tmp_path / "data.csv"
    _write_minimal_csv(csv_path)

    harness = BacktestHarness(
        data_file=str(csv_path),
        instrument_key="NSE_INDEX|Nifty Bank",
        strategy_dir=str(strategy_dir),
        capital=100_000_000,
    )
    assert harness is not None


def test_harness_run_returns_results(strategy_dir, tmp_path):
    """BacktestHarness.run() returns a HarnessResults with equity_curve."""
    csv_path = tmp_path / "data.csv"
    _write_minimal_csv(csv_path, periods=100)

    harness = BacktestHarness(
        data_file=str(csv_path),
        instrument_key="NSE_INDEX|Nifty Bank",
        strategy_dir=str(strategy_dir),
        capital=100_000_000,
    )
    results = harness.run()

    assert isinstance(results, HarnessResults)
    assert len(results.equity_curve) > 0
    assert isinstance(results.metrics, dict)
    assert "win_rate" in results.metrics
