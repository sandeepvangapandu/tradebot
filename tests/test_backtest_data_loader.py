"""Tests for the backtest data loader module."""

import pandas as pd
import pytest
from datetime import date

from src.backtest.data_loader import BacktestDataLoader


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample OHLCV CSV file."""
    data = {
        "timestamp": pd.date_range("2025-01-01 09:15", periods=100, freq="5min"),
        "open": [100_00] * 100,  # paisa
        "high": [101_00] * 100,
        "low": [99_00] * 100,
        "close": [100_50] * 100,
        "volume": [1000] * 100,
    }
    df = pd.DataFrame(data)
    path = tmp_path / "BANKNIFTY_5min.csv"
    df.to_csv(path, index=False)
    return path


def test_load_csv(sample_csv):
    loader = BacktestDataLoader()
    df = loader.load_csv(str(sample_csv))
    assert len(df) == 100
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "timestamp"


def test_load_csv_date_filter(sample_csv):
    loader = BacktestDataLoader()
    df = loader.load_csv(
        str(sample_csv),
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 1),
    )
    assert len(df) > 0
    assert all(d.date() == date(2025, 1, 1) for d in df.index)


def test_resample_timeframe(sample_csv):
    loader = BacktestDataLoader()
    df = loader.load_csv(str(sample_csv))
    resampled = loader.resample(df, "15min")
    # 100 bars of 5min -> ~34 bars of 15min
    assert len(resampled) < len(df)
    assert len(resampled) > 0


def test_load_csv_file_not_found():
    loader = BacktestDataLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_csv("/nonexistent/file.csv")


def test_load_csv_missing_column(tmp_path):
    """Test that missing columns raise ValueError."""
    data = {
        "timestamp": pd.date_range("2025-01-01 09:15", periods=10, freq="5min"),
        "open": [100_00] * 10,
        "high": [101_00] * 10,
        # Missing 'low', 'close', 'volume'
    }
    df = pd.DataFrame(data)
    path = tmp_path / "incomplete.csv"
    df.to_csv(path, index=False)

    loader = BacktestDataLoader()
    with pytest.raises(ValueError):
        loader.load_csv(str(path))
