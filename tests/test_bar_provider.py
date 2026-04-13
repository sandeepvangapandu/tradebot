"""Tests for BacktestBarProvider — no-lookahead OHLCV slicing and resampling."""
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from src.backtest.bar_provider import BacktestBarProvider

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def sample_data() -> dict[str, pd.DataFrame]:
    """60 1-minute bars of OHLCV data in paisa."""
    idx = pd.date_range(
        "2026-01-02 09:15:00", periods=60, freq="1min", tz=IST
    )
    df = pd.DataFrame(
        {
            "open":   [5000000] * 60,
            "high":   [5010000] * 60,
            "low":    [4990000] * 60,
            "close":  [5005000] * 60,
            "volume": [1000] * 60,
        },
        index=idx,
    )
    return {"NSE_INDEX|Nifty Bank": df}


def test_get_bars_no_lookahead(sample_data):
    """Bars returned must not exceed current_index + 1."""
    provider = BacktestBarProvider(sample_data)
    provider.advance("NSE_INDEX|Nifty Bank", 9)  # advance to bar 10 (0-indexed)
    df = provider.get_bars("NSE_INDEX|Nifty Bank", timeframe=1)
    assert len(df) == 10, f"Expected 10 bars, got {len(df)}"


def test_get_bars_resample_5min(sample_data):
    """60 1-min bars resampled to 5min should give 12 bars."""
    provider = BacktestBarProvider(sample_data)
    provider.advance("NSE_INDEX|Nifty Bank", 59)
    df = provider.get_bars("NSE_INDEX|Nifty Bank", timeframe=5)
    assert len(df) == 12, f"Expected 12 bars, got {len(df)}"


def test_get_bars_resample_ohlcv_aggregation(sample_data):
    """5-min bars must aggregate open/high/low/close/volume correctly."""
    provider = BacktestBarProvider(sample_data)
    provider.advance("NSE_INDEX|Nifty Bank", 4)  # first 5 bars
    df = provider.get_bars("NSE_INDEX|Nifty Bank", timeframe=5)
    assert len(df) == 1
    assert df.iloc[0]["open"] == 5000000
    assert df.iloc[0]["high"] == 5010000
    assert df.iloc[0]["low"] == 4990000
    assert df.iloc[0]["close"] == 5005000
    assert df.iloc[0]["volume"] == 5000  # sum of 5 bars


def test_fetch_candles_daily(sample_data):
    """fetch_candles should resample to daily bars."""
    provider = BacktestBarProvider(sample_data)
    provider.advance("NSE_INDEX|Nifty Bank", 59)
    df = provider.fetch_candles("NSE_INDEX|Nifty Bank", interval="day", days=30)
    assert len(df) == 1  # all bars in same day
    assert df.iloc[0]["open"] == 5000000
    assert df.iloc[0]["high"] == 5010000


def test_advance_updates_index(sample_data):
    """advance() must update the internal index for the given instrument."""
    provider = BacktestBarProvider(sample_data)
    provider.advance("NSE_INDEX|Nifty Bank", 19)
    df = provider.get_bars("NSE_INDEX|Nifty Bank", timeframe=1)
    assert len(df) == 20


def test_unknown_instrument_returns_empty(sample_data):
    """Requesting bars for unknown instrument returns empty DataFrame."""
    provider = BacktestBarProvider(sample_data)
    df = provider.get_bars("NSE_INDEX|Unknown", timeframe=1)
    assert df.empty
