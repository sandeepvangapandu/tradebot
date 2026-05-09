"""Tests for the technical indicators engine.

Tests EMA, MACD, RSI, and VWAP calculations using pandas-ta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest


class TestEMA:
    """Test Exponential Moving Average calculations."""

    def test_ema_calculation_basic(self, sample_ohlc_data: pd.DataFrame) -> None:
        """Test basic EMA calculation."""
        df = sample_ohlc_data.copy()
        df["ema_9"] = ta.ema(df["close"], length=9)
        df["ema_21"] = ta.ema(df["close"], length=21)

        # EMA should be NaN for first few periods
        assert df["ema_9"].iloc[:8].isna().all()
        assert df["ema_21"].iloc[:20].isna().all()

        # After warmup, EMA should have values
        assert df["ema_9"].iloc[20:].notna().all()
        assert df["ema_21"].iloc[25:].notna().all()

    def test_ema_values_reasonable(self, sample_ohlc_data: pd.DataFrame) -> None:
        """Test that EMA values are within reasonable bounds."""
        df = sample_ohlc_data.copy()
        df["ema_20"] = ta.ema(df["close"], length=20)

        valid_ema = df["ema_20"].dropna()
        min_close = df["close"].min()
        max_close = df["close"].max()

        # EMA should stay within price bounds
        assert valid_ema.min() >= min_close * 0.95
        assert valid_ema.max() <= max_close * 1.05

    def test_ema_responds_to_price_changes(self) -> None:
        """Test that EMA responds to price changes."""
        # Create data with a sharp price jump
        prices = [100.0] * 20 + [150.0] * 20
        df = pd.DataFrame({"close": prices})
        df["ema_10"] = ta.ema(df["close"], length=10)

        # EMA should increase after price jump
        ema_before = df["ema_10"].iloc[19]
        ema_after = df["ema_10"].iloc[39]
        assert ema_after > ema_before

    def test_ema_crossover_detection(self) -> None:
        """Test detecting EMA crossovers."""
        # Create data where fast EMA crosses above slow EMA
        prices = list(range(100, 80, -1)) + list(range(80, 120))  # Dip then rise
        df = pd.DataFrame({"close": prices})
        df["ema_fast"] = ta.ema(df["close"], length=5)
        df["ema_slow"] = ta.ema(df["close"], length=10)

        # Find crossover points
        df["cross_above"] = (df["ema_fast"] > df["ema_slow"]) & (
            df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)
        )

        # Should have at least one crossover
        assert df["cross_above"].any()


class TestMACD:
    """Test MACD (Moving Average Convergence Divergence) calculations."""

    def test_macd_calculation_structure(self, sample_ohlc_data: pd.DataFrame) -> None:
        """Test MACD returns correct structure with MACD, Signal, and Histogram."""
        df = sample_ohlc_data.copy()
        macd_result = ta.macd(df["close"], fast=12, slow=26, signal=9)

        # Should return DataFrame with three columns
        assert isinstance(macd_result, pd.DataFrame)
        assert macd_result.shape[1] == 3

        # Check column names (pandas-ta naming convention)
        expected_cols = ["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"]
        for col in expected_cols:
            assert col in macd_result.columns

    def test_macd_line_values(self, sample_ohlc_data: pd.DataFrame) -> None:
        """Test MACD line values are reasonable."""
        df = sample_ohlc_data.copy()
        macd_result = ta.macd(df["close"], fast=12, slow=26, signal=9)

        macd_line = macd_result["MACD_12_26_9"]
        signal_line = macd_result["MACDs_12_26_9"]
        histogram = macd_result["MACDh_12_26_9"]

        # MACD line should have values after warmup
        assert macd_line.iloc[30:].notna().all()

        # Histogram should equal MACD - Signal
        expected_histogram = macd_line - signal_line
        pd.testing.assert_series_equal(
            histogram.dropna(),
            expected_histogram.dropna(),
            check_names=False
        )

    def test_macd_crossover_detection(self) -> None:
        """Test detecting MACD crossovers."""
        # Create oscillating price data
        t = np.linspace(0, 4 * np.pi, 100)
        prices = 100 + 10 * np.sin(t)
        df = pd.DataFrame({"close": prices})

        macd_result = ta.macd(df["close"], fast=12, slow=26, signal=9)
        macd_line = macd_result["MACD_12_26_9"]
        signal_line = macd_result["MACDs_12_26_9"]

        # Detect bullish cross (MACD crosses above Signal)
        bullish_cross = (macd_line > signal_line) & (
            macd_line.shift(1) <= signal_line.shift(1)
        )

        # Detect bearish cross (MACD crosses below Signal)
        bearish_cross = (macd_line < signal_line) & (
            macd_line.shift(1) >= signal_line.shift(1)
        )

        # Should have both types of crossovers in oscillating data
        assert bullish_cross.sum() > 0
        assert bearish_cross.sum() > 0

    def test_macd_trending_market(self) -> None:
        """Test MACD behavior in strongly trending market."""
        # Strong uptrend
        prices = np.linspace(100, 200, 100)
        df = pd.DataFrame({"close": prices})

        macd_result = ta.macd(df["close"], fast=12, slow=26, signal=9)
        macd_line = macd_result["MACD_12_26_9"]

        # In strong uptrend, MACD should be positive after warmup
        valid_macd = macd_line.dropna()
        assert (valid_macd > 0).all()


class TestRSI:
    """Test Relative Strength Index calculations."""

    def test_rsi_calculation_range(self, sample_ohlc_data: pd.DataFrame) -> None:
        """Test RSI values are always between 0 and 100."""
        df = sample_ohlc_data.copy()
        df["rsi"] = ta.rsi(df["close"], length=14)

        valid_rsi = df["rsi"].dropna()
        assert valid_rsi.min() >= 0
        assert valid_rsi.max() <= 100

    def test_rsi_warmup_period(self, sample_ohlc_data: pd.DataFrame) -> None:
        """Test RSI has NaN values during warmup period.

        pandas-ta RSI (EMA-based) produces only 1 leading NaN then begins
        computing. We verify that at least the first row is NaN and that
        the RSI has fully converged by row 20.
        """
        df = sample_ohlc_data.copy()
        df["rsi"] = ta.rsi(df["close"], length=14)

        # pandas-ta RSI emits at least one leading NaN
        assert df["rsi"].iloc[0:1].isna().all()
        # After sufficient warmup (row 20+) all values should be present
        assert df["rsi"].iloc[20:].notna().all()

    def test_rsi_overbought_oversold(self) -> None:
        """Test RSI correctly identifies overbought/oversold conditions."""
        # Create strong uptrend (should be overbought)
        uptrend = [100] + [100 + i * 2 for i in range(1, 50)]
        df_up = pd.DataFrame({"close": uptrend})
        df_up["rsi"] = ta.rsi(df_up["close"], length=14)

        # RSI should be high in strong uptrend
        assert df_up["rsi"].iloc[-1] > 70

        # Create strong downtrend (should be oversold)
        downtrend = [200] + [200 - i * 2 for i in range(1, 50)]
        df_down = pd.DataFrame({"close": downtrend})
        df_down["rsi"] = ta.rsi(df_down["close"], length=14)

        # RSI should be low in strong downtrend
        assert df_down["rsi"].iloc[-1] < 30

    def test_rsi_sideways_market(self) -> None:
        """Test RSI in sideways/ranging market."""
        # Create oscillating prices
        t = np.linspace(0, 4 * np.pi, 100)
        prices = 100 + 5 * np.sin(t)
        df = pd.DataFrame({"close": prices})
        df["rsi"] = ta.rsi(df["close"], length=14)

        valid_rsi = df["rsi"].dropna()
        # In sideways market, RSI should oscillate between 30 and 70
        assert valid_rsi.min() < 40
        assert valid_rsi.max() > 60


class TestVWAP:
    """Test Volume Weighted Average Price calculations."""

    def test_vwap_calculation(self, sample_ohlc_data: pd.DataFrame) -> None:
        """Test VWAP calculation with OHLCV data."""
        df = sample_ohlc_data.copy()

        # Calculate typical price
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3

        # Calculate VWAP manually
        df["tp_vol"] = df["typical_price"] * df["volume"]
        df["vwap_manual"] = df["tp_vol"].cumsum() / df["volume"].cumsum()

        # Calculate using pandas-ta
        df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

        # Both should be similar
        pd.testing.assert_series_equal(
            df["vwap"].dropna(),
            df["vwap_manual"].dropna(),
            check_names=False,
            rtol=1e-5
        )

    def test_vwap_vs_simple_average(self, sample_ohlc_data: pd.DataFrame) -> None:
        """Test VWAP differs from simple moving average."""
        df = sample_ohlc_data.copy()

        df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        df["sma_20"] = ta.sma(df["close"], length=20)

        # VWAP and SMA should be different
        diff = (df["vwap"] - df["sma_20"]).abs()
        assert diff.dropna().mean() > 0

    def test_vwap_with_volume_spikes(self) -> None:
        """Test VWAP responds to volume spikes.

        pandas-ta vwap requires a DatetimeIndex; supply one so the function
        returns non-None values.
        """
        prices = [100.0] * 20
        volumes = [1000] * 10 + [10000] + [1000] * 9
        prices[10] = 110  # High price with high volume

        dates = pd.date_range(start="2024-01-01 09:15", periods=20, freq="5min")
        df = pd.DataFrame(
            {
                "high": [p + 1 for p in prices],
                "low": [p - 1 for p in prices],
                "close": prices,
                "volume": volumes,
            },
            index=dates,
        )

        df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

        # VWAP should be pulled toward the high volume price
        vwap_after_spike = df["vwap"].iloc[15]
        assert vwap_after_spike > 100

    def test_vwap_anchored(self, sample_ohlc_data: pd.DataFrame) -> None:
        """Test anchored VWAP calculation."""
        df = sample_ohlc_data.copy()

        # Calculate VWAP anchored at start
        df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"], anchor="D")

        # VWAP should have values
        assert df["vwap"].notna().all()


class TestIndicatorEngineIntegration:
    """Test integration of multiple indicators."""

    def test_multiple_indicators_same_dataframe(self, sample_ohlc_data: pd.DataFrame) -> None:
        """Test calculating multiple indicators on same DataFrame."""
        df = sample_ohlc_data.copy()

        # Add multiple indicators
        df["ema_9"] = ta.ema(df["close"], length=9)
        df["ema_21"] = ta.ema(df["close"], length=21)
        df["rsi"] = ta.rsi(df["close"], length=14)
        df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

        macd_result = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd_result], axis=1)

        # All indicators should have some valid values
        assert df["ema_9"].notna().sum() > 0
        assert df["ema_21"].notna().sum() > 0
        assert df["rsi"].notna().sum() > 0
        assert df["vwap"].notna().sum() > 0
        assert df["MACD_12_26_9"].notna().sum() > 0

    def test_indicator_signals_alignment(self, sample_ohlc_data: pd.DataFrame) -> None:
        """Test that indicator signals align logically."""
        df = sample_ohlc_data.copy()

        df["rsi"] = ta.rsi(df["close"], length=14)
        df["ema_fast"] = ta.ema(df["close"], length=9)
        df["ema_slow"] = ta.ema(df["close"], length=21)

        # Find points where RSI > 70 (overbought) and price > EMA
        overbought_above_ema = (
            (df["rsi"] > 70) & (df["close"] > df["ema_fast"])
        )

        # In overbought conditions above EMA, we expect potential reversal
        # This is a logical consistency check
        if overbought_above_ema.any():
            # Price should be relatively high
            overbought_prices = df.loc[overbought_above_ema, "close"]
            assert overbought_prices.min() > df["close"].quantile(0.5)
