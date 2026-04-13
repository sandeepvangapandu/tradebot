"""Historical bar provider implementing BarBuilder + HistoricalDataFetcher interfaces.

Used by the BacktestHarness to feed historical OHLCV data to TradeAnalyzer
and StrategyEngine without lookahead bias.

All prices in PAISA (integer). Timestamps are IST-aware.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger


_OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


class BacktestBarProvider:
    """Drop-in replacement for BarBuilder and HistoricalDataFetcher.

    Holds full pre-loaded OHLCV DataFrames per instrument and serves
    slices up to the current bar index — strictly no lookahead.

    Args:
        data: Mapping of instrument_key → full OHLCV DataFrame (1-min, paisa,
              IST-aware DatetimeIndex).
    """

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = data
        # Current bar index per instrument (inclusive upper bound)
        self._current_index: dict[str, int] = {k: 0 for k in data}

    def advance(self, instrument_key: str, index: int) -> None:
        """Advance the time window to bar ``index`` (0-based, inclusive).

        Must be called by the harness before any evaluation for bar ``index``.

        Args:
            instrument_key: Instrument to advance.
            index: Current bar index (0-based).
        """
        self._current_index[instrument_key] = index

    # ------------------------------------------------------------------
    # BarBuilder interface (used by TradeAnalyzer._gather_data)
    # ------------------------------------------------------------------

    def get_bars(
        self,
        instrument_key: str,
        timeframe: int,
        include_current: bool = False,
    ) -> pd.DataFrame:
        """Return OHLCV bars up to the current index, resampled to timeframe.

        Args:
            instrument_key: Instrument identifier.
            timeframe: Target timeframe in minutes (1, 5, or 15).
            include_current: Ignored — current bar is always included via advance().

        Returns:
            DataFrame with OHLCV columns and DatetimeIndex, or empty DataFrame
            if instrument is unknown.
        """
        if instrument_key not in self._data:
            logger.debug(f"BacktestBarProvider: unknown instrument {instrument_key}")
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        idx = self._current_index.get(instrument_key, 0)
        window = self._data[instrument_key].iloc[: idx + 1]

        if timeframe == 1:
            return window

        resampled = (
            window
            .resample(f"{timeframe}min")
            .agg(_OHLCV_AGG)
            .dropna()
        )
        return resampled

    # ------------------------------------------------------------------
    # HistoricalDataFetcher interface (used by TradeAnalyzer._gather_data)
    # ------------------------------------------------------------------

    def fetch_candles(
        self,
        instrument_key: str,
        interval: str = "day",
        days: int = 30,
    ) -> pd.DataFrame:
        """Return daily bars from historical data up to current index.

        Args:
            instrument_key: Instrument identifier.
            interval: Ignored — always returns daily bars.
            days: Maximum number of daily bars to return.

        Returns:
            DataFrame with daily OHLCV bars, most recent ``days`` rows.
        """
        if instrument_key not in self._data:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        idx = self._current_index.get(instrument_key, 0)
        window = self._data[instrument_key].iloc[: idx + 1]
        daily = window.resample("1D").agg(_OHLCV_AGG).dropna()
        return daily.tail(days)
