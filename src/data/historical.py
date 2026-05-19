"""Historical data fetcher using Upstox History V3 API.

Fetches candle data for indicator warmup and backtesting.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import upstox_client
from loguru import logger

from config.settings import get_settings
from src.utils.rate_limiter import RateLimiter

IST = ZoneInfo("Asia/Kolkata")

# Rate limiter: 40 requests per second
_limiter = RateLimiter(rate=40.0, capacity=40.0)


class HistoricalDataFetcher:
    """Fetches historical candle data from Upstox API.

    Args:
        access_token: Valid Upstox OAuth2 access token.
    """

    def __init__(self, access_token: str) -> None:
        self._access_token = access_token
        self._configuration = upstox_client.Configuration()
        self._configuration.access_token = access_token
        self._api = upstox_client.HistoryApi(upstox_client.ApiClient(self._configuration))

    def fetch_candles(
        self,
        instrument_key: str,
        interval: str = "1minute",
        to_date: datetime | None = None,
        from_date: datetime | None = None,
        days: int = 30,
    ) -> pd.DataFrame:
        """Fetch historical candles for an instrument.

        Args:
            instrument_key: Upstox instrument key.
            interval: Candle interval (1minute, 30minute, day, week, month).
            to_date: End date (defaults to now).
            from_date: Start date (defaults to days before to_date).
            days: Number of days to fetch if from_date not specified.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            Prices are in PAISA (int).
        """
        # Default dates
        if to_date is None:
            to_date = datetime.now(IST)
        if from_date is None:
            from_date = to_date - timedelta(days=days)

        # Upstox History v3 requires yyyy-mm-dd strings, not unix epoch ints.
        to_date_str = to_date.strftime("%Y-%m-%d")
        from_date_str = from_date.strftime("%Y-%m-%d")

        logger.info(
            "Fetching {} candles for {} from {} to {}",
            interval,
            instrument_key,
            from_date.strftime("%Y-%m-%d"),
            to_date.strftime("%Y-%m-%d"),
        )

        try:
            _limiter.acquire()
            response = self._api.get_historical_candle_data1(
                instrument_key=instrument_key,
                interval=interval,
                to_date=to_date_str,
                from_date=from_date_str,
                api_version="v2",
            )

            candles = response.data.candles if response.data else []
            if not candles:
                logger.warning("No candles returned for {}", instrument_key)
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

            # Parse candles: [timestamp, open, high, low, close, volume, oi]
            data = []
            for candle in candles:
                if len(candle) >= 6:
                    ts = datetime.fromisoformat(candle[0].replace("Z", "+00:00")).astimezone(IST)
                    # Prices come as floats in rupees, convert to paisa (int)
                    open_p = int(float(candle[1]) * 100)
                    high_p = int(float(candle[2]) * 100)
                    low_p = int(float(candle[3]) * 100)
                    close_p = int(float(candle[4]) * 100)
                    volume = int(candle[5])
                    data.append([ts, open_p, high_p, low_p, close_p, volume])

            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df.set_index("timestamp", inplace=True)

            logger.info("Fetched {} candles for {}", len(df), instrument_key)
            return df

        except Exception as exc:
            logger.error("Failed to fetch historical data for {}: {}", instrument_key, exc)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    def fetch_intraday_candles(
        self,
        instrument_key: str,
        interval: str = "1minute",
    ) -> pd.DataFrame:
        """Fetch intraday candles for today via Upstox dedicated endpoint.

        Upstox History v3 excludes the current trading day; intraday data
        comes from a separate `get_intra_day_candle_data` endpoint.

        Args:
            instrument_key: Upstox instrument key.
            interval: Candle interval (1minute, 30minute).

        Returns:
            DataFrame with today's intraday candles, indexed by timestamp.
            Prices are in PAISA (int).
        """
        try:
            _limiter.acquire()
            response = self._api.get_intra_day_candle_data(
                instrument_key=instrument_key,
                interval=interval,
                api_version="v2",
            )
            candles = response.data.candles if response.data else []
            if not candles:
                logger.warning("No intraday candles returned for {}", instrument_key)
                return pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )

            data = []
            for candle in candles:
                if len(candle) >= 6:
                    ts = datetime.fromisoformat(
                        candle[0].replace("Z", "+00:00")
                    ).astimezone(IST)
                    open_p = int(float(candle[1]) * 100)
                    high_p = int(float(candle[2]) * 100)
                    low_p = int(float(candle[3]) * 100)
                    close_p = int(float(candle[4]) * 100)
                    volume = int(candle[5])
                    data.append([ts, open_p, high_p, low_p, close_p, volume])

            df = pd.DataFrame(
                data, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            logger.info("Fetched {} intraday candles for {}", len(df), instrument_key)
            return df
        except Exception as exc:
            logger.error(
                "Failed to fetch intraday data for {}: {}", instrument_key, exc
            )
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

    def warmup_bars(
        self,
        instrument_keys: list[str],
        timeframes: list[int] = [1, 5, 15],
        days: int = 5,
    ) -> dict[str, dict[int, pd.DataFrame]]:
        """Fetch warmup data for multiple instruments.

        Args:
            instrument_keys: List of instrument keys.
            timeframes: List of timeframes in minutes.
            days: Days of history to fetch.

        Returns:
            Dict mapping instrument_key -> timeframe -> DataFrame
        """
        result: dict[str, dict[int, pd.DataFrame]] = {}

        for key in instrument_keys:
            result[key] = {}
            for tf in timeframes:
                interval = f"{tf}minute" if tf < 60 else "1hour"
                df = self.fetch_candles(key, interval=interval, days=days)
                result[key][tf] = df

        return result
