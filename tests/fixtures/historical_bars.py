"""Historical OHLCV bar fixture factories.

Generates deterministic synthetic OHLCV DataFrames using a
geometric-Brownian-motion price path so indicators computed on the data
behave realistically (trends, mean-reversion noise, etc.).

All prices are in **paisa** in the generated DataFrames, matching the
bot's internal convention.  Volume is in shares/contracts.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")

# Market opens at 09:15 IST
_MARKET_OPEN_HOUR = 9
_MARKET_OPEN_MINUTE = 15
_MARKET_CLOSE_HOUR = 15
_MARKET_CLOSE_MINUTE = 30


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _market_minutes_in_day() -> int:
    """Return the number of 1-minute bars in a full trading session."""
    open_min = _MARKET_OPEN_HOUR * 60 + _MARKET_OPEN_MINUTE
    close_min = _MARKET_CLOSE_HOUR * 60 + _MARKET_CLOSE_MINUTE
    return close_min - open_min  # 375 minutes


def _trading_timestamps(
    start: datetime,
    end: datetime,
    timeframe_minutes: int,
) -> list[datetime]:
    """Generate IST-aware timestamps for every bar between start and end.

    Only timestamps that fall within market hours (09:15–15:30) on
    weekdays are included.

    Args:
        start: Start datetime (timezone-aware or naive; treated as IST).
        end: End datetime.
        timeframe_minutes: Bar length in minutes.

    Returns:
        Sorted list of bar-open timestamps.
    """
    ts: list[datetime] = []
    current = start.replace(second=0, microsecond=0)

    # Advance to next market open if before 09:15
    while current <= end:
        # Skip weekends
        if current.weekday() >= 5:  # Saturday=5, Sunday=6
            current += timedelta(days=1)
            current = current.replace(
                hour=_MARKET_OPEN_HOUR, minute=_MARKET_OPEN_MINUTE
            )
            continue

        h, m = current.hour, current.minute
        open_total = _MARKET_OPEN_HOUR * 60 + _MARKET_OPEN_MINUTE
        close_total = _MARKET_CLOSE_HOUR * 60 + _MARKET_CLOSE_MINUTE
        cur_total = h * 60 + m

        if cur_total < open_total:
            current = current.replace(
                hour=_MARKET_OPEN_HOUR, minute=_MARKET_OPEN_MINUTE
            )
            continue
        if cur_total >= close_total:
            # Jump to next day's open
            current = (current + timedelta(days=1)).replace(
                hour=_MARKET_OPEN_HOUR, minute=_MARKET_OPEN_MINUTE
            )
            continue

        ts.append(current)
        current += timedelta(minutes=timeframe_minutes)

    return ts


def _gbm_prices(
    n: int,
    base_price: float,
    volatility_pct: float,
    drift_per_bar: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return an array of GBM-simulated close prices.

    Args:
        n: Number of prices to generate.
        base_price: Starting price (paisa).
        volatility_pct: Per-bar volatility as a percentage (e.g. 0.5 = 0.5%).
        drift_per_bar: Per-bar drift (log) as a fraction of base price.
        rng: NumPy random generator.

    Returns:
        1-D float array of length ``n``.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    sigma = volatility_pct / 100.0
    dt = 1.0
    log_returns = (drift_per_bar - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * rng.standard_normal(n)
    prices = base_price * np.exp(np.cumsum(log_returns))
    prices = np.concatenate([[float(base_price)], prices])[:-1]
    return prices


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def make_ohlcv_bars(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe_minutes: int = 1,
    base_price: int = 100_000,
    volatility_pct: float = 0.5,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV bar DataFrame for a single instrument.

    Args:
        symbol: Instrument symbol (used only for the ``symbol`` column).
        start: Start datetime (inclusive).
        end: End datetime (inclusive).
        timeframe_minutes: Bar duration in minutes.
        base_price: Starting price in **paisa** (default 100_000 = ₹1,000).
        volatility_pct: Per-bar price volatility percentage.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns:
        ``timestamp``, ``symbol``, ``open``, ``high``, ``low``,
        ``close``, ``volume``.  All price columns are in paisa.
        Index is the integer range index.
    """
    rng = np.random.default_rng(seed)
    timestamps = _trading_timestamps(start, end, timeframe_minutes)

    if not timestamps:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        )

    n = len(timestamps)
    closes = _gbm_prices(n, float(base_price), volatility_pct, rng=rng)

    bar_vol_frac = volatility_pct / 100.0 * 0.5  # intra-bar range ~ half sigma
    highs = closes * (1 + np.abs(rng.normal(0, bar_vol_frac, n)))
    lows = closes * (1 - np.abs(rng.normal(0, bar_vol_frac, n)))

    # Open is close of previous bar with small overnight gap handled naturally
    opens = np.roll(closes, 1)
    opens[0] = closes[0] * (1 + rng.normal(0, bar_vol_frac))

    # Ensure OHLC integrity: high >= max(open, close), low <= min(open, close)
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    volume_base = 500_000  # ~5 lakh shares/contracts per bar as base
    volumes = (volume_base * rng.lognormal(0, 0.5, n)).astype(int)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": symbol,
        "open": opens.astype(int),
        "high": highs.astype(int),
        "low": lows.astype(int),
        "close": closes.astype(int),
        "volume": volumes,
    })
    return df


def make_warmup_bars(
    instrument_keys: list[str],
    days: int = 60,
    timeframe_minutes: int = 1,
    *,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate warmup OHLCV bars for a list of instruments.

    Args:
        instrument_keys: List of instrument keys (e.g. ``["NSE_EQ|INE002A01018"]``).
        days: Number of calendar days of history to generate.
        timeframe_minutes: Bar duration in minutes.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping each instrument key to its OHLCV DataFrame.
    """
    end = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
    start = end - timedelta(days=days)

    # Typical base prices per instrument key prefix (paisa)
    _BASE_PRICES: dict[str, int] = {
        "NSE_INDEX|Nifty 50": 2_450_000,    # ₹24,500
        "NSE_INDEX|Nifty Bank": 5_200_000,  # ₹52,000
        "NSE_EQ|INE002A01018": 290_000,     # RELIANCE ₹2,900
        "NSE_EQ|INE040A01034": 160_000,     # HDFCBANK ₹1,600
        "NSE_EQ|INE090A01021": 130_000,     # ICICIBANK ₹1,300
        "NSE_EQ|INE467B01029": 370_000,     # TCS ₹3,700
        "NSE_EQ|INE009A01021": 180_000,     # INFY ₹1,800
        "NSE_EQ|INE030A01027": 230_000,     # HINDUNILVR ₹2,300
        "NSE_EQ|INE154A01025": 45_000,      # ITC ₹450
        "NSE_EQ|INE238A01034": 120_000,     # AXISBANK ₹1,200
        "NSE_EQ|INE237A01036": 180_000,     # KOTAKBANK ₹1,800
        "NSE_EQ|INE062A01020": 85_000,      # SBIN ₹850
    }

    result: dict[str, pd.DataFrame] = {}
    for i, key in enumerate(instrument_keys):
        base_price = _BASE_PRICES.get(key, 100_000)
        df = make_ohlcv_bars(
            symbol=key,
            start=start,
            end=end,
            timeframe_minutes=timeframe_minutes,
            base_price=base_price,
            volatility_pct=0.3,
            seed=seed + i,
        )
        result[key] = df

    return result
