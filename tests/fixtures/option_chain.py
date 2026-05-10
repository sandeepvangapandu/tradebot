"""Option chain fixture factories.

Generates realistic synthetic option chain DataFrames whose column names
match the Upstox instrument master format used by
``src/data/instruments.py``.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOP_10_UNDERLYINGS: list[str] = [
    "RELIANCE",
    "HDFCBANK",
    "ICICIBANK",
    "TCS",
    "INFY",
    "HINDUNILVR",
    "ITC",
    "AXISBANK",
    "KOTAKBANK",
    "SBIN",
]

INDICES: list[str] = ["NIFTY", "BANKNIFTY"]

# Default lot sizes per underlying (matching NSE standards)
_LOT_SIZES: dict[str, int] = {
    "NIFTY": 50,
    "BANKNIFTY": 15,
    "RELIANCE": 250,
    "HDFCBANK": 550,
    "ICICIBANK": 1375,
    "TCS": 150,
    "INFY": 300,
    "HINDUNILVR": 300,
    "ITC": 3200,
    "AXISBANK": 1200,
    "KOTAKBANK": 400,
    "SBIN": 1500,
}

# Underlying keys per symbol
_UNDERLYING_KEYS: dict[str, str] = {
    "NIFTY": "NSE_INDEX|Nifty 50",
    "BANKNIFTY": "NSE_INDEX|Nifty Bank",
    "RELIANCE": "NSE_EQ|INE002A01018",
    "HDFCBANK": "NSE_EQ|INE040A01034",
    "ICICIBANK": "NSE_EQ|INE090A01021",
    "TCS": "NSE_EQ|INE467B01029",
    "INFY": "NSE_EQ|INE009A01021",
    "HINDUNILVR": "NSE_EQ|INE030A01027",
    "ITC": "NSE_EQ|INE154A01025",
    "AXISBANK": "NSE_EQ|INE238A01034",
    "KOTAKBANK": "NSE_EQ|INE237A01028",
    "SBIN": "NSE_EQ|INE062A01020",
}

# Strike intervals per underlying
_STRIKE_INTERVALS: dict[str, int] = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
    "RELIANCE": 20,
    "HDFCBANK": 20,
    "ICICIBANK": 10,
    "TCS": 50,
    "INFY": 20,
    "HINDUNILVR": 50,
    "ITC": 5,
    "AXISBANK": 10,
    "KOTAKBANK": 20,
    "SBIN": 10,
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def atm_strike(spot_rupees: float, interval: int = 100) -> float:
    """Return the ATM strike nearest to ``spot_rupees`` rounded to ``interval``.

    Args:
        spot_rupees: Current spot price in rupees.
        interval: Strike interval in rupees (e.g. 50 for NIFTY, 100 for BANKNIFTY).

    Returns:
        ATM strike price in rupees.
    """
    return round(spot_rupees / interval) * interval


def realistic_iv(strike: float, spot: float, dte: int, *, seed: int = 42) -> float:
    """Return a smile-shaped implied volatility for an option.

    The IV surface follows a parabolic smile:
    ``IV = base_iv * (1 + skew * moneyness + curvature * moneyness**2)``

    Args:
        strike: Option strike price in rupees.
        spot: Spot price in rupees.
        dte: Days to expiry.
        seed: Unused — kept for API symmetry.

    Returns:
        Implied volatility as a decimal (e.g. 0.18 = 18 %).
    """
    if spot <= 0:
        return 0.15

    base_iv = 0.15 + 0.05 * math.exp(-dte / 30.0)  # term structure
    moneyness = math.log(strike / spot) if spot > 0 else 0.0
    skew = -0.5      # negative skew (index-like)
    curvature = 8.0  # parabola width
    iv = base_iv * (1.0 + skew * moneyness + curvature * moneyness**2)
    return max(0.05, min(iv, 2.0))  # clamp to [5%, 200%]


def _bsm_price(
    spot: float,
    strike: float,
    iv: float,
    dte: int,
    r: float = 0.065,
    option_type: str = "CE",
) -> float:
    """Compute Black-Scholes-Merton option price.

    Args:
        spot: Spot price.
        strike: Strike price.
        iv: Implied volatility (decimal).
        dte: Days to expiry.
        r: Risk-free rate (decimal).
        option_type: ``"CE"`` or ``"PE"``.

    Returns:
        Theoretical option price in rupees.
    """
    if dte <= 0 or iv <= 0:
        intrinsic = max(spot - strike, 0) if option_type == "CE" else max(strike - spot, 0)
        return intrinsic

    T = dte / 365.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv**2) * T) / (iv * sqrt_T)
    d2 = d1 - iv * sqrt_T

    from statistics import NormalDist
    nd = NormalDist()
    N = nd.cdf

    if option_type == "CE":
        price = spot * N(d1) - strike * math.exp(-r * T) * N(d2)
    else:
        price = strike * math.exp(-r * T) * N(-d2) - spot * N(-d1)

    return max(price, 0.05)  # minimum premium ₹0.05


def make_option_chain(
    underlying_symbol: str,
    spot_price_rupees: float,
    expiry_date: str,
    num_strikes_each_side: int = 10,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic option chain DataFrame.

    Columns match the Upstox instrument master format (``src/data/instruments.py``):
    ``segment``, ``name``, ``instrument_key``, ``trading_symbol``,
    ``lot_size``, ``tick_size``, ``expiry``, ``strike_price``,
    ``option_type``, ``underlying_key``, ``underlying_symbol``,
    ``exchange_token``, ``isin``, ``instrument_type``, ``short_name``,
    ``freeze_quantity``.

    Additional columns for option data:
    ``ltp``, ``iv``, ``delta``, ``oi``, ``volume``.

    Args:
        underlying_symbol: e.g. ``"NIFTY"``, ``"RELIANCE"``.
        spot_price_rupees: Current spot price in rupees.
        expiry_date: Expiry date string ``"YYYY-MM-DD"``.
        num_strikes_each_side: Number of strikes on each side of ATM.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with CE and PE rows for each strike.
    """
    rng = np.random.default_rng(seed)

    sym = underlying_symbol.upper()
    interval = _STRIKE_INTERVALS.get(sym, 50)
    lot_size = _LOT_SIZES.get(sym, 100)
    underlying_key = _UNDERLYING_KEYS.get(sym, f"NSE_EQ|{sym}")

    atm = atm_strike(spot_price_rupees, interval)

    # Parse dte
    try:
        expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
        dte = max(1, (expiry_dt - datetime.now()).days)
    except ValueError:
        dte = 7  # default weekly

    strikes = sorted([
        atm + interval * i
        for i in range(-num_strikes_each_side, num_strikes_each_side + 1)
    ])

    rows: list[dict[str, Any]] = []
    token_base = hash(sym) % 100_000

    for strike_idx, strike in enumerate(strikes):
        for opt_type in ("CE", "PE"):
            iv = realistic_iv(strike, spot_price_rupees, dte, seed=seed)
            ltp = _bsm_price(spot_price_rupees, strike, iv, dte, option_type=opt_type)
            # Add some random noise to LTP
            ltp = max(0.05, ltp * (1 + rng.uniform(-0.02, 0.02)))

            # Delta (rough approximation for display)
            from statistics import NormalDist
            nd = NormalDist()
            if dte > 0 and iv > 0:
                T = dte / 365.0
                d1 = (math.log(spot_price_rupees / strike + 1e-9) + (0.065 + 0.5 * iv**2) * T) / (iv * math.sqrt(T))
                delta = nd.cdf(d1) if opt_type == "CE" else nd.cdf(d1) - 1.0
            else:
                delta = 0.5 if opt_type == "CE" else -0.5

            expiry_short = expiry_date.replace("-", "")[2:]  # YYMMDD
            trading_sym = f"{sym}{expiry_short}{int(strike)}{opt_type}"
            instrument_key = f"NSE_FO|{token_base + strike_idx * 2 + (0 if opt_type == 'CE' else 1)}"
            exchange_token = str(token_base + strike_idx * 100 + (0 if opt_type == "CE" else 1))

            oi_base = max(1000, int(100_000 * math.exp(-abs(strike - atm) / (interval * 3))))
            oi = int(oi_base * (1 + rng.uniform(-0.3, 0.3)))
            volume = int(oi * rng.uniform(0.05, 0.25))

            rows.append({
                # --- instrument master columns ---
                "segment": "NSE_FO",
                "name": sym,
                "instrument_key": instrument_key,
                "trading_symbol": trading_sym,
                "lot_size": lot_size,
                "tick_size": 0.05,
                "expiry": expiry_date,
                "strike_price": float(strike),
                "option_type": opt_type,
                "underlying_key": underlying_key,
                "underlying_symbol": sym,
                "exchange_token": exchange_token,
                "isin": None,
                "instrument_type": opt_type,
                "short_name": sym,
                "freeze_quantity": lot_size * 100,
                # --- market data columns ---
                "ltp": round(ltp, 2),
                "iv": round(iv * 100, 2),       # as percentage
                "delta": round(delta, 4),
                "oi": oi,
                "volume": volume,
            })

    df = pd.DataFrame(rows)
    df.sort_values(["strike_price", "option_type"], inplace=True)
    return df.reset_index(drop=True)
