"""Mini instrument master fixture.

Returns a DataFrame of ~16 instruments:
  - NIFTY and BANKNIFTY index entries
  - 10 top-stock equity entries
  - ATM CE and PE for NIFTY and BANKNIFTY (current weekly, ~4 instruments)

Column names exactly match ``src/data/instruments.py``'s ``_INSTRUMENT_COLUMNS``
plus the derived ``option_type`` column so the fixture plugs directly into
``InstrumentManager._instruments``.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_NEXT_THURSDAY = (
    datetime.now()
    + timedelta(days=(3 - datetime.now().weekday()) % 7 or 7)
).strftime("%Y-%m-%d")


def make_mini_master(*, seed: int = 42) -> pd.DataFrame:
    """Return a ~16-row instrument master DataFrame.

    Args:
        seed: Unused — kept for API symmetry.

    Returns:
        DataFrame with columns matching ``_INSTRUMENT_COLUMNS`` from
        ``src/data/instruments.py`` and a derived ``option_type`` column.
    """
    rows = [
        # ----------------------------------------------------------------
        # Indices
        # ----------------------------------------------------------------
        {
            "segment": "NSE_INDEX",
            "name": "Nifty 50",
            "instrument_key": "NSE_INDEX|Nifty 50",
            "trading_symbol": "NIFTY",
            "lot_size": 50,
            "tick_size": 0.05,
            "expiry": None,
            "strike_price": 0.0,
            "option_type": None,
            "underlying_key": None,
            "underlying_symbol": "NIFTY",
            "exchange_token": "99926000",
            "isin": None,
            "instrument_type": "INDEX",
            "short_name": "NIFTY",
            "freeze_quantity": 0,
        },
        {
            "segment": "NSE_INDEX",
            "name": "Nifty Bank",
            "instrument_key": "NSE_INDEX|Nifty Bank",
            "trading_symbol": "BANKNIFTY",
            "lot_size": 15,
            "tick_size": 0.05,
            "expiry": None,
            "strike_price": 0.0,
            "option_type": None,
            "underlying_key": None,
            "underlying_symbol": "BANKNIFTY",
            "exchange_token": "99926009",
            "isin": None,
            "instrument_type": "INDEX",
            "short_name": "BANKNIFTY",
            "freeze_quantity": 0,
        },
        # ----------------------------------------------------------------
        # Top-10 equities
        # ----------------------------------------------------------------
        {
            "segment": "NSE_EQ",
            "name": "RELIANCE INDUSTRIES",
            "instrument_key": "NSE_EQ|INE002A01018",
            "trading_symbol": "RELIANCE",
            "lot_size": 1,
            "tick_size": 0.05,
            "expiry": None,
            "strike_price": 0.0,
            "option_type": None,
            "underlying_key": None,
            "underlying_symbol": "RELIANCE",
            "exchange_token": "2885",
            "isin": "INE002A01018",
            "instrument_type": "EQUITY",
            "short_name": "RELIANCE",
            "freeze_quantity": 0,
        },
        {
            "segment": "NSE_EQ",
            "name": "HDFC BANK",
            "instrument_key": "NSE_EQ|INE040A01034",
            "trading_symbol": "HDFCBANK",
            "lot_size": 1,
            "tick_size": 0.05,
            "expiry": None,
            "strike_price": 0.0,
            "option_type": None,
            "underlying_key": None,
            "underlying_symbol": "HDFCBANK",
            "exchange_token": "1333",
            "isin": "INE040A01034",
            "instrument_type": "EQUITY",
            "short_name": "HDFCBANK",
            "freeze_quantity": 0,
        },
        {
            "segment": "NSE_EQ",
            "name": "ICICI BANK",
            "instrument_key": "NSE_EQ|INE090A01021",
            "trading_symbol": "ICICIBANK",
            "lot_size": 1,
            "tick_size": 0.05,
            "expiry": None,
            "strike_price": 0.0,
            "option_type": None,
            "underlying_key": None,
            "underlying_symbol": "ICICIBANK",
            "exchange_token": "4963",
            "isin": "INE090A01021",
            "instrument_type": "EQUITY",
            "short_name": "ICICIBANK",
            "freeze_quantity": 0,
        },
        {
            "segment": "NSE_EQ",
            "name": "TATA CONSULTANCY SERVICES",
            "instrument_key": "NSE_EQ|INE467B01029",
            "trading_symbol": "TCS",
            "lot_size": 1,
            "tick_size": 0.05,
            "expiry": None,
            "strike_price": 0.0,
            "option_type": None,
            "underlying_key": None,
            "underlying_symbol": "TCS",
            "exchange_token": "11536",
            "isin": "INE467B01029",
            "instrument_type": "EQUITY",
            "short_name": "TCS",
            "freeze_quantity": 0,
        },
        {
            "segment": "NSE_EQ",
            "name": "INFOSYS",
            "instrument_key": "NSE_EQ|INE009A01021",
            "trading_symbol": "INFY",
            "lot_size": 1,
            "tick_size": 0.05,
            "expiry": None,
            "strike_price": 0.0,
            "option_type": None,
            "underlying_key": None,
            "underlying_symbol": "INFY",
            "exchange_token": "1594",
            "isin": "INE009A01021",
            "instrument_type": "EQUITY",
            "short_name": "INFY",
            "freeze_quantity": 0,
        },
        {
            "segment": "NSE_EQ",
            "name": "HINDUSTAN UNILEVER",
            "instrument_key": "NSE_EQ|INE030A01027",
            "trading_symbol": "HINDUNILVR",
            "lot_size": 1,
            "tick_size": 0.05,
            "expiry": None,
            "strike_price": 0.0,
            "option_type": None,
            "underlying_key": None,
            "underlying_symbol": "HINDUNILVR",
            "exchange_token": "356",
            "isin": "INE030A01027",
            "instrument_type": "EQUITY",
            "short_name": "HINDUNILVR",
            "freeze_quantity": 0,
        },
        {
            "segment": "NSE_EQ",
            "name": "ITC",
            "instrument_key": "NSE_EQ|INE154A01025",
            "trading_symbol": "ITC",
            "lot_size": 1,
            "tick_size": 0.05,
            "expiry": None,
            "strike_price": 0.0,
            "option_type": None,
            "underlying_key": None,
            "underlying_symbol": "ITC",
            "exchange_token": "1660",
            "isin": "INE154A01025",
            "instrument_type": "EQUITY",
            "short_name": "ITC",
            "freeze_quantity": 0,
        },
        {
            "segment": "NSE_EQ",
            "name": "AXIS BANK",
            "instrument_key": "NSE_EQ|INE238A01034",
            "trading_symbol": "AXISBANK",
            "lot_size": 1,
            "tick_size": 0.05,
            "expiry": None,
            "strike_price": 0.0,
            "option_type": None,
            "underlying_key": None,
            "underlying_symbol": "AXISBANK",
            "exchange_token": "5900",
            "isin": "INE238A01034",
            "instrument_type": "EQUITY",
            "short_name": "AXISBANK",
            "freeze_quantity": 0,
        },
        {
            "segment": "NSE_EQ",
            "name": "KOTAK MAHINDRA BANK",
            "instrument_key": "NSE_EQ|INE237A01028",
            "trading_symbol": "KOTAKBANK",
            "lot_size": 1,
            "tick_size": 0.05,
            "expiry": None,
            "strike_price": 0.0,
            "option_type": None,
            "underlying_key": None,
            "underlying_symbol": "KOTAKBANK",
            "exchange_token": "1922",
            "isin": "INE237A01028",
            "instrument_type": "EQUITY",
            "short_name": "KOTAKBANK",
            "freeze_quantity": 0,
        },
        {
            "segment": "NSE_EQ",
            "name": "STATE BANK OF INDIA",
            "instrument_key": "NSE_EQ|INE062A01020",
            "trading_symbol": "SBIN",
            "lot_size": 1,
            "tick_size": 0.05,
            "expiry": None,
            "strike_price": 0.0,
            "option_type": None,
            "underlying_key": None,
            "underlying_symbol": "SBIN",
            "exchange_token": "3045",
            "isin": "INE062A01020",
            "instrument_type": "EQUITY",
            "short_name": "SBIN",
            "freeze_quantity": 0,
        },
        # ----------------------------------------------------------------
        # NIFTY ATM options (weekly)
        # ----------------------------------------------------------------
        {
            "segment": "NSE_FO",
            "name": "NIFTY",
            "instrument_key": "NSE_FO|NIFTY24500CE",
            "trading_symbol": f"NIFTY{_NEXT_THURSDAY.replace('-','')[2:]}24500CE",
            "lot_size": 50,
            "tick_size": 0.05,
            "expiry": _NEXT_THURSDAY,
            "strike_price": 24500.0,
            "option_type": "CE",
            "underlying_key": "NSE_INDEX|Nifty 50",
            "underlying_symbol": "NIFTY",
            "exchange_token": "99001001",
            "isin": None,
            "instrument_type": "CE",
            "short_name": "NIFTY",
            "freeze_quantity": 2800,
        },
        {
            "segment": "NSE_FO",
            "name": "NIFTY",
            "instrument_key": "NSE_FO|NIFTY24500PE",
            "trading_symbol": f"NIFTY{_NEXT_THURSDAY.replace('-','')[2:]}24500PE",
            "lot_size": 50,
            "tick_size": 0.05,
            "expiry": _NEXT_THURSDAY,
            "strike_price": 24500.0,
            "option_type": "PE",
            "underlying_key": "NSE_INDEX|Nifty 50",
            "underlying_symbol": "NIFTY",
            "exchange_token": "99001002",
            "isin": None,
            "instrument_type": "PE",
            "short_name": "NIFTY",
            "freeze_quantity": 2800,
        },
        # ----------------------------------------------------------------
        # BANKNIFTY ATM options (weekly)
        # ----------------------------------------------------------------
        {
            "segment": "NSE_FO",
            "name": "BANKNIFTY",
            "instrument_key": "NSE_FO|BANKNIFTY52000CE",
            "trading_symbol": f"BANKNIFTY{_NEXT_THURSDAY.replace('-','')[2:]}52000CE",
            "lot_size": 15,
            "tick_size": 0.05,
            "expiry": _NEXT_THURSDAY,
            "strike_price": 52000.0,
            "option_type": "CE",
            "underlying_key": "NSE_INDEX|Nifty Bank",
            "underlying_symbol": "BANKNIFTY",
            "exchange_token": "99002001",
            "isin": None,
            "instrument_type": "CE",
            "short_name": "BANKNIFTY",
            "freeze_quantity": 900,
        },
        {
            "segment": "NSE_FO",
            "name": "BANKNIFTY",
            "instrument_key": "NSE_FO|BANKNIFTY52000PE",
            "trading_symbol": f"BANKNIFTY{_NEXT_THURSDAY.replace('-','')[2:]}52000PE",
            "lot_size": 15,
            "tick_size": 0.05,
            "expiry": _NEXT_THURSDAY,
            "strike_price": 52000.0,
            "option_type": "PE",
            "underlying_key": "NSE_INDEX|Nifty Bank",
            "underlying_symbol": "BANKNIFTY",
            "exchange_token": "99002002",
            "isin": None,
            "instrument_type": "PE",
            "short_name": "BANKNIFTY",
            "freeze_quantity": 900,
        },
    ]

    df = pd.DataFrame(rows)

    # Ensure dtypes match InstrumentManager expectations
    df["lot_size"] = df["lot_size"].astype(int)
    df["strike_price"] = df["strike_price"].astype(float)
    df["freeze_quantity"] = df["freeze_quantity"].astype(int)
    df["tick_size"] = df["tick_size"].astype(float)

    return df
