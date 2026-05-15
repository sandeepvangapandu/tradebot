#!/usr/bin/env python3
"""Build the 30-day Nifty 50 1-min CSV used by backtest harness tests.

Usage:
    python3 scripts/build_backtest_fixture.py

Writes to repo root as ``nifty50_1min_30days.csv`` so
tests/test_backtest_harness.py picks it up (REAL_CSV path).

Source: Upstox History API via HistoricalDataFetcher. Requires a valid
cached token (data/token_cache.json). Run after a successful bot session
or after manual_login.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from loguru import logger

from src.auth.token_manager import TokenManager
from src.data.historical import HistoricalDataFetcher

IST = ZoneInfo("Asia/Kolkata")
OUT_PATH = Path(__file__).resolve().parent.parent / "nifty50_1min_30days.csv"
INSTRUMENT = "NSE_INDEX|Nifty 50"


def main() -> int:
    tm = TokenManager()
    try:
        access_token = tm.get_valid_token()
    except Exception as exc:
        logger.error("Cannot get Upstox token — run manual_login first: {}", exc)
        return 1

    fetcher = HistoricalDataFetcher(access_token=access_token)

    df = fetcher.fetch_candles(
        instrument_key=INSTRUMENT,
        interval="1minute",
        days=30,
    )
    if df.empty:
        logger.error("No data returned")
        return 2

    # Prices come as paisa int from fetch_candles; convert back to rupees
    # because backtest tests pass prices_in_rupees=True.
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = df[col] / 100.0

    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})

    df.to_csv(OUT_PATH, index=False)
    logger.info("Wrote {} rows to {}", len(df), OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
