"""Phase D Universe Scanner CLI runner.

Usage
-----
    python -m src.research.run_universe_scan [--date YYYY-MM-DD]

Runs the daily universe scan for the Top-10 NSE equity universe:
  1. Loads the NSE instrument master from the local cache.
  2. Fetches historical bars for each constituent (via Upstox API if a valid
     access token is available, otherwise falls back to the local ``bars`` table).
  3. Computes liquidity, volatility, and relative-strength ranks.
  4. Persists results to the four ranking tables in Postgres.
  5. Prints a formatted ranking table to stdout.

Intended to be called by the daily pre-market scheduler (not during live
trading hours) so that strategies can consume fresh rankings at open.

Environment variables required
-------------------------------
LOCAL_DB_URL or SUPABASE_DB_URL (depending on ENV)
    PostgreSQL connection URL.
UPSTOX_ACCESS_TOKEN (optional)
    If provided, live historical data is fetched from Upstox.  When absent
    the runner falls back to reading bars from the ``bars`` table in Postgres
    using a simple DB-backed provider.

Exit codes
----------
0  Success
1  Missing instrument cache files
2  DB connection error
3  Ranking computation error
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Bar provider builders
# ---------------------------------------------------------------------------

def _build_upstox_provider(access_token: str):
    """Return a historical_provider backed by the live Upstox History API.

    Args:
        access_token: Valid Upstox OAuth2 access token.

    Returns:
        Callable matching the historical_provider contract.
    """
    from src.data.historical import HistoricalDataFetcher

    fetcher = HistoricalDataFetcher(access_token=access_token)

    def provider(instrument_key: str, interval: str, days: int) -> pd.DataFrame:
        try:
            df = fetcher.fetch_candles(
                instrument_key=instrument_key,
                interval="day" if interval == "day" else interval,
                days=days,
            )
            return df
        except Exception as exc:
            logger.warning("Upstox fetch failed for {}: {}", instrument_key, exc)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    return provider


def _build_db_provider(engine):
    """Return a historical_provider that reads from the local ``bars`` table.

    This is the fallback when no Upstox access token is available.  Only
    'day' interval is supported by this provider.

    Args:
        engine: SQLAlchemy sync engine.

    Returns:
        Callable matching the historical_provider contract.
    """
    from sqlalchemy import text

    def provider(instrument_key: str, interval: str, days: int) -> pd.DataFrame:
        try:
            sql = text(
                """
                SELECT ts, open, high, low, close, volume
                FROM bars
                WHERE instrument_key = :key
                  AND timeframe = :tf
                ORDER BY ts DESC
                LIMIT :n
                """
            )
            tf = "1d" if interval == "day" else interval
            with engine.connect() as conn:
                rows = conn.execute(
                    sql, {"key": instrument_key, "tf": tf, "n": days + 10}
                ).fetchall()

            if not rows:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

            df = pd.DataFrame(
                rows, columns=["ts", "open", "high", "low", "close", "volume"]
            )
            df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(IST)
            df = df.sort_values("ts").set_index("ts")
            return df
        except Exception as exc:
            logger.warning("DB bar fetch failed for {}: {}", instrument_key, exc)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    return provider


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_scan(trade_date: date) -> int:
    """Execute the full universe scan for ``trade_date`` and print results.

    Args:
        trade_date: Date to tag rankings with.

    Returns:
        Exit code (0 = success, non-zero = error).
    """
    logger.info("Phase D Universe Scanner starting for date {}", trade_date)

    # --- Instrument manager ---
    from src.data.instruments import InstrumentManager

    im = InstrumentManager()
    try:
        im.load_instruments()
    except Exception as exc:
        logger.error("Failed to load instrument master: {}", exc)
        return 1

    # --- DB engine ---
    try:
        from src.storage.db import get_sync_engine
        engine = get_sync_engine()
    except Exception as exc:
        logger.error("DB connection error: {}", exc)
        return 2

    # --- Historical provider ---
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN", "").strip()
    if access_token:
        logger.info("Using Upstox live historical provider")
        provider = _build_upstox_provider(access_token)
    else:
        logger.info(
            "UPSTOX_ACCESS_TOKEN not set — falling back to DB bars table"
        )
        provider = _build_db_provider(engine)

    # --- Scanner ---
    from src.research.universe_scanner import UniverseScanner

    scanner = UniverseScanner(
        instrument_manager=im,
        historical_provider=provider,
        db_engine=engine,
    )

    # --- Run ---
    try:
        df = scanner.rank_all(trade_date)
    except Exception as exc:
        logger.error("Ranking computation failed: {}", exc)
        return 3

    if df.empty:
        logger.warning("rank_all returned empty DataFrame — check universe_constituents table")
        return 0

    # --- Print table ---
    display_cols = [
        "symbol",
        "liquidity_rank",
        "volatility_rank",
        "rs_rank",
        "composite_score",
        "adv_paisa",
        "rs_score",
    ]
    display = df[display_cols].copy()

    # Convert ADV from paisa to crore for readability
    display["adv_cr"] = (display["adv_paisa"] / 1e7).round(2)
    display["composite_score"] = display["composite_score"].round(4)
    display["rs_score"] = display["rs_score"].round(2)

    # Sort by composite score descending
    display = display.sort_values("composite_score", ascending=False).reset_index(drop=True)
    display.index += 1  # 1-based rank

    print()
    print(f"=== Phase D Universe Ranking — {trade_date} ===")
    print()
    print(
        display[
            ["symbol", "liquidity_rank", "volatility_rank", "rs_rank",
             "composite_score", "adv_cr", "rs_score"]
        ].rename(
            columns={
                "symbol": "Symbol",
                "liquidity_rank": "LiqRank",
                "volatility_rank": "VolRank",
                "rs_rank": "RSRank",
                "composite_score": "Score",
                "adv_cr": "ADV(Cr)",
                "rs_score": "RS%",
            }
        ).to_string()
    )
    print()
    logger.info("Universe scan complete. {} symbols ranked.", len(df))
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.research.run_universe_scan",
        description="Run the Phase D daily universe scan and print rankings.",
    )
    parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        default=None,
        help=(
            "Trade date for the scan (default: today in IST). "
            "Use a past date to regenerate historical rankings."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)

    if args.date:
        try:
            trade_date = date.fromisoformat(args.date)
        except ValueError:
            print(f"ERROR: Invalid date format '{args.date}'. Use YYYY-MM-DD.", file=sys.stderr)
            sys.exit(1)
    else:
        trade_date = datetime.now(tz=IST).date()

    exit_code = run_scan(trade_date)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
