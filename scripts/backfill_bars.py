#!/usr/bin/env python3
"""Backfill historical bars into the Postgres `bars` table.

Pulls 1-minute candles from Upstox History API for a configurable instrument
list, resamples in-memory to 5m / 15m / 1h, and bulk-inserts into the Postgres
`bars` table (Phase 0 schema). Idempotent — uses INSERT ON CONFLICT DO NOTHING.

Run once before bot startup so Bloomberg modules (universe_scanner,
volume_profile, portfolio_risk correlations) have
data immediately instead of waiting 60 days of live ticks.

USAGE
=====

    # Default: Top-10 + indices, last 60 days, all timeframes (1m/5m/15m/1h)
    python3 scripts/backfill_bars.py

    # Custom universe + range
    python3 scripts/backfill_bars.py --days 90 \
        --instruments "NSE_INDEX|Nifty 50,NSE_INDEX|Nifty Bank,NSE_EQ|INE002A01018"

    # Just one timeframe (faster)
    python3 scripts/backfill_bars.py --timeframes 5m,15m

    # Dry-run (show counts, don't insert)
    python3 scripts/backfill_bars.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

# Add repo root to path + load .env BEFORE importing storage layer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # python-dotenv optional

from src.auth.token_manager import TokenManager
from src.data.historical import HistoricalDataFetcher
from src.storage.db import get_sync_engine

IST = ZoneInfo("Asia/Kolkata")

# Default universe = Top-10 NSE equities (ISIN-verified) + NIFTY/BANKNIFTY indices
DEFAULT_UNIVERSE = [
    "NSE_INDEX|Nifty 50",
    "NSE_INDEX|Nifty Bank",
    "NSE_EQ|INE002A01018",   # RELIANCE
    "NSE_EQ|INE040A01034",   # HDFCBANK
    "NSE_EQ|INE090A01021",   # ICICIBANK
    "NSE_EQ|INE467B01029",   # TCS
    "NSE_EQ|INE009A01021",   # INFY
    "NSE_EQ|INE030A01027",   # HINDUNILVR
    "NSE_EQ|INE154A01025",   # ITC
    "NSE_EQ|INE238A01034",   # AXISBANK
    "NSE_EQ|INE237A01036",   # KOTAKBANK
    "NSE_EQ|INE062A01020",   # SBIN
]

# Timeframe alias → (resample rule, label stored in bars.timeframe)
TIMEFRAME_RULES: dict[str, tuple[str, str]] = {
    "1m":  ("1min",  "1m"),
    "5m":  ("5min",  "5m"),
    "15m": ("15min", "15m"),
    "30m": ("30min", "30m"),
    "1h":  ("60min", "1h"),
}

DEFAULT_TIMEFRAMES = ["1m", "5m", "15m", "1h"]


def resample_ohlcv(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1m bars to a coarser interval using OHLCV rules.

    Input: DataFrame indexed by timestamp (DatetimeIndex), columns
    open/high/low/close/volume. Output: same shape with `timestamp` column
    (reset_index) for downstream insert.
    """
    if df_1m.empty:
        return df_1m
    df = df_1m.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        # Tolerate either column or index named 'timestamp'
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        else:
            return pd.DataFrame()
    df = df.sort_index()
    agg = df.resample(rule, label="left", closed="left").agg(
        {
            "open":  "first",
            "high":  "max",
            "low":   "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna(subset=["open"])
    agg = agg.reset_index().rename(columns={"index": "timestamp"})
    if "timestamp" not in agg.columns and agg.columns[0] != "timestamp":
        agg = agg.rename(columns={agg.columns[0]: "timestamp"})
    return agg


def bulk_insert_bars(
    engine,
    instrument_key: str,
    timeframe_label: str,
    df: pd.DataFrame,
    batch_size: int = 1000,
    dry_run: bool = False,
) -> int:
    """INSERT each row via ON CONFLICT DO NOTHING. Returns count attempted."""
    if df.empty:
        return 0
    if dry_run:
        logger.info(
            "[dry-run] Would insert {} {} bars for {}",
            len(df), timeframe_label, instrument_key,
        )
        return len(df)

    from sqlalchemy import text
    sql = text(
        "INSERT INTO bars (instrument_key, timeframe, ts, open, high, low, close, volume) "
        "VALUES (:ik, :tf, :ts, :o, :h, :l, :c, :v) "
        "ON CONFLICT (instrument_key, timeframe, ts) DO NOTHING"
    )

    inserted = 0
    rows = df.to_dict(orient="records")
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        try:
            with engine.begin() as conn:
                conn.execute(
                    sql,
                    [
                        {
                            "ik": instrument_key,
                            "tf": timeframe_label,
                            "ts": row["timestamp"],
                            "o":  int(row["open"]),
                            "h":  int(row["high"]),
                            "l":  int(row["low"]),
                            "c":  int(row["close"]),
                            "v":  int(row.get("volume", 0)),
                        }
                        for row in batch
                    ],
                )
                inserted += len(batch)
        except Exception as exc:
            logger.warning("Batch insert failed for {} {}: {}", instrument_key, timeframe_label, exc)
    return inserted


def fetch_long_1m(
    fetcher: HistoricalDataFetcher,
    instrument_key: str,
    days: int,
    chunk_days: int = 25,
) -> pd.DataFrame:
    """Fetch >30 days of 1m bars by chunking (Upstox 1m limit = 30 days)."""
    end = datetime.now(IST)
    start = end - timedelta(days=days)
    chunks: list[pd.DataFrame] = []
    cur_to = end
    while cur_to > start:
        cur_from = max(start, cur_to - timedelta(days=chunk_days))
        try:
            df = fetcher.fetch_candles(
                instrument_key=instrument_key,
                interval="1minute",
                to_date=cur_to,
                from_date=cur_from,
            )
            if not df.empty:
                chunks.append(df)
        except Exception as exc:
            logger.warning("Chunk fetch failed for {} {}-{}: {}",
                           instrument_key, cur_from.date(), cur_to.date(), exc)
        cur_to = cur_from - timedelta(seconds=1)
    if not chunks:
        return pd.DataFrame()
    combined = pd.concat(chunks)
    # Drop dup timestamps (chunks overlap by 1s boundary)
    if isinstance(combined.index, pd.DatetimeIndex):
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return combined


def backfill_one(
    fetcher: HistoricalDataFetcher,
    engine,
    instrument_key: str,
    days: int,
    timeframes: list[str],
    dry_run: bool,
) -> dict:
    """Pull 1m bars from Upstox (chunked if >30 days), resample, persist."""
    result = {"instrument_key": instrument_key, "rows": {}, "elapsed_s": 0.0}
    t0 = time.time()
    try:
        if days > 30:
            df_1m = fetch_long_1m(fetcher, instrument_key, days)
        else:
            df_1m = fetcher.fetch_candles(
                instrument_key=instrument_key,
                interval="1minute",
                days=days,
            )
    except Exception as exc:
        logger.error("Fetch failed for {}: {}", instrument_key, exc)
        result["error"] = str(exc)
        return result

    if df_1m.empty:
        logger.warning("No data for {}", instrument_key)
        result["error"] = "no_data"
        return result

    # Normalize 1m to a flat DataFrame with `timestamp` column
    if isinstance(df_1m.index, pd.DatetimeIndex):
        df_1m_flat = df_1m.reset_index().rename(columns={df_1m.index.name or "index": "timestamp"})
        if "timestamp" not in df_1m_flat.columns:
            df_1m_flat = df_1m_flat.rename(columns={df_1m_flat.columns[0]: "timestamp"})
    else:
        df_1m_flat = df_1m

    for tf_alias in timeframes:
        rule, label = TIMEFRAME_RULES[tf_alias]
        df = df_1m_flat if tf_alias == "1m" else resample_ohlcv(df_1m_flat, rule)
        n = bulk_insert_bars(engine, instrument_key, label, df, dry_run=dry_run)
        result["rows"][tf_alias] = n
        logger.info("  {} {}: {} rows", instrument_key, tf_alias, n)

    result["elapsed_s"] = round(time.time() - t0, 2)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill bars into Postgres")
    parser.add_argument("--days", type=int, default=60,
                        help="Lookback days (default 60)")
    parser.add_argument("--instruments", type=str, default=None,
                        help="Comma-separated instrument_keys; omit to use Top-10+indices")
    parser.add_argument("--timeframes", type=str, default=",".join(DEFAULT_TIMEFRAMES),
                        help="Comma-separated timeframes from {1m,5m,15m,30m,1h}")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write to DB, just count")
    args = parser.parse_args()

    # Parse args
    instruments = (args.instruments.split(",") if args.instruments
                   else DEFAULT_UNIVERSE)
    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]
    bad_tf = [tf for tf in timeframes if tf not in TIMEFRAME_RULES]
    if bad_tf:
        logger.error("Unknown timeframe(s): {}. Supported: {}",
                     bad_tf, list(TIMEFRAME_RULES))
        return 1

    # Auth + clients
    token_mgr = TokenManager()
    try:
        access_token = token_mgr.get_valid_token()
    except Exception as exc:
        logger.error("Auth failed: {}", exc)
        return 2
    fetcher = HistoricalDataFetcher(access_token=access_token)
    engine = None if args.dry_run else get_sync_engine()

    logger.info("=" * 60)
    logger.info("Bars backfill")
    logger.info("  instruments: {}", len(instruments))
    logger.info("  days:        {}", args.days)
    logger.info("  timeframes:  {}", timeframes)
    logger.info("  dry-run:     {}", args.dry_run)
    logger.info("=" * 60)

    summary = []
    total_rows = 0
    t_all = time.time()
    for ik in instruments:
        result = backfill_one(fetcher, engine, ik, args.days, timeframes, args.dry_run)
        summary.append(result)
        total_rows += sum(result.get("rows", {}).values())

    elapsed_total = round(time.time() - t_all, 1)
    ok_count = sum(1 for r in summary if "error" not in r)
    error_count = len(summary) - ok_count

    logger.info("=" * 60)
    logger.info(
        "Done: {} ok / {} error | total rows {} | {}s",
        ok_count, error_count, total_rows, elapsed_total,
    )
    logger.info("=" * 60)

    # Persist run summary to logs
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"backfill_bars_{datetime.now(IST):%Y%m%d_%H%M%S}.json"
    out_path.write_text(json.dumps({
        "ts": datetime.now(IST).isoformat(),
        "args": vars(args),
        "ok": ok_count,
        "error": error_count,
        "total_rows": total_rows,
        "elapsed_s": elapsed_total,
        "details": summary,
    }, indent=2, default=str))
    logger.info("Run summary: {}", out_path)

    return 0 if error_count == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
