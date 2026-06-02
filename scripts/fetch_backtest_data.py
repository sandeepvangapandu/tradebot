#!/usr/bin/env python3
"""Fetch 18 months of 1-min OHLCV data from Upstox for all active backtest instruments.

Writes CSV files to data/backtest/ (indices) and data/backtest/equity/ (equities).
Format: timestamp,open,high,low,close,volume — prices in PAISA (₹ × 100).

Requires a valid cached Upstox token (data/token_cache.json).
Run after a successful bot session or `python3 scripts/exchange_token.py`.

Usage:
    python3 scripts/fetch_backtest_data.py
    python3 scripts/fetch_backtest_data.py --months 6
    python3 scripts/fetch_backtest_data.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

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
CHUNK_DAYS = 25  # Upstox 1-min limit is 30 days per request; use 25 for safety

# ---------------------------------------------------------------------------
# Active instrument catalogue
# ---------------------------------------------------------------------------

# Active instruments for backtest — survivor universe (PF > 1.0 only)
# out_file is a template: {mo} gets replaced with the actual months count at runtime
INSTRUMENTS = [
    # Indices (straddle proxy backtest)
    {
        "name":           "BankNifty",
        "instrument_key": "NSE_INDEX|Nifty Bank",
        "out_file":       "data/backtest/banknifty_1m_{mo}mo.csv",
        "type":           "index",
    },
    {
        "name":           "Nifty50",
        "instrument_key": "NSE_INDEX|Nifty 50",
        "out_file":       "data/backtest/nifty50_1m_{mo}mo.csv",
        "type":           "index",
    },
    {
        "name":           "FinNifty",
        "instrument_key": "NSE_INDEX|Nifty Fin Service",
        "out_file":       "data/backtest/finnifty_1m_{mo}mo.csv",
        "type":           "index",
    },
    # Equities — survivor universe (PF > 1.0 confirmed over 18mo)
    {
        "name":           "ADANIENT",
        "instrument_key": "NSE_EQ|INE423A01024",
        "out_file":       "data/backtest/equity/adanient_1m_{mo}mo.csv",
        "type":           "equity",
    },
]


def fetch_chunked(
    fetcher: HistoricalDataFetcher,
    instrument_key: str,
    months: int,
    dry_run: bool = False,
) -> pd.DataFrame | None:
    """Fetch 1-min bars for `months` months via chunked 25-day requests."""
    end = datetime.now(IST).replace(hour=15, minute=30, second=0, microsecond=0)
    start = end - timedelta(days=int(months * 30.5))

    if dry_run:
        chunks_needed = (end - start).days // CHUNK_DAYS + 1
        logger.info(f"  [dry-run] would fetch ~{chunks_needed} chunks from {start.date()} to {end.date()}")
        return None

    chunks: list[pd.DataFrame] = []
    cur_to = end
    total_chunks = 0

    while cur_to > start:
        cur_from = max(start, cur_to - timedelta(days=CHUNK_DAYS))
        try:
            df = fetcher.fetch_candles(
                instrument_key=instrument_key,
                interval="1minute",
                to_date=cur_to,
                from_date=cur_from,
            )
            if not df.empty:
                chunks.append(df)
                total_chunks += 1
                logger.info(f"  chunk {total_chunks}: {cur_from.date()} → {cur_to.date()} = {len(df)} bars")
            else:
                logger.warning(f"  chunk {cur_from.date()} → {cur_to.date()}: empty")
        except Exception as exc:
            logger.warning(f"  chunk {cur_from.date()} → {cur_to.date()} failed: {exc}")

        cur_to = cur_from - timedelta(seconds=1)
        time.sleep(0.1)  # stay under 40 req/s rate limit

    if not chunks:
        return None

    combined = pd.concat(chunks)

    # Deduplicate and sort
    if isinstance(combined.index, pd.DatetimeIndex):
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()
        combined = combined.reset_index().rename(columns={combined.index.name or "index": "timestamp"})

    if "timestamp" not in combined.columns:
        combined = combined.rename(columns={combined.columns[0]: "timestamp"})

    combined = combined.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    return combined


def save_csv(df: pd.DataFrame, out_file: str) -> None:
    """Write DataFrame to CSV ensuring prices are in paisa (already converted by fetcher)."""
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"  Saved {len(df):,} bars → {out}")


def update_backtest_symlinks(instruments: list[dict], months: int) -> None:
    """Update backtest scripts to reference newly-fetched CSV filenames."""
    new_suffix = f"_1m_{months}mo.csv"
    for script in ["scripts/backtest_all.py", "scripts/backtest_poe.py"]:
        bt_script = Path(script)
        if not bt_script.exists():
            continue
        content = bt_script.read_text()
        # Replace any existing _1m_NNmo.csv suffix with the new one
        import re
        updated = re.sub(r"_1m_\d+mo\.csv", new_suffix, content)
        if updated != content:
            bt_script.write_text(updated)
            logger.info(f"  Updated {script}: *_1m_NNmo.csv → {new_suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch 1-min backtest data from Upstox (max 24 months for 1-min interval)")
    parser.add_argument("--months", type=int, default=24, help="Months of history to fetch (default 24, max 24 for 1-min)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fetched without API calls")
    parser.add_argument("--instruments", type=str, default=None,
                        help="Comma-separated names to fetch (default: all). E.g. TCS,SBIN")
    args = parser.parse_args()

    target_names = {n.strip().upper() for n in args.instruments.split(",")} if args.instruments else None
    instruments = [
        {**i, "out_file": i["out_file"].replace("{mo}", str(args.months))}
        for i in INSTRUMENTS
        if target_names is None or i["name"].upper() in target_names
    ]

    if not instruments:
        print("No matching instruments found.")
        sys.exit(1)

    print(f"\nFetching {args.months}-month 1-min data for {len(instruments)} instruments\n")

    if not args.dry_run:
        try:
            token_mgr = TokenManager()
            access_token = token_mgr.get_valid_token()
        except Exception as exc:
            print(f"\nERROR: Auth failed — {exc}")
            print("Run: python3 scripts/exchange_token.py   (or let the bot auto-refresh)")
            sys.exit(2)
        fetcher = HistoricalDataFetcher(access_token=access_token)
    else:
        fetcher = None

    results = []
    for inst in instruments:
        print(f"\n[{inst['name']}] {inst['instrument_key']}")
        t0 = time.time()
        df = fetch_chunked(fetcher, inst["instrument_key"], args.months, dry_run=args.dry_run)

        if args.dry_run:
            # estimate: ~22 chunks × ~270 bars/chunk trading days
            est_bars = int(args.months * 30.5 / CHUNK_DAYS) * 270 * CHUNK_DAYS // 375 * 375
            print(f"  [dry-run] ~{est_bars:,} bars expected | ~{int(args.months*30.5/CHUNK_DAYS)+1} API calls")
            results.append({"name": inst["name"], "rows": est_bars, "ok": True})
        elif df is not None and not df.empty:
            save_csv(df, inst["out_file"])
            elapsed = round(time.time() - t0, 1)
            rows = len(df)
            first = df["timestamp"].iloc[0] if "timestamp" in df.columns else "?"
            last  = df["timestamp"].iloc[-1] if "timestamp" in df.columns else "?"
            print(f"  OK  {rows:,} bars | {first} → {last} | {elapsed}s")
            results.append({"name": inst["name"], "rows": rows, "ok": True})
        else:
            print(f"  FAILED — no data returned")
            results.append({"name": inst["name"], "rows": 0, "ok": False})

    # Update backtest scripts to reference the newly-fetched CSV filenames
    if not args.dry_run and any(r["ok"] for r in results):
        print(f"\nUpdating backtest scripts to reference {args.months}mo CSVs...")
        update_backtest_symlinks(instruments, args.months)

    print(f"\n{'='*50}")
    print(f"Done: {sum(r['ok'] for r in results)}/{len(results)} instruments fetched")
    for r in results:
        status = "OK" if r["ok"] else "FAIL"
        print(f"  [{status}] {r['name']}: {r['rows']:,} bars")
    print()


if __name__ == "__main__":
    main()
