"""Kronos shadow-mode CLI runner.

Loads the Kronos model once, iterates over the instrument universe (Top-10 NSE
large-caps + indices), pulls last N bars from the ``bars`` table, runs
prediction for each instrument, and persists results to ``kronos_forecasts``.

Designed for scheduling every 5 minutes during market hours via cron/APScheduler.

Usage::

    python -m src.research.run_kronos_shadow --model-size small --horizon 12 --timeframe 5m

Options:
    --model-size    Kronos model variant: mini | small | base  (default: small)
    --horizon       Bars ahead to forecast                      (default: 12)
    --timeframe     Candle timeframe label                      (default: 5m)
    --context       Max historical bars to feed model          (default: 400)
    --device        PyTorch device: cpu | cuda:0 | mps          (default: cpu)
    --dry-run       Run without persisting to DB

This is SHADOW MODE only — no orders are generated from Kronos predictions.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Instrument universe — Top-10 NSE large-caps + indices
# ---------------------------------------------------------------------------

UNIVERSE = [
    "NSE_EQ|INE002A01018",   # Reliance Industries
    "NSE_EQ|INE040A01034",   # HDFC Bank
    "NSE_EQ|INE009A01021",   # Infosys
    "NSE_EQ|INE467B01029",   # TCS
    "NSE_EQ|INE030A01027",   # ICICI Bank
    "NSE_EQ|INE062A01020",   # Kotak Mahindra Bank
    "NSE_EQ|INE090A01021",   # Wipro
    "NSE_EQ|INE397D01024",   # HCL Technologies
    "NSE_EQ|INE238A01034",   # Axis Bank
    "NSE_EQ|INE001A01036",   # Bajaj Finance
    "NSE_INDEX|Nifty 50",
    "NSE_INDEX|Nifty Bank",
]


def _get_db_engine():
    """Return SQLAlchemy engine from DATABASE_URL env var."""
    from sqlalchemy import create_engine
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL environment variable not set.")
        sys.exit(1)
    return create_engine(db_url, pool_pre_ping=True)


def _fetch_bars(engine, instrument_key: str, context: int, timeframe: str) -> pd.DataFrame | None:
    """Fetch last `context` bars for instrument from the bars table.

    Returns None if insufficient data.
    """
    try:
        from sqlalchemy import text
        # Map timeframe label to interval string for filtering
        tf_map = {"1m": "1 minute", "5m": "5 minutes", "15m": "15 minutes",
                  "1h": "1 hour", "1d": "1 day"}
        interval = tf_map.get(timeframe, "5 minutes")

        sql = text("""
            SELECT ts, open, high, low, close, volume,
                   COALESCE(amount, close * volume) AS amount
            FROM bars
            WHERE instrument_key = :ikey
              AND timeframe = :tf
            ORDER BY ts DESC
            LIMIT :lim
        """)
        with engine.connect() as conn:
            rows = conn.execute(sql, {"ikey": instrument_key, "tf": timeframe, "lim": context}).fetchall()

        if not rows:
            logger.warning("No bars for {} timeframe={}", instrument_key, timeframe)
            return None

        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "amount"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Asia/Kolkata")
        df = df.sort_values("ts").set_index("ts")
        return df

    except Exception as exc:
        logger.warning("Failed to fetch bars for {}: {}", instrument_key, exc)
        return None


def run_shadow(
    model_size: str = "small",
    horizon: int = 12,
    timeframe: str = "5m",
    context: int = 400,
    device: str = "cpu",
    dry_run: bool = False,
) -> None:
    """Main shadow-mode loop — loads model, iterates universe, persists forecasts."""
    from src.research.kronos_predictor import KronosForecaster

    logger.info(
        "Kronos shadow runner starting | model={} horizon={} timeframe={} context={} dry_run={}",
        model_size, horizon, timeframe, context, dry_run,
    )

    engine = None if dry_run else _get_db_engine()

    forecaster = KronosForecaster(
        model_size=model_size,
        device=device,
        max_context=context,
        db_engine=engine,
    )

    # Load model weights once (downloads from HuggingFace on first run)
    try:
        forecaster.load()
    except Exception as exc:
        logger.error("Failed to load Kronos model: {}", exc)
        sys.exit(1)

    cycle_start = time.time()
    results = {"ok": 0, "skip": 0, "error": 0}

    for ikey in UNIVERSE:
        try:
            if engine is not None:
                bars = _fetch_bars(engine, ikey, context, timeframe)
            else:
                # Dry-run: generate synthetic bars for smoke test
                logger.info("[dry-run] Generating synthetic bars for {}", ikey)
                ts = pd.date_range(
                    datetime.now(tz=IST).replace(second=0, microsecond=0),
                    periods=50,
                    freq="-5min",
                    tz="Asia/Kolkata",
                )[::-1]
                import numpy as np
                rng = np.random.default_rng()
                close = 50000.0 + rng.normal(0, 100, 50).cumsum()
                bars = pd.DataFrame(
                    {
                        "open": close - 20,
                        "high": close + 50,
                        "low": close - 50,
                        "close": close,
                        "volume": rng.integers(1000, 3000, 50).astype(float),
                        "amount": close * rng.integers(1000, 3000, 50).astype(float),
                    },
                    index=ts,
                )

            if bars is None or len(bars) < 10:
                logger.warning("Skipping {} — insufficient bars ({})", ikey, len(bars) if bars is not None else 0)
                results["skip"] += 1
                continue

            summary = forecaster.predict_summary(
                instrument_key=ikey,
                bars=bars,
                timeframe=timeframe,
                horizon=horizon,
                persist=(not dry_run),
            )

            if summary:
                logger.info(
                    "{} => {} | change={:.2f}% | range={:.2f}% | {}ms",
                    ikey,
                    summary.get("predicted_direction"),
                    summary.get("predicted_change_pct", 0.0),
                    summary.get("predicted_range_pct", 0.0),
                    summary.get("inference_ms", 0),
                )
                results["ok"] += 1
            else:
                results["error"] += 1

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
            break
        except Exception as exc:
            logger.exception("Unexpected error for {}: {}", ikey, exc)
            results["error"] += 1

    elapsed = time.time() - cycle_start
    logger.info(
        "Cycle complete in {:.1f}s | ok={} skip={} error={}",
        elapsed, results["ok"], results["skip"], results["error"],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_kronos_shadow",
        description="Run Kronos foundation-model forecasts in shadow mode.",
    )
    parser.add_argument("--model-size", default="small", choices=["mini", "small", "base"],
                        help="Kronos model variant (default: small)")
    parser.add_argument("--horizon", type=int, default=12,
                        help="Bars ahead to forecast (default: 12)")
    parser.add_argument("--timeframe", default="5m",
                        choices=["1m", "5m", "15m", "1h", "1d"],
                        help="Candle timeframe label (default: 5m)")
    parser.add_argument("--context", type=int, default=400,
                        help="Max historical bars fed to model (default: 400)")
    parser.add_argument("--device", default="cpu",
                        help="PyTorch device: cpu | cuda:0 | mps (default: cpu)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without persisting to DB; uses synthetic bars")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_shadow(
        model_size=args.model_size,
        horizon=args.horizon,
        timeframe=args.timeframe,
        context=args.context,
        device=args.device,
        dry_run=args.dry_run,
    )
