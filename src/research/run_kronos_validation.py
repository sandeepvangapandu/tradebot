"""CLI runner for Kronos prediction validation.

Computes directional-accuracy and close-MAE metrics against actual bars,
optionally persists results to ``kronos_accuracy_daily``, and prints a
human-readable report.

Usage examples::

    # Compute yesterday's accuracy and print a one-line summary (default)
    python -m src.research.run_kronos_validation

    # Compute accuracy for a specific date
    python -m src.research.run_kronos_validation --date 2026-05-13

    # Generate a 28-day Markdown report to stdout
    python -m src.research.run_kronos_validation --report --days 28

    # Filter to a single instrument over the default 30-day window
    python -m src.research.run_kronos_validation --report --instrument "NSE_INDEX|Nifty 50"
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Logging setup — before importing project modules so level propagates.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kronos_validation_cli")

IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Lazy engine builder — only imported when actually needed so that unit-test
# runs that import this module don't require a live DB connection.
# ---------------------------------------------------------------------------


def _build_engine():
    """Try to build a SQLAlchemy engine from DATABASE_URL env var.

    Returns:
        SQLAlchemy engine, or ``None`` when the env var is absent or the
        import fails.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        logger.warning(
            "DATABASE_URL is not set — running without DB persistence.  "
            "Set DATABASE_URL to a valid SQLAlchemy URL for production use."
        )
        return None

    try:
        from sqlalchemy import create_engine as _ce

        engine = _ce(db_url)
        logger.debug("DB engine created from DATABASE_URL")
        return engine
    except Exception:
        logger.exception("Failed to create DB engine from DATABASE_URL=%r", db_url)
        return None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_kronos_validation",
        description="Evaluate Kronos model predictions against actual bar outcomes.",
    )
    parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        help=(
            "Compute accuracy for this specific trade date (IST). "
            "Defaults to yesterday."
        ),
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print a full Markdown accuracy report instead of just the daily summary.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        metavar="N",
        help="Rolling-window length for --report (default: 30).",
    )
    parser.add_argument(
        "--instrument",
        metavar="INSTRUMENT_KEY",
        help="Filter to a single instrument (e.g. 'NSE_INDEX|Nifty 50').",
    )
    return parser


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI runner.

    Args:
        argv: Optional argument list (useful for testing). Defaults to
            ``sys.argv[1:]`` when ``None``.

    Returns:
        Exit code — ``0`` on success, non-zero on error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    from src.research.kronos_validator import KronosValidator

    engine = _build_engine()
    validator = KronosValidator(db_engine=engine)

    # ------------------------------------------------------------------
    # Determine trade_date
    # ------------------------------------------------------------------
    if args.date:
        try:
            trade_date = date.fromisoformat(args.date)
        except ValueError:
            logger.error("Invalid --date value %r. Expected YYYY-MM-DD.", args.date)
            return 2
    else:
        # Default: yesterday in IST
        today_ist = datetime.now(tz=IST).date()
        trade_date = today_ist - timedelta(days=1)

    # ------------------------------------------------------------------
    # Report mode
    # ------------------------------------------------------------------
    if args.report:
        report = validator.generate_report(days=args.days)
        print(report)
        return 0

    # ------------------------------------------------------------------
    # Daily compute mode (default)
    # ------------------------------------------------------------------
    logger.info("Computing Kronos accuracy for %s ...", trade_date)
    metrics_list = validator.compute_daily(trade_date)

    if not metrics_list:
        print(
            f"No Kronos predictions found for {trade_date}. "
            "Ensure the model ran in shadow mode on this date and that "
            "DATABASE_URL is set correctly."
        )
        return 0

    # Print summary table
    print(
        f"\nKronos Accuracy — {trade_date}"
        + (f" | instrument: {args.instrument}" if args.instrument else "")
    )
    print("-" * 72)
    print(
        f"{'Instrument':<40} {'Horizon':>7} {'Count':>6} "
        f"{'Dir%':>7} {'MAE%':>7}"
    )
    print("-" * 72)

    for m in sorted(metrics_list, key=lambda x: (x.instrument_key, x.horizon_bars)):
        if args.instrument and m.instrument_key != args.instrument:
            continue
        flag = " *" if m.above_baseline else ""
        print(
            f"{m.instrument_key:<40} {m.horizon_bars:>7} {m.prediction_count:>6} "
            f"{m.direction_accuracy_pct:>6.1f}% {m.close_mae_pct:>6.3f}%{flag}"
        )

    print("-" * 72)
    total = sum(m.prediction_count for m in metrics_list)
    correct = sum(m.direction_correct for m in metrics_list)
    overall = correct / total * 100.0 if total > 0 else 0.0
    from src.research.kronos_validator import KronosValidator as _KV

    rec = _KV._recommendation(overall)
    print(
        f"{'TOTAL':<40} {'':>7} {total:>6} {overall:>6.1f}%"
        f"  → {rec}"
    )
    print()

    # Also print rolling summary if DB is available
    if engine is not None:
        summary = validator.get_accuracy_summary(
            days=30, instrument_key=args.instrument
        )
        print(
            f"Rolling 30-day accuracy: {summary['overall_accuracy_pct']:.1f}%  "
            f"({summary['total_predictions']:,} predictions)  "
            f"Recommendation: {summary['recommendation']}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
