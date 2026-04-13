"""CLI entry point for the BacktestHarness.

Usage:
    python -m src.backtest.runner \\
        --data nifty50_1min_30days.csv \\
        --instrument "NSE_INDEX|Nifty Bank" \\
        --capital 1000000 \\
        --from 2026-01-01 \\
        --to 2026-03-31 \\
        --strategies config/strategies
"""

from __future__ import annotations

import argparse
import sys
from datetime import date


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a full-fidelity backtest using the live trading components.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to 1-minute OHLCV CSV file (paisa prices, IST timestamps)",
    )
    parser.add_argument(
        "--instrument", default="NSE_INDEX|Nifty Bank",
        help='Instrument key, e.g. "NSE_INDEX|Nifty Bank" (default)',
    )
    parser.add_argument(
        "--strategies", default="config/strategies",
        help="Directory containing strategy JSON files (default: config/strategies)",
    )
    parser.add_argument(
        "--capital", type=int, default=None,
        help="Initial capital in PAISA (default: from .env CAPITAL setting)",
    )
    parser.add_argument(
        "--from", dest="start_date", type=_parse_date, default=None,
        help="Start date YYYY-MM-DD (inclusive)",
    )
    parser.add_argument(
        "--to", dest="end_date", type=_parse_date, default=None,
        help="End date YYYY-MM-DD (inclusive)",
    )
    parser.add_argument(
        "--prices-in-rupees", action="store_true",
        help="CSV prices are in rupees — multiply by 100 to convert to paisa",
    )
    parser.add_argument(
        "--options-proxy", action="store_true",
        help="Transform index OHLCV into synthetic ATM option premium OHLCV",
    )
    parser.add_argument(
        "--straddle-proxy", action="store_true",
        help="Transform index OHLCV into synthetic ATM straddle premium OHLCV",
    )

    args = parser.parse_args()

    from src.backtest.harness import BacktestHarness

    harness = BacktestHarness(
        data_file=args.data,
        instrument_key=args.instrument,
        strategy_dir=args.strategies,
        capital=args.capital,
        start_date=args.start_date,
        end_date=args.end_date,
        prices_in_rupees=args.prices_in_rupees,
        options_proxy=args.options_proxy,
        straddle_proxy=args.straddle_proxy,
    )

    results = harness.run()
    m = results.metrics

    print("\n" + "─" * 50)
    print(f"  BACKTEST RESULTS")
    if args.start_date or args.end_date:
        print(f"  {args.start_date or 'start'} → {args.end_date or 'end'}")
    print("─" * 50)
    print(f"  Total Trades     : {m.get('total_trades', 0)}")
    print(f"  Win Rate         : {m.get('win_rate', 0):.1f}%")
    print(f"  Profit Factor    : {m.get('profit_factor', 0):.2f}")
    print(f"  Net P&L          : ₹ {m.get('net_pnl_rupees', 0):,.2f}")
    print(f"  Max Drawdown     : ₹ {m.get('max_drawdown_rupees', 0):,.2f} "
          f"({m.get('max_drawdown_pct', 0):.1f}%)")
    print(f"  Sharpe Ratio     : {m.get('sharpe_ratio', 0):.2f}")
    print(f"  Total Fees       : ₹ {m.get('total_fees_rupees', 0):,.2f}")
    print(f"  Return           : {m.get('return_pct', 0):+.2f}%")
    print(f"  Bars Processed   : {results.bars_processed}")
    print("─" * 50 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
