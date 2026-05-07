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
        "--data", default=None,
        help="Path to single 1-minute OHLCV CSV file (paisa prices, IST timestamps)",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Directory containing multiple CSV files for parallel backtesting",
    )
    parser.add_argument(
        "--parallel-workers", type=int, default=4,
        help="Max number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--instrument", default="NSE_INDEX|Nifty Bank",
        help='Instrument key for single file run, e.g. "NSE_INDEX|Nifty Bank" (default)',
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
    parser.add_argument(
        "--strategy-only", default=None,
        help="Run only the specified strategy (by name). Useful for isolated backtests.",
    )

    args = parser.parse_args()
    
    if not args.data and not args.data_dir:
        parser.error("Must provide either --data or --data-dir")

    from src.backtest.harness import BacktestHarness, run_multi_symbol_backtest

    def _print_metrics(title: str, m: dict, bars: int):
        print("\n" + "─" * 50)
        print(f"  {title}")
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
        print(f"  Bars Processed   : {bars}")
        print("─" * 50 + "\n")

    if args.data_dir:
        from pathlib import Path
        data_dir = Path(args.data_dir)
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {args.data_dir}")
            return 1
            
        data_files = {}
        for f in csv_files:
            # Map filename to instrument roughly
            name = f.stem.split("_")[0].upper()
            inst_key = f"NSE_INDEX|{name}"
            data_files[inst_key] = str(f)
            
        print(f"Starting parallel backtest for {len(data_files)} instruments with {args.parallel_workers} workers...")
        results_map = run_multi_symbol_backtest(
            data_files=data_files,
            strategy_dir=args.strategies,
            capital=args.capital,
            start_date=args.start_date,
            end_date=args.end_date,
            max_workers=args.parallel_workers,
        )
        
        # Portfolio aggregation
        port_trades = 0
        port_pnl = 0.0
        port_fees = 0.0
        port_wins = 0
        port_losses = 0
        port_gross_profit = 0.0
        port_gross_loss = 0.0
        
        for inst, res in results_map.items():
            _print_metrics(f"RESULTS FOR {inst}", res.metrics, res.bars_processed)
            m = res.metrics
            port_trades += m.get("total_trades", 0)
            port_pnl += m.get("net_pnl_rupees", 0.0)
            port_fees += m.get("total_fees_rupees", 0.0)
            
            # Reconstruct wins/losses for port profit factor and win rate
            trades_list = res.trades
            for t in trades_list:
                pnl = t.get("net_pnl", 0) / 100.0
                if pnl > 0:
                    port_wins += 1
                    port_gross_profit += pnl
                else:
                    port_losses += 1
                    port_gross_loss += abs(pnl)
                    
        port_wr = (port_wins / port_trades * 100) if port_trades > 0 else 0.0
        port_pf = (port_gross_profit / port_gross_loss) if port_gross_loss > 0 else float('inf')
        
        port_metrics = {
            "total_trades": port_trades,
            "win_rate": port_wr,
            "profit_factor": port_pf,
            "net_pnl_rupees": port_pnl,
            "total_fees_rupees": port_fees,
            "max_drawdown_rupees": 0, # Difficult to accurately aggregate without merged equity curve
            "max_drawdown_pct": 0,
            "sharpe_ratio": 0,
            "return_pct": (port_pnl / ((args.capital or 100000000)/100)) * 100
        }
        _print_metrics("AGGREGATED PORTFOLIO RESULTS", port_metrics, 0)

    else:
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
            strategy_only=args.strategy_only,
        )

        results = harness.run()
        _print_metrics("BACKTEST RESULTS", results.metrics, results.bars_processed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
