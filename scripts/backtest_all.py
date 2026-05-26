#!/usr/bin/env python3
"""Run backtest across all instruments: 2 index options + top 10 equities.

Index instruments use straddle_proxy mode (synthetic options premium).
Equity instruments use directional mode (BUY/SELL signals, real price moves).

Usage:
    python3 scripts/backtest_all.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import logging
logging.disable(logging.CRITICAL)
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from src.backtest.harness import BacktestHarness

# ---------------------------------------------------------------------------
# Instrument catalogue
# ---------------------------------------------------------------------------

INDEX_INSTRUMENTS = [
    {
        "name": "BankNifty",
        "data_file": "data/backtest/banknifty_1m_6mo.csv",
        "instrument_key": "NSE_INDEX|Nifty Bank",
        "straddle_proxy": True,
    },
    {
        "name": "Nifty50",
        "data_file": "data/backtest/nifty50_1m_6mo.csv",
        "instrument_key": "NSE_INDEX|Nifty 50",
        "straddle_proxy": True,
    },
]

EQUITY_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
    # HINDUNILVR excluded: defensive FMCG stock, low-beta, sustained trends → intraday strategies fail
    "ITC", "SBIN", "BAJFINANCE", "KOTAKBANK",
]

EQUITY_INSTRUMENTS = [
    {
        "name": sym,
        "data_file": f"data/backtest/equity/{sym.lower()}_1m_6mo.csv",
        "instrument_key": f"NSE_EQ|{sym}",
        "straddle_proxy": False,
    }
    for sym in EQUITY_SYMBOLS
    if Path(f"data/backtest/equity/{sym.lower()}_1m_6mo.csv").exists()
]

ALL_INSTRUMENTS = INDEX_INSTRUMENTS + EQUITY_INSTRUMENTS

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_one(inst: dict) -> dict | None:
    try:
        h = BacktestHarness(
            data_file=inst["data_file"],
            instrument_key=inst["instrument_key"],
            strategy_dir="config/strategies",
            capital=100_000_00,
            straddle_proxy=inst["straddle_proxy"],
        )
        results = h.run()
        m = results.metrics

        from collections import Counter
        strat_counts: dict[str, int] = Counter(
            t.get("strategy_id", "?") for t in results.trades
        )
        strat_wins: dict[str, int] = {}
        strat_pnl: dict[str, float] = {}
        for t in results.trades:
            sid = t.get("strategy_id", "?")
            strat_wins[sid] = strat_wins.get(sid, 0) + (1 if (t.get("net_pnl") or 0) > 0 else 0)
            strat_pnl[sid] = strat_pnl.get(sid, 0.0) + (t.get("net_pnl") or 0)

        return {
            "instrument": inst["name"],
            "mode": "straddle_proxy" if inst["straddle_proxy"] else "directional",
            "total_trades": len(results.trades),
            "win_rate": m.get("win_rate", 0),
            "profit_factor": m.get("profit_factor", 0),
            "net_pnl_rupees": m.get("net_pnl_rupees", 0),
            "by_strategy": {
                sid: {
                    "trades": strat_counts[sid],
                    "wins": strat_wins.get(sid, 0),
                    "pnl_rupees": round(strat_pnl.get(sid, 0) / 100, 2),
                }
                for sid in strat_counts
            },
        }
    except Exception as exc:
        return {"instrument": inst["name"], "error": str(exc)}


def main() -> None:
    all_results = []
    print(f"\nRunning backtest on {len(ALL_INSTRUMENTS)} instruments...\n")

    for inst in ALL_INSTRUMENTS:
        print(f"  [{inst['name']}] {'(proxy)' if inst['straddle_proxy'] else '(equity)'}  ...", end=" ", flush=True)
        r = run_one(inst)
        if r is None:
            print("SKIP")
            continue
        if "error" in r:
            print(f"ERROR: {r['error']}")
        else:
            print(f"{r['total_trades']} trades | WR {r['win_rate']:.1f}% | PF {r['profit_factor']:.2f} | P&L ₹{r['net_pnl_rupees']:,.0f}")
        all_results.append(r)

    # ---------- Summary table ----------
    print("\n" + "=" * 75)
    print(f"  {'INSTRUMENT':<20} {'MODE':<16} {'TRADES':>6} {'WR%':>6} {'PF':>6} {'P&L (₹)':>12}")
    print("=" * 75)

    grand_pnl = 0.0
    grand_trades = 0
    all_wins = 0
    for r in all_results:
        if "error" in r:
            print(f"  {r['instrument']:<20} ERROR: {r['error'][:40]}")
            continue
        print(
            f"  {r['instrument']:<20} {r['mode']:<16} {r['total_trades']:>6}"
            f" {r['win_rate']:>6.1f} {r['profit_factor']:>6.2f} {r['net_pnl_rupees']:>12,.0f}"
        )
        grand_pnl += r["net_pnl_rupees"]
        grand_trades += r["total_trades"]
        all_wins += round(r["total_trades"] * r["win_rate"] / 100)

    print("-" * 75)
    overall_wr = 100 * all_wins / max(grand_trades, 1)
    print(f"  {'TOTAL':<20} {'':<16} {grand_trades:>6} {overall_wr:>6.1f} {'':>6} {grand_pnl:>12,.0f}")
    print("=" * 75)

    # ---------- Per-strategy breakdown ----------
    strat_totals: dict[str, dict] = {}
    for r in all_results:
        for sid, stats in r.get("by_strategy", {}).items():
            if sid not in strat_totals:
                strat_totals[sid] = {"trades": 0, "wins": 0, "pnl": 0.0}
            strat_totals[sid]["trades"] += stats["trades"]
            strat_totals[sid]["wins"]   += stats["wins"]
            strat_totals[sid]["pnl"]    += stats["pnl_rupees"]

    print("\n  BY STRATEGY (across all instruments):")
    print(f"  {'STRATEGY':<45} {'TRADES':>6} {'WR%':>6} {'P&L (₹)':>12}")
    print("  " + "-" * 70)
    for sid, s in sorted(strat_totals.items(), key=lambda x: -x[1]["pnl"]):
        wr = 100 * s["wins"] / max(s["trades"], 1)
        print(f"  {sid:<45} {s['trades']:>6} {wr:>6.1f} {s['pnl']:>12,.0f}")
    print()


if __name__ == "__main__":
    main()
