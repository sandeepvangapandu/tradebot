#!/usr/bin/env python3
"""Run backtest across all instruments: 2 index options + top 10 equities.

Index instruments use straddle_proxy mode (synthetic options premium).
Equity instruments use directional mode (BUY/SELL signals, real price moves).

Usage:
    python3 scripts/backtest_all.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import time as dt_time
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
        # Anchor straddle strike at 10:00 AM (strategy entry window) not 9:15 AM open
        # so the proxy doesn't simulate a deeply ITM straddle after early moves (High #4 fix)
        "entry_anchor_time": dt_time(10, 0),
    },
    {
        "name": "Nifty50",
        "data_file": "data/backtest/nifty50_1m_6mo.csv",
        "instrument_key": "NSE_INDEX|Nifty 50",
        "straddle_proxy": True,
        "entry_anchor_time": dt_time(10, 0),
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
# Worker initializer (runs once per subprocess on spawn)
# ---------------------------------------------------------------------------

def _worker_init() -> None:
    _root = str(Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(_root) / ".env")
    except ImportError:
        pass
    import logging as _logging
    _logging.disable(_logging.CRITICAL)
    try:
        from loguru import logger as _log
        _log.remove()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Per-instrument runner
# ---------------------------------------------------------------------------

def run_one(inst: dict) -> dict | None:
    # Each worker gets its own temp SQLite to avoid write-lock contention
    tmp_db_path = None
    try:
        fd, tmp_db_path = tempfile.mkstemp(suffix=".db", prefix="bt_")
        os.close(fd)
        database_url = f"sqlite:///{tmp_db_path}"

        h = BacktestHarness(
            data_file=inst["data_file"],
            instrument_key=inst["instrument_key"],
            strategy_dir="config/strategies",
            capital=100_000_00,
            straddle_proxy=inst["straddle_proxy"],
            entry_anchor_time=inst.get("entry_anchor_time"),
            database_url=database_url,
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
    finally:
        if tmp_db_path:
            try:
                os.unlink(tmp_db_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    n_workers = min(os.cpu_count() or 4, len(ALL_INSTRUMENTS))
    print(f"\nRunning backtest on {len(ALL_INSTRUMENTS)} instruments (parallel, workers={n_workers})...\n")

    # Preserve insertion order for summary table
    results_map: dict[str, dict] = {}

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as pool:
        future_to_inst = {pool.submit(run_one, inst): inst for inst in ALL_INSTRUMENTS}
        for future in as_completed(future_to_inst):
            inst = future_to_inst[future]
            try:
                r = future.result()
            except Exception as exc:
                r = {"instrument": inst["name"], "error": str(exc)}

            if r is None:
                print(f"  [{inst['name']}] SKIP")
                continue
            if "error" in r:
                print(f"  [{inst['name']}] {'(proxy)' if inst['straddle_proxy'] else '(equity)'}  ERROR: {r['error'][:60]}")
            else:
                print(
                    f"  [{inst['name']}] {'(proxy)' if inst['straddle_proxy'] else '(equity)'}  "
                    f"{r['total_trades']} trades | WR {r['win_rate']:.1f}% | "
                    f"PF {r['profit_factor']:.2f} | P&L ₹{r['net_pnl_rupees']:,.0f}"
                )
            results_map[inst["name"]] = r

    # Ordered summary: maintain catalogue order
    all_results = [results_map[inst["name"]] for inst in ALL_INSTRUMENTS if inst["name"] in results_map]

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
