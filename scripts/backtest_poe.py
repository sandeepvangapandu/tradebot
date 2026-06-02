#!/usr/bin/env python3
"""Backtest with percent_of_equity sizing vs fixed_quantity.

Runs the same instrument set as backtest_all.py but overrides position sizing
to percent_of_equity calibrated to match initial fixed_qty at ₹5L capital.
Compounding kicks in: as equity grows, lot sizes scale proportionally.

Usage:
    python3 scripts/backtest_poe.py
"""
from __future__ import annotations

import json
import os
import shutil
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

# Calibrated percent_of_equity values to match fixed_qty at ₹5L starting capital.
# formula: pct = fixed_qty × avg_proxy_price_paisa / initial_capital_paisa × 100
# BankNifty proxy: 50 lots × ~47,000 paisa / 50,000,000 paisa = ~4.7% → 5%
# Nifty50 proxy:   10 lots × ~22,000 paisa / 50,000,000 paisa = ~0.44% → 0.5%
# FinNifty proxy:  10 lots × ~20,000 paisa / 50,000,000 paisa = ~0.40% → 0.5%
# Equity (equal-value per trade — 5% of ₹5L = ₹25k per position):
#   ADANIENT @₹2800 → ~9 shares; SBIN @₹600 → ~42 shares; TCS @₹3500 → ~7 shares
POE_PERCENTS: dict[str, float] = {
    "expiry_straddle_sell_banknifty": 5.0,
    "expiry_straddle_sell_nifty50":   0.5,
    "expiry_straddle_sell_finnifty":  0.5,
    "equity_supertrend_breakout":     25.0,  # ~50 shares for ADANIENT @₹2800, scales with equity
}

INITIAL_CAPITAL = 500_000_00  # ₹5L in paisa

INDEX_INSTRUMENTS = [
    {
        "name": "BankNifty",
        "data_file": "data/backtest/banknifty_1m_24mo.csv",
        "instrument_key": "NSE_INDEX|Nifty Bank",
        "straddle_proxy": True,
        "entry_anchor_time": dt_time(10, 0),
    },
    {
        "name": "Nifty50",
        "data_file": "data/backtest/nifty50_1m_24mo.csv",
        "instrument_key": "NSE_INDEX|Nifty 50",
        "straddle_proxy": True,
        "entry_anchor_time": dt_time(10, 0),
    },
    {
        "name": "FinNifty",
        "data_file": "data/backtest/finnifty_1m_24mo.csv",
        "instrument_key": "NSE_INDEX|Nifty Fin Service",
        "straddle_proxy": True,
        "entry_anchor_time": dt_time(10, 0),
    },
]

EQUITY_SYMBOLS = [
    "TCS", "SBIN", "INFY",
    "ADANIENT", "BHARTIARTL", "TATAMOTORS",
]

EQUITY_INSTRUMENTS = [
    {
        "name": sym,
        "data_file": f"data/backtest/equity/{sym.lower()}_1m_24mo.csv",
        "instrument_key": f"NSE_EQ|{sym}",
        "straddle_proxy": False,
    }
    for sym in EQUITY_SYMBOLS
    if Path(f"data/backtest/equity/{sym.lower()}_1m_24mo.csv").exists()
]

ALL_INSTRUMENTS = INDEX_INSTRUMENTS + EQUITY_INSTRUMENTS


def _build_poe_strategies_dir() -> Path:
    """Copy strategy configs to a temp dir with percent_of_equity sizing applied."""
    src = Path("config/strategies")
    tmp = Path(tempfile.mkdtemp(prefix="bt_poe_"))
    shutil.copytree(src, tmp / "strategies")
    strat_dir = tmp / "strategies"

    for json_file in strat_dir.glob("*.json"):
        with open(json_file) as f:
            cfg = json.load(f)

        strategy_key = json_file.stem  # e.g. "expiry_straddle_sell_banknifty"
        if strategy_key not in POE_PERCENTS:
            continue
        if not cfg.get("enabled", True):
            continue  # skip disabled strategies

        pct = POE_PERCENTS[strategy_key]
        cfg["position_sizing"] = {
            "method": "percent_of_equity",
            "percent_of_equity": pct,
            "min_quantity": 1,
        }
        with open(json_file, "w") as f:
            json.dump(cfg, f, indent=2)

    return strat_dir


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


def run_one(inst: dict, strat_dir: str) -> dict | None:
    tmp_db_path = None
    try:
        import tempfile as _tmp, os as _os
        fd, tmp_db_path = _tmp.mkstemp(suffix=".db", prefix="bt_poe_")
        _os.close(fd)

        h = BacktestHarness(
            data_file=inst["data_file"],
            instrument_key=inst["instrument_key"],
            strategy_dir=strat_dir,
            capital=INITIAL_CAPITAL,
            straddle_proxy=inst["straddle_proxy"],
            entry_anchor_time=inst.get("entry_anchor_time"),
            database_url=f"sqlite:///{tmp_db_path}",
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
                import os as _os
                _os.unlink(tmp_db_path)
            except OSError:
                pass


def main() -> None:
    strat_dir_path = _build_poe_strategies_dir()
    strat_dir = str(strat_dir_path)

    print(f"\nRunning POE backtest on {len(ALL_INSTRUMENTS)} instruments (parallel)...")
    print(f"Sizing: percent_of_equity (calibrated to match fixed_qty at ₹5L start)\n")

    results_map: dict[str, dict] = {}
    n_workers = min(os.cpu_count() or 4, len(ALL_INSTRUMENTS))

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as pool:
        future_to_inst = {pool.submit(run_one, inst, strat_dir): inst for inst in ALL_INSTRUMENTS}
        for future in as_completed(future_to_inst):
            inst = future_to_inst[future]
            try:
                r = future.result()
            except Exception as exc:
                r = {"instrument": inst["name"], "error": str(exc)}

            if r is None:
                continue
            if "error" in r:
                print(f"  [{inst['name']}] ERROR: {r['error'][:80]}")
            else:
                print(
                    f"  [{inst['name']}] {'(proxy)' if inst['straddle_proxy'] else '(equity)'}  "
                    f"{r['total_trades']} trades | WR {r['win_rate']:.1f}% | "
                    f"PF {r['profit_factor']:.2f} | P&L ₹{r['net_pnl_rupees']:,.0f}"
                )
            results_map[inst["name"]] = r

    all_results = [results_map[inst["name"]] for inst in ALL_INSTRUMENTS if inst["name"] in results_map]

    print("\n" + "=" * 75)
    print(f"  {'INSTRUMENT':<20} {'MODE':<16} {'TRADES':>6} {'WR%':>6} {'PF':>6} {'P&L (₹)':>12}")
    print("=" * 75)

    grand_pnl = grand_trades = all_wins = 0
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

    strat_totals: dict[str, dict] = {}
    for r in all_results:
        for sid, stats in r.get("by_strategy", {}).items():
            if sid not in strat_totals:
                strat_totals[sid] = {"trades": 0, "wins": 0, "pnl": 0.0}
            strat_totals[sid]["trades"] += stats["trades"]
            strat_totals[sid]["wins"]   += stats["wins"]
            strat_totals[sid]["pnl"]    += stats["pnl_rupees"]

    print("\n  BY STRATEGY:")
    print(f"  {'STRATEGY':<45} {'TRADES':>6} {'WR%':>6} {'P&L (₹)':>12}")
    print("  " + "-" * 70)
    for sid, s in sorted(strat_totals.items(), key=lambda x: -x[1]["pnl"]):
        wr = 100 * s["wins"] / max(s["trades"], 1)
        print(f"  {sid:<45} {s['trades']:>6} {wr:>6.1f} {s['pnl']:>12,.0f}")

    print(f"\n  FIXED_QTY BASELINE (₹5L, 18mo):  ₹341,632 (658 trades, 57.9% WR)")
    print(f"  POE RESULT (equity=25%):          ₹{grand_pnl:,.0f} ({grand_trades} trades, {overall_wr:.1f}% WR)")
    delta = grand_pnl - 341_632
    print(f"  DELTA vs fixed_qty:               ₹{delta:+,.0f} ({delta/341632*100:+.1f}%)\n")

    # Cleanup temp dir
    try:
        shutil.rmtree(strat_dir_path.parent)
    except Exception:
        pass


if __name__ == "__main__":
    main()
