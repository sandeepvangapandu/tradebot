#!/usr/bin/env python3
"""Full pipeline backtest — mirrors live TradingBot startup/run/shutdown.

Runs all 4 instruments sequentially through BacktestHarness with INFO-level
logging (same format as live terminal), per-trade CSV export, signal stats,
and anomaly detection.

Usage:
    python3 scripts/backtest_full_pipeline.py

Outputs:
    logs/backtest_pipeline_YYYYMMDD_HHMMSS.log   — full INFO log
    logs/backtest_trades_YYYYMMDD_HHMMSS.csv     — per-trade detail
    logs/backtest_signals_YYYYMMDD_HHMMSS.csv    — signal pass/block stats
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile
import time
from datetime import datetime, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from loguru import logger

IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Log setup — file + stderr, same format as live bot
# ---------------------------------------------------------------------------
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
    "<level>{message}</level>"
)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = Path("logs") / f"backtest_pipeline_{ts}.log"
TRADES_CSV = Path("logs") / f"backtest_trades_{ts}.csv"
SIGNALS_CSV = Path("logs") / f"backtest_signals_{ts}.csv"

logger.remove()
logger.add(sys.stderr, level="INFO", format=LOG_FORMAT, colorize=True)
logger.add(str(LOG_FILE), level="INFO", format=LOG_FORMAT, colorize=False)

# ---------------------------------------------------------------------------
# Instrument catalogue (same as backtest_all.py)
# ---------------------------------------------------------------------------
INSTRUMENTS = [
    {
        "name": "BankNifty",
        "data_file": "data/backtest/banknifty_1m_24mo.csv",
        "instrument_key": "NSE_INDEX|Nifty Bank",
        "straddle_proxy": True,
        "entry_anchor_time": dt_time(10, 0),
        "proxy_label": "PROXY:ATR_SYNTH",
    },
    {
        "name": "Nifty50",
        "data_file": "data/backtest/nifty50_1m_24mo.csv",
        "instrument_key": "NSE_INDEX|Nifty 50",
        "straddle_proxy": True,
        "entry_anchor_time": dt_time(10, 0),
        "proxy_label": "PROXY:ATR_SYNTH",
    },
    {
        "name": "FinNifty",
        "data_file": "data/backtest/finnifty_1m_24mo.csv",
        "instrument_key": "NSE_INDEX|Nifty Fin Service",
        "straddle_proxy": True,
        "entry_anchor_time": dt_time(10, 0),
        "proxy_label": "PROXY:ATR_SYNTH",
    },
    {
        "name": "ADANIENT",
        "data_file": "data/backtest/equity/adanient_1m_24mo.csv",
        "instrument_key": "NSE_EQ|ADANIENT",
        "straddle_proxy": False,
        "entry_anchor_time": None,
        "proxy_label": "REAL_PRICE",
    },
]

CAPITAL = 500_000_00  # ₹5,00,000 in paisa

# ---------------------------------------------------------------------------
# Anomaly checks on trades
# ---------------------------------------------------------------------------

def check_anomalies(trades: list[dict], instrument_name: str, proxy_label: str) -> list[str]:
    flags = []
    for t in trades:
        tid = t.get("trade_id", "?")
        entry_p = t.get("entry_price", 0)
        exit_p = t.get("exit_price", 0)
        fees = t.get("fees", 0)
        net = t.get("net_pnl", 0)
        gross = t.get("gross_pnl", net + fees)
        entry_t = t.get("entry_time", "")
        exit_t = t.get("exit_time", "")
        qty = t.get("quantity", 0)

        if entry_p <= 0:
            flags.append(f"[ANOMALY] {instrument_name} trade {tid}: zero/negative entry_price={entry_p}")
        if exit_p <= 0:
            flags.append(f"[ANOMALY] {instrument_name} trade {tid}: zero/negative exit_price={exit_p}")
        if fees < 0:
            flags.append(f"[ANOMALY] {instrument_name} trade {tid}: negative fees={fees}")
        if fees > abs(gross) and gross != 0:
            flags.append(f"[ANOMALY] {instrument_name} trade {tid}: fees({fees}) > gross_pnl({gross})")
        if qty <= 0:
            flags.append(f"[ANOMALY] {instrument_name} trade {tid}: zero/negative quantity={qty}")
        # 1-bar cooldown: entry and exit on exact same minute
        if entry_t and exit_t and entry_t[:16] == exit_t[:16]:
            flags.append(f"[ANOMALY] {instrument_name} trade {tid}: entry+exit same minute — 1-bar cooldown violated")

    return flags


# ---------------------------------------------------------------------------
# Per-instrument runner
# ---------------------------------------------------------------------------

def run_instrument(inst: dict, trades_writer, signals_summary: list) -> dict | None:
    from src.backtest.harness import BacktestHarness

    name = inst["name"]
    proxy_label = inst["proxy_label"]

    logger.info("═" * 70)
    logger.info("▶ TradingBot STARTUP | instrument={} | mode={}", name, proxy_label)
    logger.info("  data_file    : {}", inst["data_file"])
    logger.info("  capital      : ₹{:,.0f}", CAPITAL / 100)
    logger.info("  straddle_proxy: {}", inst["straddle_proxy"])
    logger.info("  price_type   : {}", proxy_label)
    logger.info("═" * 70)

    if not Path(inst["data_file"]).exists():
        logger.error("Data file not found: {} — SKIPPING", inst["data_file"])
        return {"instrument": name, "error": "data file missing"}

    fd, tmp_db = tempfile.mkstemp(suffix=".db", prefix=f"bt_{name}_")
    os.close(fd)

    t0 = time.monotonic()
    try:
        h = BacktestHarness(
            data_file=inst["data_file"],
            instrument_key=inst["instrument_key"],
            strategy_dir="config/strategies",
            capital=CAPITAL,
            straddle_proxy=inst["straddle_proxy"],
            entry_anchor_time=inst.get("entry_anchor_time"),
            database_url=f"sqlite:///{tmp_db}",
        )
        results = h.run()
    except Exception as exc:
        logger.exception("BacktestHarness failed for {}: {}", name, exc)
        return {"instrument": name, "error": str(exc)}
    finally:
        try:
            os.unlink(tmp_db)
        except OSError:
            pass

    elapsed = time.monotonic() - t0
    m = results.metrics

    # --- Per-trade CSV rows ---
    for t in results.trades:
        entry_p = t.get("entry_price", 0)
        exit_p = t.get("exit_price", 0)
        fees = t.get("fees", 0)
        net = t.get("net_pnl", 0)
        gross = net + fees
        trades_writer.writerow({
            "instrument": name,
            "proxy_label": proxy_label,
            "strategy": t.get("strategy_id", ""),
            "trade_id": t.get("trade_id", ""),
            "side": t.get("side", ""),
            "entry_time": t.get("entry_time", ""),
            "entry_price_paisa": entry_p,
            "entry_price_rs": round(entry_p / 100, 2) if entry_p else 0,
            "exit_time": t.get("exit_time", ""),
            "exit_price_paisa": exit_p,
            "exit_price_rs": round(exit_p / 100, 2) if exit_p else 0,
            "quantity": t.get("quantity", 0),
            "exit_reason": t.get("exit_reason", ""),
            "gross_pnl_paisa": gross,
            "gross_pnl_rs": round(gross / 100, 2),
            "fees_paisa": fees,
            "fees_rs": round(fees / 100, 2),
            "net_pnl_paisa": net,
            "net_pnl_rs": round(net / 100, 2),
            "win": 1 if net > 0 else 0,
        })

    # --- Anomaly scan ---
    anomalies = check_anomalies(results.trades, name, proxy_label)

    # --- Exit reason breakdown ---
    exit_reasons: dict[str, int] = {}
    for t in results.trades:
        r = t.get("exit_reason", "unknown") or "unknown"
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    # --- Log instrument summary ---
    logger.info("")
    logger.info("■ TradingBot SHUTDOWN | instrument={}", name)
    logger.info("  bars_processed : {:,}", results.bars_processed)
    logger.info("  total_trades   : {}", m.get("total_trades", 0))
    logger.info("  win_rate       : {:.1f}%", m.get("win_rate", 0))
    logger.info("  profit_factor  : {:.2f}", m.get("profit_factor", 0))
    logger.info("  net_pnl        : ₹{:,.2f}", m.get("net_pnl_rupees", 0))
    logger.info("  total_fees     : ₹{:,.2f}", m.get("total_fees_rupees", 0))
    logger.info("  max_drawdown   : ₹{:,.2f} ({:.1f}%)",
                m.get("max_drawdown_rupees", 0), m.get("max_drawdown_pct", 0))
    logger.info("  return_pct     : {:.2f}%", m.get("return_pct", 0))
    logger.info("  sharpe_ratio   : {:.2f}", m.get("sharpe_ratio", 0))
    logger.info("  elapsed        : {:.1f}s", elapsed)
    logger.info("  exit_reasons   : {}", exit_reasons)

    if anomalies:
        for a in anomalies:
            logger.warning(a)
    else:
        logger.info("  anomaly_check  : PASS — no issues detected")

    logger.info("  proxy_note     : {} — P&L uses {} premiums not real IV",
                proxy_label,
                "ATR-derived synthetic" if inst["straddle_proxy"] else "actual market")

    signals_summary.append({
        "instrument": name,
        "proxy_label": proxy_label,
        "trades": m.get("total_trades", 0),
        "win_rate": m.get("win_rate", 0),
        "profit_factor": m.get("profit_factor", 0),
        "net_pnl_rs": m.get("net_pnl_rupees", 0),
        "fees_rs": m.get("total_fees_rupees", 0),
        "max_dd_rs": m.get("max_drawdown_rupees", 0),
        "max_dd_pct": m.get("max_drawdown_pct", 0),
        "return_pct": m.get("return_pct", 0),
        "sharpe": m.get("sharpe_ratio", 0),
        "anomalies": len(anomalies),
        "exit_reasons": str(exit_reasons),
    })

    return {
        "instrument": name,
        "proxy_label": proxy_label,
        "total_trades": m.get("total_trades", 0),
        "win_rate": m.get("win_rate", 0),
        "profit_factor": m.get("profit_factor", 0),
        "net_pnl_rupees": m.get("net_pnl_rupees", 0),
        "fees_rupees": m.get("total_fees_rupees", 0),
        "max_dd_rupees": m.get("max_drawdown_rupees", 0),
        "max_dd_pct": m.get("max_drawdown_pct", 0),
        "return_pct": m.get("return_pct", 0),
        "sharpe": m.get("sharpe_ratio", 0),
        "anomalies": len(anomalies),
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("╔══════════════════════════════════════════════════════════════════════╗")
    logger.info("║   FULL PIPELINE BACKTEST — 24-MONTH DATA — {:<31}║", ts)
    logger.info("║   4 instruments | sequential | INFO logging | anomaly detection      ║")
    logger.info("╚══════════════════════════════════════════════════════════════════════╝")
    logger.info("Log file    : {}", LOG_FILE)
    logger.info("Trades CSV  : {}", TRADES_CSV)
    logger.info("Capital/run : ₹{:,.0f}  ({} paisa)", CAPITAL / 100, CAPITAL)
    logger.info("")

    all_results: list[dict] = []
    signals_summary: list[dict] = []
    total_t0 = time.monotonic()

    # Trades CSV — open once, all instruments write rows
    trade_fields = [
        "instrument", "proxy_label", "strategy", "trade_id", "side",
        "entry_time", "entry_price_paisa", "entry_price_rs",
        "exit_time", "exit_price_paisa", "exit_price_rs",
        "quantity", "exit_reason",
        "gross_pnl_paisa", "gross_pnl_rs",
        "fees_paisa", "fees_rs",
        "net_pnl_paisa", "net_pnl_rs",
        "win",
    ]

    with open(TRADES_CSV, "w", newline="") as trades_f:
        tw = csv.DictWriter(trades_f, fieldnames=trade_fields)
        tw.writeheader()

        for inst in INSTRUMENTS:
            result = run_instrument(inst, tw, signals_summary)
            if result:
                all_results.append(result)

    # Signals summary CSV
    if signals_summary:
        with open(SIGNALS_CSV, "w", newline="") as sf:
            sw = csv.DictWriter(sf, fieldnames=list(signals_summary[0].keys()))
            sw.writeheader()
            sw.writerows(signals_summary)

    total_elapsed = time.monotonic() - total_t0

    # -----------------------------------------------------------------------
    # Final summary table
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════════════════════════╗")
    logger.info("║  FINAL BACKTEST SUMMARY — 24 months — capital ₹{:<36}║", f"{CAPITAL/100:,.0f}/instrument")
    logger.info("╠══════════════════════════════════════════════════════════════════════════════════╣")
    logger.info("║  {:<18} {:<12} {:>6} {:>6} {:>6} {:>12} {:>10} {:>6}  ║",
                "INSTRUMENT", "TYPE", "TRADES", "WR%", "PF", "P&L (₹)", "MAXDD (₹)", "ANN")
    logger.info("╠══════════════════════════════════════════════════════════════════════════════════╣")

    grand_pnl = 0.0
    grand_trades = 0
    grand_wins = 0
    grand_fees = 0.0
    total_anomalies = 0

    for r in all_results:
        if "error" in r:
            logger.error("║  {:<18} ERROR: {:<55}  ║", r["instrument"], r.get("error", "")[:55])
            continue
        wins = round(r["total_trades"] * r["win_rate"] / 100)
        logger.info("║  {:<18} {:<12} {:>6} {:>6.1f} {:>6.2f} {:>12,.0f} {:>10,.0f} {:>5.1f}%  ║",
                    r["instrument"],
                    r["proxy_label"],
                    r["total_trades"],
                    r["win_rate"],
                    r["profit_factor"],
                    r["net_pnl_rupees"],
                    r["max_dd_rupees"],
                    r["return_pct"],
                    )
        grand_pnl += r["net_pnl_rupees"]
        grand_trades += r["total_trades"]
        grand_wins += wins
        grand_fees += r.get("fees_rupees", 0)
        total_anomalies += r.get("anomalies", 0)

    overall_wr = 100 * grand_wins / max(grand_trades, 1)
    logger.info("╠══════════════════════════════════════════════════════════════════════════════════╣")
    logger.info("║  {:<18} {:<12} {:>6} {:>6.1f} {:>6} {:>12,.0f} {:>10} {:>6}  ║",
                "TOTAL", "", grand_trades, overall_wr, "", grand_pnl, "", "")
    logger.info("╚══════════════════════════════════════════════════════════════════════════════════╝")
    logger.info("")
    logger.info("  Total fees charged   : ₹{:,.2f}", grand_fees)
    logger.info("  Anomalies detected   : {}", total_anomalies)
    logger.info("  Total elapsed        : {:.1f}s ({:.1f} min)", total_elapsed, total_elapsed / 60)
    logger.info("  Log file             : {}", LOG_FILE)
    logger.info("  Trades CSV           : {}", TRADES_CSV)
    logger.info("  Signals CSV          : {}", SIGNALS_CSV)
    logger.info("")
    logger.info("  PROXY NOTE: BankNifty/Nifty50/FinNifty P&L uses ATR-derived synthetic")
    logger.info("  option premiums. Live P&L will differ based on actual IV and bid-ask.")
    logger.info("  ADANIENT uses real historical equity prices — no proxy.")


if __name__ == "__main__":
    main()
