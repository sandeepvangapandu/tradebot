#!/usr/bin/env python3
"""Replay 24 months of OHLCV as Upstox-shaped ticks through paper trading.

This is an evidence harness, not a strategy engine replacement. It feeds each
historical bar through ``TradingBot._on_market_tick()`` using Upstox V3-like
payloads, then lets the existing StrategyEngine, OrderManager, PaperBroker,
RiskManager, and PositionManager handle the rest.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = dt_time(9, 15)
NO_NEW_ENTRY_AFTER = dt_time(15, 0)
CAPITAL_PAISA = 500_000_00

DATA = {
    "NSE_INDEX|Nifty Bank": {
        "instrument": "BankNifty",
        "data_file": "data/backtest/banknifty_1m_24mo.csv",
        "entry_anchor_time": dt_time(10, 0),
    },
    "NSE_INDEX|Nifty 50": {
        "instrument": "Nifty50",
        "data_file": "data/backtest/nifty50_1m_24mo.csv",
        "entry_anchor_time": dt_time(10, 0),
    },
    "NSE_INDEX|Nifty Fin Service": {
        "instrument": "FinNifty",
        "data_file": "data/backtest/finnifty_1m_24mo.csv",
        "entry_anchor_time": dt_time(10, 0),
    },
    "NSE_EQ|ADANIENT": {
        "instrument": "ADANIENT",
        "data_file": "data/backtest/equity/adanient_1m_24mo.csv",
        "entry_anchor_time": None,
    },
}


def _strategy_instrument_key(cfg: dict[str, Any]) -> str:
    instrument = cfg.get("instrument") or {}
    if isinstance(instrument, dict) and instrument.get("instrument_key"):
        return str(instrument["instrument_key"])

    strategy_id = str(cfg.get("strategy_id") or cfg.get("name") or "")
    if "Equity" in strategy_id:
        return "NSE_EQ|ADANIENT"
    if "FinNifty" in strategy_id:
        return "NSE_INDEX|Nifty Fin Service"
    if "Nifty50" in strategy_id:
        return "NSE_INDEX|Nifty 50"
    return "NSE_INDEX|Nifty Bank"


def _is_straddle_strategy(cfg: dict[str, Any]) -> bool:
    strategy_id = str(cfg.get("strategy_id") or cfg.get("name") or "").lower()
    if "straddle" in strategy_id or "strangle" in strategy_id:
        return True

    for entry in cfg.get("entry_sets") or []:
        signal = str(entry.get("signal", "")) if isinstance(entry, dict) else ""
        if signal.upper() in {"STRADDLE", "STRANGLE"}:
            return True
    return False


def _is_directional_option_strategy(cfg: dict[str, Any]) -> bool:
    """Return True for option strategies that need single-leg premium replay."""
    if _is_straddle_strategy(cfg):
        return False

    underlying = cfg.get("underlying") or {}
    segment = str(underlying.get("segment") or cfg.get("segment") or "").upper()
    if segment not in {"NSE_FO", "NFO"}:
        return False

    instrument_key = str(underlying.get("instrument_key") or "")
    if not instrument_key.startswith("NSE_INDEX|"):
        return False

    return True


def _load_jobs(strategy_filter: set[str] | None = None) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for path in sorted((ROOT / "config/strategies").glob("*.json")):
        cfg = json.loads(path.read_text())
        strategy_id = str(cfg.get("strategy_id") or cfg.get("name") or path.stem)
        if strategy_filter and strategy_id not in strategy_filter:
            continue
        if not bool(cfg.get("enabled", True)):
            continue

        instrument_key = _strategy_instrument_key(cfg)
        if instrument_key not in DATA:
            print(f"SKIP {strategy_id}: no replay data mapping for {instrument_key}", flush=True)
            continue

        jobs.append(
            {
                "file": str(path.relative_to(ROOT)),
                "strategy_id": strategy_id,
                "instrument_key": instrument_key,
                "straddle_proxy": _is_straddle_strategy(cfg),
                "options_proxy": _is_directional_option_strategy(cfg),
            }
        )
    return jobs


def _upstox_payload(instrument_key: str, row: Any) -> dict[str, Any]:
    close_rupees = int(row.close) / 100.0
    bid = max(0.05, close_rupees - 0.05)
    ask = close_rupees + 0.05
    volume = int(getattr(row, "volume", 0) or 0)
    ts_ms = int(row.Index.timestamp() * 1000)
    ohlc = {
        "interval": "I1",
        "open": int(row.open) / 100.0,
        "high": int(row.high) / 100.0,
        "low": int(row.low) / 100.0,
        "close": close_rupees,
        "ts": ts_ms,
    }

    if instrument_key.startswith("NSE_INDEX|"):
        inner = {
            "ltpc": {"ltp": close_rupees, "ltt": ts_ms, "cp": close_rupees, "v": volume},
            "marketOHLC": {"ohlc": [ohlc]},
        }
        return {"feeds": {instrument_key: {"fullFeed": {"indexFF": inner}}}}

    inner = {
        "ltpc": {"ltp": close_rupees, "ltt": ts_ms, "cp": close_rupees, "v": volume},
        "vtt": volume,
        "marketLevel": {"bidAskQuote": [{"bidP": bid, "bidQ": 1, "askP": ask, "askQ": 1}]},
        "depth": {
            "buyDepth": [{"price": bid, "quantity": 1}],
            "sellDepth": [{"price": ask, "quantity": 1}],
        },
        "marketOHLC": {"ohlc": [ohlc]},
    }
    return {"feeds": {instrument_key: {"fullFeed": {"marketFF": inner}}}}


def _charge_total(record: dict[str, Any]) -> int:
    charges = record.get("charges", 0)
    if isinstance(charges, dict):
        return int(charges.get("total", 0) or 0)
    return int(charges or record.get("fees", 0) or 0)


def _closed_positions(position_manager: Any) -> list[Any]:
    positions = getattr(position_manager, "_positions", {})
    values = positions.values() if isinstance(positions, dict) else positions
    return [position for position in values if getattr(position, "is_closed", False)]


def _trade_from_position(position: Any) -> dict[str, Any]:
    return {
        "instrument_key": position.instrument_key,
        "strategy_id": position.strategy_id,
        "side": position.side.value,
        "entry_price": int(position.entry_price),
        "exit_price": int(position.blended_exit_price or position.exit_price or position.entry_price),
        "quantity": int(position.original_quantity),
        "net_pnl": int(position.total_realized_pnl),
        "entry_time": str(position.entry_time),
        "exit_time": str(position.exit_time) if position.exit_time else None,
        "exit_reason": str(getattr(position, "exit_reason", "unknown")),
    }


def _metrics(trades: list[dict[str, Any]], equity_curve: list[int], total_fees: int) -> dict[str, Any]:
    if not trades:
        final = equity_curve[-1] if equity_curve else CAPITAL_PAISA
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_pnl_rupees": 0.0,
            "total_fees_rupees": round(total_fees / 100, 2),
            "max_drawdown_rupees": 0.0,
            "max_drawdown_pct": 0.0,
            "return_pct": round((final - CAPITAL_PAISA) / CAPITAL_PAISA * 100, 2),
            "sharpe_ratio": 0.0,
        }

    pnls = [int(t.get("net_pnl", 0) or 0) for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    gross_loss = abs(sum(losers)) if losers else 1

    peak = CAPITAL_PAISA
    max_dd = 0
    for equity in equity_curve:
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    daily_pnl: dict[str, int] = {}
    for trade in trades:
        exit_day = str(trade.get("exit_time") or "")[:10]
        if exit_day:
            daily_pnl[exit_day] = daily_pnl.get(exit_day, 0) + int(trade.get("net_pnl", 0) or 0)

    sharpe = 0.0
    if len(daily_pnl) > 1:
        returns = [value / CAPITAL_PAISA for value in daily_pnl.values()]
        mean_r = sum(returns) / len(returns)
        variance = sum((ret - mean_r) ** 2 for ret in returns) / (len(returns) - 1)
        std_r = math.sqrt(variance) if variance > 0 else 1e-10
        sharpe = round((mean_r / std_r) * math.sqrt(252), 2)

    final = equity_curve[-1] if equity_curve else CAPITAL_PAISA
    return {
        "total_trades": len(trades),
        "win_rate": round(len(winners) / len(trades) * 100, 2),
        "profit_factor": round(sum(winners) / gross_loss, 2),
        "net_pnl_rupees": round(sum(pnls) / 100, 2),
        "total_fees_rupees": round(total_fees / 100, 2),
        "max_drawdown_rupees": round(max_dd / 100, 2),
        "max_drawdown_pct": round(max_dd / CAPITAL_PAISA * 100, 2),
        "return_pct": round((final - CAPITAL_PAISA) / CAPITAL_PAISA * 100, 2),
        "sharpe_ratio": sharpe,
    }


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    from src.backtest.harness import BacktestHarness
    from src.main import TradingBot

    log_stream = io.StringIO()
    logger.remove()
    logger.add(log_stream, level="WARNING")
    stderr_stream = io.StringIO()
    db_path: str | None = None
    started = time.time()

    try:
        fd, db_path = tempfile.mkstemp(suffix=".db", prefix="upstox_replay_")
        os.close(fd)
        data = DATA[job["instrument_key"]]
        harness = BacktestHarness(
            data_file=data["data_file"],
            instrument_key=job["instrument_key"],
            strategy_dir="config/strategies",
            capital=CAPITAL_PAISA,
            start_date=job.get("start_date"),
            end_date=job.get("end_date"),
            options_proxy=job["options_proxy"],
            straddle_proxy=job["straddle_proxy"],
            allow_directional_proxy=job["options_proxy"],
            entry_anchor_time=data.get("entry_anchor_time") if job["straddle_proxy"] else None,
            database_url=f"sqlite:///{db_path}",
            strategy_only=job["strategy_id"],
        )

        bot = TradingBot()
        bot.position_manager = harness._position_manager
        bot.paper_broker = harness._paper_broker
        bot.depth_feed = None
        bot.tick_metrics = None
        bot.vix_regime = None
        bot.micro_feature_engine = None

        total = len(harness._master_df)
        prev_day: date | None = None
        equity_curve: list[int] = []
        manual_errors: list[str] = []
        print(
            f"START {job['strategy_id']} {data['instrument']} bars={total} "
            f"mode={'straddle_proxy' if job['straddle_proxy'] else 'options_proxy' if job['options_proxy'] else 'raw'}",
            flush=True,
        )

        with contextlib.redirect_stderr(stderr_stream):
            max_bars = int(job.get("max_bars") or 0)
            for i, row in enumerate(harness._master_df.itertuples()):
                if max_bars and i >= max_bars:
                    break
                bar_time: datetime = row.Index
                bar_time_ist = bar_time.astimezone(IST) if bar_time.tzinfo else bar_time.replace(tzinfo=IST)

                harness._bar_provider.advance(job["instrument_key"], i)
                harness._raw_bar_provider.advance(job["instrument_key"], i)
                harness._position_manager.set_backtest_time(bar_time_ist)

                today = bar_time_ist.date()
                if today != prev_day:
                    harness._risk_manager.reset_daily()
                    equity = harness._paper_broker.get_portfolio_value()
                    harness._risk_manager.update_capital(equity)
                    harness._strategy_engine.update_account_equity(equity)
                    prev_day = today

                if bar_time_ist.time() < MARKET_OPEN:
                    equity_curve.append(harness._paper_broker.get_portfolio_value())
                    continue

                try:
                    bot._on_market_tick(_upstox_payload(job["instrument_key"], row))
                except Exception as exc:
                    manual_errors.append(f"tick error bar={i}: {exc!r}")
                    logger.error("tick error at bar {}: {}", i, exc)

                bar_t = bar_time_ist.time()
                harness._circuit_breaker.set_backtest_time(bar_time_ist)
                if bar_t < NO_NEW_ENTRY_AFTER and not harness._circuit_breaker.is_halted():
                    bars = {
                        job["instrument_key"]: harness._bar_provider.get_bars(
                            job["instrument_key"], timeframe=1
                        )
                    }
                    try:
                        for signal in harness._strategy_engine.evaluate_bar_sync(
                            bars=bars, bar_time=bar_time_ist
                        ):
                            signal.timestamp = bar_time_ist
                            harness._order_manager.process_signal(signal)
                    except Exception as exc:
                        manual_errors.append(f"signal error bar={i}: {exc!r}")
                        logger.error("signal error at bar {}: {}", i, exc)

                equity_curve.append(harness._paper_broker.get_portfolio_value())
                if i % 25_000 == 0:
                    print(
                        f"PROGRESS {job['strategy_id']} {i}/{total} "
                        f"events={len(harness._paper_broker.get_trade_history())} "
                        f"equity={harness._paper_broker.get_portfolio_value()/100:.2f}",
                        flush=True,
                    )

        harness._close_all_positions(type("ReplayResult", (), {"trades": []})())
        closed = [_trade_from_position(position) for position in _closed_positions(harness._position_manager)]
        broker_events = harness._paper_broker.get_trade_history()
        fees = sum(_charge_total(record) for record in broker_events)
        metrics = _metrics(closed, equity_curve, fees)

        log_text = log_stream.getvalue() + stderr_stream.getvalue()
        lines = [line for line in log_text.splitlines() if line.strip()]
        warnings = [line for line in lines if "WARNING" in line or "| WARNING" in line]
        errors = [
            line
            for line in lines
            if any(token in line for token in ("ERROR", "CRITICAL", "| ERROR", "| CRITICAL", "Traceback"))
        ]
        signatures: Counter[str] = Counter()
        for line in warnings + errors:
            message = line.split(" - ")[-1]
            if "| " in message:
                message = message.split("| ", 1)[-1]
            signatures[message[:200]] += 1

        result = {
            **job,
            "status": "OK",
            "instrument": data["instrument"],
            "data_file": data["data_file"],
            "bars": total,
            "date_start": str(harness._master_df.index[0]),
            "date_end": str(harness._master_df.index[-1]),
            "closed_positions": len(closed),
            "broker_events": len(broker_events),
            "open_positions_end": len(harness._position_manager.get_open_positions()),
            "metrics": metrics,
            "manual_error_count": len(manual_errors),
            "manual_errors": manual_errors[:20],
            "warning_count": len(warnings),
            "error_count": len(errors),
            "top_log_signatures": signatures.most_common(10),
            "elapsed_sec": round(time.time() - started, 2),
        }
        print(
            f"DONE {job['strategy_id']} {data['instrument']} bars={total} "
            f"closed={len(closed)} events={len(broker_events)} "
            f"pnl={metrics['net_pnl_rupees']} warn={len(warnings)} "
            f"err={len(errors)} elapsed={result['elapsed_sec']}s",
            flush=True,
        )
        return result
    except Exception as exc:
        return {
            **job,
            "status": "EXCEPTION",
            "error": repr(exc),
            "traceback": traceback.format_exc()[-5000:],
            "elapsed_sec": round(time.time() - started, 2),
        }
    finally:
        if db_path:
            try:
                os.unlink(db_path)
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", action="append", help="Run only the named strategy_id. May repeat.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="reports/upstox_replay_24mo.json")
    parser.add_argument("--start-date", help="Optional YYYY-MM-DD replay start date for diagnostic windows.")
    parser.add_argument("--end-date", help="Optional YYYY-MM-DD replay end date for diagnostic windows.")
    parser.add_argument("--max-bars", type=int, default=0, help="Optional max bars per strategy for smoke diagnostics.")
    args = parser.parse_args()

    jobs = _load_jobs(set(args.strategy) if args.strategy else None)
    if not jobs:
        raise SystemExit("No enabled replay jobs matched.")
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        for job in jobs:
            job["start_date"] = start_date
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        for job in jobs:
            job["end_date"] = end_date
    if args.max_bars:
        for job in jobs:
            job["max_bars"] = args.max_bars

    workers = min(args.workers, os.cpu_count() or 1, len(jobs))
    print(
        f"Running Upstox-shaped 24-month paper replay with NO TIMEOUTS: "
        f"{len(jobs)} strategy job(s), workers={workers}",
        flush=True,
    )
    print("Strategies: " + ", ".join(job["strategy_id"] for job in jobs), flush=True)

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
        future_to_job = {pool.submit(run_job, job): job for job in jobs}
        for future in as_completed(future_to_job):
            result = future.result()
            results.append(result)
            if result.get("status") != "OK":
                print(
                    f"FAILED {result.get('strategy_id')} {result.get('status')} "
                    f"{result.get('error', '')}",
                    flush=True,
                )

    results.sort(key=lambda item: item.get("strategy_id", ""))
    output = {
        "run": "upstox_shaped_24mo_paper_pipeline_no_timeouts",
        "workers": workers,
        "results": results,
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, default=str) + "\n")
    print(f"Wrote {output_path}", flush=True)
    print("JSON_RESULT_START")
    print(json.dumps(output, indent=2, default=str))
    print("JSON_RESULT_END")


if __name__ == "__main__":
    main()
