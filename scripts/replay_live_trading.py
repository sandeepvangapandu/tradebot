#!/usr/bin/env python3
"""Live trading pipeline replay — feeds historical 1-min bars through the
full TradingBot component tree exactly as live trading does.

Differences from backtest_full_pipeline.py:
  - Uses real BarBuilder (tick_queue → bars)
  - StrategyEngine.evaluate_bar_sync() uses bar timestamps (not wall clock)
  - SOD equity update + EOD intraday square-off at correct simulated times
  - Startup/shutdown logs match live bot format
  - sim_clock ensures trading-hours gates use bar time, not wall clock

Usage:
    python3 scripts/replay_live_trading.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

# ── project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from src.data.bar_builder import BarBuilder
from src.execution.order_manager import OrderManager
from src.execution.paper_broker import PaperBroker
from src.execution.partial_profit import PartialProfitManager
from src.execution.position_manager import PositionConfig, PositionManager
from src.learning.integration import LearningIntegration
from src.persistence.database import get_session, init_db
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.risk_manager import RiskManager
from src.risk.strategy_quarantine import StrategyQuarantine
from src.strategy.engine import StrategyEngine
from src.utils.sim_clock import now_ist, reset_sim_time, set_sim_time
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


# ── instrument config ─────────────────────────────────────────────────────────

@dataclass
class ReplayTarget:
    label: str
    instrument_key: str        # e.g. "NSE_INDEX|Nifty Bank"
    csv_path: str
    prices_in_paisa: bool = True
    proxy: bool = False        # True → emit synthetic ATM option prices


INSTRUMENTS = [
    ReplayTarget(
        label="BankNifty",
        instrument_key="NSE_INDEX|Nifty Bank",
        csv_path="data/backtest/banknifty_1m_24mo.csv",
        prices_in_paisa=True,
        proxy=True,
    ),
    ReplayTarget(
        label="Nifty50",
        instrument_key="NSE_INDEX|Nifty 50",
        csv_path="data/backtest/nifty50_1m_24mo.csv",
        prices_in_paisa=True,
        proxy=True,
    ),
    ReplayTarget(
        label="FinNifty",
        instrument_key="NSE_INDEX|Nifty Fin Service",
        csv_path="data/backtest/finnifty_1m_24mo.csv",
        prices_in_paisa=True,
        proxy=True,
    ),
    ReplayTarget(
        label="ADANIENT",
        instrument_key="NSE_EQ|ADANIENT",
        csv_path="data/backtest/equity/adanient_1m_24mo.csv",
        prices_in_paisa=True,
        proxy=False,
    ),
]


# ── synthetic ATM option price helpers ───────────────────────────────────────

def _intraday_decay(ts: datetime) -> float:
    """Linear premium decay factor: 1.0 at 10:00 → 0.2 at 14:00."""
    t = ts.time()
    start, end = time(10, 0), time(14, 0)
    if t <= start:
        return 1.0
    if t >= end:
        return 0.2
    elapsed = (t.hour * 60 + t.minute) - 600
    return 1.0 - 0.8 * (elapsed / 240)


def _synthetic_premium_paisa(underlying_close_paisa: int, high_paisa: int, low_paisa: int, ts: datetime) -> int:
    bar_range = high_paisa - low_paisa
    premium = max(bar_range * 1.5, underlying_close_paisa * 0.005)
    premium = int(premium * _intraday_decay(ts))
    return max(premium, 500)  # min Rs 5


def _atm_strike(underlying_paisa: int, step: int = 10000) -> int:
    """Round to nearest step (paisa). BankNifty → 100pt strikes = 10000 paisa."""
    return round(underlying_paisa / step) * step


# ── single-instrument replay ──────────────────────────────────────────────────

@dataclass
class RunResult:
    label: str
    proxy_label: str
    bars_processed: int = 0
    trades: int = 0
    wins: int = 0
    gross_pnl_paisa: int = 0
    fees_paisa: int = 0
    max_drawdown_paisa: int = 0
    elapsed_s: float = 0.0
    peak_equity: int = 0
    anomalies: list[str] = field(default_factory=list)

    @property
    def net_pnl_paisa(self) -> int:
        return self.gross_pnl_paisa - self.fees_paisa

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades * 100 if self.trades else 0.0

    @property
    def profit_factor(self) -> float:
        wins = sum(t["net"] for t in self._trade_list if t["net"] > 0) if hasattr(self, "_trade_list") else 0
        losses = sum(-t["net"] for t in self._trade_list if t["net"] < 0) if hasattr(self, "_trade_list") else 1
        return wins / losses if losses else float("inf")


def run_instrument(
    target: ReplayTarget,
    settings,
    start_date: date | None,
    end_date: date | None,
) -> RunResult:
    import time as _time
    t0 = _time.perf_counter()
    proxy_label = "PROXY:ATR_SYNTH" if target.proxy else "REAL_PRICE"

    logger.info("═" * 72)
    logger.info("▶ TradingBot STARTUP | instrument={} | mode={}", target.label, proxy_label)
    logger.info("  data_file    : {}", target.csv_path)
    logger.info("  capital      : ₹{:,.0f}  ({} paisa)", settings.capital / 100, settings.capital)
    logger.info("  strategies   : config/strategies/")
    logger.info("═" * 72)

    result = RunResult(label=target.label, proxy_label=proxy_label)

    # ── load CSV ──────────────────────────────────────────────────────────
    csv = Path(target.csv_path)
    if not csv.exists():
        logger.error("CSV not found: {}", target.csv_path)
        return result

    df = pd.read_csv(csv)
    # parse timestamp column
    ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col])
    if df[ts_col].dt.tz is None:
        df[ts_col] = df[ts_col].dt.tz_localize(IST)
    else:
        df[ts_col] = df[ts_col].dt.tz_convert(IST)
    df = df.set_index(ts_col).sort_index()
    if start_date:
        df = df[df.index.date >= start_date]
    if end_date:
        df = df[df.index.date <= end_date]
    if df.empty:
        logger.warning("No bars in date range for {}", target.label)
        return result

    total_bars = len(df)
    logger.info("Loaded {} bars ({} → {})", total_bars, df.index[0].date(), df.index[-1].date())

    # ── init components (same as BacktestHarness) ─────────────────────────
    init_db(settings.database_url)
    capital = settings.capital

    broker = PaperBroker(
        initial_capital=capital,
        slippage_pct=settings.slippage_pct / 100,
    )
    partial_profit = PartialProfitManager()
    risk_manager = RiskManager(
        capital=capital,
        max_daily_loss=settings.max_daily_loss,
        max_open_positions=settings.max_open_positions,
        max_position_size_pct=settings.max_position_size_pct,
        max_capital_deployment_pct=settings.max_capital_deployment_pct,
        max_trades_per_day=getattr(settings, "max_trades_per_day", 5),
    )
    cb = CircuitBreaker(
        max_consecutive_losses=settings.consecutive_loss_pause,
        pause_minutes=settings.pause_minutes,
        state_file=None,
    )
    risk_manager.circuit_breaker = cb
    quarantine = StrategyQuarantine(backtest_mode=True)

    # options proxy uses wider SL/TP thresholds so short straddle legs
    # aren't stopped out by normal intraday index moves
    if target.proxy:
        pos_cfg = PositionConfig(
            stop_loss_pct=0.60,
            target_pct=0.60,
            trailing_sl_activation_pct=0.30,
            trailing_sl_pct=0.15,
            enable_trailing_sl=True,
            use_atr_based_sl=False,
        )
    else:
        pos_cfg = PositionConfig()

    position_manager = PositionManager(
        broker=broker,
        risk_manager=risk_manager,
        config=pos_cfg,
        partial_profit_manager=partial_profit,
    )
    learning = LearningIntegration(
        db_session=get_session,
        enabled=False,
        backtest_mode=True,
    )

    signal_q: queue.Queue = queue.Queue()
    order_manager = OrderManager(
        signal_queue=signal_q,
        broker=broker,
        trading_mode="paper",
        db_url=settings.database_url,
        risk_manager=risk_manager,
        position_manager=position_manager,
        strategy_quarantine=quarantine,
    )

    # BarBuilder (real, same as live) driven by tick_queue
    tick_queue: queue.Queue = queue.Queue()
    bar_close_event = threading.Event()
    bar_builder = BarBuilder(
        tick_queue=tick_queue,
        bar_close_event=bar_close_event,
    )
    bar_builder.start()

    # StrategyEngine — bars_provider reads from BarBuilder
    def _bars_provider() -> dict[str, pd.DataFrame]:
        out = {}
        for ik in bar_builder.get_all_instrument_keys():
            df_b = bar_builder.get_bars(ik, timeframe=1)
            if df_b is not None and not df_b.empty:
                out[ik] = df_b
        return out

    strategy_engine = StrategyEngine(
        strategies_dir="config/strategies",
        bars_provider=_bars_provider,
        account_equity_paisa=capital,
    )
    strategy_engine.load_strategies()
    logger.info("Loaded {} strategies", len(strategy_engine.strategies))

    # ── synthetic option price tracking ──────────────────────────────────
    # For proxy instruments, we track a "virtual" option key per expiry.
    # OrderManager will try to fill these when a STRADDLE signal fires.
    # We expose them to PaperBroker via update_quote() each bar.
    current_option_keys: list[str] = []  # updated each bar for proxy instruments

    def _update_proxy_option_prices(close_paisa: int, high_paisa: int, low_paisa: int, ts: datetime) -> None:
        if not target.proxy:
            return
        premium = _synthetic_premium_paisa(close_paisa, high_paisa, low_paisa, ts)
        # Generate synthetic ATM CE/PE keys for this bar's close price
        atm = _atm_strike(close_paisa)
        expiry = _next_expiry(ts, target.label)
        expiry_str = expiry.strftime("%d%b%y").upper()  # e.g. 05JUN24
        sym = _underlying_symbol(target.label)
        ce_key = f"NSE_FO|{sym}{expiry_str}{atm // 100}CE"
        pe_key = f"NSE_FO|{sym}{expiry_str}{atm // 100}PE"
        current_option_keys.clear()
        current_option_keys.extend([ce_key, pe_key])
        for key in (ce_key, pe_key):
            broker.update_quote(key, premium, None, None)

    def _underlying_symbol(label: str) -> str:
        return {"BankNifty": "BANKNIFTY", "Nifty50": "NIFTY", "FinNifty": "FINNIFTY"}.get(label, label.upper())

    def _next_expiry(ts: datetime, label: str) -> date:
        # BankNifty → Wednesday (weekday 2)
        # Nifty50   → Thursday (weekday 3)
        # FinNifty  → Tuesday  (weekday 1)
        target_wd = {"BankNifty": 2, "Nifty50": 3, "FinNifty": 1}.get(label, 2)
        d = ts.date()
        days_ahead = (target_wd - d.weekday()) % 7
        if days_ahead == 0 and ts.time() >= time(15, 30):
            days_ahead = 7  # today's expiry already passed
        return d + timedelta(days=days_ahead)

    # ── replay loop ───────────────────────────────────────────────────────
    current_day: date | None = None
    peak_equity = capital
    trade_list: list[dict] = []
    bars_done = 0
    last_log_pct = 0

    def _sod(d: date) -> None:
        eq = broker.get_portfolio_value()
        risk_manager.reset_daily()
        risk_manager.update_capital(eq)
        strategy_engine.update_account_equity(eq)
        # Reset circuit breaker for new day
        with cb._lock:
            cb.consecutive_losses = 0
            cb._recent_losses.clear()
            if cb.halt_until is not None:
                cb.halted = False
                cb.halt_until = None
        logger.info("[SOD] {} | equity=₹{:,.0f}", d, eq / 100)

    def _eod(d: date) -> None:
        try:
            closed = position_manager.close_all_positions("EOD_SQUARE_OFF")
            for pos in closed:
                _record_trade(pos)
        except Exception as exc:
            logger.debug("[EOD] close_all error: {}", exc)
        logger.info("[EOD] {} | portfolio=₹{:,.0f}", d, broker.get_portfolio_value() / 100)

    def _record_trade(pos: Any) -> None:
        # gross from position's realized_pnl (before fees)
        gross = getattr(pos, "total_realized_pnl", None)
        if gross is None:
            gross = getattr(pos, "realized_pnl", 0)
        # fees from broker's trade history for this instrument
        ik = getattr(pos, "instrument_key", "")
        fees = int(broker.sum_fees_for_instrument(ik)) if ik else 0
        net = gross - fees
        trade_list.append({"gross": gross, "fees": fees, "net": net})
        result.trades += 1
        if net > 0:
            result.wins += 1
        result.gross_pnl_paisa += gross
        result.fees_paisa += fees
        if fees > abs(gross) and gross != 0:
            result.anomalies.append(
                f"fees({fees}) > gross_pnl({gross}) in trade {result.trades}"
            )

    for ts, row in df.iterrows():
        if current_day != ts.date():
            if current_day is not None:
                _eod(current_day)
            current_day = ts.date()
            _sod(current_day)

        # Update simulated clock (drives SetEvaluator._should_evaluate)
        set_sim_time(ts)

        # Skip outside market hours
        t = ts.time()
        if not (time(9, 15) <= t <= time(15, 30)):
            continue

        close_paisa = int(row["close"])
        high_paisa  = int(row["high"])
        low_paisa   = int(row["low"])
        open_paisa  = int(row["open"])
        volume      = int(row.get("volume", 0))
        close_rs    = close_paisa / 100.0

        ik = target.instrument_key

        # 1. Put close tick into tick_queue → BarBuilder accumulates bars
        ts_ms = str(int(ts.timestamp() * 1000))
        tick_dict = {
            "feeds": {
                ik: {
                    "ltpc": {"ltp": close_rs, "v": volume, "cp": close_rs},
                }
            },
            "currentTs": ts_ms,
        }
        tick_queue.put(tick_dict)

        # 2. Update paper broker quote (tick-by-tick for SL/TP check)
        # Replay OHLC as: open → high → low → close (worst-case ordering for shorts)
        for price in (open_paisa, high_paisa, low_paisa, close_paisa):
            if price <= 0:
                continue
            broker.update_quote(ik, price, None, None)
            position_manager.set_backtest_time(ts)
            closed_pos = position_manager.on_tick(ik, price)
            if closed_pos:
                for p in closed_pos:
                    _record_trade(p)
            if ik.startswith("NSE_INDEX|"):
                try:
                    position_manager.on_underlying_tick(ik, price)
                except Exception:
                    pass

        # 3. Update synthetic option prices (proxy instruments)
        _update_proxy_option_prices(close_paisa, high_paisa, low_paisa, ts)

        # 4. Evaluate bar synchronously using bar_time (not wall clock)
        #    give BarBuilder a moment to process the tick we just put in
        bar_close_event.wait(timeout=0.01)
        bar_close_event.clear()

        current_bars = _bars_provider()
        try:
            signals = strategy_engine.evaluate_bar_sync(current_bars, ts)
        except Exception as exc:
            logger.debug("evaluate_bar_sync error at {}: {}", ts, exc)
            signals = []

        for sig in signals:
            logger.info(
                "SIGNAL | {} | {} | {} | {}",
                target.label,
                getattr(sig, "signal_type", ""),
                getattr(sig, "instrument_key", ik),
                ts.strftime("%Y-%m-%d %H:%M"),
            )
            try:
                order_manager.process_signal(sig)
            except Exception as exc:
                logger.debug("process_signal error: {}", exc)

        # 5. Track equity curve for drawdown
        eq = broker.get_portfolio_value()
        if eq > peak_equity:
            peak_equity = eq
        drawdown = peak_equity - eq
        if drawdown > result.max_drawdown_paisa:
            result.max_drawdown_paisa = drawdown

        bars_done += 1
        pct = int(bars_done / total_bars * 100)
        if pct >= last_log_pct + 10:
            last_log_pct = pct
            logger.info(
                "REPLAY TICK | {} | {}% | bar={} | equity=₹{:,.0f} | trades={}",
                target.label, pct, ts.strftime("%Y-%m-%d %H:%M"),
                eq / 100, result.trades,
            )

    # Final EOD
    if current_day:
        _eod(current_day)

    # Close anything still open
    try:
        remaining = position_manager.close_all_positions("REPLAY_END")
        for p in remaining:
            _record_trade(p)
    except Exception:
        pass

    bar_builder.stop()
    result.bars_processed = bars_done
    result.peak_equity = peak_equity
    result._trade_list = trade_list
    result.elapsed_s = _time.perf_counter() - t0

    net_pnl_rs = result.net_pnl_paisa / 100
    fees_rs    = result.fees_paisa / 100
    maxdd_rs   = result.max_drawdown_paisa / 100
    maxdd_pct  = result.max_drawdown_paisa / capital * 100

    logger.info("■ TradingBot SHUTDOWN | instrument={}", target.label)
    logger.info("  bars_processed : {:,}", result.bars_processed)
    logger.info("  trades         : {}  |  win_rate: {:.1f}%", result.trades, result.win_rate)
    logger.info("  net_pnl        : ₹{:,.2f}", net_pnl_rs)
    logger.info("  fees           : ₹{:,.2f}", fees_rs)
    logger.info("  max_drawdown   : ₹{:,.2f} ({:.1f}%)", maxdd_rs, maxdd_pct)
    logger.info("  elapsed        : {:.1f}s", result.elapsed_s)
    if result.anomalies:
        for a in result.anomalies[:5]:
            logger.warning("[ANOMALY] {} {}", target.label, a)
        if len(result.anomalies) > 5:
            logger.warning("[ANOMALY] {} + {} more...", target.label, len(result.anomalies) - 5)

    logger.info("═" * 72)
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import time as _time

    parser = argparse.ArgumentParser(description="Replay live trading pipeline on historical data")
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--end",   type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--instrument", default=None, help="Run only one label (BankNifty/Nifty50/FinNifty/ADANIENT)")
    args = parser.parse_args()

    # configure loguru to stderr + file
    ts_tag = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/replay_{ts_tag}.log"
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True)
    logger.add(log_path, level="INFO", rotation="500 MB")

    settings = get_settings()

    logger.info("╔" + "═" * 76 + "╗")
    logger.info("║  LIVE PIPELINE REPLAY — 24-MONTH DATA — {}  ║", ts_tag)
    logger.info("║  Full TradingBot component tree | BarBuilder | StrategyEngine | OrderMgr  ║")
    logger.info("╚" + "═" * 76 + "╝")
    logger.info("Capital/instrument : ₹{:,.0f}", settings.capital / 100)
    logger.info("Log file           : {}", log_path)
    logger.info("")

    targets = INSTRUMENTS
    if args.instrument:
        targets = [t for t in INSTRUMENTS if t.label.lower() == args.instrument.lower()]
        if not targets:
            logger.error("Unknown instrument: {}", args.instrument)
            sys.exit(1)

    results: list[RunResult] = []
    grand_t0 = _time.perf_counter()

    for target in targets:
        reset_sim_time()
        r = run_instrument(target, settings, args.start, args.end)
        results.append(r)

    reset_sim_time()
    total_elapsed = _time.perf_counter() - grand_t0

    # ── final summary ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("╔" + "═" * 86 + "╗")
    logger.info("║  FINAL REPLAY SUMMARY — 24 months — capital ₹{:,.0f}/instrument  ║",
                settings.capital / 100)
    logger.info("╠" + "═" * 86 + "╣")
    logger.info("║  {:<18} {:<18} {:>7} {:>7} {:>7} {:>12} {:>12}  ║",
                "INSTRUMENT", "TYPE", "TRADES", "WR%", "NET_PNL", "MAXDD", "ELAPSED")
    logger.info("╠" + "═" * 86 + "╣")
    total_trades = 0
    total_wins   = 0
    total_net    = 0
    total_fees   = 0
    for r in results:
        net_rs = r.net_pnl_paisa / 100
        dd_rs  = r.max_drawdown_paisa / 100
        logger.info(
            "║  {:<18} {:<18} {:>7} {:>6.1f}% {:>+10,.0f} {:>12,.0f} {:>10.1f}s  ║",
            r.label, r.proxy_label, r.trades, r.win_rate, net_rs, dd_rs, r.elapsed_s,
        )
        total_trades += r.trades
        total_wins   += r.wins
        total_net    += r.net_pnl_paisa
        total_fees   += r.fees_paisa
    logger.info("╠" + "═" * 86 + "╣")
    wr_all = total_wins / total_trades * 100 if total_trades else 0
    logger.info(
        "║  {:<18} {:<18} {:>7} {:>6.1f}% {:>+10,.0f}  ║",
        "TOTAL", "", total_trades, wr_all, total_net / 100,
    )
    logger.info("╚" + "═" * 86 + "╝")
    logger.info("")
    logger.info("  Total fees charged   : ₹{:,.2f}", total_fees / 100)
    logger.info("  Total elapsed        : {:.1f}s ({:.1f} min)", total_elapsed, total_elapsed / 60)
    logger.info("  Log file             : {}", log_path)
    logger.info("")

    total_anomalies = sum(len(r.anomalies) for r in results)
    if total_anomalies:
        logger.warning("  ANOMALY CHECK: {} anomalous trades (fees > |gross_pnl|)", total_anomalies)
    else:
        logger.info("  ANOMALY CHECK: PASS — no fee-exceeds-gross-pnl trades")

    logger.info("")
    logger.info("  PROXY NOTE: BankNifty/Nifty50/FinNifty P&L uses ATR-derived synthetic")
    logger.info("  option premiums — real live P&L will differ based on actual IV/bid-ask.")
    logger.info("  ADANIENT uses real historical equity prices — no proxy.")


if __name__ == "__main__":
    main()
