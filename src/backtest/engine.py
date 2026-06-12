"""Core backtesting engine — replays historical bars through strategy conditions.

This module provides:
- ``BacktestEngine``: single-strategy backtest runner (bar-first loop).
- ``DayByDayBacktester``: multi-strategy orchestrator that splits data by
  trading day and produces a row-per-day summary table.

All monetary values are in **PAISA** (integer) — 1 Rupee = 100 paisa.
"""

import json
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any, Optional, Union
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

from config.constants import FEES, MARKET_OPEN, MARKET_CLOSE
from src.strategy.builder import StrategyBuilder, StrategyConfig
from src.strategy.conditions import ConditionEvaluator
from src.research.options_pricing import bsm_price_paisa, time_to_expiry

IST = ZoneInfo("Asia/Kolkata")

# Hard square-off: 15:15 IST — no new entries after this, force close all
HARD_SQUARE_OFF = dt_time(15, 15)
# No new entries after this time
NO_NEW_ENTRY_AFTER = dt_time(15, 0)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    """A completed backtest trade."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str  # "BUY" or "SELL"
    entry_price: int  # paisa
    exit_price: int  # paisa
    quantity: int
    pnl: int  # paisa (gross P&L after slippage)
    fees: int  # paisa (total fees: brokerage + STT + exchange + GST + stamp + SEBI)
    net_pnl: int  # paisa (pnl - fees)
    exit_reason: str  # "target", "stop_loss", "time_exit", "signal", "end_of_data"
    strategy_name: str = ""

    def to_dict(self) -> dict:
        """Convert trade to dictionary."""
        return {
            "entry_time": self.entry_time.isoformat()
            if isinstance(self.entry_time, pd.Timestamp)
            else str(self.entry_time),
            "exit_time": self.exit_time.isoformat()
            if isinstance(self.exit_time, pd.Timestamp)
            else str(self.exit_time),
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "fees": self.fees,
            "net_pnl": self.net_pnl,
            "exit_reason": self.exit_reason,
            "strategy_name": self.strategy_name,
        }

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class DayResult:
    """Summary of one trading day in the backtest."""

    date: date
    total_trades: int
    wins: int
    losses: int
    win_rate: float  # 0-100
    gross_pnl: int  # paisa (before fees)
    fees: int  # paisa
    net_pnl: int  # paisa
    max_drawdown: int  # paisa (intraday peak-to-trough)
    strategies_traded: list  # list of strategy names that produced trades
    best_trade: int  # paisa (best single trade net_pnl)
    worst_trade: int  # paisa (worst single trade net_pnl)
    capital_start: int  # paisa
    capital_end: int  # paisa


@dataclass
class DayByDayResults:
    """Full results from a day-by-day multi-strategy backtest."""

    initial_capital: int  # paisa
    final_capital: int  # paisa
    day_results: list  # list[DayResult]
    all_trades: list  # list[Trade]
    equity_curve: list  # list[int] — capital at end of each day

    def get_summary_metrics(self) -> dict:
        """Compute aggregate metrics across all days."""
        if not self.all_trades:
            return {
                "total_pnl_rupees": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_rupees": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "total_fees_rupees": 0.0,
                "return_pct": 0.0,
                "profitable_days": 0,
                "losing_days": 0,
                "total_days": len(self.day_results),
                "avg_daily_pnl_rupees": 0.0,
                "best_day_rupees": 0.0,
                "worst_day_rupees": 0.0,
            }

        net_pnls = [t.net_pnl for t in self.all_trades]
        winners = [p for p in net_pnls if p > 0]
        losers = [p for p in net_pnls if p <= 0]

        total_net_pnl = sum(net_pnls)
        total_fees = sum(t.fees for t in self.all_trades)
        win_rate = len(winners) / len(net_pnls) * 100 if net_pnls else 0.0
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max drawdown from day-end equity curve
        peak = self.initial_capital
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd

        # Sharpe from daily returns
        daily_returns = []
        for dr in self.day_results:
            if dr.capital_start > 0:
                daily_returns.append(dr.net_pnl / dr.capital_start)
        if len(daily_returns) > 1:
            mean_r = sum(daily_returns) / len(daily_returns)
            var_r = sum((r - mean_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
            std_r = math.sqrt(var_r) if var_r > 0 else 1e-10
            sharpe = (mean_r / std_r) * math.sqrt(252)
        else:
            sharpe = 0.0

        day_pnls = [dr.net_pnl for dr in self.day_results]
        profitable_days = sum(1 for p in day_pnls if p > 0)
        losing_days = sum(1 for p in day_pnls if p < 0)

        return {
            "total_pnl_rupees": round(total_net_pnl / 100, 2),
            "total_trades": len(self.all_trades),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_rupees": round(max_dd / 100, 2),
            "max_drawdown_pct": round(max_dd / self.initial_capital * 100, 2)
            if self.initial_capital
            else 0,
            "sharpe_ratio": round(sharpe, 2),
            "total_fees_rupees": round(total_fees / 100, 2),
            "return_pct": round(
                (self.final_capital - self.initial_capital) / self.initial_capital * 100, 2
            )
            if self.initial_capital
            else 0,
            "profitable_days": profitable_days,
            "losing_days": losing_days,
            "total_days": len(self.day_results),
            "avg_daily_pnl_rupees": round(sum(day_pnls) / len(day_pnls) / 100, 2)
            if day_pnls
            else 0,
            "best_day_rupees": round(max(day_pnls) / 100, 2) if day_pnls else 0,
            "worst_day_rupees": round(min(day_pnls) / 100, 2) if day_pnls else 0,
        }


@dataclass
class BacktestResults:
    """Results from a single backtest run (kept for backward compat)."""

    initial_capital: int  # paisa
    final_capital: int  # paisa
    trades: list  # list[Trade]
    equity_curve: list  # list[int]
    total_bars_processed: int

    def get_metrics(self) -> dict:
        """Compute performance metrics from trade results."""
        peak = self.initial_capital
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd

        if not self.trades:
            return {
                "total_pnl": 0,
                "total_pnl_rupees": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": max_dd,
                "max_drawdown_rupees": round(max_dd / 100, 2),
                "max_drawdown_pct": round(max_dd / self.initial_capital * 100, 2),
                "sharpe_ratio": 0.0,
                "avg_trade_duration_min": 0.0,
                "avg_winner": 0.0,
                "avg_loser": 0.0,
                "final_capital_rupees": self.final_capital / 100,
                "return_pct": round(
                    (self.final_capital - self.initial_capital) / self.initial_capital * 100, 2
                ),
                "equity_curve": self.equity_curve,
            }

        pnls = [
            t.net_pnl if hasattr(t, "net_pnl") else t.get("net_pnl", t.get("pnl", 0))
            for t in self.trades
        ]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        win_rate = len(winners) / len(pnls) if pnls else 0.0
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        if len(pnls) > 1:
            # Sharpe from DAILY aggregated returns (not per-trade)
            # Group trade P&L by exit date
            daily_pnl_map: dict[str, int] = {}
            for t in self.trades:
                et = t.exit_time if hasattr(t, "exit_time") else t.get("exit_time")
                date_key = str(et)[:10] if et else ""
                if not date_key:
                    continue
                net = t.net_pnl if hasattr(t, "net_pnl") else t.get("net_pnl", 0)
                daily_pnl_map[date_key] = daily_pnl_map.get(date_key, 0) + net

            if len(daily_pnl_map) > 1:
                daily_returns = [v / self.initial_capital for v in daily_pnl_map.values()]
                mean_ret = sum(daily_returns) / len(daily_returns)
                var = sum((r - mean_ret) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
                std_ret = math.sqrt(var) if var > 0 else 1e-10
                sharpe = (mean_ret / std_ret) * math.sqrt(252)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        durations = []
        for t in self.trades:
            et = t.exit_time if hasattr(t, "exit_time") else t.get("exit_time")
            en = t.entry_time if hasattr(t, "entry_time") else t.get("entry_time")
            if isinstance(et, pd.Timestamp) and isinstance(en, pd.Timestamp):
                durations.append((et - en).total_seconds() / 60)
        avg_dur = sum(durations) / len(durations) if durations else 0

        return {
            "total_pnl": total_pnl,
            "total_pnl_rupees": round(total_pnl / 100, 2),
            "total_trades": len(pnls),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate": round(win_rate * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown": max_dd,
            "max_drawdown_rupees": round(max_dd / 100, 2),
            "max_drawdown_pct": round(max_dd / self.initial_capital * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "avg_trade_duration_min": round(avg_dur, 1),
            "avg_winner": round(sum(winners) / len(winners), 0) if winners else 0,
            "avg_loser": round(sum(losers) / len(losers), 0) if losers else 0,
            "final_capital_rupees": round(self.final_capital / 100, 2),
            "return_pct": round(
                (self.final_capital - self.initial_capital) / self.initial_capital * 100, 2
            ),
            "equity_curve": self.equity_curve,
        }


# ---------------------------------------------------------------------------
# Fee calculation (mirrors PaperBroker for backtest parity)
# ---------------------------------------------------------------------------


def calculate_trade_fees(
    entry_price: int,
    exit_price: int,
    quantity: int,
    side: str,
    segment: str = "NSE_EQ",
    product: str = "MIS",
    is_options: bool = False,
) -> int:
    """Calculate realistic trading fees for a round-trip trade.

    Uses the same fee constants as PaperBroker to ensure backtest-live parity.
    Options and futures are handled separately as their STT and exchange charges
    differ significantly.

    Args:
        entry_price: Entry fill price in paisa.
        exit_price: Exit fill price in paisa.
        quantity: Trade quantity.
        side: "BUY" or "SELL" (entry side).
        segment: Market segment — "NSE_EQ", "NSE_FO", etc.
        product: Product type — "MIS" (intraday), "CNC" (delivery), "NRML".
        is_options: If True, applies options-specific STT (on premium, not contract value).

    Returns:
        Total fees in paisa.
    """
    is_fno = segment in ("NSE_FO", "NFO")

    # For options, premium turnover = price × qty (not notional contract value)
    # For futures/equity, turnover = full price × qty
    entry_value = entry_price * quantity  # paisa
    exit_value = exit_price * quantity    # paisa
    turnover = entry_value + exit_value   # round-trip premium turnover (paisa)
    turnover_rs = turnover / 100          # in rupees

    total_fees = 0

    # --- Brokerage (flat per leg, 2 legs = entry + exit) ---
    if product == "CNC":
        brokerage = 0
    elif is_fno:
        brokerage = FEES["brokerage_fno"] * 2
    else:
        brokerage = FEES["brokerage_equity_intraday"] * 2
    total_fees += int(brokerage)

    # --- STT (Securities Transaction Tax) ---
    if is_options:
        # Options STT: 0.0125% on sell-side premium only
        # (Sebi circular: STT on options = 0.125% on premium at exercise;
        #  for intraday options selling: 0.0125% on premium turnover on sell leg)
        sell_value = exit_value if side == "BUY" else entry_value
        stt = sell_value * FEES["stt_options_sell_pct"]
    elif is_fno:
        # Futures STT: 0.02% on sell-side notional
        sell_value = exit_value if side == "BUY" else entry_value
        stt = sell_value * FEES["stt_futures_sell_pct"]
    elif product == "MIS":
        # Equity intraday: 0.025% on sell side
        sell_value = exit_value if side == "BUY" else entry_value
        stt = sell_value * FEES["stt_equity_intraday_sell_pct"]
    else:
        # Delivery: 0.1% on sell side
        sell_value = exit_value if side == "BUY" else entry_value
        stt = sell_value * FEES["stt_equity_delivery_sell_pct"]
    total_fees += int(round(stt))

    # --- Exchange transaction charges ---
    # For options: charged on premium turnover; for equity/futures: on full turnover
    if is_fno:
        exch_rate = FEES["exchange_charge_nse_fo_pct"]
    else:
        exch_rate = FEES["exchange_charge_nse_eq_pct"]
    exchange_charges_rs = turnover_rs * exch_rate
    exchange_charges = exchange_charges_rs * 100  # back to paisa
    total_fees += int(round(exchange_charges))

    # --- GST: 18% on (brokerage + exchange charges) ---
    gst_base = brokerage + exchange_charges
    gst = gst_base * FEES["gst_pct"]
    total_fees += int(round(gst))

    # --- SEBI charges (Rs 10 per crore = 0.0000001 of turnover in rupees) ---
    sebi = turnover_rs * FEES["sebi_charge_rate_pct"] * 100  # paisa
    total_fees += max(int(round(sebi)), 0)

    # --- Stamp duty (buy side only) ---
    buy_value = entry_value if side == "BUY" else exit_value
    if is_fno:
        stamp_rate = FEES["stamp_duty_fno_pct"]
    elif product == "CNC":
        stamp_rate = FEES["stamp_duty_equity_delivery_pct"]
    else:
        stamp_rate = FEES["stamp_duty_equity_intraday_pct"]
    stamp = buy_value * stamp_rate  # stamp_rate is decimal (e.g. 0.00015)
    total_fees += int(round(stamp))

    return total_fees


# ---------------------------------------------------------------------------
# BacktestEngine (bar-first, multi-strategy)
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Replays historical bars through strategy conditions to simulate trades.

    Now uses a **bar-first** loop: for each bar, all strategies are evaluated.
    This ensures concurrent strategies compete for the same bars realistically.

    Args:
        capital: Initial capital in paisa (default 10 lakhs).
        slippage_pct: Slippage percentage (default 0.05%).
        max_position_size_pct: Max position size as % of capital (default 20%).
        max_daily_loss_pct: Max daily loss as % of capital (default 5%).
        max_open_positions: Max simultaneous open positions (default 5).
        use_realistic_fees: Use full fee model (STT, exchange, GST, etc.).
        segment: Market segment for fee calc ("NSE_EQ" or "NSE_FO").
        product: Product type ("MIS" for intraday, "CNC" for delivery).
    """

    def __init__(
        self,
        capital: int = 100_000_000,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        slippage_pct: float = 0.05,
        commission_per_order: int = 2000,
        max_position_size_pct: float = 50.0,
        max_daily_loss_pct: float = 5.0,
        max_open_positions: int = 5,
        use_realistic_fees: bool = True,
        segment: str = "NSE_EQ",
        product: str = "MIS",
        options_mode: bool = False,
        option_premium_pct: float = 0.5,
        option_delta: float = 0.5,
        # BSM (Black-Scholes-Merton) option pricing integration
        use_bsm: bool = False,
        iv_source: Any = None,  # pd.Series indexed by datetime with annual vol (decimal)
        default_iv: float = 0.25,
        risk_free_rate: float = 0.05,
        strike_interval: int = 5000,  # in paisa (50 points)
    ) -> None:
        self.initial_capital = capital
        self.current_capital = capital
        self.start_date = start_date
        self.end_date = end_date
        self.slippage_pct = slippage_pct
        self.commission_per_order = commission_per_order
        self.max_position_size_pct = max_position_size_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_open_positions = max_open_positions
        self.use_realistic_fees = use_realistic_fees
        self.segment = segment
        self.product = product
        # Options proxy mode: simulate trading ATM options using index data.
        # Converts index price movements to approximate option premium movements.
        #   option_premium_pct: ATM premium as % of underlying (e.g., 0.5% → Rs260 for BankNifty)
        #   option_delta: delta of ATM option (~0.5 for ATM, decays intraday)
        self.options_mode = options_mode
        self.option_premium_pct = option_premium_pct
        self.option_delta = option_delta
        # BSM integration
        self.use_bsm = use_bsm
        self.iv_source = iv_source
        self.default_iv = default_iv
        self.risk_free_rate = risk_free_rate
        self.strike_interval = strike_interval

        self.strategies: list[dict] = []
        self.positions: dict[str, dict] = {}  # key -> position info
        self.trades: list[Trade] = []
        self.equity_curve: list[int] = [capital]

        self.daily_pnl: int = 0
        self.daily_trades: int = 0

        # Re-entry cooldown: strategy_name -> earliest allowed re-entry time
        self._cooldown_until: dict[str, pd.Timestamp] = {}
        # Per-strategy trade count for the current run
        self._strategy_trade_count: dict[str, int] = {}

        logger.info(
            f"BacktestEngine initialized | capital={capital} paisa | "
            f"slippage={slippage_pct}% | realistic_fees={use_realistic_fees}"
        )

    # ----- strategy registration -----

    def add_strategy(self, strategy_config_path: str, instrument: str) -> None:
        """Add a strategy from a JSON file path."""
        path = Path(strategy_config_path)
        if not path.exists():
            raise FileNotFoundError(f"Strategy config not found: {strategy_config_path}")
        with open(path) as f:
            raw = json.load(f)
        strategy = StrategyConfig(**raw)
        self.strategies.append(
            {
                "config_path": strategy_config_path,
                "instrument": instrument,
                "strategy": strategy,
            }
        )
        logger.info(f"Added strategy '{strategy.name}' for {instrument}")

    def add_strategy_config(self, strategy: StrategyConfig, instrument: str) -> None:
        """Add a pre-loaded StrategyConfig directly."""
        self.strategies.append(
            {
                "config_path": None,
                "instrument": instrument,
                "strategy": strategy,
            }
        )

    # ----- main run loop (bar-first) -----

    def run(self, data: dict[str, pd.DataFrame]) -> BacktestResults:
        """Run backtest — bar-first loop evaluating all strategies per bar.

        Args:
            data: Dict mapping instrument_key → OHLCV DataFrame (paisa prices).

        Returns:
            BacktestResults with trades, equity curve, and metrics.
        """
        evaluator = ConditionEvaluator()

        # Determine the master bar sequence (union of all instruments)
        all_instruments = set()
        for si in self.strategies:
            all_instruments.add(si["instrument"])

        # Use the first (usually only) instrument's DataFrame as the bar index
        master_instrument = self.strategies[0]["instrument"] if self.strategies else None
        if master_instrument is None or master_instrument not in data:
            logger.warning("No data for master instrument")
            return self._build_results(0)

        df = data[master_instrument]
        total_bars = len(df)
        # Warmup: need enough bars for slowest indicator (MACD 26 + signal 9 = 35)
        # But for small daily dataframes (5min = 75 bars), use fewer
        warmup = min(30, max(10, total_bars // 5))

        prev_date = None  # Track date to reset daily counters at start of each new day

        for i in range(warmup, total_bars):
            bar = df.iloc[i]
            bar_time = df.index[i]
            current_price = int(bar["close"])
            window = df.iloc[: i + 1]

            # --- New-day reset ---
            bar_date = bar_time.date() if hasattr(bar_time, "date") else None
            if bar_date and bar_date != prev_date:
                if prev_date is not None:
                    logger.debug(f"[BacktestEngine] New trading day: {bar_date} — resetting daily counters")
                self.daily_pnl = 0
                self.daily_trades = 0
                self._strategy_trade_count = {}
                self._cooldown_until = {}
                prev_date = bar_date

            # --- Skip pre-market bars (before 9:15 IST) ---
            bar_time_t = bar_time.time() if hasattr(bar_time, "time") else None
            if bar_time_t and bar_time_t < MARKET_OPEN:
                self.equity_curve.append(self.current_capital)
                continue

            # --- Phase 1: Check exits for all open positions ---
            for pos_key in list(self.positions.keys()):
                position = self.positions[pos_key]
                strat = position["strategy"]
                exit_reason = self._check_exit(position, current_price, bar_time, strat)
                if exit_reason:
                    trade = self._close_position(position, current_price, bar_time, exit_reason)
                    self.current_capital += trade.net_pnl
                    self.daily_pnl += trade.net_pnl
                    self.trades.append(trade)
                    del self.positions[pos_key]

                    # Set 15-minute re-entry cooldown for this strategy
                    cooldown_minutes = 15
                    self._cooldown_until[trade.strategy_name] = bar_time + pd.Timedelta(
                        minutes=cooldown_minutes
                    )
                    # Increment per-strategy trade count
                    self._strategy_trade_count[trade.strategy_name] = (
                        self._strategy_trade_count.get(trade.strategy_name, 0) + 1
                    )

            # --- Phase 2: Check daily loss limit ---
            daily_loss_limit = int(self.initial_capital * self.max_daily_loss_pct / 100)
            daily_loss_breached = self.daily_pnl < -daily_loss_limit

            # --- Phase 3: Collect entry signals from all strategies ---
            bar_time_t = bar_time.time() if hasattr(bar_time, "time") else None
            can_enter = (
                not daily_loss_breached
                and len(self.positions) < self.max_open_positions
                and (bar_time_t is None or bar_time_t < NO_NEW_ENTRY_AFTER)
            )

            if can_enter:
                candidate_signals = []
                for strategy_info in self.strategies:
                    instrument = strategy_info["instrument"]
                    strategy = strategy_info["strategy"]

                    if not strategy.enabled:
                        continue

                    # Skip if already holding a position from this strategy
                    pos_key = f"{instrument}|{strategy.name}"
                    if pos_key in self.positions:
                        continue

                    # Re-entry cooldown: skip if we recently exited this strategy
                    cooldown_key = strategy.name
                    if cooldown_key in self._cooldown_until:
                        if bar_time < self._cooldown_until[cooldown_key]:
                            continue

                    # Per-strategy daily trade limit (use strategy config or default 5)
                    risk_mgmt = strategy.risk_management or {}
                    max_daily = (
                        risk_mgmt.get("max_trades_per_day", 5)
                        if isinstance(risk_mgmt, dict)
                        else getattr(risk_mgmt, "max_trades_per_day", 5)
                    )
                    max_daily = min(max_daily, 5)  # Hard cap at 5 trades/day/strategy
                    strat_count = self._strategy_trade_count.get(strategy.name, 0)
                    if strat_count >= max_daily:
                        continue

                    inst_df = data.get(instrument)
                    if inst_df is None:
                        continue

                    inst_window = (
                        inst_df.iloc[: i + 1]
                        if instrument == master_instrument
                        else inst_df.iloc[: i + 1]
                    )
                    window_bars = {instrument: inst_window}

                    for entry_set in strategy.entry_sets:
                        try:
                            all_met = evaluator.evaluate_all(
                                entry_set.conditions, window_bars, instrument
                            )
                        except Exception:
                            all_met = False

                        if all_met:
                            priority = getattr(entry_set, "priority", 0) or 0
                            candidate_signals.append((strategy_info, entry_set, priority))

                # Sort by priority (highest first)
                candidate_signals.sort(key=lambda x: x[2], reverse=True)

                # Execute best signals up to position limit
                # Track directions used this bar to avoid correlated entries
                directions_entered_this_bar: set = set()

                for strategy_info, entry_set, _priority in candidate_signals:
                    if len(self.positions) >= self.max_open_positions:
                        break

                    instrument = strategy_info["instrument"]
                    strategy = strategy_info["strategy"]
                    pos_key = f"{instrument}|{strategy.name}"

                    if pos_key in self.positions:
                        continue

                    # Determine direction
                    side = "BUY" if entry_set.signal in ("CE", "BUY", "BUY_CE") else "SELL"
                    direction_key = f"{instrument}|{side}"

                    # Prevent multiple strategies from entering same direction
                    # on the same instrument at the same bar (correlated risk)
                    if direction_key in directions_entered_this_bar:
                        continue

                    inst_df = data.get(instrument, df)
                    inst_price = (
                        int(inst_df.iloc[i]["close"]) if i < len(inst_df) else current_price
                    )

                    qty = self._compute_quantity(inst_price, strategy)
                    if qty <= 0:
                        continue

                    # side already determined above for direction check
                    # In options mode, entry slippage is accounted for in
                    # _close_position on the premium level. Store raw index
                    # price as entry so _check_exit has clean reference.
                    if self.options_mode:
                        fill_price = inst_price  # no index-level slippage
                    else:
                        dynamic_slippage = self.slippage_pct * self._get_slippage_factor(bar_time)
                        slippage = int(inst_price * dynamic_slippage / 100)
                        fill_price = (
                            inst_price + slippage if side == "BUY" else inst_price - slippage
                        )

                    # Check position size limit — auto-reduce quantity to fit
                    position_value = fill_price * qty
                    max_allowed = int(self.current_capital * self.max_position_size_pct / 100)
                    if position_value > max_allowed:
                        # Try to reduce qty to fit within limit
                        if fill_price > 0:
                            max_qty = max_allowed // fill_price
                            if max_qty >= 1:
                                qty = max_qty
                                logger.debug(
                                    f"[{strategy.name}] Reduced qty to {qty} to fit "
                                    f"position size limit ({max_allowed} paisa)"
                                )
                            else:
                                # Even 1 unit exceeds limit — skip but log
                                logger.debug(
                                    f"[{strategy.name}] Skipped: 1 unit @ {fill_price} "
                                    f"exceeds position limit {max_allowed} paisa "
                                    f"({fill_price / 100:.0f} Rs > {max_allowed / 100:.0f} Rs)"
                                )
                                continue
                        else:
                            continue

                    # BSM option pricing pre-computation (if enabled)
                    option_types_to_process = []
                    hedge_required = strategy.risk_management.get("hedge_required", False) if hasattr(strategy, "risk_management") else False
                    hedge_offset = strategy.params.get("hedge_offset", 500) if hasattr(strategy, "params") else 500

                    if self.options_mode and self.use_bsm:
                        if entry_set.signal == "STRADDLE":
                            option_types_to_process = ["CE", "PE"]
                            if hedge_required:
                                option_types_to_process.extend(["CE_HEDGE", "PE_HEDGE"])
                        else:
                            option_types_to_process = [entry_set.signal]
                    else:
                        # Proxy mode just treats it as a single aggregated unit
                        option_types_to_process = [entry_set.signal]

                    for opt_type in option_types_to_process:
                        pos_key_specific = f"{pos_key}|{opt_type}" if len(option_types_to_process) > 1 else pos_key
                        
                        strike = None
                        expiry_ts = None
                        entry_premium = None
                        
                        # Determine direction for the specific leg
                        # Hedges are bought (long), main legs are sold (short)
                        leg_side = side
                        if "HEDGE" in opt_type:
                            leg_side = "BUY" if side == "SELL" else "SELL"
                        
                        if self.options_mode and self.use_bsm:
                            atm_strike = int(round(inst_price / self.strike_interval)) * self.strike_interval
                            
                            if opt_type == "CE_HEDGE":
                                strike = atm_strike + hedge_offset
                            elif opt_type == "PE_HEDGE":
                                strike = atm_strike - hedge_offset
                            else:
                                strike = atm_strike
                                
                            expiry_ts = self._compute_expiry(bar_time, instrument)
                            iv_entry = self._get_iv_at(bar_time)
                            T_entry = time_to_expiry(bar_time.timestamp(), expiry_ts.timestamp())
                            kind = "call" if "CE" in opt_type else "put"
                            entry_premium = bsm_price_paisa(
                                inst_price, strike, T_entry, iv_entry, self.risk_free_rate, kind
                            )

                        self.positions[pos_key_specific] = {
                            "instrument": instrument,
                            "side": leg_side,
                            "entry_price": fill_price,
                            "entry_time": bar_time,
                            "quantity": qty,
                            "strategy": strategy,
                            "strategy_name": strategy.name,
                            # BSM metadata
                            "option_type": opt_type.split("_")[0], # CE_HEDGE -> CE
                            "option_strike": strike,
                            "option_expiry": expiry_ts,
                            "option_entry_premium": entry_premium,
                        }
                    
                    # Mark this direction as used this bar
                    directions_entered_this_bar.add(direction_key)
                    logger.debug(f"[{strategy.name}] Entered Iron Condor/Straddle {qty} @ {fill_price} paisa (Types: {option_types_to_process})")

            # --- Phase 4: Hard square-off at 15:15 ---
            if bar_time_t and bar_time_t >= HARD_SQUARE_OFF:
                for pos_key in list(self.positions.keys()):
                    position = self.positions[pos_key]
                    trade = self._close_position(position, current_price, bar_time, "time_exit")
                    self.current_capital += trade.net_pnl
                    self.daily_pnl += trade.net_pnl
                    self.trades.append(trade)
                    self._strategy_trade_count[trade.strategy_name] = (
                        self._strategy_trade_count.get(trade.strategy_name, 0) + 1
                    )
                    del self.positions[pos_key]

            # --- Update equity curve ---
            unrealized = sum(
                self._calculate_unrealized_pnl_for(pos, current_price)
                for pos in self.positions.values()
            )
            self.equity_curve.append(self.current_capital + unrealized)

        # Force close any remaining positions
        for pos_key in list(self.positions.keys()):
            position = self.positions[pos_key]
            instrument = position["instrument"]
            inst_df = data.get(instrument, df)
            last_price = int(inst_df.iloc[-1]["close"])
            last_time = inst_df.index[-1]
            trade = self._close_position(position, last_price, last_time, "end_of_data")
            self.current_capital += trade.net_pnl
            self.trades.append(trade)
            del self.positions[pos_key]

        return self._build_results(total_bars)

    # ----- helpers -----

    def _build_results(self, total_bars: int) -> BacktestResults:
        return BacktestResults(
            initial_capital=self.initial_capital,
            final_capital=self.current_capital,
            trades=self.trades,
            equity_curve=self.equity_curve,
            total_bars_processed=total_bars,
        )

    def _compute_quantity(self, price: int, strategy: StrategyConfig) -> int:
        if price <= 0:
            return 0
        sizing = strategy.position_sizing
        if sizing and sizing.quantity:
            qty = sizing.quantity
        else:
            risk_amount = int(self.current_capital * 0.01)
            sl_pct = strategy.exit_rules.stop_loss_pct if strategy.exit_rules else 2.0
            sl_per_unit = int(price * sl_pct / 100)
            if sl_per_unit <= 0:
                qty = 1
            else:
                qty = max(1, risk_amount // sl_per_unit)

        # Round to lot size so backtest matches live execution
        lot_size = self._get_lot_size_for_strategy(strategy)
        if lot_size > 1:
            qty = max(lot_size, (qty // lot_size) * lot_size)
        return qty

    def _get_lot_size_for_strategy(self, strategy: StrategyConfig) -> int:
        """Resolve lot size from strategy config or instrument name."""
        # Try explicit lot_size param first
        if hasattr(strategy, "params") and isinstance(strategy.params, dict):
            ls = strategy.params.get("lot_size")
            if ls:
                return int(ls)
        # Infer from instrument name
        inst = getattr(strategy, "name", "").upper()
        if "BANKNIFTY" in inst:
            return 35
        if "FINNIFTY" in inst:
            return 40
        if "MIDCPNIFTY" in inst:
            return 50
        if "SENSEX" in inst:
            return 20
        if "NIFTY" in inst:
            return 75
        return 1

    def _compute_pnl_pct(self, position: dict, current_price: int) -> float:
        """Compute P&L percentage for a position (options-aware)."""
        entry = position["entry_price"]
        side = position["side"]

        if self.options_mode:
            index_change_pct = (current_price - entry) / entry * 100
            premium_multiplier = self.option_delta * 100 / self.option_premium_pct
            if side == "BUY":
                return index_change_pct * premium_multiplier
            else:
                return -index_change_pct * premium_multiplier
        else:
            if side == "BUY":
                return (current_price - entry) / entry * 100
            else:
                return (entry - current_price) / entry * 100

    def _check_exit(
        self, position: dict, current_price: int, bar_time: pd.Timestamp, strategy: StrategyConfig
    ) -> Optional[str]:
        pnl_pct = self._compute_pnl_pct(position, current_price)

        exit_rules = strategy.exit_rules
        if exit_rules:
            sl = exit_rules.stop_loss_pct or 2.0

            # --- Trailing stop loss ---
            tsl_cfg = exit_rules.trailing_stop_loss or {}
            tsl_enabled = tsl_cfg.get("enabled", False)
            if tsl_enabled:
                activation_pct = tsl_cfg.get("activation_pct", 15.0)
                trail_pct = tsl_cfg.get("trail_pct", 10.0)

                # Track highest P&L seen for this position
                peak_key = "peak_pnl_pct"
                prev_peak = position.get(peak_key, 0.0)
                if pnl_pct > prev_peak:
                    position[peak_key] = pnl_pct
                    prev_peak = pnl_pct

                # If peak exceeded activation, trailing stop is active
                if prev_peak >= activation_pct:
                    trailing_sl_level = prev_peak - trail_pct
                    if pnl_pct <= trailing_sl_level:
                        return "trailing_stop"

            # Fixed stop loss
            if pnl_pct <= -sl:
                return "stop_loss"

            tgt = exit_rules.target_pct or 4.0
            if pnl_pct >= tgt:
                return "target"

            if exit_rules.time_based_exit:
                exit_time_str = exit_rules.time_based_exit
                if hasattr(bar_time, "time"):
                    parts = exit_time_str.split(":")
                    exit_time = dt_time(
                        int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0
                    )
                    if bar_time.time() >= exit_time:
                        return "time_exit"

            # Max holding duration
            if exit_rules.max_holding_minutes:
                entry_time = position["entry_time"]
                if isinstance(entry_time, pd.Timestamp) and isinstance(bar_time, pd.Timestamp):
                    held_min = (bar_time - entry_time).total_seconds() / 60
                    if held_min >= exit_rules.max_holding_minutes:
                        return "max_holding"

        return None

    def _get_iv_at(self, ts: pd.Timestamp) -> float:
        """Get implied volatility value at a given timestamp.

        Uses the configured iv_source (a pandas Series indexed by datetime).
        If source is unavailable or lookup fails, falls back to default_iv.

        Args:
            ts: Timestamp to query.

        Returns:
            Annualized volatility as decimal (e.g., 0.25).
        """
        if self.iv_source is None:
            return self.default_iv
        try:
            # Ensure timezone compatibility
            if isinstance(self.iv_source, pd.Series):
                # Use asof for nearest past value
                iv_val = self.iv_source.asof(ts)
                if pd.isna(iv_val):
                    return self.default_iv
                return float(iv_val)
        except Exception as e:
            logger.debug(f"IV lookup failed for {ts}: {e}, using default")
        return self.default_iv

    def _get_slippage_factor(self, ts: pd.Timestamp) -> float:
        """Get slippage multiplier based on current volatility (VIX).

        Higher VIX widens spreads → higher slippage factor.
        Args:
            ts: Timestamp of trade.

        Returns:
            Multiplier (1.0 = base slippage, 1.5 = 50% wider).
        """
        vix = self._get_iv_at(ts)
        if vix > 0.20:
            return 1.5
        elif vix < 0.12:
            return 1.0
        else:
            # Linear interpolation between 1.0 and 1.5 for 12%-20% VIX
            return 1.0 + 1.5 * (vix - 0.12) / (0.20 - 0.12)

    def _compute_expiry(self, ts: pd.Timestamp, instrument: str) -> pd.Timestamp:
        """Compute next options expiry timestamp (15:30 IST).

        For BankNifty: weekly expiry on Wednesday.
        For Nifty: weekly expiry on Thursday.

        Args:
            ts: Current timestamp (entry time).
            instrument: Instrument key or symbol (e.g., "NSE_INDEX|Nifty Bank").

        Returns:
            Expiry datetime (timezone-aware IST).
        """
        expiry_time = dt_time(15, 30)
        inst_upper = instrument.upper()
        if "BANKNIFTY" in inst_upper:
            expiry_weekday = 2  # Wednesday
        else:
            expiry_weekday = 3  # Thursday (NIFTY and others)
        current_weekday = ts.weekday()
        days_ahead = (expiry_weekday - current_weekday) % 7
        candidate_date = (ts + timedelta(days=days_ahead)).date()
        # If today is expiry day but after market close, move to next week
        if candidate_date == ts.date() and ts.time() >= expiry_time:
            candidate_date += timedelta(days=7)
        expiry_dt = datetime.combine(candidate_date, expiry_time, tzinfo=IST)
        return pd.Timestamp(expiry_dt)

    def _close_position(
        self, position: dict, exit_price: int, exit_time: pd.Timestamp, reason: str
    ) -> Trade:
        side = position["side"]
        entry = position["entry_price"]
        qty = position["quantity"]
        strategy_name = position.get("strategy_name", "")

        if self.options_mode:
            # Determine if using Black-Scholes or delta proxy
            if getattr(self, 'use_bsm', False):
                option_type = position.get("option_type")
                strike = position.get("option_strike")
                expiry = position.get("option_expiry")
                entry_premium_stored = position.get("option_entry_premium")
                if option_type is None or strike is None or expiry is None or entry_premium_stored is None:
                    logger.error(f"[{strategy_name}] Missing BSM metadata; falling back to proxy")
                    use_proxy = True
                else:
                    use_proxy = False
            else:
                use_proxy = True

            if use_proxy:
                # Original delta-based proxy approximation
                option_entry_premium = int(entry * self.option_premium_pct / 100)
                dynamic_slippage = self.slippage_pct * self._get_slippage_factor(exit_time)
                premium_slippage_per_side = int(option_entry_premium * dynamic_slippage / 100)
                total_premium_slippage = premium_slippage_per_side * 2
                index_change = exit_price - entry if side == "BUY" else entry - exit_price
                premium_change = int(index_change * self.option_delta)
                premium_change -= total_premium_slippage
                raw_pnl = premium_change * qty
                option_exit_premium = option_entry_premium + (
                    premium_change if side == "BUY" else -premium_change
                )
                option_exit_premium = max(option_exit_premium, 100)
                entry_display = option_entry_premium
                fill_display = option_exit_premium
                if self.use_realistic_fees:
                    fees = calculate_trade_fees(
                        option_entry_premium,
                        option_exit_premium,
                        qty,
                        side,
                        "NSE_FO",
                        self.product,
                    )
                else:
                    fees = self.commission_per_order * 2
            else:
                # BSM exit pricing
                sigma = self._get_iv_at(exit_time)
                T_exit = time_to_expiry(exit_time.timestamp(), expiry.timestamp())
                kind = "call" if option_type == "CE" else "put"
                exit_premium = bsm_price_paisa(exit_price, strike, T_exit, sigma, self.risk_free_rate, kind)
                entry_premium_val = entry_premium_stored
                # Slippage as round-trip cost on entry premium
                dynamic_slippage = self.slippage_pct * self._get_slippage_factor(exit_time)
                premium_slippage_per_side = int(entry_premium_val * dynamic_slippage / 100)
                total_premium_slippage = premium_slippage_per_side * 2
                if side == "BUY":
                    raw_pnl = (exit_premium - entry_premium_val) * qty
                else:
                    raw_pnl = (entry_premium_val - exit_premium) * qty
                raw_pnl -= total_premium_slippage * qty
                entry_display = entry_premium_val
                fill_display = exit_premium
                if self.use_realistic_fees:
                    fees = calculate_trade_fees(
                        entry_premium_val,
                        exit_premium,
                        qty,
                        side,
                        "NSE_FO",
                        self.product,
                    )
                else:
                    fees = self.commission_per_order * 2
        else:
            # Standard: P&L on the actual instrument with adverse exit slippage
            dynamic_slippage = self.slippage_pct * self._get_slippage_factor(exit_time)
            slippage = int(exit_price * dynamic_slippage / 100)
            fill = exit_price - slippage if side == "BUY" else exit_price + slippage
            if side == "BUY":
                raw_pnl = (fill - entry) * qty
            else:
                raw_pnl = (entry - fill) * qty
            entry_display = entry
            fill_display = fill
            if self.use_realistic_fees:
                fees = calculate_trade_fees(entry, fill, qty, side, self.segment, self.product)
            else:
                fees = self.commission_per_order * 2

        net_pnl = raw_pnl - fees

        return Trade(
            entry_time=position["entry_time"],
            exit_time=exit_time,
            side=side,
            entry_price=entry_display,
            exit_price=fill_display,
            quantity=qty,
            pnl=raw_pnl,
            fees=fees,
            net_pnl=net_pnl,
            exit_reason=reason,
            strategy_name=strategy_name,
        )

    def _calculate_unrealized_pnl_for(self, position: dict, current_price: int) -> int:
        entry = position["entry_price"]
        qty = position["quantity"]
        side = position["side"]
        if side == "BUY":
            return (current_price - entry) * qty
        else:
            return (entry - current_price) * qty

    # Keep old name for compat
    def _calculate_unrealized_pnl(self, instrument: str, current_price: int) -> int:
        position = self.positions.get(instrument)
        if position is None:
            return 0
        return self._calculate_unrealized_pnl_for(position, current_price)

    def get_results(self) -> BacktestResults:
        return self._build_results(len(self.equity_curve) - 1)

    def reset(self) -> None:
        self.current_capital = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.equity_curve = [self.initial_capital]
        self.daily_pnl = 0
        self.daily_trades = 0
        self._cooldown_until.clear()
        self._strategy_trade_count.clear()
        logger.info("BacktestEngine reset to initial state")


# ---------------------------------------------------------------------------
# DayByDayBacktester — multi-strategy, day-by-day orchestrator
# ---------------------------------------------------------------------------


class DayByDayBacktester:
    """Orchestrates backtesting across multiple strategies and trading days.

    Splits input data by calendar day, runs all enabled strategies on each day
    using a fresh BacktestEngine, and produces a row-per-day summary table.

    This class uses the **same** ConditionEvaluator and StrategyConfig objects
    as the live engine — ensuring complete backtest–live parity.

    Args:
        capital: Starting capital in paisa.
        slippage_pct: Slippage percentage (default 0.05%).
        strategy_dir: Path to directory of strategy JSON configs.
        max_daily_loss_pct: Stop trading if day loss exceeds this %.
        max_open_positions: Max simultaneous positions per day.
        use_realistic_fees: Use full fee model.
        segment: Market segment ("NSE_EQ" or "NSE_FO").
        product: Product type ("MIS" for intraday).
        warmup_bars_from_prev_day: How many bars from prev day to prepend for warmup.
    """

    def __init__(
        self,
        capital: int = 100_000_000,
        slippage_pct: float = 0.05,
        strategy_dir: str = "config/strategies",
        max_daily_loss_pct: float = 5.0,
        max_open_positions: int = 5,
        use_realistic_fees: bool = True,
        segment: str = "NSE_EQ",
        product: str = "MIS",
        warmup_bars_from_prev_day: int = 30,
        options_mode: bool = False,
        option_premium_pct: float = 0.5,
        option_delta: float = 0.5,
    ) -> None:
        self.initial_capital = capital
        self.slippage_pct = slippage_pct
        self.strategy_dir = strategy_dir
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_open_positions = max_open_positions
        self.use_realistic_fees = use_realistic_fees
        self.segment = segment
        self.product = product
        self.warmup_bars = warmup_bars_from_prev_day
        self.options_mode = options_mode
        self.option_premium_pct = option_premium_pct
        self.option_delta = option_delta

        self.strategies: list[StrategyConfig] = []

    def load_all_strategies(self, selected_names: Optional[list[str]] = None) -> int:
        """Load all enabled strategies from the strategy directory.

        Args:
            selected_names: Optional list of strategy names to include.
                If None, all enabled strategies are loaded.

        Returns:
            Number of strategies loaded.
        """
        all_configs = StrategyBuilder.load_strategies_from_directory(self.strategy_dir)
        self.strategies = []
        for cfg in all_configs:
            if not cfg.enabled:
                continue
            if selected_names and cfg.name not in selected_names:
                continue
            self.strategies.append(cfg)
        logger.info(f"Loaded {len(self.strategies)} strategies for backtesting")
        return len(self.strategies)

    @staticmethod
    def prepare_dataframe(
        df: pd.DataFrame,
        price_unit: str = "rupees",
        resample: str | None = None,
    ) -> pd.DataFrame:
        """Clean and prepare a CSV DataFrame for backtesting.

        - Detects and parses timestamp column.
        - Localises to IST.
        - Lowercases column names.
        - Converts prices from rupees to paisa if ``price_unit == "rupees"``.
        - Optionally resamples to a higher timeframe (e.g., "5min").

        Args:
            df: Raw DataFrame from CSV.
            price_unit: "rupees" or "paisa".
            resample: Optional resample period (e.g., "5min", "15min").
                If None, raw bars are kept as-is.

        Returns:
            Cleaned DataFrame with DatetimeIndex in IST, prices in paisa.
        """
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        # Detect timestamp column
        ts_candidates = ["timestamp", "date", "time", "datetime", "dt"]
        ts_col = None
        for c in df.columns:
            if c in ts_candidates:
                ts_col = c
                break
        if ts_col is None:
            # Try first column
            ts_col = df.columns[0]

        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.set_index(ts_col)
        df.index.name = "timestamp"

        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Kolkata")
        else:
            df.index = df.index.tz_convert("Asia/Kolkata")

        # Convert rupees → paisa
        price_cols = ["open", "high", "low", "close"]
        if price_unit == "rupees":
            for col in price_cols:
                if col in df.columns:
                    df[col] = (df[col] * 100).round().astype(int)

        # Drop non-numeric columns (e.g., 'symbol') that break indicator calculations
        keep_cols = ["open", "high", "low", "close", "volume"]
        extra_cols = [c for c in df.columns if c not in keep_cols]
        if extra_cols:
            logger.info(f"Dropping non-OHLCV columns: {extra_cols}")
            df = df.drop(columns=extra_cols)

        # Resample to higher timeframe if requested
        if resample:
            logger.info(f"Resampling data from {len(df)} bars to {resample} candles")
            df = (
                df.resample(resample)
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                .dropna()
            )
            logger.info(f"After resampling: {len(df)} bars")

        return df

    @staticmethod
    def split_by_trading_day(df: pd.DataFrame) -> OrderedDict:
        """Split a DataFrame into per-day slices within market hours.

        Args:
            df: DataFrame with DatetimeIndex in IST.

        Returns:
            OrderedDict[date, DataFrame] sorted by date.
        """
        result = OrderedDict()
        # Filter to market hours (9:15 – 15:30)
        market_mask = (df.index.time >= MARKET_OPEN) & (df.index.time <= MARKET_CLOSE)
        market_df = df[market_mask]

        for day, day_df in market_df.groupby(market_df.index.date):
            if len(day_df) > 0:
                result[day] = day_df

        return result

    def run(
        self,
        data: pd.DataFrame,
        instrument_key: str,
    ) -> DayByDayResults:
        """Run the day-by-day multi-strategy backtest.

        Args:
            data: Full OHLCV DataFrame (prices in paisa, IST DatetimeIndex).
            instrument_key: Instrument key (e.g., "NSE_INDEX|Nifty 50").

        Returns:
            DayByDayResults with per-day summaries and all trades.
        """
        if not self.strategies:
            logger.warning("No strategies loaded — call load_all_strategies() first")
            return DayByDayResults(
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital,
                day_results=[],
                all_trades=[],
                equity_curve=[self.initial_capital],
            )

        day_slices = self.split_by_trading_day(data)
        if not day_slices:
            logger.warning("No trading days found in data")
            return DayByDayResults(
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital,
                day_results=[],
                all_trades=[],
                equity_curve=[self.initial_capital],
            )

        running_capital = self.initial_capital
        all_trades: list[Trade] = []
        day_results: list[DayResult] = []
        equity_curve: list[int] = [self.initial_capital]

        # Keep previous day's data for warmup
        prev_day_tail: Optional[pd.DataFrame] = None
        day_list = list(day_slices.items())

        for idx, (trading_date, day_df) in enumerate(day_list):
            day_start_capital = running_capital

            # Prepend warmup bars from previous day (so indicators can compute)
            if prev_day_tail is not None and len(prev_day_tail) > 0:
                combined = pd.concat([prev_day_tail, day_df])
                warmup_len = len(prev_day_tail)
            else:
                combined = day_df
                warmup_len = 0

            # Create a fresh engine for this day
            engine = BacktestEngine(
                capital=day_start_capital,
                slippage_pct=self.slippage_pct,
                max_daily_loss_pct=self.max_daily_loss_pct,
                max_open_positions=self.max_open_positions,
                use_realistic_fees=self.use_realistic_fees,
                segment=self.segment,
                product=self.product,
                options_mode=self.options_mode,
                option_premium_pct=self.option_premium_pct,
                option_delta=self.option_delta,
            )

            # Register all strategies
            for strat in self.strategies:
                engine.add_strategy_config(strat, instrument_key)

            # Run the engine on combined data
            result = engine.run({instrument_key: combined})

            # Filter trades to only those entered on the current trading day
            # (warmup bars from prev day should not produce new entries)
            day_trades = []
            for t in result.trades:
                entry_date = (
                    t.entry_time.date()
                    if isinstance(t.entry_time, (pd.Timestamp, datetime))
                    else None
                )
                if entry_date == trading_date:
                    day_trades.append(t)

            # Compute day metrics
            day_net_pnl = sum(t.net_pnl for t in day_trades)
            day_gross_pnl = sum(t.pnl for t in day_trades)
            day_fees = sum(t.fees for t in day_trades)
            day_wins = sum(1 for t in day_trades if t.net_pnl > 0)
            day_losses = sum(1 for t in day_trades if t.net_pnl <= 0)
            total_day_trades = len(day_trades)
            win_rate = (day_wins / total_day_trades * 100) if total_day_trades > 0 else 0.0

            strategies_traded = list(set(t.strategy_name for t in day_trades if t.strategy_name))
            best_trade = max((t.net_pnl for t in day_trades), default=0)
            worst_trade = min((t.net_pnl for t in day_trades), default=0)

            # Intraday drawdown from engine's equity curve
            if len(engine.equity_curve) > 1:
                peak = engine.equity_curve[0]
                max_dd = 0
                for eq in engine.equity_curve:
                    if eq > peak:
                        peak = eq
                    dd = peak - eq
                    if dd > max_dd:
                        max_dd = dd
            else:
                max_dd = 0

            running_capital = day_start_capital + day_net_pnl
            equity_curve.append(running_capital)

            day_results.append(
                DayResult(
                    date=trading_date,
                    total_trades=total_day_trades,
                    wins=day_wins,
                    losses=day_losses,
                    win_rate=round(win_rate, 1),
                    gross_pnl=day_gross_pnl,
                    fees=day_fees,
                    net_pnl=day_net_pnl,
                    max_drawdown=max_dd,
                    strategies_traded=strategies_traded,
                    best_trade=best_trade,
                    worst_trade=worst_trade,
                    capital_start=day_start_capital,
                    capital_end=running_capital,
                )
            )

            all_trades.extend(day_trades)

            # Save tail for next day's warmup
            prev_day_tail = day_df.tail(self.warmup_bars)

            logger.info(
                f"Day {trading_date} | trades={total_day_trades} | "
                f"net_pnl={day_net_pnl} paisa | capital={running_capital} paisa"
            )

        return DayByDayResults(
            initial_capital=self.initial_capital,
            final_capital=running_capital,
            day_results=day_results,
            all_trades=all_trades,
            equity_curve=equity_curve,
        )
