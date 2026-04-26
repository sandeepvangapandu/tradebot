"""BacktestHarness — drives all live trading components synchronously.

Replaces BacktestEngine with a harness that instantiates the exact same
components used in live trading (PaperBroker, RiskManager, PositionManager,
PartialProfitManager, OrderManager, TradeAnalyzer, LearningIntegration) and
drives them bar-by-bar from historical OHLCV data.

All prices in PAISA. Timestamps in IST.
"""

from __future__ import annotations

import math
import queue
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

from config.settings import get_settings
from src.backtest.bar_provider import BacktestBarProvider
from src.backtest.data_loader import BacktestDataLoader
from src.execution.order_manager import OrderManager
from src.execution.paper_broker import PaperBroker
from src.execution.partial_profit import FOUR_TIER_CONFIG, PartialProfitManager
from src.execution.position_manager import PositionConfig, PositionManager
from src.learning.integration import LearningIntegration
import src.persistence.database as _db_module
from src.persistence.database import init_db
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.risk_manager import RiskManager
from src.risk.strategy_quarantine import StrategyQuarantine
from src.strategy.engine import StrategyEngine

IST = ZoneInfo("Asia/Kolkata")
NO_NEW_ENTRY_AFTER = dt_time(15, 0)
INTRADAY_SQUARE_OFF = dt_time(15, 15)
MARKET_OPEN = dt_time(9, 15)

# Optional components (gracefully absent)
try:
    from src.research.trade_analyzer import TradeAnalyzer
    _HAS_TRADE_ANALYZER = True
except ImportError:
    TradeAnalyzer = None  # type: ignore[misc,assignment]
    _HAS_TRADE_ANALYZER = False

try:
    from src.risk.position_sizer import KellyPositionSizer
    _HAS_KELLY = True
except ImportError:
    KellyPositionSizer = None  # type: ignore[misc,assignment]
    _HAS_KELLY = False


@dataclass
class HarnessResults:
    """Results from a BacktestHarness.run() call.

    Attributes:
        equity_curve: Capital in paisa at the end of each processed bar.
        trades: List of closed trade dicts.
        metrics: Aggregate performance metrics (win_rate, profit_factor, etc.).
        bars_processed: Total number of bars evaluated.
    """

    equity_curve: list[int] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    bars_processed: int = 0

    def _compute_metrics(self, initial_capital: int) -> None:
        """Populate self.metrics from self.trades and self.equity_curve."""
        if not self.trades:
            self.metrics = {
                "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "net_pnl_rupees": 0.0, "total_fees_rupees": 0.0,
                "max_drawdown_rupees": 0.0, "max_drawdown_pct": 0.0,
                "return_pct": 0.0, "sharpe_ratio": 0.0,
            }
            return

        pnls = [t.get("net_pnl", 0) for t in self.trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        gross_profit = sum(winners)
        gross_loss = abs(sum(losers)) if losers else 1

        # Max drawdown from equity curve
        peak = initial_capital
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd

        final = self.equity_curve[-1] if self.equity_curve else initial_capital
        ret_pct = (final - initial_capital) / initial_capital * 100 if initial_capital else 0

        self.metrics = {
            "total_trades": len(pnls),
            "win_rate": round(len(winners) / len(pnls) * 100, 2),
            "profit_factor": round(gross_profit / gross_loss, 2),
            "net_pnl_rupees": round(sum(pnls) / 100, 2),
            "total_fees_rupees": round(sum(t.get("fees", 0) for t in self.trades) / 100, 2),
            "max_drawdown_rupees": round(max_dd / 100, 2),
            "max_drawdown_pct": round(max_dd / initial_capital * 100, 2) if initial_capital else 0,
            "return_pct": round(ret_pct, 2),
            "sharpe_ratio": 0.0,
        }

        # Sharpe from trade returns
        if len(pnls) > 1:
            returns = [p / initial_capital for p in pnls]
            mean_r = sum(returns) / len(returns)
            var_r = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
            std_r = math.sqrt(var_r) if var_r > 0 else 1e-10
            self.metrics["sharpe_ratio"] = round((mean_r / std_r) * math.sqrt(252), 2)


class BacktestHarness:
    """Drives all live components synchronously from historical OHLCV data.

    Instantiates the same components as TradingBot.startup() (PaperBroker,
    RiskManager, PositionManager, PartialProfitManager, OrderManager,
    TradeAnalyzer, LearningIntegration) and steps through bars in order,
    calling the same methods the live event loop would call.

    Args:
        data_file: Path to 1-minute OHLCV CSV file.
        instrument_key: Instrument identifier (e.g. "NSE_INDEX|Nifty Bank").
        strategy_dir: Directory containing strategy JSON configs.
        capital: Initial capital in paisa. Defaults to settings.capital.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        database_url: SQLAlchemy database URL. Defaults to settings.database_url.
        prices_in_rupees: If True, multiply CSV prices by 100 to convert to paisa.
        options_proxy: If True, transform index OHLCV into synthetic ATM-option
            premium prices so the full live pipeline trades at realistic
            premium values (matching engine.py's options proxy semantics).
        option_premium_pct: ATM option premium as % of underlying (default 0.5%
            → Rs 250 premium on a Rs 50,000 BankNifty).
        option_delta: Delta of the ATM option (default 0.5). Used to scale
            index moves into premium moves: Δpremium = delta × Δindex.
    """

    def __init__(
        self,
        data_file: str,
        instrument_key: str,
        strategy_dir: str = "config/strategies",
        capital: Optional[int] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        database_url: Optional[str] = None,
        prices_in_rupees: bool = False,
        options_proxy: bool = False,
        straddle_proxy: bool = False,
        option_premium_pct: float = 0.5,
        option_delta: float = 0.5,
    ) -> None:
        self._settings = get_settings()
        self._capital = capital or self._settings.capital
        self._instrument_key = instrument_key
        self._strategy_dir = strategy_dir
        self._start_date = start_date
        self._end_date = end_date
        self._options_proxy = options_proxy
        self._straddle_proxy = straddle_proxy
        self._option_premium_pct = option_premium_pct
        self._option_delta = option_delta

        # Load historical data
        loader = BacktestDataLoader()
        df = loader.load_csv(data_file, start_date=start_date, end_date=end_date)
        if prices_in_rupees:
            for col in ["open", "high", "low", "close"]:
                df[col] = (df[col] * 100).astype(int)

        if straddle_proxy:
            df = self._transform_to_straddle_proxy(df)
        elif options_proxy:
            df = self._transform_to_options_proxy(df)

        self._master_df = df
        self._data: dict[str, pd.DataFrame] = {instrument_key: df}

        # Bar provider (shared BarBuilder + HistoricalDataFetcher interface)
        self._bar_provider = BacktestBarProvider(self._data)

        # Initialize database (SessionLocal is set as a module-level var by init_db)
        db_url = database_url or self._settings.database_url
        init_db(db_url)
        self._db_session = _db_module.SessionLocal()

        # Wire up live components
        self._setup_components()
        self._setup_strategy_engine()

    # ------------------------------------------------------------------
    # Options proxy transformation
    # ------------------------------------------------------------------

    def _transform_to_options_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform index OHLCV into synthetic ATM option premium OHLCV.

        For each trading day, uses the day's first-bar open as the reference
        index price. The synthetic premium for any subsequent bar is:

            premium(i) = premium_pct × ref + delta × (index(i) - ref)

        This gives each session a fresh ~Rs 250 premium baseline on BankNifty
        (assuming 0.5 % premium_pct and 50,000 index) that expands/contracts
        by delta × index_change. A 1 % index move translates to a
        (delta / premium_pct) × 1 % ≈ 100 % premium move, matching how real
        ATM options behave intraday.

        All prices remain integers in PAISA. Premiums are floored at 100
        paisa (Rs 1) to prevent zero/negative premium after extreme drawdowns.

        Args:
            df: Raw index OHLCV dataframe (paisa prices, IST index).

        Returns:
            New dataframe with ``open``/``high``/``low``/``close`` replaced by
            synthetic option premium values. ``volume`` is preserved.
        """
        df = df.copy()
        # Reference = each day's opening index price (the first bar of that date)
        day_key = df.index.date
        day_open = pd.Series(df["open"].values, index=day_key).groupby(level=0).first()
        # Broadcast back to the full bar index
        ref_series = pd.Series(
            [day_open.loc[d] for d in day_key], index=df.index, dtype="int64"
        )

        scale = self._option_delta  # Δpremium = delta × Δindex
        base_factor = self._option_premium_pct / 100.0  # e.g. 0.005

        # Synthetic premium formula (vectorised)
        def _to_premium(series: pd.Series) -> pd.Series:
            entry_premium = ref_series.astype("float64") * base_factor
            delta_move = (series.astype("float64") - ref_series.astype("float64")) * scale
            out = entry_premium + delta_move
            return out.clip(lower=100).round().astype("int64")

        # High/low stay ordered because the transform is monotonic in index level.
        new_open = _to_premium(df["open"])
        new_high = _to_premium(df["high"])
        new_low = _to_premium(df["low"])
        new_close = _to_premium(df["close"])

        df["open"] = new_open
        df["high"] = new_high
        df["low"] = new_low
        df["close"] = new_close

        logger.warning(
            "WARNING: Experimental synthetic options proxy enabled | premium_pct={}% | delta={} | "
            "first-day premium baseline ≈ {} paisa. This transforms index OHLCV into synthetic ATM premium scaling.",
            self._option_premium_pct,
            self._option_delta,
            int(day_open.iloc[0] * base_factor) if len(day_open) else 0,
        )
        return df

    def _transform_to_straddle_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform index OHLCV into synthetic ATM Straddle premium OHLCV.

        For a 0DTE straddle, premium consists of Time Value + Intrinsic Value.
        At open (9:15 AM), Intrinsic = 0. Total Premium = 2 * (premium_pct * ref).
        At close (15:30 PM), Time Value = 0. Total Premium = |Index - ref|.
        
        We linearly interpolate Time Value over the day from Open to Close,
        and add pure absolute distance for Intrinsic Value.
        
        Args:
            df: Raw index OHLCV dataframe.

        Returns:
            New dataframe representing combined Straddle premium.
        """
        df = df.copy()
        
        # Reference = opening index price
        day_key = df.index.date
        day_open = pd.Series(df["open"].values, index=day_key).groupby(level=0).first()
        
        # Broadcast reference back to full index
        ref_series = pd.Series(
            [day_open.loc[d] for d in day_key], index=df.index, dtype="float64"
        )
        
        # Total initial Straddle premium = 2 * (1 side premium)
        # Using twice the base factor
        base_factor = (self._option_premium_pct * 2.0) / 100.0
        initial_premium = ref_series * base_factor

        # Time fraction remaining (1.0 at 9:15, 0.0 at 15:30)
        # 9:15 AM = 9.25 hours
        # 15:30 PM = 15.5 hours
        # Total duration = 6.25 hours = 375 minutes
        times = df.index.time
        minutes_from_midnight = [t.hour * 60 + t.minute for t in times]
        
        # Normalize to 0 -> 1 for decay
        decay_factors = []
        for m in minutes_from_midnight:
            elapsed = m - (9 * 60 + 15)  # Minutes since 9:15
            remaining = 375 - elapsed
            factor = max(0.0, remaining / 375.0)
            decay_factors.append(factor)
            
        time_fraction = pd.Series(decay_factors, index=df.index, dtype="float64")
        
        def _to_straddle_premium(series: pd.Series) -> pd.Series:
            # Time Value = Initial Premium * Time Fraction
            time_val = initial_premium * time_fraction
            
            # Intrinsic Value = |Current Index - Restrike Index|
            intrinsic_val = (series.astype("float64") - ref_series).abs()
            
            out = time_val + intrinsic_val
            return out.clip(lower=100).round().astype("int64")

        # Because this is an absolute value function `|price - ref|`, it's not strictly monotonic.
        # But for bar OHLC mapping, treating min/max directly on high/low is generally acceptable
        # since time decay is smooth and intrinsic is V-shaped.
        new_open = _to_straddle_premium(df["open"])
        new_high = _to_straddle_premium(df["high"])
        new_low = _to_straddle_premium(df["low"])
        new_close = _to_straddle_premium(df["close"])
        
        # Swap high/low if inverted due to |x - ref|
        true_high = pd.concat([new_open, new_high, new_low, new_close], axis=1).max(axis=1)
        true_low = pd.concat([new_open, new_high, new_low, new_close], axis=1).min(axis=1)

        df["open"] = new_open
        df["high"] = true_high
        df["low"] = true_low
        df["close"] = new_close

        logger.warning(
            "WARNING: Experimental synthetic STRADDLE proxy enabled | premium_pct={} | "
            "first-day combined premium ≈ {} paisa.",
            self._option_premium_pct * 2.0,
            int(day_open.iloc[0] * base_factor) if len(day_open) else 0,
        )
        return df

    # ------------------------------------------------------------------
    # Component setup (mirrors TradingBot.startup, no auth/WebSocket)
    # ------------------------------------------------------------------

    def _setup_components(self) -> None:
        """Instantiate all live execution and risk components."""
        settings = self._settings

        self._paper_broker = PaperBroker(
            initial_capital=self._capital,
            slippage_pct=settings.slippage_pct,
        )
        self._partial_profit = PartialProfitManager()
        self._risk_manager = RiskManager(
            capital=self._capital,
            max_daily_loss=settings.max_daily_loss,
            max_open_positions=settings.max_open_positions,
            max_position_size_pct=settings.max_position_size_pct,
            max_capital_deployment_pct=settings.max_capital_deployment_pct,
            max_trades_per_day=1 if (self._options_proxy or self._straddle_proxy) else 3,  # 1 straddle per day
        )
        self._circuit_breaker = CircuitBreaker(
            max_consecutive_losses=settings.consecutive_loss_pause,
            pause_minutes=settings.pause_minutes,
        )
        self._strategy_quarantine = StrategyQuarantine()

        # Learning system — backtest_mode=True skips lesson persistence
        self._learning = LearningIntegration(
            db_session=self._db_session,
            enabled=settings.learning_enabled,
            backtest_mode=True,
        )

        # Position config — widen thresholds when running in options proxy mode
        # so trailing-stop / SL / target act on the premium scale (1% index ≈ 100%
        # premium), not the index scale. Otherwise defaults (2% activation / 1%
        # trail / 2% SL) whipsaw on every intraday bounce.
        if self._options_proxy or self._straddle_proxy:
            position_config = PositionConfig(
                stop_loss_pct=0.60,           # 60% premium expansion before stop (option selling)
                target_pct=0.60,              # Collect 60% of original premium
                trailing_sl_activation_pct=0.30,  # arm trail after +30% premium decay
                trailing_sl_pct=0.15,         # trail 15% from peak
                enable_trailing_sl=True,
                use_atr_based_sl=False,       # let strategy exit_rules drive SL
                enable_partial_profit_booking=True,
                enable_volatility_target_adjustment=False,
                enable_options_time_decay_exit=False,
            )
            logger.info(
                "Options proxy PositionConfig active | SL=60% | target=60% | "
                "trail_activate=30% | trail=15% | ATR-SL disabled"
            )
        else:
            position_config = PositionConfig()

        self._position_manager = PositionManager(
            broker=self._paper_broker,
            risk_manager=self._risk_manager,
            config=position_config,
            partial_profit_manager=self._partial_profit,
            on_position_close=self._learning.on_position_closed,
        )

        # Kelly position sizer (optional)
        position_sizer = None
        if _HAS_KELLY:
            try:
                position_sizer = KellyPositionSizer()
            except Exception as exc:
                logger.warning(f"KellyPositionSizer unavailable: {exc}")

        # Research module (optional — gracefully degrades without live data)
        trade_analyzer = None
        if _HAS_TRADE_ANALYZER and settings.research_enabled:
            try:
                trade_analyzer = TradeAnalyzer(
                    settings=settings,
                    bar_builder=self._bar_provider,
                    historical_fetcher=self._bar_provider,
                    instrument_manager=None,  # gracefully degrades
                )
                logger.info("TradeAnalyzer active in backtest (OHLCV analyzers only)")
            except Exception as exc:
                logger.warning(f"TradeAnalyzer unavailable: {exc}")

        # AI Agent Pipeline (optional — requires API key for chosen provider)
        agent_pipeline = None
        llm_provider = getattr(settings, "llm_provider", "groq")
        if llm_provider == "openrouter":
            _api_key = getattr(settings, "openrouter_api_key", "")
            _model = getattr(settings, "openrouter_model", "nvidia/nemotron-3-super-120b-a12b:free")
            _fallback = getattr(settings, "openrouter_fallback_model", "google/gemma-4-31b-it:free")
            _rpm = getattr(settings, "openrouter_rate_limit_rpm", 20)
        else:
            _api_key = settings.groq_api_key
            _model = settings.groq_model
            _fallback = settings.groq_fallback_model
            _rpm = settings.groq_rate_limit_rpm

        if settings.agent_pipeline_enabled and _api_key:
            try:
                from src.agents.llm_client import LLMClient
                from src.agents.pipeline import AgentPipeline
                from src.memory.memory_db import MemoryDB

                if llm_provider == "openrouter":
                    _temperature = getattr(settings, "openrouter_temperature", 0.1)
                    _max_tokens = getattr(settings, "openrouter_max_tokens", 1024)
                else:
                    _temperature = settings.groq_temperature
                    _max_tokens = settings.groq_max_tokens

                llm_client = LLMClient(
                    api_key=_api_key,
                    model=_model,
                    fallback_model=_fallback,
                    temperature=_temperature,
                    max_tokens=_max_tokens,
                    rate_limit_rpm=_rpm,
                    provider=llm_provider,
                )
                memory_db = MemoryDB(
                    decay_rate=settings.memory_decay_rate,
                    decay_start_days=settings.memory_decay_start_days,
                )
                agent_pipeline = AgentPipeline(
                    llm_client=llm_client,
                    memory_db=memory_db,
                    regime_confidence_threshold=settings.regime_confidence_threshold,
                )
                logger.info(
                    "AgentPipeline active in backtest | provider={} model={}",
                    llm_provider, _model,
                )
            except Exception as exc:
                logger.warning(f"AgentPipeline unavailable in backtest: {exc}")
        else:
            logger.info("AgentPipeline disabled (no API key or agent_pipeline_enabled=False)")

        self._order_manager = OrderManager(
            signal_queue=queue.Queue(),  # unused — we call process_signal() directly
            broker=self._paper_broker,
            trading_mode="paper",
            db_url=self._settings.database_url,
            trade_analyzer=trade_analyzer,
            risk_manager=self._risk_manager,
            position_manager=self._position_manager,
            position_sizer=position_sizer,
            strategy_quarantine=self._strategy_quarantine,
            research_enabled=(trade_analyzer is not None),
            agent_pipeline=agent_pipeline,
        )

    def _setup_strategy_engine(self) -> None:
        """Load strategies from config directory."""
        self._strategy_engine = StrategyEngine(
            strategies_dir=self._strategy_dir,
            bars_provider=lambda: self._data,
        )
        n = self._strategy_engine.load_strategies()
        logger.info(f"Loaded {n} strategies for backtest")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        progress_callback: Optional[callable] = None,
    ) -> HarnessResults:
        """Run the backtest bar-by-bar.

        Args:
            progress_callback: Optional callable(current_bar, total_bars) for
                progress updates.

        Returns:
            HarnessResults with equity curve, closed trades, and metrics.
        """
        results = HarnessResults()
        prev_date: Optional[date] = None

        logger.info(
            f"Backtest starting | instrument={self._instrument_key} | "
            f"bars={len(self._master_df)} | capital={self._capital / 100:.0f} INR"
        )

        total_bars = len(self._master_df)
        for i, (bar_time, bar) in enumerate(self._master_df.iterrows()):
            bar_time_ist: datetime = (
                bar_time.astimezone(IST) if bar_time.tzinfo else bar_time.replace(tzinfo=IST)
            )
            current_price = int(bar["close"])

            # --- Advance bar provider (no lookahead) ---
            self._bar_provider.advance(self._instrument_key, i)

            # --- Update broker LTP so MARKET orders get correct fill price ---
            self._paper_broker.update_ltp(self._instrument_key, current_price)

            # --- Set backtest time so PositionManager uses bar time, not wall-clock ---
            self._position_manager.set_backtest_time(bar_time_ist)

            # --- Phase 1: New trading day reset ---
            today = bar_time_ist.date()
            if today != prev_date:
                if prev_date is not None:
                    logger.info(f"[BACKTEST] New day: {today} (prev: {prev_date})")
                self._risk_manager.reset_daily()
                prev_date = today

            # --- Phase 2: Skip pre-market bars ---
            if bar_time_ist.time() < MARKET_OPEN:
                results.equity_curve.append(self._paper_broker.get_funds())
                continue

            # --- Phase 3: Position exits (SL, target, trailing SL, partial profit) ---
            try:
                closed = self._position_manager.on_tick(
                    self._instrument_key, current_price
                )
                for pos in (closed or []):
                    results.trades.append({
                        "instrument_key": pos.instrument_key,
                        "strategy_id": pos.strategy_id,
                        "side": pos.side.value,
                        "entry_price": pos.entry_price,
                        "exit_price": pos.blended_exit_price or pos.entry_price,
                        "quantity": pos.quantity,
                        "net_pnl": pos.total_realized_pnl,
                        "fees": 0,  # PaperBroker deducts fees internally
                        "entry_time": str(pos.entry_time),
                        "exit_time": str(pos.exit_time) if pos.exit_time else None,
                        "exit_reason": getattr(pos, "exit_reason", "unknown"),
                    })
            except Exception as exc:
                logger.error(f"[BACKTEST] PositionManager.on_tick error at bar {i}: {exc}")

            # --- Phase 4: Session gate ---
            bar_t = bar_time_ist.time()
            if bar_t >= NO_NEW_ENTRY_AFTER:
                results.equity_curve.append(self._paper_broker.get_funds())
                results.bars_processed += 1
                continue

            if self._circuit_breaker.is_halted():
                results.equity_curve.append(self._paper_broker.get_funds())
                results.bars_processed += 1
                continue

            # --- Phase 5: Signal generation (same ConditionEvaluator as live) ---
            # Pass the bar_provider slice (no lookahead) not self._data (full df)
            current_bars = {
                self._instrument_key: self._bar_provider.get_bars(
                    self._instrument_key, timeframe=1
                )
            }
            try:
                signals = self._strategy_engine.evaluate_bar_sync(
                    bars=current_bars,
                    bar_time=bar_time_ist,
                )
                for signal in signals:
                    signal.timestamp = bar_time_ist
                    self._order_manager.process_signal(signal)
            except Exception as exc:
                logger.error(f"[BACKTEST] Signal generation error at bar {i}: {exc}")

            # --- Phase 6: Equity snapshot ---
            results.equity_curve.append(self._paper_broker.get_funds())

            # --- Progress callback ---
            if progress_callback and i % 100 == 0:
                progress_callback(i, total_bars)

            results.bars_processed += 1

        # Final progress update
        if progress_callback:
            progress_callback(total_bars, total_bars)

        # Force close any remaining positions at last bar
        self._close_all_positions(results)

        # Compute aggregate metrics
        results._compute_metrics(self._capital)

        logger.info(
            f"Backtest complete | bars={results.bars_processed} | "
            f"trades={len(results.trades)} | "
            f"P&L={results.metrics.get('net_pnl_rupees', 0):.2f} INR"
        )
        return results

    def _close_all_positions(self, results: HarnessResults) -> None:
        """Force-close all open positions at end of backtest data."""
        try:
            closed = self._position_manager.close_all_positions("Backtest data ended")
            for pos in (closed or []):
                results.trades.append({
                    "instrument_key": pos.instrument_key,
                    "strategy_id": pos.strategy_id,
                    "side": pos.side.value,
                    "entry_price": pos.entry_price,
                    "exit_price": pos.blended_exit_price or pos.entry_price,
                    "quantity": pos.quantity,
                    "net_pnl": pos.total_realized_pnl,
                    "fees": 0,
                    "entry_time": str(pos.entry_time),
                    "exit_time": str(pos.exit_time) if pos.exit_time else None,
                    "exit_reason": "backtest_end",
                })
        except Exception as exc:
            logger.warning(f"[BACKTEST] End-of-data close_all failed: {exc}")
