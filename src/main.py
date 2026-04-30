"""Main orchestrator for the Upstox trading bot.

Startup sequence:
1. Load settings from .env
2. Initialize logger
3. Authenticate
4. Download/load instruments
5. Initialize database
6. Start MarketDataFeed WebSocket
7. Start PortfolioFeed WebSocket
8. Start BarBuilder thread
9. Load strategies and start evaluation threads
10. Initialize RiskManager
11. Start OrderManager
12. Configure scheduler
13. Register signal handlers

Shutdown sequence:
1. Stop strategy threads
2. Exit open positions
3. Save strategy state
4. Stop WebSocket connections
5. Stop scheduler
6. Log daily P&L
7. Close database connections
"""

from __future__ import annotations

import queue
import signal
import sys
import threading
import time
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from config.settings import Settings, get_settings
from src.auth.token_manager import TokenManager
from src.data.bar_builder import BarBuilder
from src.data.historical import HistoricalDataFetcher
from src.data.instruments import InstrumentManager
from src.data.portfolio_feed import PortfolioFeed
from src.data.websocket_feed import MarketDataFeed
from src.execution.order_manager import OrderManager
from src.execution.order_tracker import OrderTracker
from src.execution.paper_broker import PaperBroker
from src.execution.partial_profit import PartialProfitManager
from src.execution.position_manager import PositionManager
from src.persistence.database import get_session, init_db
from src.persistence.trade_log import TradeLogger as TradeLog
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.risk_manager import RiskManager
from src.risk.strategy_quarantine import StrategyQuarantine
from src.strategy.engine import StrategyEngine
from src.utils.exceptions import TradingBotError
from src.utils.health_monitor import HealthMonitor
from src.utils.holidays import is_trading_day
from src.utils.logger import setup_logger
from src.utils.scheduler import get_scheduler

from src.agents.llm_client import LLMClient
from src.agents.pipeline import AgentPipeline
from src.memory.memory_db import MemoryDB
from src.memory.outcome_analyzer import OutcomeAnalyzer
from src.memory.mistake_classifier import MistakeClassifier

IST = ZoneInfo("Asia/Kolkata")


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self) -> None:
        self.settings: Settings = get_settings()
        self.token_manager: TokenManager | None = None
        self.instrument_manager: InstrumentManager | None = None
        self.risk_manager: RiskManager | None = None
        self.circuit_breaker: CircuitBreaker | None = None
        self.strategy_quarantine: StrategyQuarantine | None = None
        self.paper_broker: PaperBroker | None = None
        self.order_manager: OrderManager | None = None
        self.position_manager: PositionManager | None = None
        self.partial_profit_manager: PartialProfitManager | None = None
        self.order_tracker: OrderTracker | None = None
        self.market_feed: MarketDataFeed | None = None
        self.portfolio_feed: PortfolioFeed | None = None
        self.bar_builder: BarBuilder | None = None
        self.strategy_engine: StrategyEngine | None = None
        self.trade_log: TradeLog | None = None
        self.scheduler: BackgroundScheduler | None = None
        self.health_monitor: HealthMonitor | None = None

        # AI agent pipeline
        self.llm_client: LLMClient | None = None
        self.agent_pipeline: AgentPipeline | None = None
        self.memory_db: MemoryDB | None = None
        self.outcome_analyzer: OutcomeAnalyzer | None = None
        self.mistake_classifier: MistakeClassifier | None = None

        # Signal queue for strategy -> order communication
        self.signal_queue: queue.Queue[Any] = queue.Queue()
        self.tick_queue: queue.Queue[Any] = queue.Queue()
        self.bar_close_event = threading.Event()

        self._running = False
        self._shutdown_event = threading.Event()

    def startup(self) -> None:
        """Execute startup sequence."""
        logger.info("=" * 60)
        logger.info("Starting Upstox Trading Bot")
        logger.info("=" * 60)

        # 1. Initialize logger
        setup_logger(
            log_level=self.settings.log_level,
            log_file=self.settings.log_file,
        )
        logger.info("Logger initialized")

        # 2. Authenticate
        logger.info("Authenticating with Upstox...")
        self.token_manager = TokenManager()
        access_token = self.token_manager.get_valid_token()
        logger.info("Authentication successful")

        # 3. Check if today is a trading day
        today = datetime.now(IST).date()
        ignore_holidays = "--ignore-holidays" in sys.argv
        if not is_trading_day(today):
            if ignore_holidays:
                logger.warning(
                    "Today {} is not a trading day, but --ignore-holidays is set. "
                    "Continuing for a dry run; no live ticks will flow.",
                    today,
                )
            else:
                logger.error("Today is not a trading day (holiday or weekend). Bot shutting down.")
                raise TradingBotError(f"Today {today} is not a trading day")

        # 3. Initialize database
        logger.info("Initializing database...")
        init_db(self.settings.database_url)
        self.trade_log = TradeLog(session_factory=get_session)
        logger.info("Database initialized")

        # 4. Download/load instruments
        logger.info("Loading instrument master...")
        self.instrument_manager = InstrumentManager()
        self.instrument_manager.download_instruments(["NSE", "BSE"])
        self.instrument_manager.load_instruments()
        logger.info("Instruments loaded")

        # 5. Initialize risk manager
        self.risk_manager = RiskManager(
            capital=self.settings.capital,
            max_daily_loss=self.settings.max_daily_loss,
            max_open_positions=self.settings.max_open_positions,
            max_position_size_pct=self.settings.max_position_size_pct,
            max_capital_deployment_pct=self.settings.max_capital_deployment_pct,
        )
        self.circuit_breaker = CircuitBreaker(
            max_consecutive_losses=self.settings.consecutive_loss_pause,
            pause_minutes=self.settings.pause_minutes,
        )
        self.strategy_quarantine = StrategyQuarantine()
        logger.info("Risk manager initialized")

        # Check if circuit breaker is already halted (from previous run)
        if self.circuit_breaker.is_halted():
            logger.error("Circuit breaker is currently halted. Cannot start trading.")
            raise TradingBotError("Circuit breaker is halted. Please resume manually.")

        # 6. Initialize broker, partial profit manager, and order/position managers
        self.paper_broker = PaperBroker(
            initial_capital=self.settings.capital,
            slippage_pct=self.settings.slippage_pct,
        )
        self.partial_profit_manager = PartialProfitManager()
        self.position_manager = PositionManager(
            broker=self.paper_broker,
            risk_manager=self.risk_manager,
            partial_profit_manager=self.partial_profit_manager,
        )
        self.order_tracker = OrderTracker(
            broker=self.paper_broker,
        )
        self.order_manager = OrderManager(
            signal_queue=self.signal_queue,
            broker=self.paper_broker,
            trading_mode=self.settings.trading_mode,
            instrument_manager=self.instrument_manager,
            risk_manager=self.risk_manager,
            position_manager=self.position_manager,
            strategy_quarantine=self.strategy_quarantine,
        )
        logger.info("Order and position managers initialized")

        # 7. Start BarBuilder
        self.bar_builder = BarBuilder(
            tick_queue=self.tick_queue,
            bar_close_event=self.bar_close_event,
        )
        self.bar_builder.start()
        logger.info("BarBuilder started")

        # 7b. Backfill historical bars so indicators have warmup before
        # live ticks start. Without this, freshly-started bots cannot
        # generate signals for ~50+ minutes (RSI/EMA/MACD warmup window).
        try:
            underlyings_for_warmup = [
                "NSE_INDEX|Nifty Bank",
                "NSE_INDEX|Nifty 50",
            ]
            fetcher = HistoricalDataFetcher(access_token)
            import pandas as _pd

            for ik in underlyings_for_warmup:
                # Upstox History v3 only supports 1m / 30m / day intervals,
                # and the historical endpoint excludes the current session.
                # Fetch 1m for prior 5 days + today's intraday, then resample
                # to 5m and 15m locally before seeding the bar builder.
                hist_df = fetcher.fetch_candles(ik, interval="1minute", days=5)
                today_df = fetcher.fetch_intraday_candles(
                    ik, interval="1minute"
                )

                parts = [d for d in (hist_df, today_df) if d is not None and not d.empty]
                if not parts:
                    logger.warning("No warmup bars available for {}", ik)
                    continue

                df_1m = _pd.concat(parts).sort_index()
                df_1m = df_1m[~df_1m.index.duplicated(keep="last")]
                today_n = 0 if today_df is None or today_df.empty else len(today_df)

                # Seed 1m
                n1 = self.bar_builder.seed_bars(ik, 1, df_1m)
                logger.info(
                    "Seeded {} 1m bars for {} (warmup, {} from today)",
                    n1, ik, today_n,
                )

                # Resample to 5m and 15m
                agg = {"open": "first", "high": "max", "low": "min",
                       "close": "last", "volume": "sum"}
                for tf in (5, 15):
                    rs = df_1m.resample(f"{tf}min").agg(agg).dropna()
                    if rs.empty:
                        continue
                    n = self.bar_builder.seed_bars(ik, tf, rs)
                    logger.info(
                        "Seeded {} {}m bars for {} (resampled from 1m)",
                        n, tf, ik,
                    )
        except Exception as exc:
            logger.warning(
                "Historical warmup failed: {} — strategies will warm up "
                "from live ticks instead.", exc
            )

        # 8. Start WebSocket feeds
        self.market_feed = MarketDataFeed(access_token)
        self.market_feed.tick_queue = self.tick_queue
        self.market_feed.register_callback(self._on_market_tick)

        self.portfolio_feed = PortfolioFeed(access_token)
        self.portfolio_feed.register_order_callback(self._on_order_update)
        self.portfolio_feed.register_position_callback(self._on_position_update)

        # Subscribe to underlyings
        underlyings = ["NSE_INDEX|Nifty Bank", "NSE_INDEX|Nifty 50"]
        self.market_feed.start(underlyings, mode="full")
        self.portfolio_feed.start()
        logger.info("WebSocket feeds started")

        # 9. Load strategies
        strategy_dir = Path("config/strategies")
        if strategy_dir.exists():
            def _bars_provider() -> dict[str, Any]:
                """Return latest 1-min bars for every instrument the bar builder is tracking."""
                if not self.bar_builder:
                    return {}
                out: dict[str, Any] = {}
                for ik in self.bar_builder.get_all_instrument_keys():
                    df = self.bar_builder.get_bars(ik, timeframe=1)
                    if df is not None and not df.empty:
                        out[ik] = df
                return out

            self.strategy_engine = StrategyEngine(
                strategies_dir=strategy_dir,
                bars_provider=_bars_provider,
            )
            self.strategy_engine.load_strategies()
            self.strategy_engine.start()
            logger.info("Strategy engine started")

        # 10. Start order manager
        self.order_manager.start()
        logger.info("Order manager started")

        # 11. Configure scheduler
        self.scheduler = get_scheduler()
        self._setup_scheduler()
        logger.info("Scheduler configured")

        # 12. Initialize health monitor
        self.health_monitor = HealthMonitor()
        self.health_monitor.register_component("market_feed", heartbeat_timeout=30)
        self.health_monitor.register_component("database", heartbeat_timeout=120)
        self.health_monitor.register_component("scheduler", heartbeat_timeout=120)
        self.health_monitor.start_monitoring(interval=30)

        # Wire market_feed heartbeat: every tick bumps the watchdog so the
        # health monitor can detect a truly stalled WebSocket vs. a quiet
        # market. Without this, market_feed is always reported FAILED.
        if self.market_feed is not None:
            _hm = self.health_monitor
            self.market_feed.register_callback(
                lambda _msg: _hm.heartbeat("market_feed")
            )

        # database + scheduler heartbeats: lightweight 30s interval job that
        # pings DB and bumps both watchdogs. Without this, both components
        # are perpetually marked FAILED even when healthy.
        from sqlalchemy import text as _sql_text

        def _heartbeat_pulse() -> None:
            logger.debug("heartbeat pulse firing")
            try:
                with get_session() as session:
                    session.execute(_sql_text("SELECT 1"))
                self.health_monitor.heartbeat("database")
            except Exception as exc:
                logger.warning("DB heartbeat ping failed: {}", exc)
            self.health_monitor.heartbeat("scheduler")

        if self.scheduler is not None:
            try:
                self.scheduler.add_interval_job(
                    _heartbeat_pulse, seconds=30, job_id="health_heartbeat"
                )
                logger.info("Health heartbeat pulse scheduled (30s interval)")
            except Exception as exc:
                logger.error("Failed to schedule heartbeat pulse: {}", exc)

        logger.info("Health monitor initialized and started")

        # 13. Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logger.info("Signal handlers registered")

        # 14. Initialize AI agent pipeline
        if self.settings.agent_pipeline_enabled:
            llm_provider = self.settings.llm_provider
            if llm_provider == "openrouter":
                _api_key = self.settings.openrouter_api_key
                _model = self.settings.openrouter_model
                _fallback = self.settings.openrouter_fallback_model
                _rpm = self.settings.openrouter_rate_limit_rpm
            else:
                _api_key = self.settings.groq_api_key
                _model = self.settings.groq_model
                _fallback = self.settings.groq_fallback_model
                _rpm = self.settings.groq_rate_limit_rpm

            if llm_provider == "openrouter":
                _temperature = self.settings.openrouter_temperature
                _max_tokens = self.settings.openrouter_max_tokens
            else:
                _temperature = self.settings.groq_temperature
                _max_tokens = self.settings.groq_max_tokens

            self.llm_client = LLMClient(
                api_key=_api_key,
                model=_model,
                fallback_model=_fallback,
                temperature=_temperature,
                max_tokens=_max_tokens,
                rate_limit_rpm=_rpm,
                provider=llm_provider,
            )
            self.memory_db = MemoryDB(
                decay_rate=self.settings.memory_decay_rate,
                decay_start_days=self.settings.memory_decay_start_days,
            )
            self.agent_pipeline = AgentPipeline(
                llm_client=self.llm_client,
                memory_db=self.memory_db,
                regime_confidence_threshold=self.settings.regime_confidence_threshold,
            )
            self.outcome_analyzer = OutcomeAnalyzer()
            self.mistake_classifier = MistakeClassifier()

            # Wire pipeline into OrderManager for live signal validation
            self.order_manager._agent_pipeline = self.agent_pipeline

            logger.info(
                "AI Agent Pipeline initialized & wired into OrderManager | LLM configured={}",
                self.llm_client.is_configured,
            )

        self._running = True
        logger.info("=" * 60)
        logger.info("Trading Bot Startup Complete")
        logger.info("=" * 60)

    def _setup_scheduler(self) -> None:
        """Configure scheduled jobs."""
        if not self.scheduler:
            return

        # Daily token refresh at 08:45 IST
        self.scheduler.add_daily_job(
            func=self._scheduled_token_refresh,
            hour=8,
            minute=45,
            job_id="token_refresh",
        )
        # Daily instrument download at 06:15 IST
        self.scheduler.add_daily_job(
            func=self._scheduled_instrument_download,
            hour=6,
            minute=15,
            job_id="instrument_download",
        )
        # Intraday square-off at 15:15 IST
        self.scheduler.add_daily_job(
            func=self._scheduled_square_off,
            hour=15,
            minute=15,
            job_id="square_off",
        )
        # Daily P&L summary at 15:30 IST
        self.scheduler.add_daily_job(
            func=self._scheduled_daily_summary,
            hour=15,
            minute=30,
            job_id="daily_summary",
        )

        self.scheduler.start()

    def _scheduled_token_refresh(self) -> None:
        """Scheduled job: refresh access token.

        Token refresh fires before market open (08:45 IST). If it fails,
        the bot would continue with a stale token and silently reject every
        live order. Halt the order pipeline so the user is forced to act.
        """
        logger.info("[SCHEDULER] Refreshing access token")
        if not self.token_manager:
            return
        try:
            self.token_manager.refresh_token()
            if not self.token_manager.is_token_valid():
                raise RuntimeError("token invalid immediately after refresh")
        except Exception as exc:
            logger.critical(
                "Scheduled token refresh FAILED: {}. Halting order manager. "
                "Run `python3 -m src.auth.manual_login` and restart the bot.",
                exc,
            )
            if self.order_manager:
                self.order_manager.stop()

    def _scheduled_instrument_download(self) -> None:
        """Scheduled job: download instrument files."""
        logger.info("[SCHEDULER] Downloading instrument files")
        if self.instrument_manager:
            try:
                self.instrument_manager.download_instruments(["NSE", "BSE"])
                self.instrument_manager.load_instruments()
            except Exception as exc:
                logger.error("Scheduled instrument download failed: {}", exc)

    def _scheduled_square_off(self) -> None:
        """Scheduled job: square off all intraday positions."""
        logger.info("[SCHEDULER] Squaring off all intraday positions")
        if self.position_manager:
            try:
                # Check if today is a trading day
                if not is_trading_day(datetime.now(IST).date()):
                    logger.info("Today is not a trading day, skipping square-off")
                    return
                self.position_manager.close_all_positions(reason="Intraday square-off at 15:15")
            except Exception as exc:
                logger.error("Scheduled square-off failed: {}", exc)

    def _scheduled_daily_summary(self) -> None:
        """Scheduled job: log daily P&L summary."""
        logger.info("[SCHEDULER] Logging daily P&L summary")
        if self.trade_log and self.paper_broker:
            try:
                # Check if today is a trading day
                if not is_trading_day(datetime.now(IST).date()):
                    logger.info("Today is not a trading day, skipping daily summary")
                    return
                positions = self.paper_broker.get_positions()
                realized_pnl = sum(p.realized_pnl for p in positions)
                unrealized_pnl = sum(p.unrealized_pnl for p in positions)
                wins = sum(1 for p in positions if p.realized_pnl > 0)
                self.trade_log.log_daily_summary(
                    date=datetime.now(IST).date(),
                    realized=int(realized_pnl),
                    unrealized=int(unrealized_pnl),
                    trades=len(positions),
                    wins=wins,
                )
            except Exception as exc:
                logger.error("Scheduled daily summary failed: {}", exc)

    def _on_market_tick(self, tick: Any) -> None:
        """Callback for market tick updates.

        Upstox V3 protobuf ticks are decoded into a dict-like structure with
        `feeds[instrument_key]`. Extract instrument_key and last_price (in
        paisa) and forward to PositionManager.on_tick(instrument_key, price).
        """
        if not self.position_manager:
            return
        try:
            feeds = tick.get("feeds") if isinstance(tick, dict) else None
            if not feeds:
                return
            for instrument_key, payload in feeds.items():
                # Try common shapes: {"ltpc": {"ltp": 123.4}} or {"fullFeed": {...}}
                ltp = None
                if isinstance(payload, dict):
                    ltpc = payload.get("ltpc") or {}
                    ltp = ltpc.get("ltp")
                    if ltp is None:
                        full = payload.get("fullFeed", {}).get("indexFF") or payload.get("fullFeed", {}).get("marketFF") or {}
                        ltpc = (full or {}).get("ltpc") or {}
                        ltp = ltpc.get("ltp")
                if ltp is None:
                    continue
                # Convert rupees → paisa (PositionManager expects paisa int)
                price_paisa = int(round(float(ltp) * 100))
                self.position_manager.on_tick(instrument_key, price_paisa)
        except Exception as exc:
            logger.debug("Tick dispatch error: {}", exc)

    def _on_order_update(self, update: dict[str, Any]) -> None:
        """Callback for order updates from portfolio feed."""
        if self.order_tracker:
            self.order_tracker.on_order_update(update)

    def _on_position_update(self, update: dict[str, Any]) -> None:
        """Callback for position updates from portfolio feed."""
        if self.position_manager:
            self.position_manager.on_position_update(update)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info("Received signal {}, initiating shutdown...", signum)
        self._shutdown_event.set()

    def shutdown(self) -> None:
        """Execute graceful shutdown sequence."""
        logger.info("=" * 60)
        logger.info("Shutting Down Trading Bot")
        logger.info("=" * 60)

        self._running = False

        # 1. Stop strategy engine
        if self.strategy_engine:
            logger.info("Stopping strategy engine...")
            self.strategy_engine.stop()

        # 2. Stop order manager
        if self.order_manager:
            logger.info("Stopping order manager...")
            self.order_manager.stop()

        # 3. Exit open positions (if configured)
        if self.position_manager and self.settings.trading_mode == "paper":
            logger.info("Exiting open positions...")
            self.position_manager.close_all_positions(reason="Bot shutdown")

        # 4. Stop WebSocket feeds
        if self.market_feed:
            logger.info("Stopping market feed...")
            self.market_feed.stop()

        if self.portfolio_feed:
            logger.info("Stopping portfolio feed...")
            self.portfolio_feed.stop()

        # 5. Stop bar builder
        if self.bar_builder:
            logger.info("Stopping bar builder...")
            self.bar_builder.stop()

        # 6. Stop health monitor
        if self.health_monitor:
            logger.info("Stopping health monitor...")
            self.health_monitor.stop_monitoring()

        # 7. Stop scheduler
        if self.scheduler:
            logger.info("Stopping scheduler...")
            self.scheduler.stop()

        # 8. Log final summary
        if self.trade_log and self.paper_broker:
            logger.info("Logging final P&L summary...")
            try:
                positions = self.paper_broker.get_positions()
                realized_pnl = sum(p.realized_pnl for p in positions)
                unrealized_pnl = sum(p.unrealized_pnl for p in positions)
                wins = sum(1 for p in positions if p.realized_pnl > 0)
                self.trade_log.log_daily_summary(
                    date=datetime.now(IST).date(),
                    realized=int(realized_pnl),
                    unrealized=int(unrealized_pnl),
                    trades=len(positions),
                    wins=wins,
                )
            except Exception as exc:
                logger.warning("Final summary logging failed: {}", exc)

        # Save circuit breaker state
        if self.circuit_breaker:
            logger.info("Saving circuit breaker state...")
            # This would typically save to a file or database
            # For now, just log the state
            logger.info(
                f"Circuit breaker state - halted: {self.circuit_breaker.is_halted()}, consecutive_losses: {self.circuit_breaker.consecutive_losses}"
            )

        logger.info("=" * 60)
        logger.info("Trading Bot Shutdown Complete")
        logger.info("=" * 60)

    def run(self) -> None:
        """Main loop."""
        try:
            self.startup()

            # Keep main thread alive
            while self._running and not self._shutdown_event.is_set():
                self._shutdown_event.wait(timeout=1.0)

        except Exception as exc:
            logger.exception("Fatal error in main loop: {}", exc)
            raise
        finally:
            self.shutdown()


def main() -> int:
    """Entry point."""
    bot = TradingBot()
    try:
        bot.run()
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as exc:
        logger.exception("Fatal error: {}", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
