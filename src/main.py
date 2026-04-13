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

from src.utils.holidays import is_trading_day

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
        if not is_trading_day(today):
            logger.error("Today is not a trading day (holiday or weekend). Bot shutting down.")
            raise TradingBotError(f"Today {today} is not a trading day")

        # 3. Initialize database
        logger.info("Initializing database...")
        init_db()
        self.trade_log = TradeLog()
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
            consecutive_loss_limit=self.settings.consecutive_loss_pause,
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
        self.partial_profit_manager = PartialProfitManager(config=FOUR_TIER_CONFIG)
        self.position_manager = PositionManager(
            broker=self.paper_broker,
            risk_manager=self.risk_manager,
            partial_profit_manager=self.partial_profit_manager,
        )
        self.order_tracker = OrderTracker(
            trade_log=self.trade_log,
        )
        self.order_manager = OrderManager(
            broker=self.paper_broker,
            instrument_manager=self.instrument_manager,
            risk_manager=self.risk_manager,
            position_manager=self.position_manager,
            signal_queue=self.signal_queue,
            trade_log=self.trade_log,
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
            self.strategy_engine = StrategyEngine(
                strategy_directory=str(strategy_dir),
                signal_queue=self.signal_queue,
                bar_close_event=self.bar_close_event,
                instrument_manager=self.instrument_manager,
                bar_builder=self.bar_builder,
                circuit_breaker=self.circuit_breaker,
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
        logger.info("Health monitor initialized and started")

        # 13. Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logger.info("Signal handlers registered")

        # 14. Initialize AI agent pipeline
        if self.settings.agent_pipeline_enabled:
            self.llm_client = LLMClient(
                api_key=self.settings.groq_api_key,
                model=self.settings.groq_model,
                fallback_model=self.settings.groq_fallback_model,
                temperature=self.settings.groq_temperature,
                max_tokens=self.settings.groq_max_tokens,
                rate_limit_rpm=self.settings.groq_rate_limit_rpm,
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
            logger.info(
                "AI Agent Pipeline initialized | LLM configured={}",
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
        self.scheduler.add_job(
            func=self._scheduled_token_refresh,
            trigger=CronTrigger(hour=8, minute=45, timezone=IST),
            id="token_refresh",
            replace_existing=True,
        )

        # Daily instrument download at 06:15 IST
        self.scheduler.add_job(
            func=self._scheduled_instrument_download,
            trigger=CronTrigger(hour=6, minute=15, timezone=IST),
            id="instrument_download",
            replace_existing=True,
        )

        # Intraday square-off at 15:15 IST
        self.scheduler.add_job(
            func=self._scheduled_square_off,
            trigger=CronTrigger(hour=15, minute=15, timezone=IST),
            id="square_off",
            replace_existing=True,
        )

        # Daily P&L summary at 15:30 IST
        self.scheduler.add_job(
            func=self._scheduled_daily_summary,
            trigger=CronTrigger(hour=15, minute=30, timezone=IST),
            id="daily_summary",
            replace_existing=True,
        )

        self.scheduler.start()

    def _scheduled_token_refresh(self) -> None:
        """Scheduled job: refresh access token."""
        logger.info("[SCHEDULER] Refreshing access token")
        if self.token_manager:
            try:
                self.token_manager.refresh_token()
            except Exception as exc:
                logger.error("Scheduled token refresh failed: {}", exc)

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
                self.position_manager.square_off_all("Intraday square-off at 15:15")
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
                self.trade_log.log_daily_summary(
                    date=datetime.now(IST).date(),
                    realized_pnl=realized_pnl,
                    trade_count=len(positions),
                )
            except Exception as exc:
                logger.error("Scheduled daily summary failed: {}", exc)

    def _on_market_tick(self, tick: dict[str, Any]) -> None:
        """Callback for market tick updates."""
        # Position manager monitors for SL/target
        if self.position_manager:
            self.position_manager.on_tick(tick)

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
            self.position_manager.square_off_all("Bot shutdown")

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
            self.scheduler.shutdown(wait=False)

        # 8. Log final summary
        if self.trade_log and self.paper_broker:
            logger.info("Logging final P&L summary...")
            positions = self.paper_broker.get_positions()
            realized_pnl = sum(p.realized_pnl for p in positions)
            self.trade_log.log_daily_summary(
                date=datetime.now(IST).date(),
                realized_pnl=realized_pnl,
                trade_count=len(positions),
            )

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
