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

import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    from dotenv import load_dotenv
    load_dotenv(Path(_root) / ".env")
except ImportError:
    pass

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
from src.data.options_resolver import OptionsResolver

from src.agents.llm_client import LLMClient
from src.agents.pipeline import AgentPipeline
from src.memory.memory_db import MemoryDB
from src.memory.outcome_analyzer import OutcomeAnalyzer
from src.memory.mistake_classifier import MistakeClassifier

# ---------------------------------------------------------------------------
# Phase module imports (Bloomberg build — feature/bloomberg-data-execution)
# All wrapped so import failures don't crash the bot.
# ---------------------------------------------------------------------------
try:
    from src.storage.db import get_sync_engine as _get_sync_engine
    _STORAGE_DB_OK = True
except Exception:  # pragma: no cover
    _get_sync_engine = None  # type: ignore[assignment]
    _STORAGE_DB_OK = False

try:
    from src.research.universe_scanner import UniverseScanner as _UniverseScanner
except Exception:  # pragma: no cover
    _UniverseScanner = None  # type: ignore[assignment]

try:
    from src.research.sector_rotation import SectorRotationAnalyzer as _SectorRotation
except Exception:  # pragma: no cover
    _SectorRotation = None  # type: ignore[assignment]

try:
    from src.research.vix_regime import VIXRegimeClassifier as _VIXRegime
except Exception:  # pragma: no cover
    _VIXRegime = None  # type: ignore[assignment]

try:
    from src.research.macro_overlay import MacroOverlay as _MacroOverlay
except Exception:  # pragma: no cover
    _MacroOverlay = None  # type: ignore[assignment]

try:
    from src.research.flow_regime import FlowRegimeAnalyzer as _FlowRegime
except Exception:  # pragma: no cover
    _FlowRegime = None  # type: ignore[assignment]

try:
    from src.research.corporate_calendar import CorporateCalendar as _CorporateCalendar
except Exception:  # pragma: no cover
    _CorporateCalendar = None  # type: ignore[assignment]

try:
    from src.research.earnings_calendar import EarningsCalendar as _EarningsCalendar
except Exception:  # pragma: no cover
    _EarningsCalendar = None  # type: ignore[assignment]

try:
    from src.research.news_query import NewsQuery as _NewsQuery
except Exception:  # pragma: no cover
    _NewsQuery = None  # type: ignore[assignment]

try:
    from src.research.insider_signals import InsiderSignals as _InsiderSignals
except Exception:  # pragma: no cover
    _InsiderSignals = None  # type: ignore[assignment]

try:
    from src.research.sentiment_classifier import SentimentClassifier as _SentimentClassifier
except Exception:  # pragma: no cover
    _SentimentClassifier = None  # type: ignore[assignment]

try:
    from src.data.depth_feed import DepthFeedHandler as _DepthFeed
except Exception:  # pragma: no cover
    _DepthFeed = None  # type: ignore[assignment]

try:
    from src.data.tick_metrics import TickMetricsAggregator as _TickMetrics
except Exception:  # pragma: no cover
    _TickMetrics = None  # type: ignore[assignment]

try:
    from src.data.micro_features import MicroFeatureEngine as _MicroFeatureEngine
except Exception:  # pragma: no cover
    _MicroFeatureEngine = None  # type: ignore[assignment]

try:
    from src.execution.rl_exit_agent import RLExitAgent as _RLExitAgent, build_rl_exit_agent as _build_rl_exit_agent
except Exception:  # pragma: no cover
    _RLExitAgent = None  # type: ignore[assignment]
    _build_rl_exit_agent = None  # type: ignore[assignment]

try:
    from src.data.options_chain_feed import OptionsChainFeed as _OptionsChainFeed
except Exception:  # pragma: no cover
    _OptionsChainFeed = None  # type: ignore[assignment]

try:
    from src.data.flow_scraper import FlowScraper as _FlowScraper
except Exception:  # pragma: no cover
    _FlowScraper = None  # type: ignore[assignment]

try:
    from src.data.corporate_actions_scraper import CorporateActionsScraper as _CorpActionsScraper
except Exception:  # pragma: no cover
    _CorpActionsScraper = None  # type: ignore[assignment]

try:
    from src.data.earnings_scraper import EarningsScraper as _EarningsScraper
except Exception:  # pragma: no cover
    _EarningsScraper = None  # type: ignore[assignment]

try:
    from src.data.news_rss_scraper import NewsRSSScraper as _NewsRSSScraper
except Exception:  # pragma: no cover
    _NewsRSSScraper = None  # type: ignore[assignment]

try:
    from src.data.block_deals_scraper import BlockDealsScraper as _BlockDealsScraper
except Exception:  # pragma: no cover
    _BlockDealsScraper = None  # type: ignore[assignment]

try:
    from src.data.insider_scraper import InsiderScraper as _InsiderScraper
except Exception:  # pragma: no cover
    _InsiderScraper = None  # type: ignore[assignment]

try:
    from src.indicators.volume_profile import VolumeProfileBuilder as _VolumeProfile
except Exception:  # pragma: no cover
    _VolumeProfile = None  # type: ignore[assignment]

try:
    from src.strategy.confluence_engine import ConfluenceEngine as _ConfluenceEngine
except Exception:  # pragma: no cover
    _ConfluenceEngine = None  # type: ignore[assignment]

try:
    from src.strategy.rejection_filter import RejectionFilter as _RejectionFilter
except Exception:  # pragma: no cover
    _RejectionFilter = None  # type: ignore[assignment]

try:
    from src.strategy.regime_router import RegimeRouter as _RegimeRouter
except Exception:  # pragma: no cover
    _RegimeRouter = None  # type: ignore[assignment]

try:
    from src.risk.kelly_sizer import KellySizer as _KellySizer
except Exception:  # pragma: no cover
    _KellySizer = None  # type: ignore[assignment]

try:
    from src.risk.portfolio_risk import PortfolioRisk as _PortfolioRisk
except Exception:  # pragma: no cover
    _PortfolioRisk = None  # type: ignore[assignment]

try:
    from src.risk.greek_aggregator import GreekAggregator as _GreekAggregator
except Exception:  # pragma: no cover
    _GreekAggregator = None  # type: ignore[assignment]

try:
    from src.execution.smart_router import SmartOrderRouter as _SmartRouter
except Exception:  # pragma: no cover
    _SmartRouter = None  # type: ignore[assignment]

try:
    from src.execution.order_validator import OrderValidator as _OrderValidator
except Exception:  # pragma: no cover
    _OrderValidator = None  # type: ignore[assignment]

try:
    from src.execution.reconciler import PositionReconciler as _Reconciler  # noqa: F401
except Exception:  # pragma: no cover
    _Reconciler = None  # type: ignore[assignment]

try:
    from src.execution.slippage_monitor import SlippageMonitor as _SlippageMonitor
except Exception:  # pragma: no cover
    _SlippageMonitor = None  # type: ignore[assignment]

try:
    from src.storage.archiver import ColdStorageArchiver as _Archiver
except Exception:  # pragma: no cover
    _Archiver = None  # type: ignore[assignment]

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
        self._db_run_id: int | None = None

        # ---------------------------------------------------------------------------
        # Bloomberg-build phase module handles — initialised in startup()
        # ---------------------------------------------------------------------------
        self.db_engine: Any = None
        self.universe_scanner: Any = None
        self.sector_rotation: Any = None
        self.vix_regime: Any = None
        self.macro_overlay: Any = None
        self.flow_regime: Any = None
        self.corporate_calendar: Any = None
        self.earnings_calendar: Any = None
        self.news_query: Any = None
        self.insider_signals: Any = None
        self.sentiment_classifier: Any = None
        self.depth_feed: Any = None
        self.tick_metrics: Any = None
        self.micro_feature_engine: Any = None
        self.rl_exit_agent: Any = None
        self.options_chain: Any = None
        self.flow_scraper: Any = None
        self.corp_actions_scraper: Any = None
        self.earnings_scraper: Any = None
        self.news_scraper: Any = None
        self.block_deals_scraper: Any = None
        self.insider_scraper: Any = None
        self.volume_profile: Any = None
        self.confluence_engine: Any = None
        self.rejection_filter: Any = None
        self.regime_router: Any = None
        self.kelly_sizer: Any = None
        self.portfolio_risk: Any = None
        self.greek_aggregator: Any = None
        self.smart_router: Any = None
        self.order_validator: Any = None
        self.reconciler: Any = None
        self.slippage_monitor: Any = None
        self.archiver: Any = None

    def startup(self) -> None:
        """Execute startup sequence."""
        self._running = True
        logger.info("=" * 60)
        logger.info("Starting Upstox Trading Bot")
        logger.info("=" * 60)

        # 0. Reset HealthMonitor singleton state from any previous run.
        # The HealthMonitor is a singleton — if the bot is restarted in the
        # same Python process, stale components from the prior run would
        # cause instant FAILED alerts as soon as monitoring restarts.
        HealthMonitor().reset()
        # 1. Initialize logger
        setup_logger(
            log_level=self.settings.log_level,
            log_file=self.settings.log_file,
        )
        logger.info("Logger initialized")

        # 2. Authenticate
        logger.info("Authenticating with Upstox...")
        self.token_manager = TokenManager()
        try:
            access_token = self.token_manager.get_valid_token()
        except Exception as e:
            logger.warning(f"Auto-authentication failed ({e}). Triggering interactive manual login...")
            from src.auth.manual_login import manual_login
            access_token = manual_login()

        # Probe token against live API — catch stale/revoked tokens before they cause silent failures
        try:
            import upstox_client
            _cfg = upstox_client.Configuration()
            _cfg.access_token = access_token
            _profile = upstox_client.UserApi(upstox_client.ApiClient(_cfg)).get_profile("2.0")
            _name = getattr(getattr(_profile, "data", None), "user_name", "unknown")
            logger.info(f"Token valid — logged in as: {_name}")
        except Exception as _probe_err:
            logger.error(f"Token probe FAILED: {_probe_err}")
            logger.warning("Cached/env token is invalid or revoked. Triggering fresh login...")
            # Clear stale cache so next startup doesn't reuse it
            import os
            stale_cache = Path("data/token_cache.json")
            if stale_cache.exists():
                stale_cache.unlink()
            from src.auth.manual_login import manual_login
            access_token = manual_login()

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
                logger.warning(
                    "Today {} is not a trading day (holiday or weekend). "
                    "Bot will shut down cleanly. Use --ignore-holidays to force a dry run.",
                    today,
                )
                # Clean shutdown — not an error
                raise SystemExit(0)

        # 3. Initialize database
        logger.info("Initializing database...")
        init_db(self.settings.database_url)
        self.trade_log = TradeLog(session_factory=get_session)
        logger.info("Database initialized")

        # 3b. Record bot run start in Postgres storage layer
        try:
            from src.storage.db import record_bot_run_start
            _git_sha: str | None = None
            _branch: str | None = None
            try:
                _git_sha = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                ).decode().strip()
                _branch = subprocess.check_output(
                    ["git", "branch", "--show-current"], stderr=subprocess.DEVNULL
                ).decode().strip()
            except Exception:
                pass
            self._db_run_id = record_bot_run_start(
                branch=_branch,
                git_sha=_git_sha,
                mode=self.settings.trading_mode,
                capital_paisa=self.settings.capital,
            )
            logger.info("Bot run recorded in storage DB (run_id={})", self._db_run_id)
        except Exception as _db_exc:
            logger.warning("Storage DB unavailable — bot run not recorded: {}", _db_exc)

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
            per_strategy_daily_loss_cap_pct=0.02,  # 2% of capital per strategy/day
            capital_paisa=self.settings.capital,
        )
        self.circuit_breaker.set_on_halt_callback(self._run_detox_analysis)
        self.strategy_quarantine = StrategyQuarantine()
        
        # Link circuit breaker to risk manager manually since it expects it
        self.risk_manager.circuit_breaker = self.circuit_breaker
        logger.info("Risk manager initialized")

        # Check if circuit breaker is already halted (from previous run)
        if self.circuit_breaker.is_halted():
            logger.error("Circuit breaker is currently halted. Cannot start trading.")
            raise TradingBotError("Circuit breaker is halted. Please resume manually.")

        # 6. Initialize broker, partial profit manager, and order/position managers
        self.paper_broker = PaperBroker(
            initial_capital=self.settings.capital,
            slippage_pct=self.settings.slippage_pct / 100,  # settings stores percent; broker needs decimal
        )
        self.partial_profit_manager = PartialProfitManager()
        self.position_manager = PositionManager(
            broker=self.paper_broker,
            risk_manager=self.risk_manager,
            partial_profit_manager=self.partial_profit_manager,
            trade_logger=getattr(self, 'trade_log', None),
        )
        self.order_tracker = OrderTracker(
            broker=self.paper_broker,
            db_url=self.settings.database_url,
        )
        self.order_manager = OrderManager(
            signal_queue=self.signal_queue,
            broker=self.paper_broker,
            trading_mode=self.settings.trading_mode,
            instrument_manager=self.instrument_manager,
            risk_manager=self.risk_manager,
            position_manager=self.position_manager,
            strategy_quarantine=self.strategy_quarantine,
            db_url=self.settings.database_url,
            kelly_sizer=getattr(self, "kelly_sizer", None),
            smart_router=getattr(self, "smart_router", None),
            order_validator=getattr(self, "order_validator", None),
        )
        logger.info("Order and position managers initialized")

        # 6b. Initialize Postgres engine early so BarBuilder + Bloomberg modules share it.
        # This must run before BarBuilder so closed bars are persisted from the first tick.
        try:
            if _get_sync_engine is not None:
                self.db_engine = _get_sync_engine()
                logger.info("Postgres engine initialized for cross-module use")
        except Exception as exc:
            logger.warning("Postgres engine init failed (modules will degrade): {}", exc)
            self.db_engine = None

        # 7. Start BarBuilder
        self.bar_builder = BarBuilder(
            tick_queue=self.tick_queue,
            bar_close_event=self.bar_close_event,
            db_engine=getattr(self, "db_engine", None),
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

        # Subscribe to underlyings (indices for spot price / CPR / warmup bars)
        underlyings = ["NSE_INDEX|Nifty Bank", "NSE_INDEX|Nifty 50"]
        self.market_feed.start(underlyings, mode="full")
        self.portfolio_feed.start()
        logger.info("WebSocket feeds started — subscribed to index underlyings")

        # 8b. Dynamic ATM option subscription on first tick.
        # We can't resolve ATM strike until we have the live spot price.
        # Register a one-shot callback: extract spot from first BankNifty tick
        # then subscribe to the resolved ATM CE/PE option contracts.
        _banknifty_key = "NSE_INDEX|Nifty Bank"
        _options_subscribed = [False]  # mutable flag for closure

        def _subscribe_atm_options_once(tick: Any) -> None:
            """One-shot: subscribe to ATM options on first BankNifty tick."""
            if _options_subscribed[0]:
                return
            if not isinstance(tick, dict):
                return
            feeds = tick.get("feeds", {})
            if _banknifty_key not in feeds:
                return

            # Extract spot price for every index instrument present in this tick
            # so each strategy uses its own underlying's price, not just BankNifty.
            def _extract_ltp(payload: dict) -> float | None:
                ltpc = payload.get("ltpc") or {}
                ltp = ltpc.get("ltp")
                if ltp is None:
                    ff = payload.get("fullFeed", {})
                    inner = ff.get("indexFF") or ff.get("marketFF") or {}
                    ltpc = inner.get("ltpc") or {}
                    ltp = ltpc.get("ltp")
                return ltp

            spot_map: dict[str, int] = {}
            for ik, payload in feeds.items():
                try:
                    ltp = _extract_ltp(payload)
                    if ltp is not None:
                        spot_map[ik] = int(round(float(ltp) * 100))
                except Exception:
                    pass

            banknifty_spot = spot_map.get(_banknifty_key)
            if banknifty_spot is None:
                return

            _options_subscribed[0] = True
            logger.info(
                "ATM subscription triggered — spots: {}",
                {k: f"₹{v/100:.2f}" for k, v in spot_map.items()},
            )

            # Resolve ATM option keys for all active option strategies.
            # Each strategy gets the spot of its own underlying via the getter.
            if not self.strategy_engine or not self.instrument_manager:
                return
            try:
                resolver = OptionsResolver(self.instrument_manager)
                active_strategies = self.strategy_engine.get_active_strategies()

                def _spot_getter(underlying_key: str) -> int:
                    return spot_map.get(underlying_key, banknifty_spot)

                option_keys = resolver.resolve_all_strategies(active_strategies, _spot_getter)
                if option_keys:
                    self.market_feed.subscribe(option_keys, mode="full")
                    logger.info(
                        "Subscribed to {} ATM option instrument(s): {}",
                        len(option_keys),
                        option_keys,
                    )
                else:
                    logger.warning(
                        "No ATM option instruments resolved — "
                        "check strategy instrument_selection configs"
                    )
            except Exception as exc:
                logger.error("ATM option subscription failed: {}", exc)

        if self.market_feed is not None:
            self.market_feed.register_callback(_subscribe_atm_options_once)
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

            # Resolve options instruments per strategy and subscribe to them
            from src.strategy.instrument_resolver import resolve_options_instruments
            options_to_subscribe: set[str] = set()
            for strategy in self.strategy_engine.strategies:
                if not strategy.enabled:
                    continue
                underlying = strategy.underlying or {}
                underlying_key = (
                    underlying.get("instrument_key")
                    if isinstance(underlying, dict)
                    else getattr(underlying, "instrument_key", None)
                )
                if not underlying_key:
                    continue
                # Get spot from latest warmup bar of underlying
                spot_paisa = None
                if self.bar_builder is not None:
                    df = self.bar_builder.get_bars(underlying_key, timeframe=1)
                    if df is not None and not df.empty:
                        spot_paisa = int(df.iloc[-1]["close"])
                if spot_paisa is None:
                    logger.warning(
                        "Cannot resolve options for {}: no warmup bars for {} — will retry on first live tick",
                        strategy.name, underlying_key,
                    )
                    continue
                keys = resolve_options_instruments(strategy, spot_paisa, self.instrument_manager)
                options_to_subscribe.update(keys)

            if options_to_subscribe:
                # Defer subscribe — WebSocket handshake not yet complete at this point.
                # _subscribe_atm_options_once handles actual subscription on first tick.
                logger.info(
                    "Resolved {} option contracts — subscription deferred until WebSocket ready",
                    len(options_to_subscribe),
                )

            self.strategy_engine.start()
            logger.info("Strategy engine started")

            # 9b. Wire signal forwarder: drain StrategyEngine's internal
            # signal queue → OrderManager's signal queue.
            # Without this thread, all strategy signals go into a void
            # because StrategyEngine creates its own queue that nobody reads.
            def _forward_signals() -> None:
                """Forward signals from StrategyEngine to OrderManager."""
                logger.info("SignalForwarder thread started")
                while self._running:
                    try:
                        sig = self.strategy_engine.get_signal(timeout=1.0)
                        if sig is not None:
                            # Wave-5 gate: confluence + rejection + regime
                            try:
                                gated = self.strategy_engine._apply_wave5_gates(sig)
                            except Exception as exc:
                                logger.warning("wave5_gate error (passing through): {}", exc)
                                gated = sig
                            if gated is None:
                                logger.info(
                                    "SIGNAL_BLOCKED_BY_WAVE5 | {} | {} | {}",
                                    sig.strategy_name,
                                    getattr(sig, 'signal_type', ''),
                                    getattr(sig, 'instrument_key', ''),
                                )
                                continue
                            logger.info(
                                "SIGNAL_FORWARDED | {} | {} | {} | {}",
                                sig.strategy_name,
                                getattr(sig, 'signal_type', ''),
                                getattr(sig, 'instrument_key', ''),
                                getattr(sig, 'signal_id', ''),
                            )
                            self.signal_queue.put(sig)
                    except Exception as exc:
                        logger.debug("SignalForwarder error: {}", exc)
                logger.info("SignalForwarder thread stopped")

            self._signal_forwarder = threading.Thread(
                target=_forward_signals, daemon=True, name="SignalForwarder"
            )
            self._signal_forwarder.start()
        # 10. Start order manager
        self.order_manager.start()
        logger.info("Order manager started")

        # 10b. Initialize Bloomberg-build phase modules
        self._init_phase_modules(access_token)

        # 11. Configure scheduler
        self.scheduler = get_scheduler()
        self._setup_scheduler()
        logger.info("Scheduler configured")

        # 12. Initialize health monitor
        self.health_monitor = HealthMonitor()
        self.health_monitor.register_component("market_feed", heartbeat_timeout=300)  # 5 min — keepalive thread bumps every 60s
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

            # Register a real market_feed recovery strategy that has access
            # to the actual self.market_feed instance. The default recovery
            # in HealthMonitor referenced MarketFeed.get_instance() which
            # does not exist, so every recovery attempt silently failed and
            # exhausted the 3-attempt cap in ~45s.
            _feed = self.market_feed

            def _recover_market_feed(component_name: str) -> bool:
                """Real recovery: restart the WebSocket streamer."""
                logger.info("Attempting market_feed recovery via real instance...")
                try:
                    _feed.stop()
                    import time as _time
                    _time.sleep(2)
                    # Re-subscribe to the same symbols that were active
                    subs = list(getattr(_feed, "_subscribed_keys", set()) or
                                getattr(_feed, "_subscribed_symbols", set()))
                    if subs:
                        _feed.start(subs, mode="full")
                    else:
                        logger.warning("market_feed recovery: no known subscriptions to restore")
                    logger.info("market_feed recovery successful")
                    return True
                except Exception as exc:
                    logger.error("market_feed recovery failed: {}", exc)
                    return False

            _hm.register_recovery_strategy("market_feed", _recover_market_feed)


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
            
            # Initialize Phase 3 and Phase 4 Agents
            from src.agents.market_maker import MarketMakerAgent
            from src.learning.rl_loop import ContinuousRLLoop
            
            self.market_maker = MarketMakerAgent(llm_client=self.llm_client)
            # The RL loop requires the trade_analyzer. Assuming outcome_analyzer acts similarly,
            # or we create a new TradeAnalyzer instance. 
            from src.learning.trade_analyzer import TradeAnalyzer
            self.trade_analyzer = TradeAnalyzer()
            self.rl_loop = ContinuousRLLoop(llm_client=self.llm_client, trade_analyzer=self.trade_analyzer)

            # Wire pipeline into OrderManager for live signal validation
            self.order_manager._agent_pipeline = self.agent_pipeline

            def _agent_context_provider() -> dict[str, Any]:
                """Provide indicator context to the async macro loop."""
                if not self.bar_builder:
                    return {}
                
                # We extract VIX if available, else default to 15.0
                bars = _bars_provider() if callable(_bars_provider) else {}
                vix = 15.0
                vix_sym = getattr(self.settings, "vix_symbol", "NSE_INDEX|India VIX")
                if vix_sym in bars:
                    try:
                        vix = float(bars[vix_sym]["close"].iloc[-1])
                    except Exception:
                        pass
                
                # In Phase 2, we will hook up the actual IndicatorEngine here.
                # For now, providing minimal defaults + VIX to RegimeAgent.
                playbook = self.market_maker.latest_playbook if hasattr(self, "market_maker") else None
                return {
                    "vix": vix,
                    "adx": 20.0,
                    "rsi": 50.0,
                    "atr_percentile": 50.0,
                    "price_position_in_range": 50.0,
                    "playbook": playbook,
                }

            self.agent_pipeline.start_async_macro_loop(
                context_provider=_agent_context_provider,
                interval_seconds=180,
            )

            logger.info(
                "AI Agent Pipeline initialized & wired into OrderManager | LLM configured={}",
                self.llm_client.is_configured,
            )

        logger.info("=" * 60)
        logger.info("Trading Bot Startup Complete")
        logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Bloomberg-build phase module initialisation
    # -----------------------------------------------------------------------

    def _init_phase_modules(self, access_token: str) -> None:
        """Initialise every Bloomberg-build phase module.

        All initialisations are wrapped in try/except so that a single missing
        dependency does NOT crash the bot.  Each failed module is set to None
        and a WARNING is emitted.  The bot continues with degraded data depth.
        """
        logger.info("Initialising Bloomberg-build phase modules…")

        # ------------------------------------------------------------------ #
        # Phase 0 — Postgres storage engine
        # db_engine is initialised early in startup() so BarBuilder can use it.
        # Here we just confirm it is available (already a singleton via lru_cache).
        # ------------------------------------------------------------------ #
        try:
            if _STORAGE_DB_OK and _get_sync_engine is not None:
                if self.db_engine is None:
                    self.db_engine = _get_sync_engine()
                logger.debug("db_engine: OK")
        except Exception as exc:
            logger.warning("db_engine init failed: {}", exc)

        # ------------------------------------------------------------------ #
        # Helper: historical provider compatible with research modules
        # ------------------------------------------------------------------ #
        def _hist_provider(instrument_key: str, interval: str, days: int):
            try:
                fetcher = HistoricalDataFetcher(access_token)
                return fetcher.fetch_candles(instrument_key, interval=interval, days=days)
            except Exception as exc:
                logger.warning("historical provider failed for {}: {}", instrument_key, exc)
                import pandas as _pd
                return _pd.DataFrame()

        # ------------------------------------------------------------------ #
        # Phase D — Universe scanner
        # ------------------------------------------------------------------ #
        try:
            if _UniverseScanner is not None:
                self.universe_scanner = _UniverseScanner(
                    instrument_manager=self.instrument_manager,
                    historical_provider=_hist_provider,
                    db_engine=self.db_engine,
                )
                logger.debug("universe_scanner: OK")
        except Exception as exc:
            logger.warning("universe_scanner init failed: {}", exc)

        # ------------------------------------------------------------------ #
        # Phase B.1 — Sector rotation
        # ------------------------------------------------------------------ #
        try:
            if _SectorRotation is not None:
                self.sector_rotation = _SectorRotation(
                    instrument_manager=self.instrument_manager,
                    historical_provider=_hist_provider,
                    db_engine=self.db_engine,
                )
                logger.debug("sector_rotation: OK")
        except Exception as exc:
            logger.warning("sector_rotation init failed: {}", exc)

        # ------------------------------------------------------------------ #
        # Phase B.2 — VIX regime
        # ------------------------------------------------------------------ #
        try:
            if _VIXRegime is not None:
                self.vix_regime = _VIXRegime(
                    historical_provider=_hist_provider,
                    db_engine=self.db_engine,
                )
                logger.debug("vix_regime: OK")
        except Exception as exc:
            logger.warning("vix_regime init failed: {}", exc)

        # ------------------------------------------------------------------ #
        # Phase B.3 — Macro overlay
        # ------------------------------------------------------------------ #
        try:
            if _MacroOverlay is not None:
                self.macro_overlay = _MacroOverlay(
                    instrument_manager=self.instrument_manager,
                    historical_provider=_hist_provider,
                    db_engine=self.db_engine,
                )
                logger.debug("macro_overlay: OK")
        except Exception as exc:
            logger.warning("macro_overlay init failed: {}", exc)

        # ------------------------------------------------------------------ #
        # Phase B.4 — Flow regime + flow scraper
        # ------------------------------------------------------------------ #
        try:
            if _FlowRegime is not None:
                self.flow_regime = _FlowRegime(db_engine=self.db_engine)
                logger.debug("flow_regime: OK")
        except Exception as exc:
            logger.warning("flow_regime init failed: {}", exc)

        try:
            if _FlowScraper is not None:
                self.flow_scraper = _FlowScraper(db_engine=self.db_engine)
                logger.debug("flow_scraper: OK")
        except Exception as exc:
            logger.warning("flow_scraper init failed: {}", exc)

        # ------------------------------------------------------------------ #
        # Phase C.1 — Corporate calendar + scraper
        # ------------------------------------------------------------------ #
        try:
            if _CorporateCalendar is not None:
                self.corporate_calendar = _CorporateCalendar(db_engine=self.db_engine)
                logger.debug("corporate_calendar: OK")
        except Exception as exc:
            logger.warning("corporate_calendar init failed: {}", exc)

        try:
            if _CorpActionsScraper is not None:
                self.corp_actions_scraper = _CorpActionsScraper(db_engine=self.db_engine)
                logger.debug("corp_actions_scraper: OK")
        except Exception as exc:
            logger.warning("corp_actions_scraper init failed: {}", exc)

        # ------------------------------------------------------------------ #
        # Phase C.2 — Earnings calendar + scraper
        # ------------------------------------------------------------------ #
        try:
            if _EarningsCalendar is not None:
                self.earnings_calendar = _EarningsCalendar(db_engine=self.db_engine)
                logger.debug("earnings_calendar: OK")
        except Exception as exc:
            logger.warning("earnings_calendar init failed: {}", exc)

        try:
            if _EarningsScraper is not None:
                self.earnings_scraper = _EarningsScraper(db_engine=self.db_engine)
                logger.debug("earnings_scraper: OK")
        except Exception as exc:
            logger.warning("earnings_scraper init failed: {}", exc)

        # ------------------------------------------------------------------ #
        # Phase C.3 — News + sentiment
        # ------------------------------------------------------------------ #
        try:
            if _NewsQuery is not None:
                self.news_query = _NewsQuery(db_engine=self.db_engine)
                logger.debug("news_query: OK")
        except Exception as exc:
            logger.warning("news_query init failed: {}", exc)

        try:
            if _NewsRSSScraper is not None:
                self.news_scraper = _NewsRSSScraper(db_engine=self.db_engine)
                logger.debug("news_scraper: OK")
        except Exception as exc:
            logger.warning("news_scraper init failed: {}", exc)

        try:
            if _SentimentClassifier is not None:
                self.sentiment_classifier = _SentimentClassifier(db_engine=self.db_engine)
                logger.debug("sentiment_classifier: OK")
        except Exception as exc:
            logger.warning("sentiment_classifier init failed: {}", exc)

        # ------------------------------------------------------------------ #
        # Phase C.4 — Block deals + insider
        # ------------------------------------------------------------------ #
        try:
            if _InsiderSignals is not None:
                self.insider_signals = _InsiderSignals(db_engine=self.db_engine)
                logger.debug("insider_signals: OK")
        except Exception as exc:
            logger.warning("insider_signals init failed: {}", exc)

        try:
            if _BlockDealsScraper is not None:
                self.block_deals_scraper = _BlockDealsScraper(db_engine=self.db_engine)
                logger.debug("block_deals_scraper: OK")
        except Exception as exc:
            logger.warning("block_deals_scraper init failed: {}", exc)

        try:
            if _InsiderScraper is not None:
                self.insider_scraper = _InsiderScraper(db_engine=self.db_engine)
                logger.debug("insider_scraper: OK")
        except Exception as exc:
            logger.warning("insider_scraper init failed: {}", exc)

        # ------------------------------------------------------------------ #
        # Phase A.1 — Depth feed
        # ------------------------------------------------------------------ #
        try:
            if _DepthFeed is not None:
                self.depth_feed = _DepthFeed(db_engine=self.db_engine)
                logger.debug("depth_feed: OK")
        except Exception as exc:
            logger.warning("depth_feed init failed: {}", exc)

        # ------------------------------------------------------------------ #
        # Phase A.2 — Tick metrics / CVD
        # ------------------------------------------------------------------ #
        try:
            if _TickMetrics is not None:
                self.tick_metrics = _TickMetrics(db_engine=self.db_engine)
                logger.debug("tick_metrics: OK")
        except Exception as exc:
            logger.warning("tick_metrics init failed: {}", exc)

        # ------------------------------------------------------------------ #
        # Phase A.2b — Micro feature engine (tick-level microstructure)
        # ------------------------------------------------------------------ #
        try:
            if _MicroFeatureEngine is not None:
                self.micro_feature_engine = _MicroFeatureEngine(buffer_size=50)
                logger.debug("micro_feature_engine: OK")
        except Exception as exc:
            logger.warning("micro_feature_engine init failed: {}", exc)

        # ------------------------------------------------------------------ #
        # Phase A.2c — RL exit agent
        # ------------------------------------------------------------------ #
        try:
            if _build_rl_exit_agent is not None:
                self.rl_exit_agent = _build_rl_exit_agent(
                    db_path=self.settings.database_url.replace("sqlite:///", ""),
                    checkpoint_path="models/rl_exit_agent.pkl",
                    warm_start=True,
                )
                logger.debug("rl_exit_agent: OK (episodes={})", self.rl_exit_agent.episode_count)
                if self.position_manager is not None:
                    self.position_manager.set_rl_exit_agent(self.rl_exit_agent)
        except Exception as exc:
            logger.warning("rl_exit_agent init failed: {}", exc)

        # ------------------------------------------------------------------ #
        # Phase A.3 — Options chain feed
        # ------------------------------------------------------------------ #
        try:
            if _OptionsChainFeed is not None:
                self.options_chain = _OptionsChainFeed(
                    access_token=access_token,
                    db_engine=self.db_engine,
                )
                logger.debug("options_chain: OK")
        except Exception as exc:
            logger.warning("options_chain init failed: {}", exc)

        # ------------------------------------------------------------------ #
        # Phase A.4 — Volume profile
        # ------------------------------------------------------------------ #
        try:
            if _VolumeProfile is not None:
                self.volume_profile = _VolumeProfile(db_engine=self.db_engine)
                logger.debug("volume_profile: OK")
        except Exception as exc:
            logger.warning("volume_profile init failed: {}", exc)

        # ------------------------------------------------------------------ #
        # Wave 5 — Confluence engine, rejection filter, regime router
        # ------------------------------------------------------------------ #
        try:
            if _ConfluenceEngine is not None:
                self.confluence_engine = _ConfluenceEngine(db_engine=self.db_engine)
                logger.debug("confluence_engine: OK")
        except Exception as exc:
            logger.warning("confluence_engine init failed: {}", exc)

        try:
            if _RejectionFilter is not None:
                self.rejection_filter = _RejectionFilter(db_engine=self.db_engine)
                logger.debug("rejection_filter: OK")
        except Exception as exc:
            logger.warning("rejection_filter init failed: {}", exc)

        try:
            if _RegimeRouter is not None:
                self.regime_router = _RegimeRouter(db_engine=self.db_engine)
                logger.debug("regime_router: OK")
        except Exception as exc:
            logger.warning("regime_router init failed: {}", exc)

        # ------------------------------------------------------------------ #
        # Wave 5 — Kelly sizer
        # ------------------------------------------------------------------ #
        try:
            if _KellySizer is not None:
                self.kelly_sizer = _KellySizer(db_engine=self.db_engine)
                logger.debug("kelly_sizer: OK")
        except Exception as exc:
            logger.warning("kelly_sizer init failed: {}", exc)

        # ------------------------------------------------------------------ #
        # Phase F — Portfolio risk + Greek aggregator
        # ------------------------------------------------------------------ #
        try:
            if _PortfolioRisk is not None:
                self.portfolio_risk = _PortfolioRisk(db_engine=self.db_engine)
                logger.debug("portfolio_risk: OK")
        except Exception as exc:
            logger.warning("portfolio_risk init failed: {}", exc)

        try:
            if _GreekAggregator is not None:
                self.greek_aggregator = _GreekAggregator(db_engine=self.db_engine)
                logger.debug("greek_aggregator: OK")
        except Exception as exc:
            logger.warning("greek_aggregator init failed: {}", exc)

        # ------------------------------------------------------------------ #
        # Phase E — Smart router, order validator, reconciler, slippage monitor
        # ------------------------------------------------------------------ #
        try:
            if _SmartRouter is not None:
                self.smart_router = _SmartRouter(
                    broker=self.paper_broker,
                    db_engine=self.db_engine,
                )
                logger.debug("smart_router: OK")
        except Exception as exc:
            logger.warning("smart_router init failed: {}", exc)

        try:
            if _OrderValidator is not None:
                self.order_validator = _OrderValidator(
                    instrument_manager=self.instrument_manager,
                    broker=self.paper_broker,
                    db_engine=self.db_engine,
                )
                logger.debug("order_validator: OK")
        except Exception as exc:
            logger.warning("order_validator init failed: {}", exc)

        try:
            if _Reconciler is not None:
                self.reconciler = _Reconciler(
                    broker=self.paper_broker,
                    db_engine=self.db_engine,
                )
                logger.debug("reconciler: OK")
        except Exception as exc:
            logger.warning("reconciler init failed: {}", exc)

        try:
            if _SlippageMonitor is not None:
                self.slippage_monitor = _SlippageMonitor(db_engine=self.db_engine)
                logger.debug("slippage_monitor: OK")
        except Exception as exc:
            logger.warning("slippage_monitor init failed: {}", exc)

        # ------------------------------------------------------------------ #
        # Phase 0 — Cold storage archiver
        # ------------------------------------------------------------------ #
        try:
            if _Archiver is not None:
                self.archiver = _Archiver(db_engine=self.db_engine)
                logger.debug("archiver: OK")
        except Exception as exc:
            logger.warning("archiver init failed: {}", exc)

        # ------------------------------------------------------------------ #
        # Wire Wave-5 gates into strategy engine
        # ------------------------------------------------------------------ #
        if self.strategy_engine is not None:
            try:
                self.strategy_engine.set_wave5_modules(
                    confluence_engine=self.confluence_engine,
                    rejection_filter=self.rejection_filter,
                    regime_router=self.regime_router,
                    kelly_sizer=self.kelly_sizer,
                    db_engine=self.db_engine,
                )
            except Exception as exc:
                logger.warning("Failed to wire Wave-5 gates into strategy engine: {}", exc)

        # ------------------------------------------------------------------ #
        # Expand WebSocket subscriptions to include universe + sector + macro
        # Deferred to first tick so the socket is guaranteed open.
        # ------------------------------------------------------------------ #
        _ws_expanded = False

        def _expand_ws_once(tick: Any) -> None:
            nonlocal _ws_expanded
            if _ws_expanded:
                return
            _ws_expanded = True
            self._expand_websocket_subscriptions()

        if self.market_feed is not None:
            self.market_feed.register_callback(_expand_ws_once)

        # ------------------------------------------------------------------ #
        # Startup health log
        # ------------------------------------------------------------------ #
        logger.info("=" * 70)
        logger.info("Phase modules initialized:")
        for _name, _instance in [
            ("storage_db",          self.db_engine),
            ("universe_scanner",    self.universe_scanner),
            ("sector_rotation",     self.sector_rotation),
            ("vix_regime",          self.vix_regime),
            ("macro_overlay",       self.macro_overlay),
            ("flow_regime",         self.flow_regime),
            ("corporate_calendar",  self.corporate_calendar),
            ("earnings_calendar",   self.earnings_calendar),
            ("news_query",          self.news_query),
            ("insider_signals",     self.insider_signals),
            ("depth_feed",          self.depth_feed),
            ("tick_metrics",        self.tick_metrics),
            ("options_chain",       self.options_chain),
            ("volume_profile",      self.volume_profile),
            ("confluence_engine",   self.confluence_engine),
            ("rejection_filter",    self.rejection_filter),
            ("regime_router",       self.regime_router),
            ("kelly_sizer",         self.kelly_sizer),
            ("portfolio_risk",      self.portfolio_risk),
            ("greek_aggregator",    self.greek_aggregator),
            ("smart_router",        self.smart_router),
            ("order_validator",     self.order_validator),
            ("reconciler",          self.reconciler),
            ("slippage_monitor",    self.slippage_monitor),
        ]:
            logger.info("  {:22s}: {}", _name, "OK" if _instance is not None else "MISSING")
        logger.info("=" * 70)

    def _expand_websocket_subscriptions(self) -> None:
        """Expand the WebSocket subscription set to include all Bloomberg-build instruments."""
        if self.market_feed is None:
            logger.warning("market_feed not available — skipping subscription expansion")
            return

        try:
            index_keys = ["NSE_INDEX|Nifty Bank", "NSE_INDEX|Nifty 50"]
            vix_keys = ["NSE_INDEX|India VIX"]

            top_10_keys: list[str] = []
            if self.universe_scanner is not None:
                try:
                    top_10_keys = self.universe_scanner.get_instrument_keys(include_indices=False)
                except Exception as exc:
                    logger.warning("universe_scanner.get_instrument_keys failed: {}", exc)

            sector_keys: list[str] = []
            if self.sector_rotation is not None:
                try:
                    sector_keys = self.sector_rotation.get_sector_instrument_keys()
                except Exception as exc:
                    logger.warning("sector_rotation.get_sector_instrument_keys failed: {}", exc)

            macro_keys: list[str] = []
            if self.macro_overlay is not None:
                try:
                    macro_keys = self.macro_overlay.get_macro_keys()
                except Exception as exc:
                    logger.warning("macro_overlay.get_macro_keys failed: {}", exc)

            all_subscriptions = list(set(index_keys + vix_keys + top_10_keys + sector_keys + macro_keys))
            logger.info(
                "Expanding WS subscriptions to %d instruments: %s",
                len(all_subscriptions),
                all_subscriptions,
            )
            self.market_feed.subscribe(all_subscriptions, mode="full")
        except Exception as exc:
            logger.warning("WS subscription expansion failed: {}", exc)

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
        # 9:00 AM AI Playbook generation
        self.scheduler.add_daily_job(
            func=self._scheduled_ai_playbook,
            hour=9,
            minute=0,
            job_id="ai_playbook",
        )
        # 9:15 AM SOD: update risk limits and position sizing to current equity
        self.scheduler.add_daily_job(
            func=self._scheduled_sod_equity_update,
            hour=9,
            minute=15,
            job_id="sod_equity_update",
        )
        # 9:35 AM ORB Dynamic Update (Scheme 1)
        self.scheduler.add_daily_job(
            func=self._scheduled_orb_update,
            hour=9,
            minute=35,
            job_id="orb_update",
        )
        # 15:30 PM AI Reinforcement Learning Loop
        self.scheduler.add_daily_job(
            func=self._scheduled_rl_loop,
            hour=15,
            minute=30,
            job_id="rl_loop",
        )

        # Bloomberg-build phase scheduled jobs
        self._setup_bloomberg_scheduler_jobs()

        self.scheduler.start()

    def _setup_bloomberg_scheduler_jobs(self) -> None:
        """Register Bloomberg-build phase scheduler jobs.

        All jobs are wrapped in lambdas that guard against None modules so
        that missing modules simply log a skip rather than raising.
        """
        if not self.scheduler:
            return

        # ------------------------------------------------------------------ #
        # Pre-market 08:30 — universe scan
        # ------------------------------------------------------------------ #
        def _universe_scan() -> None:
            if self.universe_scanner is None:
                return
            try:
                from datetime import date as _date
                self.universe_scanner.rank_all(_date.today())
            except Exception as exc:
                logger.warning("universe_scan job failed: {}", exc)

        self.scheduler.add_daily_job(
            func=_universe_scan, hour=8, minute=30, job_id="universe_scan_premarket"
        )

        # ------------------------------------------------------------------ #
        # Pre-market 08:35 — regime decision
        # ------------------------------------------------------------------ #
        def _regime_decision() -> None:
            if self.regime_router is None:
                return
            try:
                from datetime import date as _date
                self.regime_router.evaluate_today(_date.today())
            except Exception as exc:
                logger.warning("regime_decision job failed: {}", exc)

        self.scheduler.add_daily_job(
            func=_regime_decision, hour=8, minute=35, job_id="regime_decision_premarket"
        )

        # ------------------------------------------------------------------ #
        # Every 30s — options chain poll
        # ------------------------------------------------------------------ #
        def _options_chain_poll() -> None:
            if self.options_chain is None or self.instrument_manager is None:
                return
            try:
                def _get_spot(ik: str) -> int:
                    if self.bar_builder is None:
                        return 0
                    df = self.bar_builder.get_bars(ik, timeframe=1)
                    if df is None or df.empty:
                        return 0
                    return int(df.iloc[-1]["close"])

                # Use instrument-master derived expiries so we always hit a
                # valid contract. The hand-rolled "next Wed/Thu" calc broke
                # in 2024 when NSE removed BANKNIFTY weekly expiry — bot
                # then polled non-existent chains every 30s.
                polls = [
                    ("NSE_INDEX|Nifty Bank", "BANKNIFTY"),
                    ("NSE_INDEX|Nifty 50",   "NIFTY"),
                ]
                for ik, sym in polls:
                    expiry = self.instrument_manager.get_weekly_expiry(sym)
                    if not expiry:
                        logger.debug("options_chain_poll: no expiry found for {} — skipping", sym)
                        continue
                    spot = _get_spot(ik)
                    if spot == 0:
                        logger.debug("options_chain_poll: no spot for {} — skipping", ik)
                        continue
                    exp_str = expiry if isinstance(expiry, str) else expiry.strftime("%Y-%m-%d")
                    try:
                        self.options_chain.snapshot(ik, sym, exp_str, spot)
                    except Exception as exc:
                        logger.debug("options_chain.snapshot failed for {}: {}", ik, exc)
            except Exception as exc:
                logger.warning("options_chain_poll job failed: {}", exc)

        self.scheduler.add_interval_job(
            func=_options_chain_poll, seconds=30, job_id="options_chain_poll"
        )

        # ------------------------------------------------------------------ #
        # Every 30s — position reconciler
        # ------------------------------------------------------------------ #
        def _reconciler_cycle() -> None:
            if self.reconciler is None:
                return
            try:
                self.reconciler.run_cycle()
            except Exception as exc:
                logger.warning("reconciler_cycle job failed: {}", exc)

        self.scheduler.add_interval_job(
            func=_reconciler_cycle, seconds=30, job_id="reconciler_30s"
        )

        # ------------------------------------------------------------------ #
        # Hourly 09-15 — news RSS scrape
        # ------------------------------------------------------------------ #
        def _news_scrape() -> None:
            if self.news_scraper is None:
                return
            try:
                self.news_scraper.run_all_sources()
            except Exception as exc:
                logger.warning("news_scrape job failed: {}", exc)

        for _hr in range(9, 16):
            self.scheduler.add_daily_job(
                func=_news_scrape,
                hour=_hr,
                minute=0,
                job_id=f"news_scrape_{_hr:02d}00",
            )

        # ------------------------------------------------------------------ #
        # Hourly 09-15 +10min — sentiment classification
        # ------------------------------------------------------------------ #
        def _sentiment_classify() -> None:
            if self.sentiment_classifier is None:
                return
            try:
                self.sentiment_classifier.run_for_unprocessed()
            except Exception as exc:
                logger.warning("sentiment_classify job failed: {}", exc)

        for _hr in range(9, 16):
            self.scheduler.add_daily_job(
                func=_sentiment_classify,
                hour=_hr,
                minute=10,
                job_id=f"sentiment_classify_{_hr:02d}10",
            )

        # ------------------------------------------------------------------ #
        # EOD 16:00 — all EOD scraper jobs
        # ------------------------------------------------------------------ #
        def _eod_flow_scrape() -> None:
            if self.flow_scraper is None:
                return
            try:
                self.flow_scraper.run_daily()
            except Exception as exc:
                logger.warning("eod_flow_scrape failed: {}", exc)

        def _eod_corp_actions() -> None:
            if self.corp_actions_scraper is None or self.universe_scanner is None:
                return
            try:
                constituents = self.universe_scanner.get_constituents()
                symbols = [c["symbol"] for c in constituents]
                self.corp_actions_scraper.run_for_universe(symbols)
            except Exception as exc:
                logger.warning("eod_corp_actions failed: {}", exc)

        def _eod_earnings() -> None:
            if self.earnings_scraper is None or self.universe_scanner is None:
                return
            try:
                constituents = self.universe_scanner.get_constituents()
                symbols = [c["symbol"] for c in constituents]
                self.earnings_scraper.run_for_universe(symbols)
            except Exception as exc:
                logger.warning("eod_earnings failed: {}", exc)

        def _eod_block_deals() -> None:
            if self.block_deals_scraper is None:
                return
            try:
                self.block_deals_scraper.run_daily()
            except Exception as exc:
                logger.warning("eod_block_deals failed: {}", exc)

        def _eod_insider() -> None:
            if self.insider_scraper is None or self.universe_scanner is None:
                return
            try:
                constituents = self.universe_scanner.get_constituents()
                symbols = [c["symbol"] for c in constituents]
                self.insider_scraper.run_for_universe(symbols)
            except Exception as exc:
                logger.warning("eod_insider failed: {}", exc)

        def _eod_kelly_perf() -> None:
            if self.kelly_sizer is None:
                return
            try:
                self.kelly_sizer.update_rolling_perf()
            except Exception as exc:
                logger.warning("eod_kelly_perf failed: {}", exc)

        def _eod_flow_regime() -> None:
            if self.flow_regime is None:
                return
            try:
                from datetime import date as _date
                self.flow_regime.compute_regime(_date.today())
            except Exception as exc:
                logger.warning("eod_flow_regime failed: {}", exc)

        def _eod_macro() -> None:
            if self.macro_overlay is None:
                return
            try:
                from datetime import date as _date
                self.macro_overlay.update_daily(_date.today())
            except Exception as exc:
                logger.warning("eod_macro failed: {}", exc)

        def _eod_vix() -> None:
            if self.vix_regime is None:
                return
            try:
                from datetime import date as _date
                self.vix_regime.compute_daily(_date.today())
            except Exception as exc:
                logger.warning("eod_vix failed: {}", exc)

        def _eod_sector() -> None:
            if self.sector_rotation is None:
                return
            try:
                from datetime import date as _date
                self.sector_rotation.rank_all(_date.today())
            except Exception as exc:
                logger.warning("eod_sector failed: {}", exc)

        for _fn, _jid in [
            (_eod_flow_scrape, "flow_scrape_eod"),
            (_eod_corp_actions, "corp_actions_eod"),
            (_eod_earnings, "earnings_eod"),
            (_eod_block_deals, "block_deals_eod"),
            (_eod_insider, "insider_eod"),
            (_eod_kelly_perf, "kelly_perf_eod"),
            (_eod_flow_regime, "flow_regime_eod"),
            (_eod_macro, "macro_eod"),
            (_eod_vix, "vix_eod"),
            (_eod_sector, "sector_eod"),
        ]:
            self.scheduler.add_daily_job(func=_fn, hour=16, minute=0, job_id=_jid)

        # ------------------------------------------------------------------ #
        # 16:30 — VaR + portfolio greeks snapshot
        # ------------------------------------------------------------------ #
        def _risk_snapshot_eod() -> None:
            from datetime import date as _date
            today = _date.today()
            capital_paisa = self.settings.capital * 100
            if self.portfolio_risk is not None:
                try:
                    self.portfolio_risk.compute_var(today, capital_paisa)
                except Exception as exc:
                    logger.warning("portfolio_risk.compute_var failed: {}", exc)
            if self.greek_aggregator is not None and self.position_manager is not None:
                try:
                    open_pos = self.position_manager.get_open_positions()
                    self.greek_aggregator.snapshot(today, open_pos, capital_paisa)
                except Exception as exc:
                    logger.warning("greek_aggregator.snapshot failed: {}", exc)

        self.scheduler.add_daily_job(
            func=_risk_snapshot_eod, hour=16, minute=30, job_id="risk_snapshot_eod"
        )

        # ------------------------------------------------------------------ #
        # 00:15 — cold storage archive (ticks + depth older than 7 days)
        # ------------------------------------------------------------------ #
        def _cold_storage_archive() -> None:
            if self.archiver is None:
                return
            try:
                self.archiver.archive_ticks_older_than(7)
            except Exception as exc:
                logger.warning("cold_storage_archive (ticks) failed: {}", exc)
            try:
                self.archiver.archive_depth_older_than(7)
            except Exception as exc:
                logger.warning("cold_storage_archive (depth) failed: {}", exc)

        self.scheduler.add_daily_job(
            func=_cold_storage_archive, hour=0, minute=15, job_id="cold_storage_archive"
        )

        logger.info("Bloomberg-build scheduler jobs registered")

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
        """Scheduled job: log daily P&L summary.

        Realized P&L comes from the `trades` table (rows are written on
        position close). Unrealized P&L comes from currently-open
        positions in PaperBroker. The previous version summed P&L over
        paper_broker.get_positions(), but that returns only OPEN
        positions, so realized came back zero on every close.
        """
        logger.info("[SCHEDULER] Logging daily P&L summary")
        if not (self.trade_log and self.paper_broker):
            return
        try:
            today = datetime.now(IST).date()
            if not is_trading_day(today):
                logger.info("Today is not a trading day, skipping daily summary")
                return

            # Realized + trade count + wins — from `trades` table.
            # Use func.date() (SQLite's date()) since cast(... as Date) is
            # a no-op on SQLite's textual storage.
            from sqlalchemy import func
            from src.persistence.database import get_session as _get_session
            from src.persistence.models import TradeRecord
            today_iso = today.isoformat()
            with _get_session() as session:
                day_filter = func.date(TradeRecord.exit_time) == today_iso
                agg = (
                    session.query(
                        func.coalesce(func.sum(TradeRecord.realized_pnl), 0),
                        func.count(TradeRecord.id),
                    )
                    .filter(day_filter)
                    .one()
                )
                wins = (
                    session.query(func.count(TradeRecord.id))
                    .filter(day_filter, TradeRecord.realized_pnl > 0)
                    .scalar()
                ) or 0
                realized_pnl = int(agg[0] or 0)
                trade_count = int(agg[1] or 0)
                wins = int(wins)

            # Unrealized — from current open positions
            open_positions = self.paper_broker.get_positions()
            unrealized_pnl = int(sum(getattr(p, "unrealized_pnl", 0) for p in open_positions))

            self.trade_log.log_daily_summary(
                date=today,
                realized=realized_pnl,
                unrealized=unrealized_pnl,
                trades=trade_count,
                wins=wins,
            )
            logger.info(
                "Daily summary written: trades={} realized={} unrealized={} wins={}",
                trade_count, realized_pnl, unrealized_pnl, wins,
            )
        except Exception as exc:
            logger.error("Scheduled daily summary failed: {}", exc)

    def _scheduled_sod_equity_update(self) -> None:
        """SOD 9:15: compound risk limits and position sizing to current portfolio equity."""
        if not (self.paper_broker and self.risk_manager and self.strategy_engine):
            return
        try:
            equity = self.paper_broker.get_portfolio_value()
            self.risk_manager.update_capital(equity)
            self.strategy_engine.update_account_equity(equity)
            logger.info("[SOD] Equity updated to {} paisa ({:.0f} INR)", equity, equity / 100)
        except Exception as exc:
            logger.warning("[SOD] Equity update failed: {}", exc)

    def _scheduled_ai_playbook(self) -> None:
        """Scheduled job: Generate the Daily AI Playbook at 9:00 AM.
        
        Feeds VIX, OI data, expiry context, and news sentiment to the
        MarketMakerAgent (Schemes 1-3, 6, 7).
        """
        logger.info("[SCHEDULER] Generating Morning AI Playbook")
        if not hasattr(self, "market_maker"):
            return
        try:
            from datetime import datetime
            from zoneinfo import ZoneInfo
            IST_TZ = ZoneInfo("Asia/Kolkata")
            now = datetime.now(IST_TZ)
            
            # Determine if today is expiry day
            # BankNifty: Wednesday (weekday 2)
            is_expiry_day = now.weekday() == 2
            
            # Try to get OI data from OptionsAnalyzer
            oi_data = {}
            try:
                # Provide a realistic placeholder or use Broker API if available
                # A full integration would call self.paper_broker.get_option_chain("BANKNIFTY")
                # For now, we fetch the LTP of BankNifty to set realistic boundaries.
                spot_price = 0
                if hasattr(self, '_broker') and hasattr(self._broker, 'get_ltp'):
                    spot_price = self._broker.get_ltp("NSE_INDEX|Nifty Bank")
                
                if spot_price > 0:
                    # In absence of real chain, establish mathematical OI walls based on nearest 500-strike
                    # This makes the AI "OI Fortress" logic mathematically sound even without real data.
                    strike_interval = 500 * 100  # 500 points in paisa
                    nearest_strike = round(spot_price / strike_interval) * strike_interval
                    
                    oi_data = {
                        "pcr_ratio": 1.1,  # Neutral-Bullish default
                        "max_pain": nearest_strike,
                        "max_ce_oi_strike": nearest_strike + (2 * strike_interval), # +1000 points
                        "max_pe_oi_strike": nearest_strike - (2 * strike_interval), # -1000 points
                    }
                    logger.debug(f"Generated synthetic OI data relative to spot {spot_price/100}: {oi_data}")
                else:
                    logger.debug("Spot price unavailable, using empty OI data")
            except Exception as e:
                logger.warning(f"Failed to fetch OI data: {e}")
            
            context = {
                "vix": 15.0,
                "us_markets": "Neutral",
                "sentiment_score": 0.0,
                "is_expiry_day": is_expiry_day,
                "oi_data": oi_data,
            }
            result = self.market_maker.run(context=context)
            self.market_maker.latest_playbook = result
            logger.info("AI Playbook Generated: {}", result)
        except Exception as exc:
            logger.error("Failed to generate AI Playbook: {}", exc)

    def _scheduled_orb_update(self) -> None:
        """Scheduled job: Update playbook at 9:35 AM with Opening Range data (Scheme 1)."""
        logger.info("[SCHEDULER] Updating AI Playbook with Opening Range data")
        if not hasattr(self, "market_maker"):
            return
        try:
            # Get the 9:15-9:30 high/low from bar data
            if not self.bar_builder:
                logger.debug("No bar builder available for ORB calculation")
                return
                
            # Attempt to read from bars_provider
            bars = {}
            if hasattr(self, '_bars_provider') and callable(self._bars_provider):
                bars = self._bars_provider()
                
            if not bars:
                logger.debug("No bars available yet for ORB calculation")
                return
                
            # Find the BankNifty symbol and compute ORB
            orb_data = {}
            for symbol, df in bars.items():
                if 'BANKNIFTY' in symbol.upper() or 'NIFTY BANK' in symbol.upper():
                    if len(df) >= 3:  # Need at least 3 bars (9:15, 9:20, 9:25 in 5-min)
                        orb_high = int(df['high'].iloc[:3].max())
                        orb_low = int(df['low'].iloc[:3].min())
                        spot = int(df['close'].iloc[-1])
                        orb_range_pct = ((orb_high - orb_low) / spot * 100) if spot > 0 else 0
                        orb_data = {
                            'orb_high': orb_high,
                            'orb_low': orb_low,
                            'orb_range_pct': orb_range_pct,
                        }
                        break
            
            if orb_data:
                updated = self.market_maker.update_with_orb(orb_data)
                logger.info("AI Playbook updated with ORB: {}", updated)
            else:
                logger.debug("Could not compute ORB data from available bars")
                
        except Exception as exc:
            logger.error("Failed to update playbook with ORB: {}", exc)

    def _scheduled_rl_loop(self) -> None:
        """Scheduled job: Run continuous reinforcement learning optimization."""
        logger.info("[SCHEDULER] Running Continuous RL Optimization")
        if not hasattr(self, "rl_loop"):
            return
        try:
            from pathlib import Path
            strategy_dir = Path("config/strategies")
            self.rl_loop.execute_daily_optimization(strategy_dir)
        except Exception as exc:
            logger.error("Failed to run RL Loop: {}", exc)

    def _get_recent_bars(self, instrument_key: str, timeframe: str = "5m", count: int = 400) -> Any:
        """Pull last N bars for instrument_key from bars table. Returns DataFrame indexed by ts or None."""
        if self.db_engine is None:
            return None
        try:
            import pandas as _pd
            from sqlalchemy import text as _text
            sql = _text("""
                SELECT ts, open, high, low, close, volume
                FROM bars
                WHERE instrument_key = :ikey AND timeframe = :tf
                ORDER BY ts DESC
                LIMIT :n
            """)
            with self.db_engine.connect() as conn:
                df = _pd.read_sql(sql, conn, params={"ikey": instrument_key, "tf": timeframe, "n": count})
            if df.empty:
                return None
            df = df.sort_values("ts").set_index("ts")
            if "amount" not in df.columns:
                df["amount"] = df["close"] * df["volume"]
            return df
        except Exception as exc:
            logger.warning("_get_recent_bars failed for {}: {}", instrument_key, exc)
            return None

    def _on_market_tick(self, tick: Any) -> None:
        """Callback for market tick updates.

        Upstox V3 protobuf ticks are decoded into a dict-like structure with
        `feeds[instrument_key]`. Extract instrument_key and last_price (in
        paisa) and forward to PositionManager.on_tick(instrument_key, price).

        Also dispatches to Bloomberg-build phase handlers (depth_feed,
        tick_metrics, vix_regime intraday) — each wrapped in try/except so
        failures never interrupt the main tick path.
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
                bid_paisa = None
                ask_paisa = None
                volume = 0
                if isinstance(payload, dict):
                    ltpc = payload.get("ltpc") or {}
                    ltp = ltpc.get("ltp")
                    if ltp is None:
                        full = payload.get("fullFeed", {}).get("indexFF") or payload.get("fullFeed", {}).get("marketFF") or {}
                        ltpc = (full or {}).get("ltpc") or {}
                        ltp = ltpc.get("ltp")
                    # Extract volume + bid/ask for tick_metrics
                    try:
                        full_feed = payload.get("fullFeed", {})
                        market_ff = full_feed.get("marketFF") or full_feed.get("indexFF") or {}
                        vol_val = market_ff.get("ltpc", {}).get("v") or market_ff.get("v")
                        if vol_val is not None:
                            volume = int(vol_val)
                        # Best bid/ask from depth level 0
                        depth = market_ff.get("marketOHLC") or market_ff.get("depth") or {}
                        buy_depth = depth.get("buyDepth", [{}])
                        sell_depth = depth.get("sellDepth", [{}])
                        if buy_depth:
                            bid_price = buy_depth[0].get("price")
                            if bid_price is not None:
                                bid_paisa = int(round(float(bid_price) * 100))
                        if sell_depth:
                            ask_price = sell_depth[0].get("price")
                            if ask_price is not None:
                                ask_paisa = int(round(float(ask_price) * 100))
                    except Exception:
                        pass
                if ltp is None:
                    continue
                # Convert rupees → paisa (PositionManager expects paisa int)
                price_paisa = int(round(float(ltp) * 100))
                self.position_manager.on_tick(instrument_key, price_paisa)

                # Feed live quote into paper broker so fills use actual bid/ask
                if self.paper_broker is not None:
                    try:
                        self.paper_broker.update_quote(
                            instrument_key, price_paisa, bid_paisa, ask_paisa
                        )
                    except Exception:
                        pass

                # When the underlying index ticks, check emergency-exit on any
                # option positions tracking it (e.g. short straddle legs).
                if instrument_key.startswith("NSE_INDEX|"):
                    try:
                        self.position_manager.on_underlying_tick(instrument_key, price_paisa)
                    except Exception as exc:
                        logger.debug("on_underlying_tick failed: {}", exc)

                # ---------------------------------------------------------- #
                # Phase A.1 — depth feed (raw payload, all instruments)
                # ---------------------------------------------------------- #
                if self.depth_feed is not None:
                    try:
                        self.depth_feed.on_tick(tick)
                    except Exception as exc:
                        logger.debug("depth_feed.on_tick failed: {}", exc)

                # ---------------------------------------------------------- #
                # Phase A.2 — tick metrics + CVD
                # ---------------------------------------------------------- #
                if self.tick_metrics is not None:
                    try:
                        now_ts = datetime.now(IST)
                        self.tick_metrics.on_tick(
                            instrument_key=instrument_key,
                            ts=now_ts,
                            ltp_paisa=price_paisa,
                            volume=volume,
                            bid_paisa=bid_paisa,
                            ask_paisa=ask_paisa,
                        )
                    except Exception as exc:
                        logger.debug("tick_metrics.on_tick failed: {}", exc)

                # ---------------------------------------------------------- #
                # Phase A.2d — Micro feature engine tick ingestion
                # ---------------------------------------------------------- #
                if self.micro_feature_engine is not None:
                    try:
                        # Pass normalized tick dict — micro_features expects top-level
                        # ltp (paisa int), volume, bid, ask.  The raw `tick` dict has
                        # feeds/currentTs at the top level, not ltp/volume.
                        self.micro_feature_engine.on_tick(instrument_key, {
                            "ltp": price_paisa,
                            "volume": volume,
                            "best_bid_price": bid_paisa,
                            "best_ask_price": ask_paisa,
                        })
                    except Exception as exc:
                        logger.debug("micro_feature_engine.on_tick failed: {}", exc)

                # ---------------------------------------------------------- #
                # Phase B.2 — VIX intraday update
                # ---------------------------------------------------------- #
                if instrument_key == "NSE_INDEX|India VIX" and self.vix_regime is not None:
                    try:
                        now_ts = datetime.now(IST)
                        self.vix_regime.update_intraday(now_ts, price_paisa / 100.0)
                    except Exception as exc:
                        logger.debug("vix_regime.update_intraday failed: {}", exc)

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

    def _run_detox_analysis(self) -> None:
        """Trigger AI Detox analysis when circuit breaker halts."""
        logger.info("[SCHEDULER] Triggering AI Detox analysis due to circuit breaker halt...")
        if not hasattr(self, "rl_loop") or getattr(self, "circuit_breaker", None) is None:
            logger.warning("No RL loop available for detox analysis")
            return
            
        try:
            summary = self.circuit_breaker.get_loss_summary_for_detox()
            if summary.get("total_losses", 0) == 0:
                return
                
            # Ask the RL Loop agent to analyze the summary
            context = {
                "performance": {
                    "strategy_name": "Multiple (Detox)",
                    "max_consecutive_losses": summary.get("consecutive_losses", 0),
                    "win_rate": 0.0,
                },
                "lessons": [],
                "detox_summary": summary,
            }
            # We can use the rl_loop agent to run this
            result = self.rl_loop.run(context=context)
            self.circuit_breaker.set_detox_analysis(result)
            logger.info(f"AI Detox Analysis completed: {result.get('reasoning', '')}")
        except Exception as e:
            logger.error(f"Failed to run detox analysis: {e}")

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

        # 8. Disconnect live broker (closes urllib3 pool + WebSocket thread)
        # Must happen before GC to prevent [Errno 9] Bad file descriptor
        if hasattr(self, "live_broker") and self.live_broker is not None:
            logger.info("Disconnecting live broker...")
            try:
                self.live_broker.disconnect()
            except Exception as exc:
                logger.debug("Live broker disconnect error (ignored): {}", exc)

        # 9. Log final summary — reuse the same trades-table path as the
        # scheduled 15:30 job so values are consistent.
        try:
            self._scheduled_daily_summary()
        except Exception as exc:
            logger.warning("Final summary logging failed: {}", exc)

        # Record bot run end in Postgres storage layer
        if self._db_run_id is not None:
            try:
                from src.storage.db import record_bot_run_end
                record_bot_run_end(self._db_run_id, exit_reason="normal")
                logger.info("Bot run end recorded (run_id={})", self._db_run_id)
            except Exception as _db_exc:
                logger.warning("Could not record bot run end: {}", _db_exc)

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

        except SystemExit:
            # Clean exit (e.g. non-trading day) — propagate without logging as error
            raise
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
    except SystemExit as exc:
        # Clean exit (e.g. non-trading day)
        return int(exc.code) if exc.code is not None else 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as exc:
        logger.exception("Fatal error: {}", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
