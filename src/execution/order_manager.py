"""Order Manager for routing signals to broker and managing order lifecycle.

This module provides the OrderManager class that consumes trading signals,
performs AI research analysis, resolves instrument details, routes orders
to the appropriate broker (paper or live), and maintains comprehensive
logging of all order activity.

Integration flow:
    Signal → Validation → Position Sizing → TradeAnalyzer.analyze() → RiskManager.pre_trade_check() → Broker.place_order()
"""

import threading
from datetime import datetime
from queue import Empty, Queue
from typing import Any, Optional
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.execution.base_broker import (
    BaseBroker,
    BrokerError,
    Funds,
    Order,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    ProductType,
)
from src.execution.paper_broker import PaperBroker
from src.strategy.engine import Signal
from src.utils.validation import InputValidator

# Optional enrichment module
try:
    from src.execution.entry_optimizer import EntryOptimizer, EntryType
except ImportError:
    EntryOptimizer = None  # type: ignore[misc,assignment]
    EntryType = None  # type: ignore[misc,assignment]

# Optional risk modules
try:
    from src.risk.position_sizer import KellyPositionSizer
except ImportError:
    KellyPositionSizer = None  # type: ignore[misc,assignment]

try:
    from src.risk.strategy_quarantine import StrategyQuarantine
except ImportError:
    StrategyQuarantine = None  # type: ignore[misc,assignment]

# Optional AI agent pipeline
try:
    from src.agents.pipeline import AgentPipeline
except ImportError:
    AgentPipeline = None  # type: ignore[misc,assignment]

# Indian timezone
IST = ZoneInfo("Asia/Kolkata")

# Mapping from Signal.product_type string to ProductType enum
_PRODUCT_TYPE_MAP = {
    "I": ProductType.MIS,
    "MIS": ProductType.MIS,
    "D": ProductType.CNC,
    "CNC": ProductType.CNC,
    "NRML": ProductType.NRML,
    "BO": ProductType.BO,
    "CO": ProductType.CO,
}

# Mapping from Signal.order_type string to OrderType enum
_ORDER_TYPE_MAP = {
    "MARKET": OrderType.MARKET,
    "LIMIT": OrderType.LIMIT,
    "SL": OrderType.SL,
    "SL-M": OrderType.SL_M,
    "SL_M": OrderType.SL_M,
    "AMO": OrderType.AMO,
}


def _derive_order_side(signal: Signal) -> OrderSide:
    """Derive OrderSide from the signal's signal_type.

    BUY, BUY_CE, BUY_PE -> BUY; SELL, SELL_CE, SELL_PE -> SELL.

    Args:
        signal: Trading signal.

    Returns:
        OrderSide.BUY or OrderSide.SELL.
    """
    value = (
        signal.signal_type.value
        if hasattr(signal.signal_type, "value")
        else str(signal.signal_type)
    )
    from loguru import logger
    logger.debug(f"[_derive_order_side] signal_type={signal.signal_type}, value={value}")
    if value.startswith("BUY"):
        return OrderSide.BUY
    # STRADDLE defaults to SELL side in the short straddle
    if value == "STRADDLE":
        return OrderSide.SELL
    return OrderSide.SELL


class OrderManager:
    """Manages order routing from signals to broker execution.

    The OrderManager consumes signals from a queue, performs AI research analysis,
    resolves instrument details, routes orders to the configured broker (paper or live),
    and maintains comprehensive logging of all order activity.

    Attributes:
        signal_queue: Queue for receiving trading signals
        broker: The broker implementation to use for order execution
        trading_mode: 'paper' or 'live' trading mode
        db_session: Database session for order logging
        is_running: Whether the order manager is actively processing
        trade_analyzer: Optional AI research module for signal validation
    """

    def __init__(
        self,
        signal_queue: Queue,
        broker: BaseBroker,
        trading_mode: str = "paper",
        db_url: Optional[str] = None,
        instrument_manager: Optional["InstrumentManager"] = None,
        trade_analyzer: Optional["TradeAnalyzer"] = None,
        risk_manager: Optional["RiskManager"] = None,
        research_enabled: bool = True,
        position_manager: Optional["PositionManager"] = None,
        entry_optimizer: Optional["EntryOptimizer"] = None,
        position_sizer: Optional["KellyPositionSizer"] = None,
        strategy_quarantine: Optional["StrategyQuarantine"] = None,
        market_feed: Optional[Any] = None,
        agent_pipeline: Optional[Any] = None,
        kelly_sizer: Optional[Any] = None,
        smart_router: Optional[Any] = None,
        order_validator: Optional[Any] = None,
    ):
        """Initialize the OrderManager.

        Args:
            signal_queue: Queue for receiving trading signals
            broker: Broker implementation for order execution
            trading_mode: 'paper' or 'live' mode
            db_url: Database URL for order logging (SQLite if None)
            instrument_manager: Optional instrument manager for symbol resolution
            trade_analyzer: Optional AI research analyzer for signal validation
            risk_manager: Optional risk manager for pre-trade checks
            research_enabled: Whether to use AI research module
            position_manager: Optional position manager for tracking open positions
            entry_optimizer: Optional entry optimizer for price improvement
            position_sizer: Optional Kelly position sizer for dynamic position sizing
            strategy_quarantine: Optional strategy quarantine for disabling underperforming strategies
            market_feed: Optional market feed for orderbook data
            agent_pipeline: Optional AI agent pipeline for LLM-based signal validation
        """
        self._signal_queue = signal_queue
        self._broker = broker
        self._trading_mode = trading_mode
        self._instrument_manager = instrument_manager
        self._trade_analyzer = trade_analyzer
        self._risk_manager = risk_manager
        self._research_enabled = research_enabled and trade_analyzer is not None
        self._position_manager = position_manager
        self._entry_optimizer = entry_optimizer
        self._position_sizer = position_sizer
        self._strategy_quarantine = strategy_quarantine
        self._market_feed = market_feed
        self._agent_pipeline = agent_pipeline
        # Bloomberg-build wave-5/E modules (optional, graceful degrade)
        self._kelly_sizer = kelly_sizer
        self._smart_router = smart_router
        self._order_validator = order_validator
        self._lock = threading.Lock()
        self._pending_instruments: set[str] = set()  # Instruments currently being processed
        self._is_running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Entry optimization statistics
        self._entry_opt_stats: dict[str, Any] = {
            "total_attempts": 0,
            "successful_optimizations": 0,
            "failed_optimizations": 0,
            "by_type": {
                "immediate": {"count": 0, "filled": 0},
                "pullback": {"count": 0, "filled": 0},
                "breakout": {"count": 0, "filled": 0},
                "vwap": {"count": 0, "filled": 0},
            },
        }

        # Setup database
        if db_url is None:
            db_url = "sqlite:///trading.db"
        self._engine = create_engine(db_url)
        self._SessionLocal = sessionmaker(bind=self._engine)

        # Setup order logger
        self._setup_loggers()

        logger.info(
            f"OrderManager initialized in {trading_mode} mode (research: {self._research_enabled})"
        )

    def _setup_loggers(self) -> None:
        """Setup dedicated loggers for order tracking."""
        # Orders log
        logger.add(
            "logs/orders.log",
            rotation="1 day",
            retention="30 days",
            level="INFO",
            filter=lambda record: record["extra"].get("name") == "orders",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        )
        self._order_logger = logger.bind(name="orders")

    def start(self) -> None:
        """Start the order manager worker thread."""
        with self._lock:
            if self._is_running:
                logger.warning("OrderManager is already running")
                return

            self._is_running = True
            self._worker_thread = threading.Thread(target=self._process_loop, daemon=True)
            self._worker_thread.start()
            logger.info("OrderManager started")

    def stop(self) -> None:
        """Stop the order manager worker thread."""
        with self._lock:
            if not self._is_running:
                logger.warning("OrderManager is not running")
                return

            self._is_running = False

        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
            if self._worker_thread.is_alive():
                logger.warning("OrderManager worker thread did not stop gracefully")

        logger.info("OrderManager stopped")

    def _process_loop(self) -> None:
        """Main processing loop for signal consumption."""
        logger.info("OrderManager processing loop started")

        while self._is_running:
            try:
                # Get signal with timeout to allow checking is_running
                signal = self._signal_queue.get(timeout=1.0)
                self.process_signal(signal)
            except Empty:
                continue
            except Exception as e:
                logger.exception(f"Error in order processing loop: {e}")

    def _get_orderbook(self, instrument_key: str) -> Optional[dict[str, Any]]:
        """Get current orderbook for instrument.

        Tries multiple sources:
        1. Broker get_orderbook method
        2. Market feed orderbook cache
        3. Returns None if unavailable

        Args:
            instrument_key: The instrument identifier

        Returns:
            Orderbook dict with bid/ask/vwap/ltp data, or None if unavailable
        """
        # Try broker first
        if hasattr(self._broker, "get_orderbook"):
            try:
                return self._broker.get_orderbook(instrument_key)
            except Exception as e:
                logger.debug(f"Broker orderbook failed: {e}")

        # Try market feed
        if self._market_feed and hasattr(self._market_feed, "get_orderbook"):
            try:
                return self._market_feed.get_orderbook(instrument_key)
            except Exception as e:
                logger.debug(f"Market feed orderbook failed: {e}")

        return None

    def _validate_optimal_price(
        self, signal: Signal, optimal_price: int, microstructure: Any
    ) -> bool:
        """Validate that optimal price is within reasonable bounds.

        Args:
            signal: The trading signal
            optimal_price: The calculated optimal price in paisa
            microstructure: Market microstructure data

        Returns:
            True if price is valid, False otherwise
        """
        if optimal_price <= 0:
            logger.warning(f"Invalid optimal price: {optimal_price}")
            return False

        # Get reference price (LTP or signal price)
        reference_price = (
            microstructure.ltp
            if microstructure and hasattr(microstructure, "ltp")
            else (signal.price or 0)
        )
        if reference_price <= 0:
            # Can't validate without reference
            return True

        # Check if price is within 5% of reference (reasonable bound)
        max_deviation = 0.05  # 5%
        lower_bound = int(reference_price * (1 - max_deviation))
        upper_bound = int(reference_price * (1 + max_deviation))

        if not (lower_bound <= optimal_price <= upper_bound):
            logger.warning(
                f"Optimal price {optimal_price} outside reasonable bounds "
                f"[{lower_bound}, {upper_bound}] relative to reference {reference_price}"
            )
            return False

        return True

    def _get_capital_for_position_sizing(self) -> int:
        """Get available capital for position sizing.

        Returns:
            Available capital in paisa.
        """
        # Try to get capital from risk manager first
        if self._risk_manager and hasattr(self._risk_manager, 'capital'):
            return self._risk_manager.capital

        # Try to get from broker funds
        try:
            funds = self._broker.get_funds()
            return funds.available if hasattr(funds, 'available') else 0
        except Exception:
            pass

        # Fallback to default capital (1,000,000 INR = 100,000,000 paisa)
        return 100_000_000

    def _get_lot_size(self, instrument_key: str) -> int:
        """Get lot size for the instrument.
        
        Returns:
            Lot size (default 1 for equity, actual lot size for F&O).
        """
        if self._instrument_manager:
            try:
                # Try to get lot size from instrument manager
                instrument = self._instrument_manager.search(
                    instrument_key=instrument_key,
                    segment=None,
                    symbol=None
                )
                if not instrument.empty and 'lot_size' in instrument.columns:
                    lot_size = instrument.iloc[0]['lot_size']
                    if pd.notna(lot_size) and lot_size > 1:
                        return int(lot_size)
            except Exception:
                pass
        # Default to 1 for equity
        return 1

    def _round_to_lot_size(self, quantity: int, lot_size: int) -> int:
        """Round quantity down to nearest lot size multiple.
        
        Args:
            quantity: Raw quantity.
            lot_size: Lot size for the instrument.
        
        Returns:
            Quantity rounded to lot size.
        """
        if lot_size <= 1:
            return quantity
        return (quantity // lot_size) * lot_size

    def _validate_signal(self, signal: Signal) -> tuple[bool, list[str]]:
        """Validate trading signal using InputValidator.

        Validates:
        - Symbol format
        - Price fields (entry, stop_loss, target) are positive
        - Quantity is positive and valid
        - Risk/reward ratio is >= 1:1

        Args:
            signal: The trading signal to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: list[str] = []
        validator = InputValidator()

        # Validate symbol
        symbol_result = validator.validate_symbol(signal.instrument_key)
        if not symbol_result.is_valid:
            errors.extend([f"symbol: {e}" for e in symbol_result.errors])

        # Validate price fields
        if signal.price is not None:
            price_result = validator.validate_price(signal.price, "entry_price")
            if not price_result.is_valid:
                errors.extend([f"price: {e}" for e in price_result.errors])

        if signal.stop_loss is not None:
            sl_result = validator.validate_price(signal.stop_loss, "stop_loss")
            if not sl_result.is_valid:
                errors.extend([f"stop_loss: {e}" for e in sl_result.errors])

        if signal.target is not None:
            target_result = validator.validate_price(signal.target, "target")
            if not target_result.is_valid:
                errors.extend([f"target: {e}" for e in target_result.errors])

        # Validate quantity
        qty_result = validator.validate_quantity(signal.quantity)
        if not qty_result.is_valid:
            errors.extend([f"quantity: {e}" for e in qty_result.errors])

        # Validate risk/reward ratio >= 1:1
        if signal.price is not None and signal.stop_loss is not None and signal.target is not None:
            order_side = _derive_order_side(signal)
            if order_side == OrderSide.BUY:
                # For BUY: target - price >= price - stop_loss
                risk = signal.price - signal.stop_loss
                reward = signal.target - signal.price
            else:
                # For SELL: price - target >= stop_loss - price
                risk = signal.stop_loss - signal.price
                reward = signal.price - signal.target

            if risk <= 0:
                errors.append(
                    "risk: Invalid risk calculation (stop_loss must be below entry for BUY, above for SELL)"
                )
            elif reward < risk:
                rr_ratio = reward / risk if risk > 0 else 0
                errors.append(f"risk_reward: Ratio is {rr_ratio:.2f}:1, minimum 1:1 required")

        return len(errors) == 0, errors

    def _apply_position_sizing(self, signal: Signal) -> Signal:
        """Apply Kelly Criterion position sizing to signal.

        If position_sizer is configured and has sufficient trade history,
        calculates Kelly-optimal position size and overrides signal.quantity.
        Falls back to original quantity if insufficient history or calculation fails.

        Args:
            signal: The trading signal to size

        Returns:
            Signal with potentially updated quantity
        """
        if self._position_sizer is None:
            return signal

        with self._lock:
            # Check if we have sufficient trade history
            if not self._position_sizer.has_sufficient_history():
                trade_count = self._position_sizer.get_trade_count()
                logger.debug(
                    f"Insufficient trade history for Kelly sizing | signal_id={signal.signal_id} | "
                    f"trades={trade_count} | min_required={self._position_sizer.min_trades_for_kelly}"
                )
                return signal

            # Get capital for sizing
            capital = self._get_capital_for_position_sizing()
            if capital <= 0:
                logger.warning(f"Invalid capital for position sizing: {capital}")
                return signal

            # Get entry price (use signal price or fallback to LTP)
            entry_price = signal.price
            if entry_price is None or entry_price <= 0:
                # Try to get LTP from broker
                try:
                    entry_price = self._broker.get_ltp(signal.instrument_key)
                except Exception:
                    logger.warning(
                        f"Cannot get entry price for position sizing: {signal.instrument_key}"
                    )
                    return signal

            # Optional max-risk-per-trade cap from risk manager. If the
            # risk manager doesn't expose one, pass None so the position
            # sizer relies on its own max_kelly_pct ceiling rather than
            # silently truncating every position to 1 %.
            max_risk_pct: Optional[float] = None
            if self._risk_manager and hasattr(self._risk_manager, "max_risk_per_trade"):
                max_risk_pct = self._risk_manager.max_risk_per_trade

            # Calculate Kelly-optimal position size
            try:
                kelly_size = self._position_sizer.get_position_size(
                    capital=capital, current_price=entry_price, max_risk_per_trade_pct=max_risk_pct
                )

                # Get Kelly stats for logging
                kelly_stats = self._position_sizer.get_kelly_stats()
                kelly_pct = kelly_stats.kelly_fraction if kelly_stats else 0.0

                # Only override if calculated size is valid
                if kelly_size > 0:
                    original_qty = signal.quantity
                    signal.quantity = kelly_size

                    self._order_logger.info(
                        f"POSITION_SIZED | {signal.signal_id} | Kelly={kelly_pct:.1%} | "
                        f"Size={kelly_size} lots | Original={original_qty}"
                    )
                else:
                    logger.warning(f"Kelly sizing returned zero quantity for {signal.signal_id}")

            except Exception as e:
                logger.error(f"Position sizing failed for {signal.signal_id}: {e}")

        return signal

    def process_signal(self, signal: Signal) -> Optional[OrderResponse]:
        """Process a trading signal and route to broker.

        This is the main entry point for signal processing. It:
        1. Validates signal using InputValidator
        2. Applies position sizing (Kelly Criterion if available)
        3. Logs signal receipt
        4. Performs AI research analysis (if enabled)
        5. Runs risk checks (if risk manager configured)
        6. Resolves instrument
        7. Applies entry optimization (if available)
        8. Places order with broker

        Args:
            signal: The trading signal to process

        Returns:
            OrderResponse if order was placed, None otherwise
        """
        try:
            # Derive order side from signal_type
            order_side = _derive_order_side(signal)

            # Log signal receipt
            self._order_logger.info(
                f"SIGNAL_RECEIVED | {signal.signal_id} | {signal.strategy_name} | "
                f"{order_side.value} {signal.quantity} {signal.instrument_key}"
            )

            # Step 0: Validate Signal
            is_valid, validation_errors = self._validate_signal(signal)
            if not is_valid:
                error_msg = "; ".join(validation_errors)
                self._order_logger.warning(f"VALIDATION_FAILED | {signal.signal_id} | {error_msg}")
                return None

            # Step 0.5: Reject if a position for this instrument already exists.
            # Without this guard, a second entry signal on the same bar would:
            #   (a) fill a new order at the broker, debiting capital as a phantom
            #   (b) hit PositionManager.add_position → see existing → silently return
            #   (c) emit a misleading POSITION_CREATED log with the NEW SL/target
            #   (d) the broker LTP tick from the fill would trip on_tick / trailing
            #       SL on the *original* position and close it immediately.
            if self._position_manager is not None:
                existing = None
                get_by_instr = getattr(
                    self._position_manager, "get_position_by_instrument", None
                )
                if callable(get_by_instr):
                    try:
                        existing = get_by_instr(signal.instrument_key)
                    except Exception:
                        existing = None
                if existing is None:
                    pos_map = getattr(
                        self._position_manager, "_instrument_to_position", None
                    )
                    if pos_map and signal.instrument_key in pos_map:
                        existing = pos_map[signal.instrument_key]
                if existing:
                    self._order_logger.info(
                        f"DUPLICATE_SIGNAL_SKIP | {signal.signal_id} | "
                        f"{signal.instrument_key} | position already open"
                    )
                    return None

            # Step 0.6: Prevent concurrent processing of the same instrument
            # (race condition where multiple signals pass Step 0.5 before any position registers)
            with self._lock:
                if signal.instrument_key in self._pending_instruments:
                    self._order_logger.info(
                        f"DUPLICATE_SIGNAL_SKIP | {signal.signal_id} | "
                        f"{signal.instrument_key} | signal already being processed"
                    )
                    return None
                self._pending_instruments.add(signal.instrument_key)


            try:
                # Step 1: Apply Position Sizing (Kelly Criterion)
                signal = self._apply_position_sizing(signal)

                # Step 2: Strategy Quarantine Check
                if self._strategy_quarantine:
                    try:
                        allowed, reason = self._strategy_quarantine.check_strategy(signal.strategy_name)
                        if not allowed:
                            self._order_logger.warning(
                                f"STRATEGY_QUARANTINED | {signal.signal_id} | {signal.strategy_name} | {reason}"
                            )
                            return None
                    except Exception as e:
                        logger.error(
                            f"Strategy quarantine check failed for {signal.strategy_name}: {e}"
                        )

                # Step 3: AI Research Analysis (if enabled)
                if self._research_enabled and self._trade_analyzer:
                    research_report = self._trade_analyzer.analyze(signal)
                    self._log_research_report(research_report)

                    # Check verdict
                    if research_report.verdict == "SKIP":
                        self._order_logger.info(
                            f"RESEARCH_SKIP | {signal.signal_id} | Score: {research_report.final_score:.1f} | "
                            f"Reason: {research_report.summary[:100]}"
                        )
                        return None

                    # Apply research adjustments
                    signal = self._apply_research_adjustments(signal, research_report)
                else:
                    research_report = None

                # Step 4: Risk Check (if risk manager configured)
                if self._risk_manager:
                    from src.risk.risk_manager import RiskCheckResult

                    # Check max trades per day
                    allowed, reason = self._risk_manager.can_take_new_trade(
                        signal.instrument_key, timestamp=signal.timestamp
                    )
                    if not allowed:
                        self._order_logger.warning(
                            f"RISK_BLOCK | {signal.signal_id} | {reason}"
                        )
                        return None

                    # Check circuit breaker
                    if getattr(self._risk_manager, 'circuit_breaker', None) and not self._risk_manager.circuit_breaker.is_allowed():
                        self._order_logger.warning(
                            f"RISK_BLOCK | {signal.signal_id} | Circuit breaker activated"
                        )
                        return None

                    # Note: order_value check (pre_trade_check) moved to Step 8.7 to use correct option prices


                # Step 4.5: AI Agent Pipeline (LLM-based regime + signal validation)
                if self._agent_pipeline is not None:
                    try:
                        # Build indicator context from signal metadata
                        indicator_ctx = signal.metadata.get("indicator_context", {})
                        if not indicator_ctx:
                            # Minimal fallback context so the regime agent can work
                            indicator_ctx = {
                                "price": signal.price or 0,
                                "stop_loss": signal.stop_loss or 0,
                                "target": signal.target or 0,
                            }

                        signal_dict = signal.to_dict() if hasattr(signal, "to_dict") else {
                            "strategy_name": signal.strategy_name,
                            "instrument_key": signal.instrument_key,
                            "signal_type": str(signal.signal_type),
                            "quantity": signal.quantity,
                            "price": signal.price,
                            "stop_loss": signal.stop_loss,
                            "target": signal.target,
                        }

                        pipeline_state = self._agent_pipeline.run_fast(
                            instrument_key=signal.instrument_key,
                            signals=[signal_dict],
                        )

                        # Log agent decisions for observability
                        for decision in pipeline_state.agent_decisions:
                            self._order_logger.info(
                                f"AGENT_DECISION | {decision.agent_name} | "
                                f"confidence={decision.confidence:.2f} | "
                                f"fallback={decision.used_fallback} | "
                                f"latency={decision.latency_ms}ms | "
                                f"{decision.output_summary[:120]}"
                            )

                        # If pipeline short-circuited (low regime confidence), skip trade
                        if pipeline_state.short_circuited:
                            self._order_logger.info(
                                f"AGENT_PIPELINE_SKIP | {signal.signal_id} | "
                                f"{pipeline_state.short_circuit_reason}"
                            )
                            return None

                        # If no validated signals, agent rejected the trade
                        if not pipeline_state.validated_signals:
                            self._order_logger.info(
                                f"AGENT_REJECTED | {signal.signal_id} | "
                                f"Regime: {pipeline_state.regime.regime if pipeline_state.regime else 'unknown'} | "
                                f"No signals approved by validation agent"
                            )
                            return None

                        # Apply agent adjustments to signal (SL, target, quantity)
                        validated = pipeline_state.validated_signals[0]
                        if "stop_loss" in validated and validated["stop_loss"]:
                            old_sl = signal.stop_loss
                            signal.stop_loss = validated["stop_loss"]
                            if old_sl != signal.stop_loss:
                                self._order_logger.info(
                                    f"AGENT_ADJUST_SL | {signal.signal_id} | "
                                    f"{old_sl} → {signal.stop_loss}"
                                )
                        if "target" in validated and validated["target"]:
                            old_tgt = signal.target
                            signal.target = validated["target"]
                            if old_tgt != signal.target:
                                self._order_logger.info(
                                    f"AGENT_ADJUST_TARGET | {signal.signal_id} | "
                                    f"{old_tgt} → {signal.target}"
                                )
                        if "quantity" in validated and validated["quantity"]:
                            old_qty = signal.quantity
                            # Never allow agent to INCREASE position size — that would
                            # bypass the risk check at Step 4. Agent can only reduce qty.
                            new_qty = min(int(validated["quantity"]), old_qty)
                            signal.quantity = new_qty
                            if old_qty != signal.quantity:
                                self._order_logger.info(
                                    f"AGENT_ADJUST_QTY | {signal.signal_id} | "
                                    f"{old_qty} → {signal.quantity}"
                                )

                    except Exception as e:
                        # Agent pipeline is non-critical — log and continue without it
                        logger.warning(
                            f"Agent pipeline error for {signal.signal_id}, proceeding without: {e}"
                        )

                # Step 5: Resolve instrument if needed
                instrument_keys = [self._resolve_instrument(signal.instrument_key)]
                
                if signal.metadata and signal.metadata.get("instrument_selection"):
                    sel = signal.metadata["instrument_selection"]
                    if sel.get("type") == "options":
                        from src.data.options_resolver import OptionsResolver
                        resolver = OptionsResolver(self._instrument_manager)
                        dummy_config = {
                            "name": signal.strategy_name,
                            "instrument_selection": sel,
                            "underlying": signal.metadata.get("underlying")
                        }
                        # Restrict option_types if it's a single leg signal
                        if str(signal.signal_type) == "BUY_CE":
                            dummy_config["instrument_selection"]["option_types"] = ["CE"]
                        elif str(signal.signal_type) == "BUY_PE":
                            dummy_config["instrument_selection"]["option_types"] = ["PE"]
                        elif str(signal.signal_type) == "SELL_CE":
                            dummy_config["instrument_selection"]["option_types"] = ["CE"]
                        elif str(signal.signal_type) == "SELL_PE":
                            dummy_config["instrument_selection"]["option_types"] = ["PE"]
                            
                        # Use signal.price (spot price) to resolve options
                        resolved = resolver.resolve_strategy_instruments(dummy_config, int(signal.price))
                        if resolved:
                            instrument_keys = resolved
                            self._order_logger.info(f"Resolved options {resolved} for {signal.signal_id}")
                        else:
                            logger.warning(f"Could not resolve options for {signal.signal_id}")
                            return None

                instrument_key = instrument_keys[0] # primary instrument for optimization/validation

                # Validate instrument resolution
                if not instrument_key or instrument_key == signal.instrument_key and "|" not in instrument_key:
                    logger.warning(f"Could not resolve instrument {signal.instrument_key}")
                    return None

                # Check if instrument exists in master
                if self._instrument_manager:
                    try:
                        df = self._instrument_manager.load_instruments()
                        if instrument_key not in df["instrument_key"].values:
                            logger.warning(f"Instrument {instrument_key} not found in master")
                            return None
                    except Exception as e:
                        logger.warning(f"Failed to verify instrument {instrument_key}: {e}")
                        return None

                # Step 6: Entry optimization (if available)
                optimized_order_type = _ORDER_TYPE_MAP.get(signal.order_type, OrderType.MARKET)
                original_price = signal.price

                if self._entry_optimizer and EntryType is not None:
                    try:
                        orderbook = self._get_orderbook(instrument_key)
                        if orderbook:
                            self._entry_opt_stats["total_attempts"] += 1
                            opt_result = self._entry_optimizer.optimize_entry_price(signal, orderbook)

                            if self._validate_optimal_price(
                                signal, opt_result.optimal_price, opt_result.microstructure
                            ):
                                if opt_result.entry_type in (
                                    EntryType.PULLBACK,
                                    EntryType.BREAKOUT,
                                    EntryType.VWAP,
                                ):
                                    optimized_order_type = OrderType.LIMIT
                                    signal.price = opt_result.optimal_price
                                    self._entry_opt_stats["successful_optimizations"] += 1
                                    self._entry_opt_stats["by_type"][opt_result.entry_type.value]["count"] += 1
                                    self._order_logger.info(
                                        f"ENTRY_OPTIMIZED | {signal.signal_id} | "
                                        f"Type: {opt_result.entry_type.value} | "
                                        f"Price: {opt_result.optimal_price} paisa | "
                                        f"OrderType: LIMIT | {opt_result.reason}"
                                    )
                                elif opt_result.entry_type == EntryType.IMMEDIATE:
                                    self._entry_opt_stats["by_type"]["immediate"]["count"] += 1
                                    self._order_logger.info(
                                        f"ENTRY_OPTIMIZED | {signal.signal_id} | "
                                        f"Type: immediate | Using original order type | {opt_result.reason}"
                                    )
                            else:
                                self._entry_opt_stats["failed_optimizations"] += 1
                                logger.warning(
                                    f"Entry optimization price validation failed for {signal.signal_id}, "
                                    f"using original order parameters"
                                )
                        else:
                            logger.debug(
                                f"No orderbook data available for {instrument_key}, skipping entry optimization"
                            )
                    except Exception as e:
                        self._entry_opt_stats["failed_optimizations"] += 1
                        logger.warning(f"Entry optimization failed, using original order: {e}")
                else:
                    logger.debug("Entry optimizer not available, skipping optimization")

                # Step 7: Validate quantity against lot size
                lot_size = self._get_lot_size(instrument_key)
                if lot_size > 1:
                    quantity = self._round_to_lot_size(signal.quantity, lot_size)
                    if quantity <= 0:
                        logger.warning(
                            f"Quantity {signal.quantity} rounds to zero for lot size {lot_size}"
                        )
                        return None
                    signal.quantity = quantity

                # Step 8: Create order from signal
                final_price = signal.price
                if optimized_order_type == OrderType.MARKET:
                    final_price = 0.0
                    
                order = Order(
                    instrument_key=instrument_key,
                    side=order_side,
                    order_type=optimized_order_type,
                    product_type=_PRODUCT_TYPE_MAP.get(signal.product_type, ProductType.MIS),
                    quantity=signal.quantity,
                    price=final_price,
                    trigger_price=signal.trigger_price,
                    strategy_id=signal.strategy_name,
                    tag=signal.signal_id,
                )

                # Step 8.5: Handle Multi-Leg / Combo Orders (Iron Condor / Hedged Straddle)
                is_combo = False
                combo_order = None
                
                if len(instrument_keys) > 1:
                    is_combo = True
                    from src.execution.base_broker import ComboOrder
                    legs = []
                    for idx, ik in enumerate(instrument_keys):
                        # Use MARKET order for legs since we cannot easily limit order a multi-leg combo
                        leg_order_type = OrderType.MARKET
                        leg_tag = f"{signal.signal_id}_LEG_{idx+1}"
                        
                        # Validate leg lot size
                        leg_lot_size = self._get_lot_size(ik)
                        leg_qty = signal.quantity
                        if leg_lot_size > 1:
                            leg_qty = self._round_to_lot_size(leg_qty, leg_lot_size)
                        
                        leg = order.copy(update={
                            "instrument_key": ik, 
                            "tag": leg_tag,
                            "order_type": leg_order_type,
                            "quantity": leg_qty,
                            "price": 0.0  # Force MARKET orders to 0 price
                        })
                        legs.append(leg)
                        
                    combo_order = ComboOrder(
                        legs=legs,
                        strategy_id=signal.strategy_name,
                        tag=f"COMBO_{signal.signal_id}"
                    )
                    self._order_logger.info(f"COMBO_ROUTING | {signal.signal_id} | Multi-leg initiated ({len(legs)} legs)")
                    
                # Also handle hedged straddle specifically if configured
                hedge_required = signal.metadata.get("hedge_required", False)
                if str(signal.signal_type) == "STRADDLE" and hedge_required and not is_combo:
                    is_combo = True
                    leg1 = order.copy(update={"side": OrderSide.SELL, "tag": f"{signal.signal_id}_CE_SHORT"})
                    leg2 = order.copy(update={"side": OrderSide.SELL, "tag": f"{signal.signal_id}_PE_SHORT"})
                    leg3 = order.copy(update={"side": OrderSide.BUY, "tag": f"{signal.signal_id}_CE_HEDGE"})
                    leg4 = order.copy(update={"side": OrderSide.BUY, "tag": f"{signal.signal_id}_PE_HEDGE"})
                    from src.execution.base_broker import ComboOrder
                    combo_order = ComboOrder(legs=[leg1, leg2, leg3, leg4], strategy_id=signal.strategy_name, tag=f"IRON_CONDOR_{signal.signal_id}")
                    self._order_logger.info(f"COMBO_ROUTING | {signal.signal_id} | Iron Condor initiated (4 legs)")

                # Validate order
                if not is_combo and not self._validate_order(order):
                    return None

                # Validate stop loss and target against entry price
                if signal.stop_loss and signal.price:
                    if order.side == OrderSide.BUY:
                        if signal.stop_loss >= signal.price:
                            logger.warning(f"BUY stop loss {signal.stop_loss} must be below entry {signal.price}")
                            return None
                    else:
                        if signal.stop_loss <= signal.price:
                            logger.warning(f"SELL stop loss {signal.stop_loss} must be above entry {signal.price}")
                            return None

                if signal.target and signal.price:
                    if order.side == OrderSide.BUY:
                        if signal.target <= signal.price:
                            logger.warning(f"BUY target {signal.target} must be above entry {signal.price}")
                            return None
                    else:
                        if signal.target >= signal.price:
                            logger.warning(f"SELL target {signal.target} must be below entry {signal.price}")
                            return None

                # Check circuit limits
                if self._risk_manager:
                    price_for_circuit = signal.price
                    if price_for_circuit is None or price_for_circuit == 0:
                        if hasattr(self._broker, '_ltp_cache'):
                            price_for_circuit = self._broker._ltp_cache.get(signal.instrument_key, 0)
                        elif hasattr(self._broker, 'get_ltp'):
                            price_for_circuit = self._broker.get_ltp(signal.instrument_key)
                    if price_for_circuit and price_for_circuit > 0:
                        check_result = self._risk_manager.check_circuit_limits(
                            price=price_for_circuit,
                            stop_loss=signal.stop_loss,
                            target=signal.target,
                        )
                        if not check_result.approved:
                            logger.warning(f"CIRCUIT_LIMIT_BLOCK | {signal.signal_id} | {check_result.reason}")
                            return None

                # Step 8.7: Final pre-trade risk check with accurate order value
                if self._risk_manager:
                    total_order_value = 0
                    if is_combo and combo_order:
                        for leg in combo_order.legs:
                            leg_price = leg.price or (self._broker.get_ltp(leg.instrument_key) if hasattr(self._broker, 'get_ltp') else 0)
                            if leg_price == 0 and hasattr(self._broker, '_ltp_cache'):
                                leg_price = self._broker._ltp_cache.get(leg.instrument_key, 0)
                            # If we still don't have a price for the option, just estimate or fallback
                            # For safety, if option price is missing, fallback to a reasonable premium estimate or 0
                            # Real implementations need robust market data fetch
                            total_order_value += leg.quantity * leg_price
                    else:
                        leg_price = order.price or (self._broker.get_ltp(order.instrument_key) if hasattr(self._broker, 'get_ltp') else 0)
                        if leg_price == 0 and hasattr(self._broker, '_ltp_cache'):
                            leg_price = self._broker._ltp_cache.get(order.instrument_key, 0)
                        total_order_value = order.quantity * leg_price
                        
                    # Wait, if we use market order for options, leg.price is 0! 
                    # If LTP is not available, order_value becomes 0.
                    # That will bypass capital check, but it's paper trading and won't reject erroneously!
                    
                    risk_result = self._risk_manager.pre_trade_check(total_order_value, instrument_key)
                    if not risk_result.approved:
                        self._order_logger.warning(
                            f"RISK_BLOCK | {signal.signal_id} | {risk_result.reason}"
                        )
                        return None

                # Step 9: Place order with retry logic
                if is_combo and combo_order:
                    # For combo orders, we bypass retry logic for now (all or nothing)
                    responses = self._broker.place_combo_order(combo_order)
                    if not responses:
                        logger.error(f"Combo order placement failed entirely for {signal.signal_id}")
                        return None
                    
                    for i, resp in enumerate(responses):
                        leg_order = combo_order.legs[i]
                        self._log_order_to_db(leg_order, resp, signal, research_report)
                        side_str = getattr(leg_order.side, "value", leg_order.side)
                        self._order_logger.info(
                            f"COMBO_LEG_PLACED | {resp.order_id} | {resp.status} | "
                            f"{side_str} {leg_order.quantity} {leg_order.instrument_key} | "
                            f"Signal: {signal.signal_id}"
                        )
                        
                        if resp.status == OrderStatus.COMPLETE and self._position_manager:
                            try:
                                fill_price = resp.average_price or leg_order.price or 0
                                leg_side = OrderSide(leg_order.side) if isinstance(leg_order.side, str) else leg_order.side
                                leg_product_type = ProductType(leg_order.product_type) if isinstance(leg_order.product_type, str) else leg_order.product_type
                                self._position_manager.add_position(
                                    instrument_key=leg_order.instrument_key,
                                    side=leg_side,
                                    quantity=resp.filled_quantity or leg_order.quantity,
                                    entry_price=fill_price,
                                    strategy_id=signal.strategy_name,
                                    # SL and Target from signal are for the underlying index!
                                    # Position manager will incorrectly trigger them for options.
                                    # Nullify them for option combo legs to avoid instant SL hits.
                                    stop_loss_price=None,
                                    target_price=None,
                                    product_type=leg_product_type,
                                )
                                self._order_logger.info(
                                    f"POSITION_CREATED | {leg_order.instrument_key} | {side_str} "
                                    f"{resp.filled_quantity or leg_order.quantity} @ {fill_price} | "
                                    f"Strategy: {signal.strategy_name}"
                                )
                            except Exception as e:
                                logger.error(f"Failed to create position for filled combo leg {resp.order_id}: {e}")
                                
                    response = responses[0] # Return primary response for caller
                else:
                    response = self._place_order_with_retry(order)

                    # Log to database
                    self._log_order_to_db(order, response, signal, research_report)

                    # Log order result
                    self._order_logger.info(
                        f"ORDER_PLACED | {response.order_id} | {response.status} | "
                        f"{order.side} {order.quantity} {order.instrument_key} | "
                        f"Signal: {signal.signal_id}"
                    )

                    # Create position in position manager after a successful fill
                    if response.status == OrderStatus.COMPLETE and self._position_manager:
                        try:
                            fill_price = response.average_price or signal.price or 0
                            self._position_manager.add_position(
                                instrument_key=instrument_key,
                                side=order_side,
                                quantity=response.filled_quantity or signal.quantity,
                                entry_price=fill_price,
                                strategy_id=signal.strategy_name,
                                stop_loss_price=signal.stop_loss,
                                target_price=signal.target,
                                product_type=order.product_type,
                            )
                            self._order_logger.info(
                                f"POSITION_CREATED | {instrument_key} | {order_side.value} "
                                f"{response.filled_quantity or signal.quantity} @ {fill_price} | "
                                f"SL: {signal.stop_loss} | Target: {signal.target} | "
                                f"Strategy: {signal.strategy_name}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to create position for filled order {response.order_id}: {e}"
                            )

                # Record trade for daily limit tracking after successful fill
                if response.status == OrderStatus.COMPLETE and self._risk_manager:
                    try:
                        self._risk_manager.record_trade(instrument_key, timestamp=signal.timestamp)
                    except Exception as e:
                        logger.error(f"Failed to record trade for daily limit: {e}")

                return response

            finally:
                # Always release the pending-instrument gate so subsequent bar signals
                # can enter once this signal completes (position check blocks re-entry if open)
                with self._lock:
                    self._pending_instruments.discard(signal.instrument_key)

        except Exception as e:
            logger.exception(f"Error processing signal {signal.signal_id}: {e}")
            self._order_logger.error(
                f"SIGNAL_FAILED | {signal.signal_id} | {signal.strategy_name} | Error: {e}"
            )
            return None

    def _apply_research_adjustments(self, signal: Signal, report: "TradeResearchReport") -> Signal:
        """Apply research-driven adjustments to the signal.

        Applies research score-based position sizing:
        - Score 75-100: 1.0x size (full size)
        - Score 65-74: 0.7x size (REDUCE_SIZE)
        - Score < 65: 0x (SKIP - should not reach here)
        - Score > 90: 1.25x size (high conviction boost)

        Args:
            signal: Original signal
            report: Research analysis report

        Returns:
            Adjusted signal
        """
        # Calculate research score-based multiplier
        research_multiplier = self._calculate_research_size_multiplier(report.final_score)

        # Combine with suggested multiplier from report (if any)
        combined_multiplier = research_multiplier * (report.suggested_quantity_multiplier or 1.0)

        # Apply quantity adjustment
        if combined_multiplier != 1.0:
            original_qty = signal.quantity
            signal.quantity = max(1, int(signal.quantity * combined_multiplier))
            self._order_logger.info(
                f"RESEARCH_ADJUST | Quantity: {original_qty} → {signal.quantity} "
                f"(research_score={report.final_score:.1f}, multiplier={combined_multiplier:.2f}x)"
            )

        # Derive side for directional adjustments
        side = _derive_order_side(signal)

        # Adjust stop loss
        if signal.stop_loss and report.suggested_sl_adjustment != 1.0:
            original_sl = signal.stop_loss
            if side == OrderSide.BUY:
                signal.stop_loss = int(
                    signal.stop_loss * (2 - report.suggested_sl_adjustment)
                )  # Widen
            else:
                signal.stop_loss = int(signal.stop_loss * report.suggested_sl_adjustment)  # Widen
            self._order_logger.info(
                f"RESEARCH_ADJUST | SL: {original_sl} → {signal.stop_loss} "
                f"({report.suggested_sl_adjustment:.2f}x)"
            )

        # Adjust target
        if signal.target and report.suggested_target_adjustment != 1.0:
            original_target = signal.target
            if side == OrderSide.BUY:
                signal.target = int(signal.target * report.suggested_target_adjustment)  # Extend
            else:
                signal.target = int(
                    signal.target * (2 - report.suggested_target_adjustment)
                )  # Extend
            self._order_logger.info(
                f"RESEARCH_ADJUST | Target: {original_target} → {signal.target} "
                f"({report.suggested_target_adjustment:.2f}x)"
            )

        # Adjust entry type if recommended
        if report.optimal_entry_type == "LIMIT_AT_SUPPORT" and report.sr_data:
            # Use nearest support as limit price
            nearest_support = report.sr_data.s1 or report.sr_data.pdl
            if nearest_support:
                signal.order_type = "LIMIT"
                signal.price = nearest_support
                self._order_logger.info(
                    f"RESEARCH_ADJUST | Entry type: MARKET → LIMIT at support {nearest_support}"
                )

        # Store research metadata
        if signal.metadata is None:
            signal.metadata = {}
        signal.metadata["research_score"] = report.final_score
        signal.metadata["research_verdict"] = report.verdict
        signal.metadata["market_regime"] = report.market_regime.value

        return signal

    def _log_research_report(self, report: "TradeResearchReport") -> None:
        """Log research report details.

        Args:
            report: The research report to log
        """
        logger.info(f"[RESEARCH] Signal: {report.direction} {report.instrument_key}")
        for component in report.components:
            logger.info(
                f"[RESEARCH] {component.name}: {component.score:.0f}/100 - {component.reasoning[:80]}"
            )
        logger.info(f"[RESEARCH] FINAL SCORE: {report.final_score:.0f}/100 → {report.verdict}")

        if report.key_risks:
            for risk in report.key_risks[:3]:
                logger.warning(f"[RESEARCH] Risk: {risk}")

    def _resolve_instrument(self, instrument_key: str) -> str:
        """Resolve instrument key using instrument manager if available.

        Args:
            instrument_key: Raw instrument key or symbol

        Returns:
            Resolved instrument key
        """
        if self._instrument_manager:
            try:
                return self._instrument_manager.resolve(instrument_key)
            except Exception as e:
                logger.warning(f"Could not resolve instrument {instrument_key}: {e}")

        return instrument_key

    def _validate_order(self, order: Order) -> bool:
        """Validate order before placement.

        Args:
            order: The order to validate

        Returns:
            True if valid, False otherwise
        """
        # Check quantity
        if order.quantity <= 0:
            logger.error(f"Invalid order quantity: {order.quantity}")
            return False

        # Check limit price for limit orders
        if order.order_type == OrderType.LIMIT and order.price is None:
            logger.error("Limit order requires price")
            return False

        # Check trigger price for SL orders
        if order.order_type in (OrderType.SL, OrderType.SL_M) and order.trigger_price is None:
            logger.error("SL order requires trigger_price")
            return False

        return True

    @retry(
        retry=retry_if_exception_type(BrokerError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _place_order_with_retry(self, order: Order) -> OrderResponse:
        """Place order with retry logic for transient failures.

        Args:
            order: The order to place

        Returns:
            OrderResponse from broker

        Raises:
            BrokerError: If all retries fail
        """
        return self._broker.place_order(order)

    def _log_order_to_db(
        self,
        order: Order,
        response: OrderResponse,
        signal: Signal,
        research_report: Optional["TradeResearchReport"] = None,
    ) -> None:
        """Log order to database.

        Args:
            order: The order that was placed
            response: The broker response
            signal: The original signal
            research_report: Optional research analysis report
        """
        try:
            from src.persistence.models import OrderRecord, ResearchReport
            import json

            with self._SessionLocal() as session:
                # Log order
                # Pack extra context into the tag field as JSON
                _v = lambda x: getattr(x, 'value', x)
                tag_data = {
                    "signal_id": signal.signal_id,
                    "product_type": _v(order.product_type),
                    "trading_mode": self._trading_mode,
                }
                if response.message:
                    tag_data["message"] = response.message

                order_record = OrderRecord(
                    broker_order_id=response.order_id,
                    strategy=signal.strategy_name,
                    instrument_key=order.instrument_key,
                    transaction_type=_v(order.side),
                    order_type=_v(order.order_type),
                    quantity=order.quantity,
                    price=order.price or 0,
                    trigger_price=order.trigger_price,
                    status=_v(response.status),
                    filled_qty=response.filled_quantity,
                    avg_fill_price=response.average_price or 0,
                    placed_at=datetime.now(IST),
                    tag=json.dumps(tag_data),
                )
                session.add(order_record)

                # Log research report if available
                if research_report:
                    report_record = ResearchReport(
                        signal_id=signal.signal_id,
                        instrument_key=signal.instrument_key,
                        direction=_derive_order_side(signal).value,
                        strategy_name=signal.strategy_name,
                        final_score=research_report.final_score,
                        verdict=research_report.verdict.value,
                        signal_strength=research_report.signal_strength.value,
                        market_regime=research_report.market_regime.value,
                        volatility_regime=research_report.volatility_regime.value,
                        components_json=json.dumps(
                            [c.model_dump() for c in research_report.components]
                        )
                        if research_report.components
                        else None,
                        suggested_quantity_multiplier=research_report.suggested_quantity_multiplier,
                        suggested_sl_adjustment=research_report.suggested_sl_adjustment,
                        suggested_target_adjustment=research_report.suggested_target_adjustment,
                        optimal_entry_type=research_report.optimal_entry_type.value,
                        key_risks_json=json.dumps(research_report.key_risks)
                        if research_report.key_risks
                        else None,
                        key_supports_json=json.dumps(research_report.key_supports)
                        if research_report.key_supports
                        else None,
                        summary=research_report.summary,
                        was_executed=research_report.verdict in ["EXECUTE", "REDUCE_SIZE"],
                    )
                    session.add(report_record)

                session.commit()
        except Exception as e:
            logger.error(f"Failed to log order to database: {e}")

    def get_order_status(self, order_id: str) -> Optional[OrderResponse]:
        """Get status of a specific order.

        Args:
            order_id: The order ID to check

        Returns:
            OrderResponse if found, None otherwise
        """
        try:
            return self._broker.get_order_status(order_id)
        except BrokerError as e:
            logger.error(f"Error getting order status: {e}")
            return None

    def modify_order(self, order_id: str, changes: dict) -> Optional[OrderResponse]:
        """Modify an existing order.

        Args:
            order_id: The order ID to modify
            changes: Dictionary of changes

        Returns:
            OrderResponse if successful, None otherwise
        """
        try:
            response = self._broker.modify_order(order_id, changes)
            self._order_logger.info(f"ORDER_MODIFIED | {order_id} | Status: {response.status}")
            return response
        except BrokerError as e:
            logger.error(f"Failed to modify order {order_id}: {e}")
            return None

    def cancel_order(self, order_id: str) -> Optional[OrderResponse]:
        """Cancel an existing order.

        Args:
            order_id: The order ID to cancel

        Returns:
            OrderResponse if successful, None otherwise
        """
        try:
            response = self._broker.cancel_order(order_id)
            self._order_logger.info(f"ORDER_CANCELLED | {order_id} | Status: {response.status}")
            return response
        except BrokerError as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return None

    def get_open_orders(self) -> list[OrderResponse]:
        """Get all open orders.

        Returns:
            List of open orders
        """
        all_orders = self._broker.get_order_book()
        return [
            order
            for order in all_orders
            if order.status
            in (OrderStatus.OPEN, OrderStatus.PARTIAL_FILL, OrderStatus.PENDING, OrderStatus.PLACED)
        ]

    def cancel_all_orders(self) -> list[OrderResponse]:
        """Cancel all open orders.

        Returns:
            List of cancellation responses
        """
        open_orders = self.get_open_orders()
        results = []

        for order in open_orders:
            result = self.cancel_order(order.order_id)
            if result:
                results.append(result)

        return results

    def exit_all_positions(self) -> list[OrderResponse]:
        """Exit all open positions by placing opposite orders.

        This method is used for emergency liquidation (kill switch).
        It places market orders to close all open positions immediately.
        All prices are in PAISA.

        Returns:
            List of order responses for the exit orders placed.
        """
        try:
            positions = self._broker.get_positions()
            results = []

            for position in positions:
                # Skip positions with zero quantity
                if position.quantity == 0:
                    continue

                # Determine exit side (opposite of current position)
                # Long position (quantity > 0) -> SELL to exit
                # Short position (quantity < 0) -> BUY to exit
                exit_side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
                exit_quantity = abs(position.quantity)

                # Create exit order as MARKET order for immediate execution
                exit_order = Order(
                    instrument_key=position.instrument_key,
                    side=exit_side,
                    order_type=OrderType.MARKET,
                    product_type=position.product_type,
                    quantity=exit_quantity,
                    price=None,  # Market order
                    trigger_price=None,
                    strategy_id=position.strategy_id or "kill_switch",
                    tag=f"exit_all|{position.strategy_id or 'unknown'}",
                )

                # Place the exit order with retry logic
                try:
                    response = self._place_order_with_retry(exit_order)
                    results.append(response)
                    self._order_logger.critical(
                        f"POSITION_EXITED | {response.order_id} | {exit_side.value} {exit_quantity} "
                        f"{position.instrument_key} | AvgPrice: {position.average_price} | "
                        f"PnL: {position.unrealized_pnl}"
                    )
                except BrokerError as e:
                    logger.error(f"Failed to exit position {position.instrument_key}: {e}")
                    self._order_logger.error(
                        f"EXIT_FAILED | {position.instrument_key} | {exit_side.value} {exit_quantity} | Error: {e}"
                    )

            logger.critical(
                f"Exit all positions completed: {len(results)}/{len(positions)} positions exited"
            )
            return results

        except BrokerError as e:
            logger.exception(f"Failed to get positions during exit_all_positions: {e}")
            return []

    def _calculate_research_size_multiplier(self, research_score: float) -> float:
        """Calculate position size multiplier based on dynamic research settings.

        Score scaling rules relative to settings:
        - Score >= score_for_full_size + 15: 1.25x size (high conviction boost)
        - Score >= score_for_full_size: 1.0x size (full size)
        - Score >= min_execute_score: 0.7x size (REDUCE_SIZE)
        - Score < min_execute_score: 0x (SKIP - trade should not execute)

        Args:
            research_score: Final research score (0-100).

        Returns:
            Size multiplier to apply to position.
        """
        from config.settings import get_settings
        settings = get_settings()

        if research_score >= settings.research_score_for_full_size + 15:
            return 1.25
        elif research_score >= settings.research_score_for_full_size:
            return 1.0
        elif research_score >= settings.research_min_score:
            return 0.7
        else:
            return 0.0

    def set_trade_analyzer(self, analyzer: "TradeAnalyzer") -> None:
        """Set or update the trade analyzer.

        Args:
            analyzer: TradeAnalyzer instance
        """
        self._trade_analyzer = analyzer
        self._research_enabled = True
        logger.info("Trade analyzer updated")

    def get_execution_stats(self) -> dict:
        """Get execution statistics.

        Returns:
            Dictionary with execution statistics
        """
        try:
            from src.persistence.models import OrderRecord, ResearchReport
            from sqlalchemy import func

            with self._SessionLocal() as session:
                # Total orders
                total_orders = session.query(func.count(OrderRecord.id)).scalar()

                # Orders by status
                status_counts = (
                    session.query(OrderRecord.status, func.count(OrderRecord.id))
                    .group_by(OrderRecord.status)
                    .all()
                )

                # Research stats
                research_stats = session.query(
                    func.avg(ResearchReport.final_score),
                    func.count(ResearchReport.id),
                ).first()

                return {
                    "total_orders": total_orders,
                    "status_breakdown": {status: count for status, count in status_counts},
                    "avg_research_score": round(research_stats[0], 2)
                    if research_stats[0]
                    else None,
                    "total_research_reports": research_stats[1],
                }
        except Exception as e:
            logger.error(f"Failed to get execution stats: {e}")
            return {}

    def get_entry_optimization_stats(self) -> dict[str, Any]:
        """Get entry optimization statistics.

        Returns:
            Dictionary with entry optimization statistics including:
            - total_attempts: Total number of optimization attempts
            - successful_optimizations: Number of successful optimizations
            - failed_optimizations: Number of failed optimizations
            - by_type: Breakdown by entry type (immediate, pullback, breakout, vwap)
        """
        with self._lock:
            stats = self._entry_opt_stats.copy()

        # Calculate success rate
        if stats["total_attempts"] > 0:
            stats["success_rate_pct"] = round(
                (stats["successful_optimizations"] / stats["total_attempts"]) * 100, 2
            )
        else:
            stats["success_rate_pct"] = 0.0

        return stats

    def reset_entry_optimization_stats(self) -> None:
        """Reset entry optimization statistics."""
        with self._lock:
            self._entry_opt_stats = {
                "total_attempts": 0,
                "successful_optimizations": 0,
                "failed_optimizations": 0,
                "by_type": {
                    "immediate": {"count": 0, "filled": 0},
                    "pullback": {"count": 0, "filled": 0},
                    "breakout": {"count": 0, "filled": 0},
                    "vwap": {"count": 0, "filled": 0},
                },
            }
        logger.info("Entry optimization statistics reset")
